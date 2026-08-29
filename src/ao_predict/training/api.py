"""Restart-safe AO Predict dense-regression training lifecycle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import secrets
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import numpy as np
import torch

from .artifacts import (
    MODEL_METADATA_KIND,
    MODEL_METADATA_VERSION,
    RECOVERY_KIND,
    RECOVERY_VERSION,
    _TrainingPaths,
    atomic_save_recovery,
    clean_temporary_paths,
    load_recovery,
    prepare_training_parent,
    producer_version,
    publish_model_package,
    remove_derived_outputs,
    resolve_training_paths,
    sha256_file,
    training_path_lock,
)
from .data import (
    _ExampleSet,
    _PreparedModelTrainingData,
    _SplitMembership,
    _StandardizationState,
    data_identity,
    explicit_membership,
    fit_standardization,
    prepare_model_training_data,
    resolve_automatic_split,
    schema_metadata,
    standardize_data,
    validate_explicit_schema,
    validation_mask_checksum,
)
from .model import build_dense_model, cpu_state_dict, derived_seed
from .types import (
    ModelTrainingValidationError,
    TrainingRecoveryMismatchError,
    TrainingTerminationReason,
    TrainingValidationRecord,
    TrainModelRequest,
    TrainModelResult,
)

_MAX_SEED = (1 << 63) - 1
_HISTORY_KEYS = frozenset(
    {
        "validation_check",
        "training_epochs",
        "optimizer_updates",
        "training_examples_seen",
        "training_objective",
        "validation_objective",
        "validation_error_percent",
        "learning_rate_before",
        "learning_rate_after",
    }
)
_RECOVERY_KEYS = frozenset(
    {
        "kind",
        "version",
        "producer_version",
        "run_identity",
        "run_identity_sha256",
        "runtime",
        "training_seed",
        "split_seed",
        "validation_mask",
        "standardization",
        "model_state_dict",
        "best_model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "batch_stream_state",
        "initialization_generator_state",
        "optimizer_updates",
        "training_examples_seen",
        "validation_checks",
        "validation_exposure",
        "interval_loss_total",
        "interval_loss_examples",
        "best_validation_check",
        "best_validation_objective",
        "best_model_validation_error_percent",
        "scheduler_has_reduced",
        "early_stopping_reference",
        "early_stopping_bad_checks",
        "history",
        "terminal",
        "termination_reason",
        "publication_state",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: object) -> bool:
    return _is_real(value) and math.isfinite(float(value))


def _validate_seed(value: object, label: str, issues: list[str]) -> None:
    if value is not None and (
        not _is_integer(value) or int(value) < 0 or int(value) > _MAX_SEED
    ):
        issues.append(f"{label} must be None or an integer in [0, {_MAX_SEED}].")


def _validate_request(request: TrainModelRequest) -> list[str]:
    issues: list[str] = []
    if not isinstance(request.model_path, (str, Path)):
        issues.append("model_path must be a string or pathlib.Path.")
    elif not os.fspath(request.model_path).strip():
        issues.append("model_path must not be empty.")
    if not isinstance(request.hidden_widths, tuple):
        issues.append("hidden_widths must be a tuple of positive integers.")
    elif not all(_is_integer(width) and width > 0 for width in request.hidden_widths):
        issues.append("hidden_widths must contain only positive integers.")
    if not _is_integer(request.batch_size) or request.batch_size <= 0:
        issues.append("batch_size must be a positive integer.")

    validation_sources = sum(
        value is not None
        for value in (
            request.validation_data,
            request.validation_count,
            request.validation_fraction,
        )
    )
    if validation_sources != 1:
        issues.append(
            "Exactly one of validation_data, validation_count, or validation_fraction is required."
        )
    if request.validation_count is not None and (
        not _is_integer(request.validation_count) or request.validation_count <= 0
    ):
        issues.append("validation_count must be a positive integer.")
    if request.validation_fraction is not None and (
        not _finite(request.validation_fraction)
        or not 0.0 < float(request.validation_fraction) < 1.0
    ):
        issues.append(
            "validation_fraction must be finite and strictly between zero and one."
        )
    if request.validation_data is not None and request.split_seed is not None:
        issues.append("split_seed is not accepted with explicit validation_data.")
    _validate_seed(request.split_seed, "split_seed", issues)
    _validate_seed(request.training_seed, "training_seed", issues)

    if not isinstance(request.overwrite, bool):
        issues.append("overwrite must be a Boolean value.")
    if not isinstance(request.device, str) or not request.device:
        issues.append("device must be a non-empty PyTorch device name.")
    if request.cpu_threads is not None and (
        not _is_integer(request.cpu_threads) or request.cpu_threads <= 0
    ):
        issues.append("cpu_threads must be None or a positive integer.")
    if request.validation_batch_size is not None and (
        not _is_integer(request.validation_batch_size)
        or request.validation_batch_size <= 0
    ):
        issues.append("validation_batch_size must be None or a positive integer.")

    positive_finite = {
        "base_learning_rate": request.base_learning_rate,
        "validation_check_epochs": request.validation_check_epochs,
    }
    for label, value in positive_finite.items():
        if not _finite(value) or float(value) <= 0.0:
            issues.append(f"{label} must be finite and positive.")
    if (
        _finite(request.validation_check_epochs)
        and float(request.validation_check_epochs) < 1.0
    ):
        issues.append("validation_check_epochs must be at least one.")
    nonnegative_finite = {
        "weight_decay": request.weight_decay,
        "minimum_learning_rate": request.minimum_learning_rate,
        "early_stopping_minimum_improvement_percent": request.early_stopping_minimum_improvement_percent,
    }
    for label, value in nonnegative_finite.items():
        if not _finite(value) or float(value) < 0.0:
            issues.append(f"{label} must be finite and non-negative.")
    if (
        _finite(request.minimum_learning_rate)
        and _finite(request.base_learning_rate)
        and float(request.minimum_learning_rate) > float(request.base_learning_rate)
    ):
        issues.append("minimum_learning_rate must not exceed base_learning_rate.")
    if not _finite(request.warmup_start_fraction) or not (
        0.0 < float(request.warmup_start_fraction) <= 1.0
    ):
        issues.append("warmup_start_fraction must be finite and in (0, 1].")
    if not _finite(request.learning_rate_reduction_factor) or not (
        0.0 < float(request.learning_rate_reduction_factor) < 1.0
    ):
        issues.append("learning_rate_reduction_factor must be finite and in (0, 1).")
    if not _finite(request.scheduler_minimum_improvement_fraction) or not (
        0.0 <= float(request.scheduler_minimum_improvement_fraction) < 1.0
    ):
        issues.append(
            "scheduler_minimum_improvement_fraction must be finite and in [0, 1)."
        )
    for label, value in {
        "warmup_epochs": request.warmup_epochs,
        "minimum_training_epochs": request.minimum_training_epochs,
    }.items():
        if not _is_integer(value) or value < 0:
            issues.append(f"{label} must be a non-negative integer.")
    for label, value in {
        "scheduler_patience_checks": request.scheduler_patience_checks,
        "early_stopping_patience_checks": request.early_stopping_patience_checks,
        "maximum_validation_checks": request.maximum_validation_checks,
    }.items():
        if not _is_integer(value) or value <= 0:
            issues.append(f"{label} must be a positive integer.")
    return issues


def _select_device(name: str, cpu_threads: int | None) -> torch.device:
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise ModelTrainingValidationError([f"device is invalid: {exc}"]) from exc
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ModelTrainingValidationError(["device type must be cpu, cuda, or mps."])
    if device.type == "cpu":
        if device.index is not None:
            raise ModelTrainingValidationError(
                ["CPU devices must not include an index."]
            )
    elif cpu_threads is not None:
        raise ModelTrainingValidationError(
            ["cpu_threads is accepted only with a CPU device."]
        )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ModelTrainingValidationError(
                [f"CUDA device {name!r} is unavailable."]
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ModelTrainingValidationError(
                [f"CUDA device index {index} is unavailable."]
            )
        device = torch.device("cuda", index)
    if device.type == "mps":
        if device.index is not None:
            raise ModelTrainingValidationError(
                ["MPS devices must not include an index."]
            )
        if not torch.backends.mps.is_available():
            raise ModelTrainingValidationError(["MPS device is unavailable."])
    return device


@contextmanager
def _deterministic_runtime() -> Iterator[None]:
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


def _runtime_fingerprint(
    device: torch.device,
    validation_batch_size: int,
) -> dict[str, object]:
    device_identity: dict[str, object] = {
        "type": device.type,
        "index": device.index,
    }
    if device.type == "cuda":
        assert device.index is not None
        device_identity.update(
            {
                "name": torch.cuda.get_device_name(device.index),
                "capability": list(torch.cuda.get_device_capability(device.index)),
            }
        )
    elif device.type == "mps":
        device_identity["platform"] = platform.platform()
    else:
        device_identity["machine"] = platform.machine()
    return {
        "ao_predict_version": producer_version(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "device": device_identity,
        "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
        "validation_batch_size": validation_batch_size,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


class _ShuffledBatchStream:
    """Private epoch-shuffled example stream with exact continuation state."""

    def __init__(self, example_count: int, batch_size: int, seed: int) -> None:
        self.example_count = example_count
        self.batch_size = batch_size
        self.generator = np.random.Generator(np.random.PCG64(seed))
        self.order = self.generator.permutation(example_count).astype(np.int64)
        self.cursor = 0
        self.epoch = 1

    def next(self) -> np.ndarray:
        if self.cursor >= self.example_count:
            self.order = self.generator.permutation(self.example_count).astype(np.int64)
            self.cursor = 0
            self.epoch += 1
        stop = min(self.cursor + self.batch_size, self.example_count)
        indexes = self.order[self.cursor : stop]
        self.cursor = stop
        return indexes

    def state_dict(self) -> dict[str, object]:
        return {
            "example_count": self.example_count,
            "batch_size": self.batch_size,
            "order": torch.from_numpy(self.order.copy()),
            "cursor": self.cursor,
            "epoch": self.epoch,
            "random_state": self.generator.bit_generator.state,
        }

    def load_state_dict(self, value: object) -> None:
        if not isinstance(value, dict):
            # This is malformed persisted state, rather than a caller type error.
            raise ValueError(  # noqa: TRY004
                "Recovery batch stream state must be a mapping."
            )
        if value.get("example_count") != self.example_count:
            raise ValueError("Recovery batch stream example count differs.")
        if value.get("batch_size") != self.batch_size:
            raise ValueError("Recovery batch stream batch size differs.")
        order = value.get("order")
        cursor = value.get("cursor")
        epoch = value.get("epoch")
        random_state = value.get("random_state")
        if not isinstance(order, torch.Tensor) or order.shape != (self.example_count,):
            raise ValueError("Recovery batch order is invalid.")
        order_array = order.cpu().numpy().astype(np.int64, copy=True)
        if not np.array_equal(np.sort(order_array), np.arange(self.example_count)):
            raise ValueError("Recovery batch order is not a complete permutation.")
        if not _is_integer(cursor) or not 0 <= cursor <= self.example_count:
            raise ValueError("Recovery batch cursor is invalid.")
        if not _is_integer(epoch) or epoch < 1:
            raise ValueError("Recovery batch epoch is invalid.")
        self.generator.bit_generator.state = random_state
        self.order = order_array
        self.cursor = cursor
        self.epoch = epoch


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _build_run_identity(
    request: TrainModelRequest,
    training_data: _PreparedModelTrainingData,
    validation_data: _PreparedModelTrainingData,
    membership: _SplitMembership,
    *,
    training_seed: int,
    validation_batch_size: int,
    training_example_count: int,
    warmup_updates: int,
    minimum_training_updates: int,
    validation_exposure: int,
) -> dict[str, object]:
    automatic = membership.validation_mask is not None
    return {
        "model": {
            "hidden_widths": list(request.hidden_widths),
            "input_width": len(training_data.config.features),
            "output_width": len(training_data.config.targets),
            "initialization": "pytorch_linear_fan_in_uniform_v1",
        },
        "data": {
            "training": data_identity(training_data),
            "validation": data_identity(validation_data),
            "partition_mode": "automatic" if automatic else "explicit",
            "validation_count": request.validation_count,
            "validation_fraction": request.validation_fraction,
            "split_seed": membership.split_seed,
            "validation_mask_sha256": validation_mask_checksum(
                membership.validation_mask
            ),
            "training_simulation_count": int(membership.training_simulations.size),
            "validation_simulation_count": int(membership.validation_simulations.size),
            "training_example_count": training_example_count,
            "validation_example_count": int(
                membership.validation_simulations.size
                * validation_data.examples_per_simulation
            ),
        },
        "training": {
            "training_seed": training_seed,
            "batch_size": request.batch_size,
            "validation_batch_size": validation_batch_size,
            "base_learning_rate": float(request.base_learning_rate),
            "weight_decay": float(request.weight_decay),
            "warmup_epochs": request.warmup_epochs,
            "warmup_start_fraction": float(request.warmup_start_fraction),
            "warmup_updates": warmup_updates,
            "minimum_training_epochs": request.minimum_training_epochs,
            "minimum_training_updates": minimum_training_updates,
            "validation_check_epochs": float(request.validation_check_epochs),
            "validation_exposure": validation_exposure,
            "learning_rate_reduction_factor": float(
                request.learning_rate_reduction_factor
            ),
            "minimum_learning_rate": float(request.minimum_learning_rate),
            "scheduler_patience_checks": request.scheduler_patience_checks,
            "scheduler_minimum_improvement_fraction": float(
                request.scheduler_minimum_improvement_fraction
            ),
            "early_stopping_patience_checks": request.early_stopping_patience_checks,
            "early_stopping_minimum_improvement_percent": float(
                request.early_stopping_minimum_improvement_percent
            ),
            "maximum_validation_checks": request.maximum_validation_checks,
            "training_example_count": training_example_count,
            "optimizer": {
                "kind": "adam",
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "amsgrad": False,
                "maximize": False,
                "decoupled_weight_decay": False,
            },
        },
    }


def _history_mapping(record: TrainingValidationRecord) -> dict[str, object]:
    return asdict(record)


def _history_record(value: object) -> TrainingValidationRecord:
    if not isinstance(value, dict):
        # This is malformed persisted state, rather than a caller type error.
        raise ValueError(  # noqa: TRY004
            "Recovery validation history entries must be mappings."
        )
    if set(value) != _HISTORY_KEYS:
        raise ValueError("Recovery validation history fields are invalid.")
    return TrainingValidationRecord(
        validation_check=int(value["validation_check"]),
        training_epochs=float(value["training_epochs"]),
        optimizer_updates=int(value["optimizer_updates"]),
        training_examples_seen=int(value["training_examples_seen"]),
        training_objective=float(value["training_objective"]),
        validation_objective=float(value["validation_objective"]),
        validation_error_percent=float(value["validation_error_percent"]),
        learning_rate_before=float(value["learning_rate_before"]),
        learning_rate_after=float(value["learning_rate_after"]),
    )


def _write_log(handle: TextIO, line: str = "") -> None:
    handle.write(line + "\n")
    handle.flush()


def _write_initial_log(
    handle: TextIO,
    paths: _TrainingPaths,
    identity: Mapping[str, object],
    runtime: Mapping[str, object],
) -> None:
    _write_log(handle, "AO Predict model training")
    _write_log(handle, f"started: {_utc_now()}")
    _write_log(handle, f"model_path: {paths.model_path}")
    _write_log(handle, f"producer_version: {runtime['ao_predict_version']}")
    _write_log(handle, f"run_identity_sha256: {_canonical_sha256(identity)}")
    _write_log(handle, "model:")
    for key, value in sorted(identity["model"].items()):
        _write_log(handle, f"  {key}: {value}")
    _write_log(handle, "configuration:")
    for key, value in sorted(identity["training"].items()):
        _write_log(handle, f"  {key}: {value}")
    _write_log(handle, "data:")
    data = identity["data"]
    assert isinstance(data, Mapping)
    for key in (
        "partition_mode",
        "split_seed",
        "validation_mask_sha256",
        "training_simulation_count",
        "validation_simulation_count",
        "training_example_count",
        "validation_example_count",
    ):
        _write_log(handle, f"  {key}: {data.get(key)}")
    training_data = data["training"]
    validation_data = data["validation"]
    assert isinstance(training_data, Mapping)
    assert isinstance(validation_data, Mapping)
    _write_log(handle, f"  training_data_sha256: {training_data['checksum']}")
    _write_log(handle, f"  validation_data_sha256: {validation_data['checksum']}")
    _write_log(handle, "runtime:")
    for key, value in sorted(runtime.items()):
        _write_log(handle, f"  {key}: {value}")


def _model_metadata(
    request: TrainModelRequest,
    prepared: _PreparedModelTrainingData,
    standardization: _StandardizationState,
    training_seed: int,
) -> dict[str, object]:
    features, targets = schema_metadata(prepared, standardization)
    return {
        "kind": MODEL_METADATA_KIND,
        "version": MODEL_METADATA_VERSION,
        "producer_version": producer_version(),
        "model": {
            "input_width": len(features),
            "hidden_widths": list(request.hidden_widths),
            "output_width": len(targets),
            "hidden_activation": "relu",
            "output_activation": "linear",
            "bias": True,
        },
        "features": features,
        "targets": targets,
        "numerical": {
            "model_dtype": "float32",
            "prediction_dtype": "float32",
            "standardization_variance": "population",
            "constant_scale": 1.0,
            "objective": "physical_relative_mse",
        },
        "training_seed": training_seed,
    }


def _relative_residuals(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_means: torch.Tensor,
    target_scales: torch.Tensor,
) -> torch.Tensor:
    physical_predictions = predictions * target_scales + target_means
    physical_targets = targets * target_scales + target_means
    if torch.any(physical_targets <= 0):
        raise ValueError("Physical training targets must remain strictly positive.")
    return (physical_predictions - physical_targets) / physical_targets


def _complete_validation(
    model: torch.nn.Module,
    examples: _ExampleSet,
    batch_size: int,
    device: torch.device,
    target_means: torch.Tensor,
    target_scales: torch.Tensor,
) -> tuple[float, float]:
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            stop = min(start + batch_size, len(examples))
            features, targets = examples.gather(np.arange(start, stop, dtype=np.int64))
            feature_tensor = torch.from_numpy(features).to(device)
            target_tensor = torch.from_numpy(targets).to(device)
            residuals = _relative_residuals(
                model(feature_tensor),
                target_tensor,
                target_means,
                target_scales,
            )
            total += float(torch.sum(torch.square(residuals)).item())
            count += residuals.numel()
    objective = total / count
    if not math.isfinite(objective):
        raise ValueError("The validation objective must be finite.")
    return objective, math.sqrt(objective) * 100.0


def _warmup_learning_rate(request: TrainModelRequest, update: int, total: int) -> float:
    if total < 1 or update >= total:
        return float(request.base_learning_rate)
    fraction = (update - 1) / (total - 1)
    scale = (
        float(request.warmup_start_fraction)
        + (1.0 - float(request.warmup_start_fraction)) * fraction
    )
    return float(request.base_learning_rate) * scale


def _recovery_mapping(
    *,
    identity: dict[str, object],
    runtime: dict[str, object],
    training_seed: int,
    membership: _SplitMembership,
    standardization: _StandardizationState,
    model: torch.nn.Module,
    best_model_state: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    stream: _ShuffledBatchStream,
    initialization_generator_state: torch.Tensor,
    optimizer_updates: int,
    training_examples_seen: int,
    validation_checks: int,
    validation_exposure: int,
    interval_loss_total: float,
    interval_loss_examples: int,
    best_validation_check: int,
    best_validation_objective: float,
    best_model_validation_error_percent: float,
    scheduler_has_reduced: bool,
    early_stopping_reference: float | None,
    early_stopping_bad_checks: int,
    history: list[TrainingValidationRecord],
    termination_reason: TrainingTerminationReason | None,
) -> dict[str, object]:
    terminal = termination_reason is not None
    return {
        "kind": RECOVERY_KIND,
        "version": RECOVERY_VERSION,
        "producer_version": producer_version(),
        "run_identity": identity,
        "run_identity_sha256": _canonical_sha256(identity),
        "runtime": runtime,
        "training_seed": training_seed,
        "split_seed": membership.split_seed,
        "validation_mask": (
            None
            if membership.validation_mask is None
            else torch.from_numpy(np.asarray(membership.validation_mask).copy())
        ),
        "standardization": standardization.as_mapping(),
        "model_state_dict": cpu_state_dict(model),
        "best_model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in best_model_state.items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "batch_stream_state": stream.state_dict(),
        "initialization_generator_state": initialization_generator_state.cpu().clone(),
        "optimizer_updates": optimizer_updates,
        "training_examples_seen": training_examples_seen,
        "validation_checks": validation_checks,
        "validation_exposure": validation_exposure,
        "interval_loss_total": interval_loss_total,
        "interval_loss_examples": interval_loss_examples,
        "best_validation_check": best_validation_check,
        "best_validation_objective": best_validation_objective,
        "best_model_validation_error_percent": best_model_validation_error_percent,
        "scheduler_has_reduced": scheduler_has_reduced,
        "early_stopping_reference": early_stopping_reference,
        "early_stopping_bad_checks": early_stopping_bad_checks,
        "history": [_history_mapping(record) for record in history],
        "terminal": terminal,
        "termination_reason": None
        if termination_reason is None
        else termination_reason.value,
        "publication_state": "terminal" if terminal else "fitting",
    }


def _validate_recovery_header(value: dict[str, object]) -> None:
    if set(value) != _RECOVERY_KEYS:
        missing = sorted(_RECOVERY_KEYS - set(value))
        extra = sorted(set(value) - _RECOVERY_KEYS)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError("Recovery fields are invalid (" + "; ".join(details) + ").")
    if value.get("kind") != RECOVERY_KIND:
        raise ValueError(f"Unsupported recovery kind: {value.get('kind')!r}.")
    if value.get("version") != RECOVERY_VERSION:
        raise ValueError(f"Unsupported recovery version: {value.get('version')!r}.")
    producer = value.get("producer_version")
    if not isinstance(producer, str) or not producer.strip():
        raise ValueError("Recovery producer_version must be a non-empty string.")
    identity = value.get("run_identity")
    if not isinstance(identity, dict):
        # This is malformed persisted state, rather than a caller type error.
        raise ValueError(  # noqa: TRY004
            "Recovery run identity must be a mapping."
        )
    if value.get("run_identity_sha256") != _canonical_sha256(identity):
        raise ValueError("Recovery run identity checksum is invalid.")
    runtime = value.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(  # noqa: TRY004
            "Recovery runtime identity must be a mapping."
        )
    if runtime.get("ao_predict_version") != producer:
        raise ValueError("Recovery producer and runtime AO Predict versions differ.")
    _validate_recovery_lifecycle_state(value)


def _validate_recovery_lifecycle_state(value: dict[str, object]) -> None:
    """Validate constrained scalar and collection state before restoration."""

    training_seed = value.get("training_seed")
    split_seed = value.get("split_seed")
    if not _is_integer(training_seed) or not 0 <= int(training_seed) <= _MAX_SEED:
        raise ValueError("Recovery training_seed is invalid.")
    if split_seed is not None and (
        not _is_integer(split_seed) or not 0 <= int(split_seed) <= _MAX_SEED
    ):
        raise ValueError("Recovery split_seed is invalid.")
    mask = value.get("validation_mask")
    if mask is not None and (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.ndim != 1
        or mask.device.type != "cpu"
    ):
        raise ValueError("Recovery validation_mask is invalid.")
    for key in (
        "model_state_dict",
        "best_model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "batch_stream_state",
        "standardization",
    ):
        if not isinstance(value.get(key), dict):
            raise ValueError(  # noqa: TRY004
                f"Recovery {key} must be a mapping."
            )
    generator_state = value.get("initialization_generator_state")
    if (
        not isinstance(generator_state, torch.Tensor)
        or generator_state.dtype != torch.uint8
        or generator_state.ndim != 1
        or generator_state.device.type != "cpu"
    ):
        raise ValueError("Recovery initialization generator state is invalid.")

    integer_minimums = {
        "optimizer_updates": 1,
        "training_examples_seen": 1,
        "validation_checks": 1,
        "validation_exposure": 0,
        "interval_loss_examples": 0,
        "best_validation_check": 1,
        "early_stopping_bad_checks": 0,
    }
    for key, minimum in integer_minimums.items():
        item = value.get(key)
        if not _is_integer(item) or int(item) < minimum:
            raise ValueError(f"Recovery {key} is invalid.")
    if value["validation_exposure"] != 0 or value["interval_loss_examples"] != 0:
        raise ValueError("Recovery must describe a completed validation boundary.")
    if value["best_validation_check"] > value["validation_checks"]:
        raise ValueError("Recovery best_validation_check is out of range.")
    if (
        not _finite(value.get("interval_loss_total"))
        or float(value["interval_loss_total"]) != 0.0
    ):
        raise ValueError("Recovery interval_loss_total is invalid.")
    for key in (
        "best_validation_objective",
        "best_model_validation_error_percent",
    ):
        if not _finite(value.get(key)) or float(value[key]) < 0.0:
            raise ValueError(f"Recovery {key} is invalid.")
    if not isinstance(value.get("scheduler_has_reduced"), bool):
        raise ValueError(  # noqa: TRY004
            "Recovery scheduler_has_reduced is invalid."
        )
    reference = value.get("early_stopping_reference")
    if reference is not None and (not _finite(reference) or float(reference) < 0.0):
        raise ValueError("Recovery early_stopping_reference is invalid.")

    history = value.get("history")
    if not isinstance(history, list) or len(history) != value["validation_checks"]:
        raise ValueError("Recovery history length is invalid.")
    records = [_history_record(item) for item in history]
    for index, record in enumerate(records, start=1):
        if record.validation_check != index:
            raise ValueError("Recovery history validation checks are not sequential.")
        numerical = (
            record.training_epochs,
            record.training_objective,
            record.validation_objective,
            record.validation_error_percent,
            record.learning_rate_before,
            record.learning_rate_after,
        )
        if not all(math.isfinite(item) and item >= 0.0 for item in numerical):
            raise ValueError("Recovery history contains invalid numerical values.")
        if record.optimizer_updates < 1 or record.training_examples_seen < 1:
            raise ValueError("Recovery history contains invalid progress counts.")
    best = records[int(value["best_validation_check"]) - 1]
    if (
        best.validation_objective != value["best_validation_objective"]
        or best.validation_error_percent != value["best_model_validation_error_percent"]
        or best.validation_objective
        != min(record.validation_objective for record in records)
    ):
        raise ValueError("Recovery best-model measurements are inconsistent.")

    terminal = value.get("terminal")
    reason = value.get("termination_reason")
    publication = value.get("publication_state")
    if not isinstance(terminal, bool):
        raise ValueError("Recovery terminal flag is invalid.")  # noqa: TRY004
    if terminal:
        try:
            TrainingTerminationReason(reason)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Terminal recovery termination reason is invalid."
            ) from exc
        if publication != "terminal":
            raise ValueError("Terminal recovery publication state is invalid.")
    elif reason is not None or publication != "fitting":
        raise ValueError("Resumable recovery publication state is invalid.")


def _recovery_mismatches(
    recovery: dict[str, object],
    identity: dict[str, object],
    runtime: dict[str, object],
) -> list[str]:
    mismatches: list[str] = []
    retained_identity = recovery.get("run_identity")
    if not isinstance(retained_identity, dict):
        return ["retained run identity is invalid"]
    for key in ("model", "data", "training"):
        if retained_identity.get(key) != identity.get(key):
            mismatches.append(f"{key} identity differs")
    retained_runtime = recovery.get("runtime")
    if retained_runtime != runtime:
        if isinstance(retained_runtime, dict):
            all_keys = sorted(set(retained_runtime) | set(runtime))
            for key in all_keys:
                if retained_runtime.get(key) != runtime.get(key):
                    mismatches.append(f"runtime {key} differs")
        else:
            mismatches.append("runtime identity is invalid")
    return mismatches


def _build_result(
    paths: _TrainingPaths,
    termination_reason: TrainingTerminationReason,
    training_seed: int,
    membership: _SplitMembership,
    optimizer_updates: int,
    training_examples_seen: int,
    validation_checks: int,
    best_validation_check: int,
    best_validation_objective: float,
    best_model_validation_error_percent: float,
    history: list[TrainingValidationRecord],
) -> TrainModelResult:
    validation_mask = membership.validation_mask
    if validation_mask is not None:
        validation_mask = validation_mask.copy()
        validation_mask.setflags(write=False)
    return TrainModelResult(
        model_path=paths.model_path,
        termination_reason=termination_reason,
        training_seed=training_seed,
        split_seed=membership.split_seed,
        validation_mask=validation_mask,
        optimizer_updates=optimizer_updates,
        training_examples_seen=training_examples_seen,
        validation_checks=validation_checks,
        best_validation_check=best_validation_check,
        best_validation_objective=best_validation_objective,
        best_model_validation_error_percent=best_model_validation_error_percent,
        validation_history=tuple(history),
    )


def train_model(request: TrainModelRequest) -> TrainModelResult:
    """Train, exactly continue, and publish one AO Predict dense model.

    The operation validates all coupled inputs before creating artifacts,
    partitions complete simulations, owns fitted standardization, and writes
    ``<model_path>.model.zip``, ``<model_path>.training.log``, and transient
    exact recovery. A compatible recovery checkpoint is continued
    automatically. Successful return occurs only after the package validates,
    the log is finalized, and recovery has been removed.

    Args:
        request: Complete data, model, runtime, and lifecycle configuration.

    Returns:
        Final progress, best-model measurements, split details, and validation
        history for the complete logical run.

    Raises:
        ModelTrainingValidationError: If coupled request or data validation
            fails.
        TrainingRecoveryMismatchError: If retained recovery cannot continue
            the requested run exactly.
        FileExistsError: If stable derived outputs collide with a fresh run.
        RuntimeError: If another operation holds the private model-path lock.
        OSError: If filesystem preparation, logging, recovery, or publication
            fails.
    """

    if not isinstance(request, TrainModelRequest):
        raise TypeError("request must be a TrainModelRequest.")
    issues = _validate_request(request)
    prepared_training = prepare_model_training_data(
        request.training_data,
        label="training_data",
        issues=issues,
    )
    prepared_validation = None
    if request.validation_data is not None:
        prepared_validation = prepare_model_training_data(
            request.validation_data,
            label="validation_data",
            issues=issues,
        )
    if prepared_training is not None and prepared_validation is not None:
        validate_explicit_schema(prepared_training, prepared_validation, issues)
    if prepared_training is not None and request.validation_data is None:
        if (
            request.validation_count is not None
            and request.validation_count >= prepared_training.simulation_count
        ):
            issues.append(
                "validation_count must leave at least one training simulation."
            )
        if request.validation_fraction is not None:
            resolved_count = math.ceil(
                float(request.validation_fraction) * prepared_training.simulation_count
            )
            if resolved_count >= prepared_training.simulation_count:
                issues.append(
                    "validation_fraction must leave at least one simulation in each partition."
                )
    if issues:
        raise ModelTrainingValidationError(issues)
    assert prepared_training is not None

    try:
        paths = resolve_training_paths(request.model_path)
    except ValueError as exc:
        raise ModelTrainingValidationError([str(exc)]) from exc
    validation_batch_size = request.validation_batch_size or 2 * request.batch_size
    device = _select_device(request.device, request.cpu_threads)
    prepare_training_parent(paths)

    with training_path_lock(paths), _deterministic_runtime():
        clean_temporary_paths(paths)
        if request.overwrite:
            remove_derived_outputs(paths)
        recovery: dict[str, object] | None = None
        if paths.recovery_path.exists():
            if not paths.log_path.exists():
                raise TrainingRecoveryMismatchError(
                    ["recovery exists without its append-only training log"]
                )
            try:
                recovery = load_recovery(paths)
                _validate_recovery_header(recovery)
            except Exception as exc:
                raise TrainingRecoveryMismatchError(
                    [f"recovery checkpoint is invalid: {exc}"]
                ) from exc
        elif paths.package_path.exists() or paths.log_path.exists():
            existing = [
                str(path)
                for path in (paths.package_path, paths.log_path)
                if path.exists()
            ]
            raise FileExistsError(
                "Refusing to start training with existing derived outputs: "
                + ", ".join(existing)
            )
        if device.type == "cpu" and request.cpu_threads is not None:
            # PyTorch owns this as process-wide state; the public contract makes
            # that scope explicit and intentionally does not restore it.
            torch.set_num_threads(request.cpu_threads)

        if recovery is None:
            training_seed = (
                request.training_seed
                if request.training_seed is not None
                else secrets.randbits(63)
            )
            if request.validation_data is None:
                membership = resolve_automatic_split(
                    prepared_training,
                    validation_count=request.validation_count,
                    validation_fraction=request.validation_fraction,
                    split_seed=request.split_seed,
                )
                prepared_validation = prepared_training
            else:
                assert prepared_validation is not None
                membership = explicit_membership(prepared_training, prepared_validation)
            standardization = fit_standardization(
                prepared_training,
                membership.training_simulations,
            )
        else:
            retained_training_seed = recovery.get("training_seed")
            retained_split_seed = recovery.get("split_seed")
            mismatch_messages: list[str] = []
            if not _is_integer(retained_training_seed):
                mismatch_messages.append("retained training_seed is invalid")
            elif (
                request.training_seed is not None
                and request.training_seed != retained_training_seed
            ):
                mismatch_messages.append("training_seed differs")
            if request.validation_data is None:
                if not _is_integer(retained_split_seed):
                    mismatch_messages.append("retained split_seed is invalid")
                elif (
                    request.split_seed is not None
                    and request.split_seed != retained_split_seed
                ):
                    mismatch_messages.append("split_seed differs")
            elif retained_split_seed is not None:
                mismatch_messages.append(
                    "explicit validation recovery has a split_seed"
                )
            if mismatch_messages:
                raise TrainingRecoveryMismatchError(mismatch_messages)
            training_seed = int(retained_training_seed)
            if request.validation_data is None:
                mask_tensor = recovery.get("validation_mask")
                if not isinstance(mask_tensor, torch.Tensor):
                    raise TrainingRecoveryMismatchError(
                        ["automatic recovery validation mask is invalid"]
                    )
                try:
                    membership = resolve_automatic_split(
                        prepared_training,
                        validation_count=request.validation_count,
                        validation_fraction=request.validation_fraction,
                        split_seed=int(retained_split_seed),
                        validation_mask=mask_tensor.cpu().numpy(),
                    )
                except ValueError as exc:
                    raise TrainingRecoveryMismatchError([str(exc)]) from exc
                prepared_validation = prepared_training
            else:
                assert prepared_validation is not None
                if recovery.get("validation_mask") is not None:
                    raise TrainingRecoveryMismatchError(
                        ["explicit validation recovery has a validation mask"]
                    )
                membership = explicit_membership(prepared_training, prepared_validation)
            retained_standardization = recovery.get("standardization")
            if not isinstance(retained_standardization, dict):
                raise TrainingRecoveryMismatchError(
                    ["retained standardization state is invalid"]
                )
            try:
                standardization = _StandardizationState.from_mapping(
                    retained_standardization,
                    feature_count=len(prepared_training.config.features),
                    target_count=len(prepared_training.config.targets),
                )
            except ValueError as exc:
                raise TrainingRecoveryMismatchError([str(exc)]) from exc

        assert prepared_validation is not None
        standardized_training = standardize_data(prepared_training, standardization)
        if prepared_validation is prepared_training:
            standardized_validation = standardized_training
        else:
            standardized_validation = standardize_data(
                prepared_validation,
                standardization,
            )
        training_examples = _ExampleSet(
            standardized_training,
            membership.training_simulations,
        )
        validation_examples = _ExampleSet(
            standardized_validation,
            membership.validation_simulations,
        )
        training_example_count = len(training_examples)
        updates_per_epoch = math.ceil(training_example_count / request.batch_size)
        warmup_updates = request.warmup_epochs * updates_per_epoch
        minimum_training_updates = request.minimum_training_epochs * updates_per_epoch
        validation_exposure_threshold = math.ceil(
            float(request.validation_check_epochs) * training_example_count
        )
        identity = _build_run_identity(
            request,
            prepared_training,
            prepared_validation,
            membership,
            training_seed=training_seed,
            validation_batch_size=validation_batch_size,
            training_example_count=training_example_count,
            warmup_updates=warmup_updates,
            minimum_training_updates=minimum_training_updates,
            validation_exposure=validation_exposure_threshold,
        )
        runtime = _runtime_fingerprint(device, validation_batch_size)
        if recovery is not None:
            mismatches = _recovery_mismatches(recovery, identity, runtime)
            if mismatches:
                raise TrainingRecoveryMismatchError(mismatches)

        initialization_seed = derived_seed(training_seed, "model-initialization")
        model, initialization_generator_state = build_dense_model(
            len(prepared_training.config.features),
            request.hidden_widths,
            len(prepared_training.config.targets),
            initialization_seed=initialization_seed,
        )
        model = model.to(device)
        foreach = device.type == "cuda"
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(request.base_learning_rate),
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=float(request.weight_decay),
            amsgrad=False,
            foreach=foreach,
            maximize=False,
            fused=False,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(request.learning_rate_reduction_factor),
            patience=request.scheduler_patience_checks - 1,
            threshold=float(request.scheduler_minimum_improvement_fraction),
            threshold_mode="rel",
            min_lr=float(request.minimum_learning_rate),
            eps=0.0,
        )
        stream = _ShuffledBatchStream(
            training_example_count,
            request.batch_size,
            derived_seed(training_seed, "batch-shuffling"),
        )
        target_means = torch.tensor(
            standardization.target_means,
            dtype=torch.float32,
            device=device,
        )
        target_scales = torch.tensor(
            standardization.target_scales,
            dtype=torch.float32,
            device=device,
        )

        optimizer_updates = 0
        training_examples_seen = 0
        validation_checks = 0
        validation_exposure = 0
        interval_loss_total = 0.0
        interval_loss_examples = 0
        best_validation_check = 0
        best_validation_objective = float("inf")
        best_model_validation_error_percent = float("inf")
        best_model_state: dict[str, torch.Tensor] = {}
        scheduler_has_reduced = False
        early_stopping_reference: float | None = None
        early_stopping_bad_checks = 0
        history: list[TrainingValidationRecord] = []
        termination_reason: TrainingTerminationReason | None = None

        if recovery is not None:
            try:
                model.load_state_dict(recovery["model_state_dict"], strict=True)
                best_state = recovery["best_model_state_dict"]
                if not isinstance(best_state, dict):
                    # This is malformed persisted state, not a caller type error.
                    raise ValueError(  # noqa: TRY004
                        "Recovery best model state is invalid."
                    )
                best_model_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in best_state.items()
                    if isinstance(name, str) and isinstance(tensor, torch.Tensor)
                }
                if len(best_model_state) != len(best_state):
                    raise ValueError(
                        "Recovery best model state contains invalid values."
                    )
                optimizer.load_state_dict(recovery["optimizer_state_dict"])
                scheduler.load_state_dict(recovery["scheduler_state_dict"])
                stream.load_state_dict(recovery["batch_stream_state"])
                retained_generator_state = recovery["initialization_generator_state"]
                if not isinstance(
                    retained_generator_state, torch.Tensor
                ) or not torch.equal(
                    retained_generator_state,
                    initialization_generator_state,
                ):
                    raise ValueError("Recovery initialization stream state differs.")
                optimizer_updates = int(recovery["optimizer_updates"])
                training_examples_seen = int(recovery["training_examples_seen"])
                validation_checks = int(recovery["validation_checks"])
                validation_exposure = int(recovery["validation_exposure"])
                interval_loss_total = float(recovery["interval_loss_total"])
                interval_loss_examples = int(recovery["interval_loss_examples"])
                best_validation_check = int(recovery["best_validation_check"])
                best_validation_objective = float(recovery["best_validation_objective"])
                best_model_validation_error_percent = float(
                    recovery["best_model_validation_error_percent"]
                )
                scheduler_has_reduced = bool(recovery["scheduler_has_reduced"])
                reference = recovery["early_stopping_reference"]
                early_stopping_reference = (
                    None if reference is None else float(reference)
                )
                early_stopping_bad_checks = int(recovery["early_stopping_bad_checks"])
                raw_history = recovery["history"]
                if not isinstance(raw_history, list):
                    # This is malformed persisted state, not a caller type error.
                    raise ValueError(  # noqa: TRY004
                        "Recovery history is invalid."
                    )
                history = [_history_record(value) for value in raw_history]
                if bool(recovery["terminal"]):
                    termination_reason = TrainingTerminationReason(
                        str(recovery["termination_reason"])
                    )
                elif recovery["termination_reason"] is not None:
                    raise ValueError("Resumable recovery has a termination reason.")
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                raise TrainingRecoveryMismatchError(
                    [f"recovery lifecycle state is invalid: {exc}"]
                ) from exc

        log_mode = "a" if recovery is not None else "x"
        with paths.log_path.open(log_mode, encoding="utf-8") as log:
            if recovery is None:
                _write_initial_log(log, paths, identity, runtime)
            else:
                _write_log(log)
                _write_log(log, f"continuation: {_utc_now()}")
                _write_log(log, f"resuming_after_validation_check: {validation_checks}")
            try:
                while termination_reason is None:
                    update = optimizer_updates + 1
                    if update <= warmup_updates:
                        learning_rate = _warmup_learning_rate(
                            request,
                            update,
                            warmup_updates,
                        )
                        for group in optimizer.param_groups:
                            group["lr"] = learning_rate

                    batch_indexes = stream.next()
                    features, targets = training_examples.gather(batch_indexes)
                    feature_tensor = torch.from_numpy(features).to(device)
                    target_tensor = torch.from_numpy(targets).to(device)
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    residuals = _relative_residuals(
                        model(feature_tensor),
                        target_tensor,
                        target_means,
                        target_scales,
                    )
                    loss = torch.mean(torch.square(residuals))
                    if loss.ndim != 0 or not torch.isfinite(loss):
                        raise ValueError(
                            "The training objective must be a finite scalar."
                        )
                    loss.backward()
                    optimizer.step()

                    batch_examples = int(batch_indexes.size)
                    optimizer_updates = update
                    training_examples_seen += batch_examples
                    validation_exposure += batch_examples
                    interval_loss_total += float(loss.item()) * batch_examples
                    interval_loss_examples += batch_examples

                    if validation_exposure < validation_exposure_threshold:
                        continue
                    validation_checks += 1
                    training_objective = interval_loss_total / interval_loss_examples
                    validation_objective, validation_error_percent = (
                        _complete_validation(
                            model,
                            validation_examples,
                            validation_batch_size,
                            device,
                            target_means,
                            target_scales,
                        )
                    )
                    if validation_objective < best_validation_objective:
                        best_validation_objective = validation_objective
                        best_model_validation_error_percent = validation_error_percent
                        best_validation_check = validation_checks
                        best_model_state = cpu_state_dict(model)

                    learning_rate_before = float(optimizer.param_groups[0]["lr"])
                    scheduler_eligible = optimizer_updates >= warmup_updates
                    if scheduler_eligible:
                        scheduler.step(validation_objective)
                    learning_rate_after = float(optimizer.param_groups[0]["lr"])
                    if learning_rate_after < learning_rate_before:
                        scheduler_has_reduced = True

                    early_stopping_eligible = (
                        optimizer_updates >= minimum_training_updates
                        and scheduler_has_reduced
                    )
                    early_stopping_triggered = False
                    if early_stopping_eligible:
                        if early_stopping_reference is None:
                            early_stopping_reference = validation_error_percent
                            early_stopping_bad_checks = 0
                        else:
                            decrease = (
                                early_stopping_reference - validation_error_percent
                            )
                            threshold = float(
                                request.early_stopping_minimum_improvement_percent
                            )
                            improved = (
                                validation_error_percent < early_stopping_reference
                                if threshold == 0.0
                                else decrease >= threshold
                            )
                            if improved:
                                early_stopping_reference = validation_error_percent
                                early_stopping_bad_checks = 0
                            else:
                                early_stopping_bad_checks += 1
                                early_stopping_triggered = (
                                    early_stopping_bad_checks
                                    >= request.early_stopping_patience_checks
                                )

                    record = TrainingValidationRecord(
                        validation_check=validation_checks,
                        training_epochs=training_examples_seen / training_example_count,
                        optimizer_updates=optimizer_updates,
                        training_examples_seen=training_examples_seen,
                        training_objective=training_objective,
                        validation_objective=validation_objective,
                        validation_error_percent=validation_error_percent,
                        learning_rate_before=learning_rate_before,
                        learning_rate_after=learning_rate_after,
                    )
                    history.append(record)
                    validation_exposure = 0
                    interval_loss_total = 0.0
                    interval_loss_examples = 0
                    if early_stopping_triggered:
                        termination_reason = TrainingTerminationReason.EARLY_STOPPING
                    elif validation_checks >= request.maximum_validation_checks:
                        termination_reason = (
                            TrainingTerminationReason.MAXIMUM_VALIDATION_CHECKS
                        )
                    recovery_value = _recovery_mapping(
                        identity=identity,
                        runtime=runtime,
                        training_seed=training_seed,
                        membership=membership,
                        standardization=standardization,
                        model=model,
                        best_model_state=best_model_state,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        stream=stream,
                        initialization_generator_state=initialization_generator_state,
                        optimizer_updates=optimizer_updates,
                        training_examples_seen=training_examples_seen,
                        validation_checks=validation_checks,
                        validation_exposure=validation_exposure,
                        interval_loss_total=interval_loss_total,
                        interval_loss_examples=interval_loss_examples,
                        best_validation_check=best_validation_check,
                        best_validation_objective=best_validation_objective,
                        best_model_validation_error_percent=best_model_validation_error_percent,
                        scheduler_has_reduced=scheduler_has_reduced,
                        early_stopping_reference=early_stopping_reference,
                        early_stopping_bad_checks=early_stopping_bad_checks,
                        history=history,
                        termination_reason=termination_reason,
                    )
                    atomic_save_recovery(paths, recovery_value)
                    _write_log(
                        log,
                        "validation "
                        f"check={validation_checks} "
                        f"epochs={record.training_epochs:.9g} "
                        f"updates={optimizer_updates} "
                        f"training_objective={training_objective:.9g} "
                        f"validation_objective={validation_objective:.9g} "
                        f"error_percent={validation_error_percent:.9g} "
                        f"learning_rate={learning_rate_before:.9g}->{learning_rate_after:.9g}",
                    )

                if not best_model_state:
                    raise RuntimeError(
                        "Training terminated without a best model state."
                    )
                package_metadata = _model_metadata(
                    request,
                    prepared_training,
                    standardization,
                    training_seed,
                )
                publish_model_package(
                    paths,
                    package_metadata,
                    best_model_state,
                )
                package_sha256 = sha256_file(paths.package_path)
                _write_log(log, f"completed: {_utc_now()}")
                _write_log(log, f"termination_reason: {termination_reason.value}")
                _write_log(log, f"best_validation_check: {best_validation_check}")
                best_record = history[best_validation_check - 1]
                _write_log(
                    log,
                    f"best_training_epochs: {best_record.training_epochs:.17g}",
                )
                _write_log(
                    log, f"best_validation_objective: {best_validation_objective:.17g}"
                )
                _write_log(
                    log,
                    "best_model_validation_error_percent: "
                    f"{best_model_validation_error_percent:.17g}",
                )
                _write_log(log, f"model_package: {paths.package_path.name}")
                _write_log(log, f"model_package_sha256: {package_sha256}")
            except BaseException as exc:
                _write_log(log, f"failure: {_utc_now()} {type(exc).__name__}: {exc}")
                raise

        paths.recovery_path.unlink()
        return _build_result(
            paths,
            termination_reason,
            training_seed,
            membership,
            optimizer_updates,
            training_examples_seen,
            validation_checks,
            best_validation_check,
            best_validation_objective,
            best_model_validation_error_percent,
            history,
        )
