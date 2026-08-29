"""Public contracts for AO Predict model training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    """One ordered scalar model input.

    Attributes:
        name: Non-empty feature name.
        values: One value per simulation or per related simulation result.
        unit: Optional physical unit label.
    """

    name: str
    values: np.ndarray
    unit: str | None = None


@dataclass(frozen=True)
class TargetConfig:
    """One ordered scalar physical model output.

    Attributes:
        name: Non-empty target name.
        values: Strictly positive observations fitted by the model.
        unit: Optional physical unit label.
    """

    name: str
    values: np.ndarray
    unit: str | None = None


@dataclass(frozen=True)
class ModelTrainingDataConfig:
    """Ordered feature and target values supplied to model training.

    Public data configurations are passive. ``train_model`` performs coupled
    shape, numerical, schema, and partition validation before fitting.

    Attributes:
        features: Ordered scalar model inputs.
        targets: Ordered scalar physical model outputs.
    """

    features: tuple[FeatureConfig, ...]
    targets: tuple[TargetConfig, ...]


@dataclass(frozen=True)
class TrainModelRequest:
    """Complete request for one restart-safe dense-regression training run.

    Exactly one of ``validation_data``, ``validation_count``, or
    ``validation_fraction`` is required. Automatic partitioning splits whole
    simulations and uses ``split_seed`` when supplied. ``cpu_threads`` changes
    PyTorch's process-wide CPU thread count.

    Attributes:
        model_path: Caller-selected path stem for the model and companions.
        training_data: Explicit training data or the complete automatic-split pool.
        hidden_widths: Dense hidden-layer widths; an empty tuple is linear.
        batch_size: Maximum number of aligned examples in a training batch.
        validation_data: Optional explicit validation partition.
        validation_count: Optional number of simulations withheld automatically.
        validation_fraction: Optional fraction of simulations withheld automatically.
        split_seed: Optional automatic-partition seed.
        overwrite: Replace an existing derived output set and start fresh.
        training_seed: Optional model-initialization and batch-order seed.
        device: Explicit CPU, CUDA, or MPS PyTorch device name.
        cpu_threads: Optional positive process-wide CPU thread count.
        validation_batch_size: Maximum validation examples per execution batch.
        base_learning_rate: Adam learning rate reached after warmup.
        weight_decay: Coupled Adam weight decay.
        warmup_epochs: Number of epochs in linear learning-rate warmup.
        warmup_start_fraction: Initial fraction of ``base_learning_rate``.
        minimum_training_epochs: Early-stopping eligibility gate in epochs.
        validation_check_epochs: Training-set exposures between validation checks.
        learning_rate_reduction_factor: Plateau learning-rate multiplier.
        minimum_learning_rate: Plateau learning-rate floor.
        scheduler_patience_checks: Consecutive unsuccessful check on which the
            scheduler reduces the rate.
        scheduler_minimum_improvement_fraction: Relative objective improvement.
        early_stopping_patience_checks: Consecutive unsuccessful check on which
            early stopping terminates training.
        early_stopping_minimum_improvement_percent: Absolute Error improvement.
        maximum_validation_checks: Final scheduled validation-check bound.
    """

    model_path: str | Path
    training_data: ModelTrainingDataConfig
    hidden_widths: tuple[int, ...]
    batch_size: int
    validation_data: ModelTrainingDataConfig | None = None
    validation_count: int | None = None
    validation_fraction: float | None = None
    split_seed: int | None = None
    overwrite: bool = False
    training_seed: int | None = None
    device: str = "cpu"
    cpu_threads: int | None = None
    validation_batch_size: int | None = None
    base_learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    warmup_start_fraction: float = 0.1
    minimum_training_epochs: int = 20
    validation_check_epochs: float = 1.0
    learning_rate_reduction_factor: float = 0.5
    minimum_learning_rate: float = 1.0e-6
    scheduler_patience_checks: int = 4
    scheduler_minimum_improvement_fraction: float = 0.001
    early_stopping_patience_checks: int = 9
    early_stopping_minimum_improvement_percent: float = 0.001
    maximum_validation_checks: int = 1_000


class TrainingTerminationReason(StrEnum):
    """Reason a successful model-training run terminated."""

    EARLY_STOPPING = "early_stopping"
    MAXIMUM_VALIDATION_CHECKS = "maximum_validation_checks"


@dataclass(frozen=True)
class TrainingValidationRecord:
    """Measurements from one complete validation check.

    The training objective covers examples fitted since the preceding check,
    or since the run began for the first record. Validation measurements cover
    the complete validation partition. Learning rates bracket scheduler action.
    """

    validation_check: int
    training_epochs: float
    optimizer_updates: int
    training_examples_seen: int
    training_objective: float
    validation_objective: float
    validation_error_percent: float
    learning_rate_before: float
    learning_rate_after: float


@dataclass(frozen=True)
class TrainModelResult:
    """Published result from one complete model-training run.

    The result is returned only after the best model package is validated and
    published, the training log is finalized, and recovery is removed.
    ``validation_mask`` is a read-only AO Predict-owned copy for automatic
    splits and is ``None`` for caller-supplied explicit partitions.
    """

    model_path: Path
    termination_reason: TrainingTerminationReason
    training_seed: int
    split_seed: int | None
    validation_mask: np.ndarray | None
    optimizer_updates: int
    training_examples_seen: int
    validation_checks: int
    best_validation_check: int
    best_validation_objective: float
    best_model_validation_error_percent: float
    validation_history: tuple[TrainingValidationRecord, ...]


class ModelTrainingValidationError(ValueError):
    """Raised when a model-training request has invalid coupled inputs."""

    def __init__(self, issues: list[str]):
        self.issues = list(issues)
        message = "Model training validation failed:\n- " + "\n- ".join(self.issues)
        super().__init__(message)


class TrainingRecoveryMismatchError(ValueError):
    """Raised when retained recovery cannot continue the requested run."""

    def __init__(self, mismatches: list[str]):
        self.mismatches = list(mismatches)
        message = "Training recovery does not match the request:\n- " + "\n- ".join(
            self.mismatches
        )
        super().__init__(message)


def model_training_data_from_rows(
    feature_rows: np.ndarray,
    target_rows: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    target_names: tuple[str, ...] | list[str],
    *,
    feature_units: Mapping[str, str] | None = None,
    target_units: Mapping[str, str] | None = None,
) -> ModelTrainingDataConfig:
    """Build one-example-per-simulation training data from row matrices.

    NumPy inputs are retained through column views and are not copied. Complete
    coupled and numerical validation remains the responsibility of
    ``train_model``.

    Args:
        feature_rows: Rank-two NumPy array with one simulation per row.
        target_rows: Rank-two NumPy array with the same row count.
        feature_names: Ordered name for each feature column.
        target_names: Ordered name for each target column.
        feature_units: Optional feature units keyed by feature name.
        target_units: Optional target units keyed by target name.

    Returns:
        Canonical feature-centered model-training data using column views.

    Raises:
        TypeError: If either values input is not a NumPy array.
        ValueError: If ranks, row counts, names, or unit keys do not match.
    """

    if not isinstance(feature_rows, np.ndarray):
        raise TypeError("feature_rows must be a NumPy array.")
    if not isinstance(target_rows, np.ndarray):
        raise TypeError("target_rows must be a NumPy array.")
    feature_array = feature_rows
    target_array = target_rows
    if feature_array.ndim != 2:
        raise ValueError("feature_rows must be a two-dimensional array.")
    if target_array.ndim != 2:
        raise ValueError("target_rows must be a two-dimensional array.")
    if feature_array.shape[0] != target_array.shape[0]:
        raise ValueError("feature_rows and target_rows must have the same row count.")
    feature_names = tuple(feature_names)
    target_names = tuple(target_names)
    if len(feature_names) != feature_array.shape[1]:
        raise ValueError("feature_names must match the feature_rows column count.")
    if len(target_names) != target_array.shape[1]:
        raise ValueError("target_names must match the target_rows column count.")
    feature_units = dict(feature_units or {})
    target_units = dict(target_units or {})
    unknown_feature_units = set(feature_units) - set(feature_names)
    unknown_target_units = set(target_units) - set(target_names)
    if unknown_feature_units:
        raise ValueError(
            "feature_units contains unknown names: "
            + ", ".join(sorted(unknown_feature_units))
        )
    if unknown_target_units:
        raise ValueError(
            "target_units contains unknown names: "
            + ", ".join(sorted(unknown_target_units))
        )
    return ModelTrainingDataConfig(
        features=tuple(
            FeatureConfig(name, feature_array[:, index], feature_units.get(name))
            for index, name in enumerate(feature_names)
        ),
        targets=tuple(
            TargetConfig(name, target_array[:, index], target_units.get(name))
            for index, name in enumerate(target_names)
        ),
    )
