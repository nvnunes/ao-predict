"""Loading, prediction, and aggregate evaluation for AO Predict models."""

from __future__ import annotations

import math
import os
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn

from .._model_package import load_model_package
from .._runtime import select_device
from .._standardization import standardize_to_float32
from .types import ModelEvaluationResult

_PACKAGE_SUFFIX = ".model.zip"
_FINITE_CHECK_CHUNK = 1_000_000


def _positive_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _validate_real_array(
    value: object,
    *,
    label: str,
    positive: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array.")
    if not (
        np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.floating)
    ) or np.issubdtype(value.dtype, np.bool_):
        raise ValueError(f"{label} must have a real, non-Boolean numeric dtype.")
    iterator = np.nditer(
        value,
        flags=["buffered", "external_loop", "zerosize_ok"],
        op_flags=[["readonly"]],
        buffersize=_FINITE_CHECK_CHUNK,
    )
    for chunk in iterator:
        if not bool(np.all(np.isfinite(chunk))):
            raise ValueError(f"{label} must contain only finite values.")
        if positive and not bool(np.all(chunk > 0)):
            raise ValueError(f"{label} must contain only strictly positive values.")
    return value


def _as_float32_batch(
    value: np.ndarray,
    label: str,
    *,
    positive: bool = False,
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.ascontiguousarray(value, dtype=np.float32)
    if not bool(np.all(np.isfinite(output))):
        raise ValueError(
            f"{label} values must be representable as finite float32 values."
        )
    if positive and not bool(np.all(output > 0)):
        raise ValueError(
            f"{label} values must be representable as strictly positive float32 values."
        )
    return output


def _validate_names(
    values: Mapping[object, object],
    expected_names: tuple[str, ...],
    label: str,
) -> None:
    actual_names = set(values)
    expected = set(expected_names)
    if actual_names != expected:
        missing = sorted(expected - actual_names)
        extra = sorted(str(name) for name in actual_names - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError(f"{label} names must match the model ({'; '.join(details)}).")


@dataclass(frozen=True)
class _PreparedFeatures:
    """Validated borrowed feature arrays with bounded batch gathering."""

    names: tuple[str, ...]
    example_shape: tuple[int, ...]
    example_count: int
    direct: np.ndarray | None
    fields: tuple[np.ndarray, ...]

    def batch(self, start: int, stop: int) -> np.ndarray:
        if self.direct is not None:
            return self.direct[start:stop]
        output = np.empty((stop - start, len(self.names)), dtype=np.float64)
        structured = len(self.example_shape) == 2
        if structured:
            related_examples = self.example_shape[1]
            flat_indices = np.arange(start, stop, dtype=np.intp)
            simulation_indices = flat_indices // related_examples
            related_indices = flat_indices % related_examples
        with np.errstate(over="ignore", invalid="ignore"):
            for column, values in enumerate(self.fields):
                if not structured:
                    output[:, column] = values[start:stop]
                elif values.ndim == 2:
                    output[:, column] = values[
                        simulation_indices,
                        related_indices,
                    ]
                else:
                    output[:, column] = values[simulation_indices]
        if not bool(np.all(np.isfinite(output))):
            raise ValueError("feature values must be representable as finite float64.")
        return output


@dataclass(frozen=True)
class _PreparedTargets:
    """Validated borrowed target arrays with bounded batch gathering."""

    names: tuple[str, ...]
    example_shape: tuple[int, ...]
    direct: np.ndarray | None
    fields: tuple[np.ndarray, ...]

    def batch(self, start: int, stop: int) -> np.ndarray:
        structured = len(self.example_shape) == 2
        if structured:
            related_examples = self.example_shape[1]
            flat_indices = np.arange(start, stop, dtype=np.intp)
            simulation_indices = flat_indices // related_examples
            related_indices = flat_indices % related_examples
        if self.direct is not None:
            if structured:
                batch = self.direct[
                    simulation_indices,
                    related_indices,
                    :,
                ]
            else:
                batch = self.direct[start:stop]
            return _as_float32_batch(batch, "target", positive=True)
        output = np.empty((stop - start, len(self.names)), dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            for column, values in enumerate(self.fields):
                if structured:
                    output[:, column] = values[
                        simulation_indices,
                        related_indices,
                    ]
                else:
                    output[:, column] = values[start:stop]
        if not bool(np.all(np.isfinite(output))):
            raise ValueError(
                "target values must be representable as finite float32 values."
            )
        if not bool(np.all(output > 0)):
            raise ValueError(
                "target values must be representable as strictly positive float32 values."
            )
        return output


def _prepare_features(
    values: np.ndarray | Mapping[str, np.ndarray],
    names: tuple[str, ...],
) -> _PreparedFeatures:
    if isinstance(values, np.ndarray):
        direct = _validate_real_array(values, label="features")
        if direct.ndim != 2:
            raise ValueError("features must be a rank-two NumPy matrix.")
        if direct.shape[1] != len(names):
            raise ValueError(
                f"features must have {len(names)} columns, not {direct.shape[1]}."
            )
        return _PreparedFeatures(
            names=names,
            example_shape=(direct.shape[0],),
            example_count=direct.shape[0],
            direct=direct,
            fields=(),
        )
    if not isinstance(values, Mapping):
        raise TypeError(
            "features must be a NumPy array or a mapping of feature arrays."
        )
    _validate_names(values, names, "feature")
    fields: list[np.ndarray] = []
    rank_two_shape: tuple[int, int] | None = None
    simulation_count: int | None = None
    for name in names:
        field = _validate_real_array(values[name], label=f"feature {name!r}")
        if field.ndim not in {1, 2}:
            raise ValueError(f"feature {name!r} must have rank one or two.")
        if field.ndim == 2:
            shape = (field.shape[0], field.shape[1])
            if rank_two_shape is None:
                rank_two_shape = shape
            elif shape != rank_two_shape:
                raise ValueError(
                    "All rank-two feature arrays must have the same shape."
                )
        if simulation_count is None:
            simulation_count = field.shape[0]
        elif field.shape[0] != simulation_count:
            raise ValueError("All feature arrays must have the same simulation count.")
        fields.append(field)
    assert simulation_count is not None
    if rank_two_shape is None:
        example_shape = (simulation_count,)
    else:
        if rank_two_shape[0] != simulation_count:
            raise ValueError("All feature arrays must have the same simulation count.")
        example_shape = rank_two_shape
    return _PreparedFeatures(
        names=names,
        example_shape=example_shape,
        example_count=math.prod(example_shape),
        direct=None,
        fields=tuple(fields),
    )


def _prepare_targets(
    values: np.ndarray | Mapping[str, np.ndarray],
    names: tuple[str, ...],
    example_shape: tuple[int, ...],
) -> _PreparedTargets:
    if isinstance(values, np.ndarray):
        direct = _validate_real_array(values, label="targets", positive=True)
        expected_shape = (*example_shape, len(names))
        if direct.shape != expected_shape:
            raise ValueError(
                f"targets must have shape {expected_shape}, not {direct.shape}."
            )
        return _PreparedTargets(names, example_shape, direct, ())
    if not isinstance(values, Mapping):
        raise TypeError("targets must be a NumPy array or a mapping of target arrays.")
    _validate_names(values, names, "target")
    fields: list[np.ndarray] = []
    for name in names:
        field = _validate_real_array(
            values[name],
            label=f"target {name!r}",
            positive=True,
        )
        if field.shape != example_shape:
            raise ValueError(
                f"target {name!r} must have shape {example_shape}, not {field.shape}."
            )
        fields.append(field)
    return _PreparedTargets(names, example_shape, None, tuple(fields))


class ModelPredictor:
    """A loaded AO Predict model with bounded prediction and evaluation.

    Instances are created by ``load_model_predictor()``. Public properties
    describe the immutable package contract and selected runtime. The PyTorch
    model and fitted standardization values remain private.
    """

    __slots__ = (
        "_batch_size",
        "_device",
        "_feature_mean",
        "_feature_names",
        "_feature_scale",
        "_feature_units",
        "_model",
        "_model_package_path",
        "_model_path",
        "_target_mean",
        "_target_names",
        "_target_scale",
        "_target_units",
    )

    def __init__(self) -> None:
        """Reject direct construction; load a validated package instead."""

        raise TypeError(
            "ModelPredictor instances are created by load_model_predictor()."
        )

    @classmethod
    def _from_loaded_package(
        cls,
        *,
        model_path: Path,
        model_package_path: Path,
        device: torch.device,
        batch_size: int,
        model: nn.Module,
        feature_definitions: list[dict[str, object]],
        target_definitions: list[dict[str, object]],
    ) -> ModelPredictor:
        self = cls.__new__(cls)
        self._model_path = model_path
        self._model_package_path = model_package_path
        self._device = str(device)
        self._batch_size = batch_size
        self._feature_names = tuple(str(item["name"]) for item in feature_definitions)
        self._feature_units = tuple(
            cast(str | None, item["unit"]) for item in feature_definitions
        )
        self._target_names = tuple(str(item["name"]) for item in target_definitions)
        self._target_units = tuple(
            cast(str | None, item["unit"]) for item in target_definitions
        )
        self._feature_mean = np.asarray(
            [float(item["mean"]) for item in feature_definitions],
            dtype=np.float64,
        )
        self._feature_scale = np.asarray(
            [float(item["scale"]) for item in feature_definitions],
            dtype=np.float64,
        )
        self._target_mean = torch.tensor(
            [float(item["mean"]) for item in target_definitions],
            dtype=torch.float32,
            device=device,
        )
        self._target_scale = torch.tensor(
            [float(item["scale"]) for item in target_definitions],
            dtype=torch.float32,
            device=device,
        )
        self._model = model.to(device)
        self._model.eval()
        return self

    @property
    def model_path(self) -> Path:
        """Return the normalized caller-facing model stem."""

        return self._model_path

    @property
    def model_package_path(self) -> Path:
        """Return the exact loaded ``.model.zip`` path."""

        return self._model_package_path

    @property
    def device(self) -> str:
        """Return the resolved PyTorch device name."""

        return self._device

    @property
    def batch_size(self) -> int:
        """Return the default maximum execution batch size."""

        return self._batch_size

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return feature names in model input order."""

        return self._feature_names

    @property
    def feature_units(self) -> tuple[str | None, ...]:
        """Return feature units in model input order."""

        return self._feature_units

    @property
    def target_names(self) -> tuple[str, ...]:
        """Return target names in model output order."""

        return self._target_names

    @property
    def target_units(self) -> tuple[str | None, ...]:
        """Return target units in model output order."""

        return self._target_units

    def _resolve_batch_size(self, batch_size: int | None) -> int:
        if batch_size is None:
            return self._batch_size
        return _positive_integer(batch_size, "batch_size")

    def _predict_batch(self, values: np.ndarray) -> torch.Tensor:
        standardized = standardize_to_float32(
            values,
            self._feature_mean,
            self._feature_scale,
            label="feature values",
        )
        tensor = torch.from_numpy(standardized).to(self._device)
        return self._model(tensor) * self._target_scale + self._target_mean

    def predict(
        self,
        features: np.ndarray | Mapping[str, np.ndarray],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Predict physical targets for direct or named feature arrays.

        Direct input has shape ``(examples, features)``. Named feature arrays
        may have shape ``(simulations,)`` or
        ``(simulations, related_examples)``; simulation-level arrays are shared
        over the related-example axis without materializing a repeated input.

        Args:
            features: A real finite NumPy matrix in model feature order, or an
                exact-name mapping of rank-one and rank-two NumPy arrays.
            batch_size: Optional positive execution-batch override. ``None``
                uses the predictor's default.

        Returns:
            A physical-unit ``float32`` NumPy array with the target axis last.

        Raises:
            TypeError: If an input is not one of the supported NumPy forms.
            ValueError: If names, dtypes, values, ranks, shapes, or batch size
                violate the loaded model contract.
        """

        resolved_batch_size = self._resolve_batch_size(batch_size)
        prepared = _prepare_features(features, self._feature_names)
        flat_output = np.empty(
            (prepared.example_count, len(self._target_names)),
            dtype=np.float32,
        )
        with torch.inference_mode():
            for start in range(0, prepared.example_count, resolved_batch_size):
                stop = min(start + resolved_batch_size, prepared.example_count)
                prediction = self._predict_batch(prepared.batch(start, stop))
                flat_output[start:stop] = prediction.detach().cpu().numpy()
        return flat_output.reshape(*prepared.example_shape, len(self._target_names))

    def predict_one(
        self,
        features: np.ndarray | Mapping[str, Real],
    ) -> np.ndarray:
        """Predict one physical target vector from positional or named values.

        Args:
            features: One real finite NumPy vector in model feature order, or
                an exact-name mapping of real finite scalar values.

        Returns:
            One physical-unit ``float32`` vector in model target order.

        Raises:
            TypeError: If the input or a named value has an unsupported type.
            ValueError: If names, dtype, values, rank, or width violate the
                loaded model contract.
        """

        if isinstance(features, np.ndarray):
            vector = _validate_real_array(features, label="features")
            if vector.ndim != 1 or vector.shape[0] != len(self._feature_names):
                raise ValueError(
                    f"features must have shape ({len(self._feature_names)},)."
                )
            matrix = vector.reshape(1, -1)
        elif isinstance(features, Mapping):
            _validate_names(features, self._feature_names, "feature")
            matrix = np.empty((1, len(self._feature_names)), dtype=np.float64)
            for index, name in enumerate(self._feature_names):
                value = features[name]
                if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
                    raise TypeError(f"feature {name!r} must be a real scalar.")
                try:
                    converted = float(value)
                except OverflowError as exc:
                    raise ValueError(f"feature {name!r} must be finite.") from exc
                if not math.isfinite(converted):
                    raise ValueError(f"feature {name!r} must be finite.")
                matrix[0, index] = converted
        else:
            raise TypeError("features must be a NumPy vector or a mapping of scalars.")
        with torch.inference_mode():
            prediction = self._predict_batch(matrix)
        return prediction.detach().cpu().numpy()[0]

    def evaluate(
        self,
        features: np.ndarray | Mapping[str, np.ndarray],
        targets: np.ndarray | Mapping[str, np.ndarray],
        *,
        batch_size: int | None = None,
    ) -> ModelEvaluationResult:
        """Evaluate aggregate physical relative error over one population.

        Args:
            features: Direct or exact-name feature arrays accepted by
                ``predict()``.
            targets: A direct array matching the prediction shape, or an
                exact-name mapping with one full-shape array per target.
            batch_size: Optional positive execution-batch override. ``None``
                uses the predictor's default.

        Returns:
            Immutable complete-population relative-error measurements.

        Raises:
            TypeError: If features or targets use unsupported object types.
            ValueError: If the population is empty or names, dtypes, values,
                ranks, shapes, or batch size violate the model contract.
        """

        resolved_batch_size = self._resolve_batch_size(batch_size)
        prepared_features = _prepare_features(features, self._feature_names)
        if prepared_features.example_count == 0:
            raise ValueError("evaluation requires at least one example.")
        prepared_targets = _prepare_targets(
            targets,
            self._target_names,
            prepared_features.example_shape,
        )
        squared_total = 0.0
        target_squared_total = [0.0] * len(self._target_names)
        with torch.inference_mode():
            for start in range(
                0,
                prepared_features.example_count,
                resolved_batch_size,
            ):
                stop = min(
                    start + resolved_batch_size,
                    prepared_features.example_count,
                )
                prediction = self._predict_batch(prepared_features.batch(start, stop))
                expected = torch.from_numpy(prepared_targets.batch(start, stop)).to(
                    self._device
                )
                squared = torch.square((prediction - expected) / expected)
                squared_total += float(torch.sum(squared).item())
                per_target = torch.sum(squared, dim=0).detach().cpu().tolist()
                for index, value in enumerate(per_target):
                    target_squared_total[index] += float(value)
        relative_mse = squared_total / (
            prepared_features.example_count * len(self._target_names)
        )
        target_relative_rmse = {
            name: math.sqrt(total / prepared_features.example_count)
            for name, total in zip(
                self._target_names,
                target_squared_total,
                strict=True,
            )
        }
        return ModelEvaluationResult(
            example_count=prepared_features.example_count,
            relative_mse=relative_mse,
            relative_rmse=math.sqrt(relative_mse),
            target_relative_rmse=target_relative_rmse,
        )


def _resolve_model_paths(value: str | Path) -> tuple[Path, Path]:
    if not isinstance(value, (str, Path)):
        raise TypeError("model_path must be a string or pathlib.Path.")
    raw = os.fspath(value)
    if not raw or not raw.strip():
        raise ValueError("model_path must not be empty.")
    separators = (os.sep,) if os.altsep is None else (os.sep, os.altsep)
    if isinstance(value, str) and raw.endswith(separators):
        raise ValueError("model_path must include a model name.")
    normalized = Path(os.path.normpath(raw))
    if normalized.name.endswith(_PACKAGE_SUFFIX):
        stem_name = normalized.name[: -len(_PACKAGE_SUFFIX)]
        if not stem_name:
            raise ValueError("model_path must include a model name.")
        return normalized.with_name(stem_name), normalized
    if normalized.name in {"", ".", ".."}:
        raise ValueError("model_path must include a model name.")
    return normalized, Path(f"{normalized}{_PACKAGE_SUFFIX}")


def load_model_predictor(
    model_path: str | Path,
    *,
    device: str = "cpu",
    cpu_threads: int | None = None,
    batch_size: int = 16_384,
) -> ModelPredictor:
    """Load and validate a model package for prediction and evaluation.

    ``model_path`` may be either the caller-facing stem or the exact
    ``.model.zip`` path. Device selection is explicit and never falls back.
    Supplying ``cpu_threads`` changes PyTorch's process-wide CPU thread count.

    Args:
        model_path: The caller-facing model stem or exact ``.model.zip`` path.
        device: An explicit available ``cpu``, ``cuda``, ``cuda:<index>``, or
            ``mps`` PyTorch device name.
        cpu_threads: Optional positive process-wide PyTorch CPU thread count.
        batch_size: Positive default maximum execution batch size.

    Returns:
        A loaded predictor retaining its model on the resolved device and its
        private preprocessing state in the representations used during bounded
        prediction.

    Raises:
        TypeError: If ``model_path`` has an unsupported object type.
        FileNotFoundError: If the syntactically resolved package is missing.
        ValueError: If a path value, runtime selection, batch value, or model
            package is invalid, unavailable, malformed, or unsupported.
    """

    model_stem, package_path = _resolve_model_paths(model_path)
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string.")
    if cpu_threads is not None:
        _positive_integer(cpu_threads, "cpu_threads")
    resolved_batch_size = _positive_integer(batch_size, "batch_size")
    try:
        package = load_model_package(package_path)
    except FileNotFoundError:
        raise
    except (zipfile.BadZipFile, RuntimeError, EOFError) as exc:
        raise ValueError(f"Model package is malformed: {exc}") from exc
    resolved_device = select_device(device, cpu_threads)
    if resolved_device.type == "cpu" and cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    metadata = package.metadata
    feature_definitions = metadata["features"]
    target_definitions = metadata["targets"]
    assert isinstance(feature_definitions, list)
    assert isinstance(target_definitions, list)
    return ModelPredictor._from_loaded_package(
        model_path=model_stem,
        model_package_path=package_path,
        device=resolved_device,
        batch_size=resolved_batch_size,
        model=package.model,
        feature_definitions=feature_definitions,
        target_definitions=target_definitions,
    )
