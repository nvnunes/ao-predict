"""Preparation, identity, partition, and standardization for model training."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .._standardization import standardize_to_float32, validate_float32_scaler
from .types import FeatureConfig, ModelTrainingDataConfig, TargetConfig

_CHECKSUM_ROW_CHUNK = 4_096
_MOMENT_ROW_CHUNK = 4_096


@dataclass(frozen=True)
class _PreparedModelTrainingData:
    """Validated borrowed arrays and their canonical training identity."""

    config: ModelTrainingDataConfig
    feature_values: tuple[np.ndarray, ...]
    target_values: tuple[np.ndarray, ...]
    target_shape: tuple[int, ...]
    simulation_count: int
    examples_per_simulation: int
    component_checksums: tuple[str, ...]
    checksum: str

    @property
    def feature_schema(self) -> tuple[tuple[str, str | None], ...]:
        return tuple((item.name, item.unit) for item in self.config.features)

    @property
    def target_schema(self) -> tuple[tuple[str, str | None], ...]:
        return tuple((item.name, item.unit) for item in self.config.targets)


@dataclass(frozen=True)
class _SplitMembership:
    """Resolved simulation membership for explicit or automatic partitions."""

    split_seed: int | None
    validation_mask: np.ndarray | None
    training_simulations: np.ndarray
    validation_simulations: np.ndarray


@dataclass(frozen=True)
class _StandardizationState:
    """Authoritative float64 population mean and scale values."""

    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    target_means: tuple[float, ...]
    target_scales: tuple[float, ...]

    def as_mapping(self) -> dict[str, list[float]]:
        """Return a plain recovery- and JSON-compatible representation."""

        return {
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "target_means": list(self.target_means),
            "target_scales": list(self.target_scales),
        }

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, object],
        *,
        feature_count: int,
        target_count: int,
    ) -> _StandardizationState:
        """Validate and reconstruct retained standardization state."""

        expected = {
            "feature_means": feature_count,
            "feature_scales": feature_count,
            "target_means": target_count,
            "target_scales": target_count,
        }
        converted: dict[str, tuple[float, ...]] = {}
        for key, count in expected.items():
            raw = value.get(key)
            if not isinstance(raw, list) or len(raw) != count:
                raise ValueError(f"Recovery standardization field {key!r} is invalid.")
            items = tuple(float(item) for item in raw)
            if not all(math.isfinite(item) for item in items):
                raise ValueError(
                    f"Recovery standardization field {key!r} is non-finite."
                )
            converted[key] = items
        if any(item <= 0.0 for item in converted["feature_scales"]):
            raise ValueError("Recovery feature scales must be positive.")
        if any(item <= 0.0 for item in converted["target_scales"]):
            raise ValueError("Recovery target scales must be positive.")
        return cls(**converted)


@dataclass(frozen=True)
class _StandardizedModelTrainingData:
    """AO Predict-owned float32 arrays with original feature compactness."""

    prepared: _PreparedModelTrainingData
    feature_values: tuple[np.ndarray, ...]
    target_values: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _ExampleSet:
    """One logical example sequence over selected complete simulations."""

    data: _StandardizedModelTrainingData
    simulation_indexes: np.ndarray

    def __len__(self) -> int:
        return int(
            self.simulation_indexes.size * self.data.prepared.examples_per_simulation
        )

    def gather(self, example_indexes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Materialize only one requested feature and target batch."""

        example_indexes = np.asarray(example_indexes, dtype=np.int64)
        examples_per_simulation = self.data.prepared.examples_per_simulation
        simulation_slots = example_indexes // examples_per_simulation
        point_indexes = example_indexes % examples_per_simulation
        simulation_indexes = self.simulation_indexes[simulation_slots]
        features = np.empty(
            (example_indexes.size, len(self.data.feature_values)),
            dtype=np.float32,
        )
        targets = np.empty(
            (example_indexes.size, len(self.data.target_values)),
            dtype=np.float32,
        )
        for column, values in enumerate(self.data.feature_values):
            if values.ndim == 1:
                features[:, column] = values[simulation_indexes]
            else:
                features[:, column] = values[simulation_indexes, point_indexes]
        for column, values in enumerate(self.data.target_values):
            if values.ndim == 1:
                targets[:, column] = values[simulation_indexes]
            else:
                targets[:, column] = values[simulation_indexes, point_indexes]
        return features, targets


def _validate_name_unit_family(
    values: Sequence[object],
    label: str,
    expected_type: type[FeatureConfig | TargetConfig],
    issues: list[str],
) -> None:
    names: list[str] = []
    for index, item in enumerate(values):
        if not isinstance(item, expected_type):
            issues.append(f"{label}[{index}] must be a {expected_type.__name__}.")
            continue
        name = getattr(item, "name", None)
        unit = getattr(item, "unit", None)
        if not isinstance(name, str) or not name.strip():
            issues.append(f"{label}[{index}].name must be a non-empty string.")
        else:
            names.append(name)
        if unit is not None and (not isinstance(unit, str) or not unit.strip()):
            issues.append(f"{label}[{index}].unit must be None or a non-empty string.")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        issues.append(
            f"{label} names must be unique; duplicates: {', '.join(duplicates)}."
        )


def _is_real_numeric_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and (
            np.issubdtype(value.dtype, np.integer)
            or np.issubdtype(value.dtype, np.floating)
        )
        and not np.issubdtype(value.dtype, np.bool_)
    )


def _array_checksum(array: np.ndarray) -> str:
    """Hash logical row-major values independent of strides and ownership."""

    digest = hashlib.sha256()
    header = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0")
    rows = array.shape[0]
    for start in range(0, rows, _CHECKSUM_ROW_CHUNK):
        chunk = np.ascontiguousarray(array[start : start + _CHECKSUM_ROW_CHUNK])
        digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def _array_is_finite(array: np.ndarray) -> bool:
    for start in range(0, array.shape[0], _CHECKSUM_ROW_CHUNK):
        chunk = array[start : start + _CHECKSUM_ROW_CHUNK]
        if not bool(np.all(np.isfinite(chunk))):
            return False
    return True


def _array_is_strictly_positive(array: np.ndarray) -> bool:
    for start in range(0, array.shape[0], _CHECKSUM_ROW_CHUNK):
        chunk = array[start : start + _CHECKSUM_ROW_CHUNK]
        if not bool(np.all(chunk > 0)):
            return False
    return True


def prepare_model_training_data(
    config: ModelTrainingDataConfig,
    *,
    label: str,
    issues: list[str],
) -> _PreparedModelTrainingData | None:
    """Validate one passive public data config and bind borrowed arrays."""

    start_issue_count = len(issues)
    if not isinstance(config, ModelTrainingDataConfig):
        issues.append(f"{label} must be a ModelTrainingDataConfig.")
        return None
    if not isinstance(config.features, tuple) or not config.features:
        issues.append(f"{label}.features must be a non-empty tuple.")
    if not isinstance(config.targets, tuple) or not config.targets:
        issues.append(f"{label}.targets must be a non-empty tuple.")
    if len(issues) != start_issue_count:
        return None
    _validate_name_unit_family(
        config.features,
        f"{label}.features",
        FeatureConfig,
        issues,
    )
    _validate_name_unit_family(
        config.targets,
        f"{label}.targets",
        TargetConfig,
        issues,
    )

    target_shape: tuple[int, ...] | None = None
    target_values: list[np.ndarray] = []
    for index, target in enumerate(config.targets):
        if not isinstance(target, TargetConfig):
            continue
        values = target.values
        item_label = f"{label}.targets[{index}].values"
        if not _is_real_numeric_array(values):
            issues.append(f"{item_label} must be a real numerical NumPy array.")
            continue
        if values.ndim not in (1, 2):
            issues.append(f"{item_label} must have one or two dimensions.")
            continue
        if 0 in values.shape:
            issues.append(f"{item_label} must not contain an empty dimension.")
            continue
        if not _array_is_finite(values):
            issues.append(f"{item_label} must contain only finite values.")
        if not _array_is_strictly_positive(values):
            issues.append(f"{item_label} must contain only strictly positive targets.")
        if target_shape is None:
            target_shape = tuple(values.shape)
        elif tuple(values.shape) != target_shape:
            issues.append(
                f"{item_label} has shape {values.shape}; expected shared target shape {target_shape}."
            )
        target_values.append(values)

    if target_shape is None:
        return None
    simulation_count = target_shape[0]
    feature_values: list[np.ndarray] = []
    for index, feature in enumerate(config.features):
        if not isinstance(feature, FeatureConfig):
            continue
        values = feature.values
        item_label = f"{label}.features[{index}].values"
        if not _is_real_numeric_array(values):
            issues.append(f"{item_label} must be a real numerical NumPy array.")
            continue
        allowed_shapes = {(simulation_count,), target_shape}
        if tuple(values.shape) not in allowed_shapes:
            issues.append(
                f"{item_label} has shape {values.shape}; expected {(simulation_count,)} or {target_shape}."
            )
            continue
        if not _array_is_finite(values):
            issues.append(f"{item_label} must contain only finite values.")
        feature_values.append(values)

    if len(issues) != start_issue_count:
        return None
    component_checksums = tuple(
        _array_checksum(values) for values in (*feature_values, *target_values)
    )
    identity = {
        "features": [
            {
                "name": item.name,
                "unit": item.unit,
                "checksum": component_checksums[index],
            }
            for index, item in enumerate(config.features)
        ],
        "targets": [
            {
                "name": item.name,
                "unit": item.unit,
                "checksum": component_checksums[len(config.features) + index],
            }
            for index, item in enumerate(config.targets)
        ],
    }
    checksum = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _PreparedModelTrainingData(
        config=config,
        feature_values=tuple(feature_values),
        target_values=tuple(target_values),
        target_shape=target_shape,
        simulation_count=simulation_count,
        examples_per_simulation=1 if len(target_shape) == 1 else target_shape[1],
        component_checksums=component_checksums,
        checksum=checksum,
    )


def validate_explicit_schema(
    training: _PreparedModelTrainingData,
    validation: _PreparedModelTrainingData,
    issues: list[str],
) -> None:
    """Require model-facing names and units to agree across partitions."""

    if training.feature_schema != validation.feature_schema:
        issues.append(
            "training_data and validation_data must have identical ordered feature names and units."
        )
    if training.target_schema != validation.target_schema:
        issues.append(
            "training_data and validation_data must have identical ordered target names and units."
        )


def resolve_automatic_split(
    prepared: _PreparedModelTrainingData,
    *,
    validation_count: int | None,
    validation_fraction: float | None,
    split_seed: int | None,
    validation_mask: np.ndarray | None = None,
) -> _SplitMembership:
    """Resolve or restore a complete-simulation automatic partition."""

    count = validation_count
    if validation_fraction is not None:
        count = math.ceil(validation_fraction * prepared.simulation_count)
    if count is None:
        raise RuntimeError("Automatic split count was not resolved.")
    if split_seed is None:
        split_seed = secrets.randbits(63)
    if validation_mask is None:
        generator = np.random.Generator(np.random.PCG64(split_seed))
        selected = generator.choice(
            prepared.simulation_count,
            size=count,
            replace=False,
        )
        validation_mask = np.zeros(prepared.simulation_count, dtype=np.bool_)
        validation_mask[selected] = True
    else:
        validation_mask = np.asarray(validation_mask, dtype=np.bool_).copy()
        if validation_mask.shape != (prepared.simulation_count,):
            raise ValueError("Recovery validation mask has the wrong shape.")
        if int(np.count_nonzero(validation_mask)) != count:
            raise ValueError("Recovery validation mask has the wrong validation count.")
    validation_mask.setflags(write=False)
    return _SplitMembership(
        split_seed=split_seed,
        validation_mask=validation_mask,
        training_simulations=np.flatnonzero(~validation_mask),
        validation_simulations=np.flatnonzero(validation_mask),
    )


def explicit_membership(
    training: _PreparedModelTrainingData,
    validation: _PreparedModelTrainingData,
) -> _SplitMembership:
    """Return complete ordered membership for explicit partitions."""

    return _SplitMembership(
        split_seed=None,
        validation_mask=None,
        training_simulations=np.arange(training.simulation_count, dtype=np.int64),
        validation_simulations=np.arange(validation.simulation_count, dtype=np.int64),
    )


def validation_mask_checksum(mask: np.ndarray | None) -> str | None:
    """Return canonical automatic-membership identity."""

    return None if mask is None else _array_checksum(mask)


def _selected_chunks(
    values: np.ndarray,
    simulation_indexes: np.ndarray,
) -> Iterable[np.ndarray]:
    for start in range(0, simulation_indexes.size, _MOMENT_ROW_CHUNK):
        indexes = simulation_indexes[start : start + _MOMENT_ROW_CHUNK]
        yield np.asarray(values[indexes], dtype=np.float64)


def _fit_population_mean_scale(
    values: np.ndarray,
    simulation_indexes: np.ndarray,
) -> tuple[float, float]:
    total = 0.0
    count = 0
    for chunk in _selected_chunks(values, simulation_indexes):
        total += float(np.sum(chunk, dtype=np.float64))
        count += int(chunk.size)
    mean = total / count
    squared_total = 0.0
    for chunk in _selected_chunks(values, simulation_indexes):
        squared_total += float(
            np.sum(np.square(chunk - mean, dtype=np.float64), dtype=np.float64)
        )
    variance = max(0.0, squared_total / count)
    scale = math.sqrt(variance)
    if scale == 0.0:
        scale = 1.0
    return float(mean), float(scale)


def fit_standardization(
    training: _PreparedModelTrainingData,
    training_simulations: np.ndarray,
) -> _StandardizationState:
    """Fit compact population standardization on training simulations only."""

    feature = tuple(
        _fit_population_mean_scale(values, training_simulations)
        for values in training.feature_values
    )
    target = tuple(
        _fit_population_mean_scale(values, training_simulations)
        for values in training.target_values
    )
    return _StandardizationState(
        feature_means=tuple(value[0] for value in feature),
        feature_scales=tuple(value[1] for value in feature),
        target_means=tuple(value[0] for value in target),
        target_scales=tuple(value[1] for value in target),
    )


def _standardize_array(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    result = np.empty(values.shape, dtype=np.float32)
    for start in range(0, values.shape[0], _MOMENT_ROW_CHUNK):
        stop = start + _MOMENT_ROW_CHUNK
        result[start:stop] = standardize_to_float32(
            values[start:stop],
            mean,
            scale,
            label="training values",
        )
    return result


def standardize_data(
    prepared: _PreparedModelTrainingData,
    state: _StandardizationState,
) -> _StandardizedModelTrainingData:
    """Create the AO Predict-owned float32 state used for fitting."""

    if len(state.feature_means) != len(prepared.feature_values):
        raise ValueError("Feature standardization state has the wrong width.")
    if len(state.target_means) != len(prepared.target_values):
        raise ValueError("Target standardization state has the wrong width.")
    return _StandardizedModelTrainingData(
        prepared=prepared,
        feature_values=tuple(
            _standardize_array(values, mean, scale)
            for values, mean, scale in zip(
                prepared.feature_values,
                state.feature_means,
                state.feature_scales,
                strict=True,
            )
        ),
        target_values=tuple(
            _standardize_array(values, mean, scale)
            for values, mean, scale in zip(
                prepared.target_values,
                state.target_means,
                state.target_scales,
                strict=True,
            )
        ),
    )


def model_numerical_compatibility_issues(
    datasets: Sequence[_StandardizedModelTrainingData],
    state: _StandardizationState,
) -> list[str]:
    """Validate derived preprocessing against the float32 model contract."""

    issues: list[str] = []
    for family, means, scales in (
        ("feature", state.feature_means, state.feature_scales),
        ("target", state.target_means, state.target_scales),
    ):
        for index, (mean, scale) in enumerate(zip(means, scales, strict=True)):
            try:
                validate_float32_scaler(
                    mean,
                    scale,
                    label=f"Fitted {family}[{index}]",
                )
            except ValueError as exc:
                issues.append(str(exc))

    if issues or not datasets:
        return issues
    target_means = np.asarray(state.target_means, dtype=np.float32)
    target_scales = np.asarray(state.target_scales, dtype=np.float32)
    for dataset_label, dataset in zip(
        ("training_data", "validation_data"),
        datasets,
        strict=True,
    ):
        for index, values in enumerate(dataset.target_values):
            mean = target_means[index]
            scale = target_scales[index]
            for start in range(0, values.shape[0], _MOMENT_ROW_CHUNK):
                stop = start + _MOMENT_ROW_CHUNK
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    physical = values[start:stop] * scale + mean
                if not bool(np.all(np.isfinite(physical))):
                    issues.append(
                        f"{dataset_label}.targets[{index}] cannot be reconstructed "
                        "as finite float32 physical values."
                    )
                    break
                if not bool(np.all(physical > 0)):
                    issues.append(
                        f"{dataset_label}.targets[{index}] cannot be reconstructed "
                        "as strictly positive float32 physical values."
                    )
                    break
    return issues


def data_identity(prepared: _PreparedModelTrainingData) -> dict[str, object]:
    """Return the exact-recovery identity of one prepared partition."""

    return {
        "checksum": prepared.checksum,
        "simulation_count": prepared.simulation_count,
        "examples_per_simulation": prepared.examples_per_simulation,
        "target_shape": list(prepared.target_shape),
        "feature_schema": [list(value) for value in prepared.feature_schema],
        "target_schema": [list(value) for value in prepared.target_schema],
    }


def schema_metadata(
    prepared: _PreparedModelTrainingData,
    state: _StandardizationState,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build deployable ordered feature and target metadata."""

    features = [
        {
            "name": item.name,
            "unit": item.unit,
            "mean": state.feature_means[index],
            "scale": state.feature_scales[index],
        }
        for index, item in enumerate(prepared.config.features)
    ]
    targets = [
        {
            "name": item.name,
            "unit": item.unit,
            "mean": state.target_means[index],
            "scale": state.target_scales[index],
        }
        for index, item in enumerate(prepared.config.targets)
    ]
    return features, targets
