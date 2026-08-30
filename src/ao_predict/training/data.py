"""Preparation, identity, partition, and standardization for model training."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from astropy import units as u

from .._standardization import standardize_to_float32, validate_float32_scaler
from .._units import unit_string
from .types import ModelTrainingDataConfig

_CHECKSUM_ROW_CHUNK = 4_096
_MOMENT_ROW_CHUNK = 4_096


@dataclass(frozen=True)
class _PreparedModelTrainingData:
    """Validated borrowed arrays and their canonical training identity."""

    config: ModelTrainingDataConfig
    feature_names: tuple[str, ...]
    feature_units: tuple[str | None, ...]
    feature_values: tuple[np.ndarray, ...]
    target_names: tuple[str, ...]
    target_units: tuple[str | None, ...]
    target_values: tuple[np.ndarray, ...]
    target_shape: tuple[int, ...]
    simulation_count: int
    examples_per_simulation: int
    component_checksums: tuple[str, ...]
    checksum: str

    @property
    def feature_schema(self) -> tuple[tuple[str, str | None], ...]:
        return tuple(zip(self.feature_names, self.feature_units, strict=True))

    @property
    def target_schema(self) -> tuple[tuple[str, str | None], ...]:
        return tuple(zip(self.target_names, self.target_units, strict=True))


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


def _validate_named_family(
    values: object,
    label: str,
    issues: list[str],
) -> tuple[tuple[str, ...], tuple[object, ...]] | None:
    if not isinstance(values, Mapping) or not values:
        issues.append(f"{label} must be a non-empty mapping.")
        return None
    names = tuple(values.keys())
    invalid = [name for name in names if not isinstance(name, str) or not name.strip()]
    if invalid:
        issues.append(f"{label} names must be non-empty strings.")
        return None
    return names, tuple(values[name] for name in names)


def _is_real_numeric_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray)
        and (
            np.issubdtype(value.dtype, np.integer)
            or np.issubdtype(value.dtype, np.floating)
        )
        and not np.issubdtype(value.dtype, np.bool_)
    )


def _prepare_value(value: object) -> tuple[np.ndarray | None, str | None]:
    if isinstance(value, u.Quantity):
        array = np.asarray(value.value)
        return (array if _is_real_numeric_array(array) else None), unit_string(value.unit)
    return (value if _is_real_numeric_array(value) else None), None


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


def _prepared_data_checksums(
    feature_names: tuple[str, ...],
    feature_units: tuple[str | None, ...],
    feature_values: tuple[np.ndarray, ...],
    target_names: tuple[str, ...],
    target_units: tuple[str | None, ...],
    target_values: tuple[np.ndarray, ...],
) -> tuple[tuple[str, ...], str]:
    """Return component and aggregate identity for normalized prepared data."""
    component_checksums = tuple(
        _array_checksum(values) for values in (*feature_values, *target_values)
    )
    identity = {
        "features": [
            {
                "name": name,
                "unit": feature_units[index],
                "checksum": component_checksums[index],
            }
            for index, name in enumerate(feature_names)
        ],
        "targets": [
            {
                "name": name,
                "unit": target_units[index],
                "checksum": component_checksums[len(feature_names) + index],
            }
            for index, name in enumerate(target_names)
        ],
    }
    checksum = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return component_checksums, checksum


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
    feature_family = _validate_named_family(config.features, f"{label}.features", issues)
    target_family = _validate_named_family(config.targets, f"{label}.targets", issues)
    if feature_family is None or target_family is None:
        return None
    feature_names, raw_feature_values = feature_family
    target_names, raw_target_values = target_family

    target_shape: tuple[int, ...] | None = None
    target_values: list[np.ndarray] = []
    target_units: list[str | None] = []
    for name, raw_values in zip(target_names, raw_target_values, strict=True):
        values, unit = _prepare_value(raw_values)
        item_label = f"{label}.targets[{name!r}]"
        if values is None:
            issues.append(f"{item_label} must be a real numerical NumPy array or Astropy Quantity.")
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
        target_units.append(unit)

    if target_shape is None:
        return None
    simulation_count = target_shape[0]
    feature_values: list[np.ndarray] = []
    feature_units: list[str | None] = []
    for name, raw_values in zip(feature_names, raw_feature_values, strict=True):
        values, unit = _prepare_value(raw_values)
        item_label = f"{label}.features[{name!r}]"
        if values is None:
            issues.append(f"{item_label} must be a real numerical NumPy array or Astropy Quantity.")
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
        feature_units.append(unit)

    if len(issues) != start_issue_count:
        return None
    prepared_feature_units = tuple(feature_units)
    prepared_feature_values = tuple(feature_values)
    prepared_target_units = tuple(target_units)
    prepared_target_values = tuple(target_values)
    component_checksums, checksum = _prepared_data_checksums(
        feature_names,
        prepared_feature_units,
        prepared_feature_values,
        target_names,
        prepared_target_units,
        prepared_target_values,
    )
    return _PreparedModelTrainingData(
        config=config,
        feature_names=feature_names,
        feature_units=prepared_feature_units,
        feature_values=prepared_feature_values,
        target_names=target_names,
        target_units=prepared_target_units,
        target_values=prepared_target_values,
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
) -> _PreparedModelTrainingData:
    """Require matching names and normalize validation values to training units."""

    if training.feature_names != validation.feature_names:
        issues.append(
            "training_data and validation_data must have identical ordered feature names."
        )
    if training.target_names != validation.target_names:
        issues.append(
            "training_data and validation_data must have identical ordered target names."
        )
    converted_features = (
        _convert_explicit_units(
            training.feature_units,
            validation.feature_units,
            validation.feature_values,
            "feature",
            issues,
        )
        if training.feature_names == validation.feature_names
        else None
    )
    converted_targets = (
        _convert_explicit_units(
            training.target_units,
            validation.target_units,
            validation.target_values,
            "target",
            issues,
        )
        if training.target_names == validation.target_names
        else None
    )
    if converted_features is None or converted_targets is None:
        return validation
    if (
        validation.feature_units == training.feature_units
        and validation.target_units == training.target_units
    ):
        return validation
    component_checksums, checksum = _prepared_data_checksums(
        validation.feature_names,
        training.feature_units,
        converted_features,
        validation.target_names,
        training.target_units,
        converted_targets,
    )
    return replace(
        validation,
        feature_units=training.feature_units,
        feature_values=converted_features,
        target_units=training.target_units,
        target_values=converted_targets,
        component_checksums=component_checksums,
        checksum=checksum,
    )


def _convert_explicit_units(
    training_units: tuple[str | None, ...],
    validation_units: tuple[str | None, ...],
    values: tuple[np.ndarray, ...],
    family: str,
    issues: list[str],
) -> tuple[np.ndarray, ...] | None:
    start_issue_count = len(issues)
    converted: list[np.ndarray] = []
    for index, (training_unit, validation_unit, array) in enumerate(
        zip(training_units, validation_units, values, strict=True)
    ):
        if training_unit is None or validation_unit is None:
            if training_unit != validation_unit:
                issues.append(
                    f"training_data and validation_data {family} {index} must both be physical quantities or both be nonphysical arrays."
                )
            else:
                converted.append(array)
            continue
        training_astropy_unit = u.Unit(training_unit)
        validation_astropy_unit = u.Unit(validation_unit)
        if training_astropy_unit == validation_astropy_unit:
            converted.append(array)
            continue
        try:
            converted.append(
                np.asarray(
                    u.Quantity(
                        array,
                        unit=validation_astropy_unit,
                        copy=False,
                    ).to_value(training_astropy_unit)
                )
            )
        except u.UnitConversionError:
            issues.append(
                f"training_data and validation_data {family} {index} units are not equivalent."
            )
    return None if len(issues) != start_issue_count else tuple(converted)


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
            "name": name,
            "unit": prepared.feature_units[index],
            "mean": state.feature_means[index],
            "scale": state.feature_scales[index],
        }
        for index, name in enumerate(prepared.feature_names)
    ]
    targets = [
        {
            "name": name,
            "unit": prepared.target_units[index],
            "mean": state.target_means[index],
            "scale": state.target_scales[index],
        }
        for index, name in enumerate(prepared.target_names)
    ]
    return features, targets
