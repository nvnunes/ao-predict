"""Tests for public model-training data and private prepared-state contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy import units as u

from ao_predict import ModelTrainingDataConfig
from ao_predict.training.data import (
    _array_checksum,
    _ExampleSet,
    fit_standardization,
    prepare_model_training_data,
    resolve_automatic_split,
    standardize_data,
    validate_explicit_schema,
)


def _prepare(config: ModelTrainingDataConfig):  # type: ignore[no-untyped-def]
    issues: list[str] = []
    prepared = prepare_model_training_data(config, label="data", issues=issues)
    assert issues == []
    assert prepared is not None
    return prepared


def test_feature_centered_data_broadcasts_only_shared_features_per_batch() -> None:
    shared = np.asarray([10.0, 20.0, 30.0])
    resolved = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    target = resolved + 1.0
    prepared = _prepare(
        ModelTrainingDataConfig(
            features={"shared": shared, "resolved": resolved},
            targets={"target": target},
        )
    )
    state = fit_standardization(prepared, np.asarray([0, 1, 2]))
    standardized = standardize_data(prepared, state)
    examples = _ExampleSet(standardized, np.asarray([0, 1, 2]))

    features, targets = examples.gather(np.asarray([1, 4]))

    assert standardized.feature_values[0].shape == (3,)
    assert standardized.feature_values[1].shape == (3, 2)
    np.testing.assert_array_equal(
        features[:, 0],
        standardized.feature_values[0][[0, 2]],
    )
    np.testing.assert_array_equal(
        features[:, 1],
        standardized.feature_values[1][[0, 2], [1, 0]],
    )
    np.testing.assert_array_equal(
        targets[:, 0],
        standardized.target_values[0][[0, 2], [1, 0]],
    )


def test_rank_one_shared_feature_moments_equal_explicit_repetition() -> None:
    shared = np.asarray([2.0, 4.0, 8.0], dtype=np.float32)
    targets = np.ones((3, 5), dtype=np.float32)
    prepared = _prepare(
        ModelTrainingDataConfig(
            features={"shared": shared},
            targets={"target": targets},
        )
    )

    state = fit_standardization(prepared, np.asarray([0, 1, 2]))
    repeated = np.repeat(shared, targets.shape[1])

    assert state.feature_means[0] == float(np.mean(repeated, dtype=np.float64))
    assert state.feature_scales[0] == pytest.approx(
        float(np.std(repeated, dtype=np.float64)),
        rel=1.0e-15,
    )


def test_constant_columns_use_scale_one_and_owned_float32_state() -> None:
    prepared = _prepare(
        ModelTrainingDataConfig(
            features={"constant": np.full(4, 7.0)},
            targets={"target": np.full(4, 3.0)},
        )
    )

    state = fit_standardization(prepared, np.arange(4))
    standardized = standardize_data(prepared, state)

    assert state.feature_scales == (1.0,)
    assert state.target_scales == (1.0,)
    assert standardized.feature_values[0].dtype == np.float32
    assert standardized.target_values[0].dtype == np.float32
    np.testing.assert_array_equal(standardized.feature_values[0], 0.0)
    np.testing.assert_array_equal(standardized.target_values[0], 0.0)


def test_explicit_validation_values_are_converted_to_training_units() -> None:
    training = _prepare(
        ModelTrainingDataConfig(
            features={"distance": np.asarray([1.0, 2.0]) * u.m},
            targets={"fwhm": np.asarray([100.0, 200.0]) * u.mas},
        )
    )
    validation = _prepare(
        ModelTrainingDataConfig(
            features={"distance": np.asarray([100.0, 200.0]) * u.cm},
            targets={"fwhm": np.asarray([0.1, 0.2]) * u.arcsec},
        )
    )
    issues: list[str] = []

    converted = validate_explicit_schema(training, validation, issues)

    assert issues == []
    assert converted.feature_units == ("m",)
    assert converted.target_units == ("mas",)
    np.testing.assert_allclose(converted.feature_values[0], [1.0, 2.0])
    np.testing.assert_allclose(converted.target_values[0], [100.0, 200.0])
    assert converted.component_checksums == (
        _array_checksum(converted.feature_values[0]),
        _array_checksum(converted.target_values[0]),
    )

    canonical_validation = _prepare(
        ModelTrainingDataConfig(
            features={"distance": np.asarray([1.0, 2.0]) * u.m},
            targets={"fwhm": np.asarray([100.0, 200.0]) * u.mas},
        )
    )
    assert converted.checksum == canonical_validation.checksum


def test_explicit_validation_borrows_values_already_in_training_units() -> None:
    training = _prepare(
        ModelTrainingDataConfig(
            features={"distance": np.asarray([1.0, 2.0]) * u.m},
            targets={"fwhm": np.asarray([100.0, 200.0]) * u.mas},
        )
    )
    validation = _prepare(
        ModelTrainingDataConfig(
            features={"distance": np.asarray([3.0, 4.0]) * u.m},
            targets={"fwhm": np.asarray([300.0, 400.0]) * u.mas},
        )
    )
    issues: list[str] = []

    converted = validate_explicit_schema(training, validation, issues)

    assert issues == []
    assert converted is validation
    assert converted.feature_values[0] is validation.feature_values[0]
    assert converted.target_values[0] is validation.target_values[0]


def test_explicit_validation_rejects_physical_and_nonphysical_mismatch() -> None:
    training = _prepare(
        ModelTrainingDataConfig(
            features={"distance": np.asarray([1.0, 2.0]) * u.m},
            targets={"target": np.asarray([1.0, 2.0])},
        )
    )
    validation = _prepare(
        ModelTrainingDataConfig(
            features={"distance": np.asarray([1.0, 2.0])},
            targets={"target": np.asarray([1.0, 2.0])},
        )
    )
    issues: list[str] = []

    converted = validate_explicit_schema(training, validation, issues)

    assert converted is validation
    assert any("both be physical quantities" in issue for issue in issues)


def test_explicit_validation_rejects_incompatible_physical_units() -> None:
    training = _prepare(
        ModelTrainingDataConfig(
            features={"distance": np.asarray([1.0, 2.0]) * u.m},
            targets={"target": np.asarray([1.0, 2.0])},
        )
    )
    validation = _prepare(
        ModelTrainingDataConfig(
            features={"distance": np.asarray([1.0, 2.0]) * u.s},
            targets={"target": np.asarray([1.0, 2.0])},
        )
    )
    issues: list[str] = []

    converted = validate_explicit_schema(training, validation, issues)

    assert converted is validation
    assert any("units are not equivalent" in issue for issue in issues)


def test_float64_standardization_preserves_variation_before_float32_conversion() -> (
    None
):
    features = np.asarray(
        [100_000_000.0, 100_000_001.0, 100_000_002.0],
        dtype=np.float64,
    )
    prepared = _prepare(
        ModelTrainingDataConfig(
            features={"offset": features},
            targets={"target": np.asarray([1.0, 2.0, 3.0])},
        )
    )

    state = fit_standardization(prepared, np.arange(3))
    standardized = standardize_data(prepared, state)

    np.testing.assert_allclose(
        standardized.feature_values[0],
        np.asarray([-1.2247449, 0.0, 1.2247449], dtype=np.float32),
        rtol=1.0e-6,
    )


def test_checksum_ignores_layout_but_includes_dtype_shape_and_values() -> None:
    rows = np.arange(24, dtype=np.float64).reshape(6, 4)
    view = rows[:, 2]
    copy = np.ascontiguousarray(view)

    assert not view.flags.c_contiguous
    assert _array_checksum(view) == _array_checksum(copy)
    assert _array_checksum(view) != _array_checksum(copy.astype(np.float32))
    assert _array_checksum(view) != _array_checksum(copy.reshape(2, 3))
    changed = copy.copy()
    changed[-1] += 1.0
    assert _array_checksum(view) != _array_checksum(changed)


def test_automatic_split_is_deterministic_and_rounds_fraction_up() -> None:
    prepared = _prepare(
        ModelTrainingDataConfig(
            features={
                "a": np.arange(30, dtype=np.float32).reshape(10, 3)[:, 0],
                "b": np.arange(30, dtype=np.float32).reshape(10, 3)[:, 1],
                "c": np.arange(30, dtype=np.float32).reshape(10, 3)[:, 2],
            },
            targets={"target": np.arange(10, dtype=np.float32) + 1.0},
        )
    )

    first = resolve_automatic_split(
        prepared,
        validation_count=None,
        validation_fraction=0.21,
        split_seed=123,
    )
    second = resolve_automatic_split(
        prepared,
        validation_count=None,
        validation_fraction=0.21,
        split_seed=123,
    )

    assert np.count_nonzero(first.validation_mask) == math.ceil(10 * 0.21)
    np.testing.assert_array_equal(first.validation_mask, second.validation_mask)
    assert set(first.training_simulations).isdisjoint(first.validation_simulations)


def test_target_broadcasting_and_non_positive_targets_are_rejected() -> None:
    issues: list[str] = []
    config = ModelTrainingDataConfig(
        features={"feature": np.ones(3)},
        targets={
            "resolved": np.ones((3, 2)),
            "shared": np.ones(3),
            "zero": np.asarray([1.0, 0.0, 2.0]),
        },
    )

    prepared = prepare_model_training_data(config, label="data", issues=issues)

    assert prepared is None
    assert any("shared target shape" in issue for issue in issues)
    assert any("strictly positive" in issue for issue in issues)


def test_invalid_feature_and_target_definition_types_are_collected() -> None:
    issues: list[str] = []
    config = ModelTrainingDataConfig(
        features=(object(),),  # type: ignore[arg-type]
        targets=(object(),),  # type: ignore[arg-type]
    )

    prepared = prepare_model_training_data(config, label="data", issues=issues)

    assert prepared is None
    assert any("features must be a non-empty mapping" in issue for issue in issues)
    assert any("targets must be a non-empty mapping" in issue for issue in issues)
