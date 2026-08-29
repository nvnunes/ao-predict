"""Tests for public model-training data and private prepared-state contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ao_predict import (
    FeatureConfig,
    ModelTrainingDataConfig,
    TargetConfig,
    model_training_data_from_rows,
)
from ao_predict.training.data import (
    _array_checksum,
    _ExampleSet,
    fit_standardization,
    prepare_model_training_data,
    resolve_automatic_split,
    standardize_data,
)


def _prepare(config: ModelTrainingDataConfig):  # type: ignore[no-untyped-def]
    issues: list[str] = []
    prepared = prepare_model_training_data(config, label="data", issues=issues)
    assert issues == []
    assert prepared is not None
    return prepared


def test_row_adapter_preserves_zero_copy_column_views() -> None:
    features = np.arange(20, dtype=np.float64).reshape(5, 4)
    targets = np.arange(10, dtype=np.float32).reshape(5, 2) + 1.0

    config = model_training_data_from_rows(
        features,
        targets,
        ("a", "b", "c", "d"),
        ("x", "y"),
        feature_units={"a": "m"},
    )

    assert np.shares_memory(config.features[0].values, features)
    assert np.shares_memory(config.targets[1].values, targets)
    assert config.features[0].unit == "m"
    assert config.features[1].unit is None


def test_feature_centered_data_broadcasts_only_shared_features_per_batch() -> None:
    shared = np.asarray([10.0, 20.0, 30.0])
    resolved = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    target = resolved + 1.0
    prepared = _prepare(
        ModelTrainingDataConfig(
            features=(
                FeatureConfig("shared", shared),
                FeatureConfig("resolved", resolved),
            ),
            targets=(TargetConfig("target", target),),
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
            features=(FeatureConfig("shared", shared),),
            targets=(TargetConfig("target", targets),),
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
            features=(FeatureConfig("constant", np.full(4, 7.0)),),
            targets=(TargetConfig("target", np.full(4, 3.0)),),
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
        model_training_data_from_rows(
            np.arange(30, dtype=np.float32).reshape(10, 3),
            np.arange(10, dtype=np.float32).reshape(10, 1) + 1.0,
            ("a", "b", "c"),
            ("target",),
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
        features=(FeatureConfig("feature", np.ones(3)),),
        targets=(
            TargetConfig("resolved", np.ones((3, 2))),
            TargetConfig("shared", np.ones(3)),
            TargetConfig("zero", np.asarray([1.0, 0.0, 2.0])),
        ),
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
    assert any("must be a FeatureConfig" in issue for issue in issues)
    assert any("must be a TargetConfig" in issue for issue in issues)


def test_row_adapter_requires_numpy_arrays_to_preserve_zero_copy_contract() -> None:
    with pytest.raises(TypeError, match="feature_rows must be a NumPy array"):
        model_training_data_from_rows(  # type: ignore[arg-type]
            [[1.0]],
            np.asarray([[1.0]]),
            ("feature",),
            ("target",),
        )
