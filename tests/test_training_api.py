"""Tests for the public AO Predict model-training lifecycle."""

from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ao_predict import (
    FeatureConfig,
    ModelTrainingDataConfig,
    ModelTrainingValidationError,
    TargetConfig,
    TrainingRecoveryMismatchError,
    TrainingTerminationReason,
    TrainModelRequest,
    model_training_data_from_rows,
    train_model,
)
from ao_predict.training.artifacts import (
    load_model_package,
    load_recovery,
    resolve_training_paths,
)
from ao_predict.training.model import build_dense_model, derived_seed

training_api = importlib.import_module("ao_predict.training.api")


def _data(
    count: int,
    *,
    offset: float = 0.0,
) -> ModelTrainingDataConfig:
    values = np.arange(count, dtype=np.float32) + 1.0 + offset
    features = np.column_stack((values, np.square(values) / 10.0)).astype(np.float32)
    targets = np.column_stack((2.0 + values / 10.0, 4.0 + values / 20.0)).astype(
        np.float32
    )
    return model_training_data_from_rows(
        features,
        targets,
        ("linear", "quadratic"),
        ("metric_a", "metric_b"),
    )


def _request(
    model_path: Path,
    *,
    maximum_validation_checks: int = 2,
    training_seed: int | None = 123,
) -> TrainModelRequest:
    return TrainModelRequest(
        model_path=model_path,
        training_data=_data(5),
        validation_data=_data(3, offset=10.0),
        hidden_widths=(4,),
        batch_size=3,
        validation_batch_size=2,
        training_seed=training_seed,
        warmup_epochs=0,
        minimum_training_epochs=0,
        maximum_validation_checks=maximum_validation_checks,
    )


def test_coupled_request_validation_reports_all_relevant_issues(tmp_path: Path) -> None:
    request = TrainModelRequest(
        model_path=tmp_path / "model",
        training_data=ModelTrainingDataConfig(
            features=(FeatureConfig("", np.asarray([1.0, np.nan])),),
            targets=(TargetConfig("target", np.asarray([1.0, 0.0])),),
        ),
        hidden_widths=(0,),
        batch_size=0,
        validation_count=2,
        validation_fraction=0.5,
        warmup_start_fraction=0.0,
    )

    with pytest.raises(ModelTrainingValidationError) as captured:
        train_model(request)

    issues = captured.value.issues
    assert any("Exactly one" in issue for issue in issues)
    assert any("hidden_widths" in issue for issue in issues)
    assert any("batch_size" in issue for issue in issues)
    assert any("strictly positive" in issue for issue in issues)
    assert any("finite" in issue for issue in issues)


def test_automatic_partition_reports_generated_seed_and_owned_mask(
    tmp_path: Path,
) -> None:
    request = replace(
        _request(tmp_path / "model", maximum_validation_checks=1),
        training_data=_data(10),
        validation_data=None,
        validation_count=3,
        split_seed=None,
    )

    result = train_model(request)

    assert result.split_seed is not None
    assert result.validation_mask is not None
    assert result.validation_mask.dtype == np.bool_
    assert np.count_nonzero(result.validation_mask) == 3
    assert not result.validation_mask.flags.writeable


def test_batch_size_counts_examples_and_retains_partial_final_batch(
    tmp_path: Path,
) -> None:
    result = train_model(_request(tmp_path / "partial", maximum_validation_checks=1))
    oversized = train_model(
        replace(
            _request(tmp_path / "oversized", maximum_validation_checks=1),
            batch_size=100,
        )
    )

    assert result.optimizer_updates == 2
    assert result.training_examples_seen == 5
    assert result.validation_history[0].training_epochs == 1.0
    assert oversized.optimizer_updates == 1
    assert oversized.training_examples_seen == 5


def test_training_does_not_mutate_global_numpy_or_torch_random_state(
    tmp_path: Path,
) -> None:
    np.random.seed(987)
    torch.manual_seed(654)
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    train_model(_request(tmp_path / "model", maximum_validation_checks=1))

    numpy_after = np.random.get_state()
    torch_after = torch.random.get_rng_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch_after)


def test_same_seed_produces_identical_history_and_weights(tmp_path: Path) -> None:
    first = train_model(_request(tmp_path / "first"))
    second = train_model(_request(tmp_path / "second"))
    first_package = load_model_package(tmp_path / "first.model.zip")
    second_package = load_model_package(tmp_path / "second.model.zip")

    assert first.validation_history == second.validation_history
    assert set(first_package.weights) == set(second_package.weights)
    for name in first_package.weights:
        assert torch.equal(first_package.weights[name], second_package.weights[name])


def test_explicit_initialization_matches_pytorch_linear_defaults() -> None:
    seed = derived_seed(123, "model-initialization")
    actual, _ = build_dense_model(2, (4,), 2, initialization_seed=seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        expected = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))

    actual_linears = [
        module for module in actual.modules() if isinstance(module, nn.Linear)
    ]
    expected_linears = [
        module for module in expected.modules() if isinstance(module, nn.Linear)
    ]
    for actual_layer, expected_layer in zip(
        actual_linears, expected_linears, strict=True
    ):
        assert torch.equal(actual_layer.weight, expected_layer.weight)
        assert torch.equal(actual_layer.bias, expected_layer.bias)


def test_validation_measurement_uses_complete_physical_partition(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path / "model", maximum_validation_checks=1)
    result = train_model(request)
    package = load_model_package(tmp_path / "model.model.zip")
    definition = package.metadata["model"]
    model, _ = build_dense_model(
        definition["input_width"],
        tuple(definition["hidden_widths"]),
        definition["output_width"],
        initialization_seed=0,
    )
    model.load_state_dict(package.weights)
    feature_rows = np.column_stack(
        [feature.values for feature in request.validation_data.features]
    )
    target_rows = np.column_stack(
        [target.values for target in request.validation_data.targets]
    )
    feature_mean = np.asarray(
        [item["mean"] for item in package.metadata["features"]], dtype=np.float64
    )
    feature_scale = np.asarray(
        [item["scale"] for item in package.metadata["features"]], dtype=np.float64
    )
    target_mean = np.asarray(
        [item["mean"] for item in package.metadata["targets"]], dtype=np.float64
    )
    target_scale = np.asarray(
        [item["scale"] for item in package.metadata["targets"]], dtype=np.float64
    )
    standardized_features = ((feature_rows - feature_mean) / feature_scale).astype(
        np.float32
    )
    with torch.no_grad():
        standardized_predictions = model(
            torch.from_numpy(standardized_features)
        ).numpy()
    predictions = standardized_predictions * target_scale + target_mean
    expected = float(np.mean(np.square((predictions - target_rows) / target_rows)))

    assert result.validation_history[0].validation_objective == pytest.approx(
        expected, rel=2.0e-7
    )
    assert result.validation_history[0].validation_error_percent == pytest.approx(
        np.sqrt(expected) * 100.0,
        rel=2.0e-7,
    )


def test_scheduler_and_early_stopping_use_action_on_this_check_semantics(
    tmp_path: Path,
) -> None:
    request = replace(
        _request(tmp_path / "model", maximum_validation_checks=3),
        scheduler_patience_checks=1,
        scheduler_minimum_improvement_fraction=0.99,
        early_stopping_patience_checks=1,
        early_stopping_minimum_improvement_percent=1_000.0,
    )

    result = train_model(request)

    assert result.termination_reason is TrainingTerminationReason.EARLY_STOPPING
    assert result.validation_checks == 3
    assert result.validation_history[0].learning_rate_after == pytest.approx(1.0e-3)
    assert result.validation_history[1].learning_rate_after == pytest.approx(5.0e-4)


def test_interrupted_and_resumed_training_matches_uninterrupted_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uninterrupted = train_model(
        _request(tmp_path / "uninterrupted", maximum_validation_checks=3)
    )
    real_save = training_api.atomic_save_recovery
    calls = 0

    def save_then_interrupt(paths, value):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        real_save(paths, value)
        if calls == 1:
            raise RuntimeError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(training_api, "atomic_save_recovery", save_then_interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train_model(_request(tmp_path / "resumed", maximum_validation_checks=3))
    prefix = (tmp_path / "resumed.training.log").read_text(encoding="utf-8")
    resumed = train_model(_request(tmp_path / "resumed", maximum_validation_checks=3))
    final_log = (tmp_path / "resumed.training.log").read_text(encoding="utf-8")
    uninterrupted_package = load_model_package(tmp_path / "uninterrupted.model.zip")
    resumed_package = load_model_package(tmp_path / "resumed.model.zip")

    assert final_log.startswith(prefix)
    assert "continuation:" in final_log
    assert resumed.validation_history == uninterrupted.validation_history
    for name in uninterrupted_package.weights:
        assert torch.equal(
            uninterrupted_package.weights[name], resumed_package.weights[name]
        )
    assert not (tmp_path / "resumed.recovery.pt").exists()


def test_terminal_publication_recovery_does_not_fit_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_publish = training_api.publish_model_package
    calls = 0

    def publish_then_interrupt(paths, metadata, weights):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        result = real_publish(paths, metadata, weights)
        if calls == 1:
            raise RuntimeError("publication interruption")
        return result

    request = _request(tmp_path / "model", maximum_validation_checks=1)
    with monkeypatch.context() as patch:
        patch.setattr(training_api, "publish_model_package", publish_then_interrupt)
        with pytest.raises(RuntimeError, match="publication interruption"):
            train_model(request)
    assert (tmp_path / "model.model.zip").exists()
    assert (tmp_path / "model.recovery.pt").exists()

    result = train_model(request)

    assert result.validation_checks == 1
    assert not (tmp_path / "model.recovery.pt").exists()
    assert "continuation:" in (tmp_path / "model.training.log").read_text(
        encoding="utf-8"
    )


def test_changed_data_is_rejected_during_exact_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_save = training_api.atomic_save_recovery

    def save_then_interrupt(paths, value):  # type: ignore[no-untyped-def]
        real_save(paths, value)
        raise RuntimeError("stop")

    request = _request(tmp_path / "model", maximum_validation_checks=2)
    with monkeypatch.context() as patch:
        patch.setattr(training_api, "atomic_save_recovery", save_then_interrupt)
        with pytest.raises(RuntimeError, match="stop"):
            train_model(request)
    changed_values = request.training_data.features[0].values.copy()
    changed_values[0] += 1.0
    changed_data = ModelTrainingDataConfig(
        features=(
            FeatureConfig("linear", changed_values),
            request.training_data.features[1],
        ),
        targets=request.training_data.targets,
    )

    with pytest.raises(TrainingRecoveryMismatchError) as captured:
        train_model(replace(request, training_data=changed_data))

    assert any("data identity differs" in item for item in captured.value.mismatches)


def test_recovery_is_weights_only_loadable_and_contains_no_source_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_save = training_api.atomic_save_recovery

    def save_then_interrupt(paths, value):  # type: ignore[no-untyped-def]
        real_save(paths, value)
        raise RuntimeError("stop")

    request = _request(tmp_path / "model", maximum_validation_checks=2)
    with monkeypatch.context() as patch:
        patch.setattr(training_api, "atomic_save_recovery", save_then_interrupt)
        with pytest.raises(RuntimeError, match="stop"):
            train_model(request)
    recovery = load_recovery(resolve_training_paths(request.model_path))

    def walk(value):  # type: ignore[no-untyped-def]
        assert not isinstance(value, np.ndarray)
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(recovery)
    assert "training_data" not in recovery
    assert "validation_data" not in recovery


def test_explicit_partitions_accept_compatible_different_ranks_and_dtypes(
    tmp_path: Path,
) -> None:
    training = ModelTrainingDataConfig(
        features=(FeatureConfig("feature", np.asarray([1, 2, 3], dtype=np.int16)),),
        targets=(TargetConfig("target", np.asarray([2, 3, 4], dtype=np.float32)),),
    )
    validation = ModelTrainingDataConfig(
        features=(
            FeatureConfig(
                "feature",
                np.asarray([[4.0, 4.5], [5.0, 5.5]], dtype=np.float64),
            ),
        ),
        targets=(
            TargetConfig(
                "target",
                np.asarray([[5.0, 5.5], [6.0, 6.5]], dtype=np.float64),
            ),
        ),
    )
    request = TrainModelRequest(
        model_path=tmp_path / "model",
        training_data=training,
        validation_data=validation,
        hidden_widths=(),
        batch_size=2,
        training_seed=19,
        warmup_epochs=0,
        minimum_training_epochs=0,
        maximum_validation_checks=1,
    )

    result = train_model(request)

    assert result.validation_checks == 1
    package = load_model_package(tmp_path / "model.model.zip")
    assert package.metadata["model"]["hidden_widths"] == []


def test_multiple_related_examples_split_as_complete_simulations(
    tmp_path: Path,
) -> None:
    shared = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    positions = np.arange(12, dtype=np.float32).reshape(4, 3) + 1.0
    data = ModelTrainingDataConfig(
        features=(
            FeatureConfig("shared", shared),
            FeatureConfig("position", positions),
        ),
        targets=(TargetConfig("metric", positions + shared[:, None]),),
    )
    request = TrainModelRequest(
        model_path=tmp_path / "model",
        training_data=data,
        validation_count=1,
        split_seed=7,
        hidden_widths=(3,),
        batch_size=4,
        training_seed=11,
        warmup_epochs=0,
        minimum_training_epochs=0,
        maximum_validation_checks=1,
    )

    result = train_model(request)

    assert np.count_nonzero(result.validation_mask) == 1
    assert result.training_examples_seen == 9
    assert result.optimizer_updates == 3


def test_generated_training_seed_is_returned_and_persisted(tmp_path: Path) -> None:
    request = _request(
        tmp_path / "model",
        maximum_validation_checks=1,
        training_seed=None,
    )

    result = train_model(request)
    package = load_model_package(tmp_path / "model.model.zip")
    log = (tmp_path / "model.training.log").read_text(encoding="utf-8")

    assert 0 <= result.training_seed < 2**63
    assert package.metadata["training_seed"] == result.training_seed
    assert f"training_seed: {result.training_seed}" in log


def test_default_validation_batch_size_and_data_identity_are_logged(
    tmp_path: Path,
) -> None:
    request = replace(
        _request(tmp_path / "model", maximum_validation_checks=1),
        validation_batch_size=None,
    )

    train_model(request)
    log = (tmp_path / "model.training.log").read_text(encoding="utf-8")

    assert "validation_batch_size: 6" in log
    assert "training_data_sha256:" in log
    assert "validation_data_sha256:" in log
    assert "training_simulation_count: 5" in log
    assert "validation_simulation_count: 3" in log


def test_cpu_threads_are_rejected_for_non_cpu_training(tmp_path: Path) -> None:
    request = replace(
        _request(tmp_path / "model", maximum_validation_checks=1),
        device="mps",
        cpu_threads=1,
    )

    with pytest.raises(ModelTrainingValidationError, match="only with a CPU"):
        train_model(request)


def test_invalid_device_is_rejected_before_output_parent_creation(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "not-created" / "model"
    request = replace(
        _request(model_path, maximum_validation_checks=1),
        device="meta",
    )

    with pytest.raises(ModelTrainingValidationError, match="cpu, cuda, or mps"):
        train_model(request)

    assert not model_path.parent.exists()


def test_warmup_schedule_reaches_both_declared_endpoints(tmp_path: Path) -> None:
    request = _request(tmp_path / "model")

    assert training_api._warmup_learning_rate(request, 1, 3) == pytest.approx(1.0e-4)
    assert training_api._warmup_learning_rate(request, 2, 3) == pytest.approx(5.5e-4)
    assert training_api._warmup_learning_rate(request, 3, 3) == pytest.approx(1.0e-3)


def test_runtime_difference_rejects_exact_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_save = training_api.atomic_save_recovery

    def save_then_interrupt(paths, value):  # type: ignore[no-untyped-def]
        real_save(paths, value)
        raise RuntimeError("stop")

    request = _request(tmp_path / "model", maximum_validation_checks=2)
    with monkeypatch.context() as patch:
        patch.setattr(training_api, "atomic_save_recovery", save_then_interrupt)
        with pytest.raises(RuntimeError, match="stop"):
            train_model(request)
    real_fingerprint = training_api._runtime_fingerprint

    def changed_fingerprint(device, validation_batch_size):  # type: ignore[no-untyped-def]
        result = real_fingerprint(device, validation_batch_size)
        result["test_runtime_property"] = "changed"
        return result

    with monkeypatch.context() as patch:
        patch.setattr(training_api, "_runtime_fingerprint", changed_fingerprint)
        with pytest.raises(TrainingRecoveryMismatchError) as captured:
            train_model(request)

    assert any(
        "runtime test_runtime_property differs" in item
        for item in captured.value.mismatches
    )


def test_unsupported_recovery_version_is_reported_as_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_save = training_api.atomic_save_recovery

    def save_then_interrupt(paths, value):  # type: ignore[no-untyped-def]
        real_save(paths, value)
        raise RuntimeError("stop")

    request = _request(tmp_path / "model", maximum_validation_checks=2)
    with monkeypatch.context() as patch:
        patch.setattr(training_api, "atomic_save_recovery", save_then_interrupt)
        with pytest.raises(RuntimeError, match="stop"):
            train_model(request)
    paths = resolve_training_paths(request.model_path)
    recovery = load_recovery(paths)
    recovery["version"] = 999
    real_save(paths, recovery)

    with pytest.raises(
        TrainingRecoveryMismatchError, match="Unsupported recovery version"
    ):
        train_model(request)
