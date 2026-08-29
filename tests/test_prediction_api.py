"""Contract tests for model loading, prediction, and evaluation."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import weakref
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ao_predict import (
    ModelEvaluationResult,
    ModelPredictor,
    TrainModelRequest,
    load_model_predictor,
    model_training_data_from_rows,
    train_model,
)
from ao_predict._model import build_dense_model, cpu_state_dict
from ao_predict.training.artifacts import (
    prepare_training_parent,
    producer_version,
    publish_model_package,
    resolve_training_paths,
)


def _model_metadata() -> dict[str, object]:
    return {
        "kind": "ao_predict_dense_regression_model",
        "version": 1,
        "producer_version": producer_version(),
        "model": {
            "input_width": 2,
            "hidden_widths": [],
            "output_width": 2,
            "hidden_activation": "relu",
            "output_activation": "linear",
            "bias": True,
        },
        "features": [
            {"name": "shared", "unit": "m", "mean": 10.0, "scale": 2.0},
            {"name": "related", "unit": None, "mean": 20.0, "scale": 4.0},
        ],
        "targets": [
            {"name": "first", "unit": "%", "mean": 100.0, "scale": 10.0},
            {"name": "second", "unit": "mas", "mean": 200.0, "scale": 20.0},
        ],
        "numerical": {
            "model_dtype": "float32",
            "prediction_dtype": "float32",
            "standardization_variance": "population",
            "constant_scale": 1.0,
            "objective": "physical_relative_mse",
        },
        "training_seed": 7,
    }


def _publish_known_model(model_path: Path) -> Path:
    model, _ = build_dense_model(2, (), 2, initialization_seed=0)
    linear = model.network[0]
    assert isinstance(linear, torch.nn.Linear)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.5]]))
        linear.bias.copy_(torch.tensor([0.25, -0.5]))
    paths = resolve_training_paths(model_path)
    prepare_training_parent(paths)
    publish_model_package(paths, _model_metadata(), cpu_state_dict(model))
    return paths.package_path


def _replace_package_member(package_path: Path, name: str, content: bytes) -> None:
    with zipfile.ZipFile(package_path) as archive:
        members = {
            member_name: archive.read(member_name)
            for member_name in archive.namelist()
        }
    members[name] = content
    if name != "manifest.json":
        manifest = json.loads(members["manifest.json"])
        manifest["members"][name] = {
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        members["manifest.json"] = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    with zipfile.ZipFile(package_path, "w") as archive:
        for member_name in ("manifest.json", "metadata.json", "weights.pt"):
            archive.writestr(member_name, members[member_name])


@pytest.fixture
def predictor(tmp_path: Path) -> ModelPredictor:
    model_path = tmp_path / "known"
    _publish_known_model(model_path)
    return load_model_predictor(model_path, batch_size=2)


def _expected(features: np.ndarray) -> np.ndarray:
    standardized = (features.astype(np.float32) - [10.0, 20.0]) / [2.0, 4.0]
    weights = np.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)
    bias = np.asarray([0.25, -0.5], dtype=np.float32)
    standardized_targets = standardized @ weights.T + bias
    return (
        standardized_targets * np.asarray([10.0, 20.0], dtype=np.float32)
        + np.asarray([100.0, 200.0], dtype=np.float32)
    ).astype(np.float32)


def test_load_exposes_runtime_and_model_contract_for_both_path_forms(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "nested" / "known"
    package_path = _publish_known_model(model_path)

    from_stem = load_model_predictor(model_path, batch_size=9)
    from_package = load_model_predictor(package_path)

    assert isinstance(from_stem, ModelPredictor)
    assert from_stem.model_path == model_path
    assert from_stem.model_package_path == package_path
    assert from_package.model_path == model_path
    assert from_package.model_package_path == package_path
    assert from_stem.device == "cpu"
    assert from_stem.batch_size == 9
    assert from_stem.feature_names == ("shared", "related")
    assert from_stem.feature_units == ("m", None)
    assert from_stem.target_names == ("first", "second")
    assert from_stem.target_units == ("%", "mas")
    with pytest.raises(TypeError, match="load_model_predictor"):
        ModelPredictor()


def test_direct_prediction_preserves_order_batches_and_float32(
    predictor: ModelPredictor,
) -> None:
    features = np.asarray(
        [[10.0, 20.0], [12.0, 24.0], [8.0, 16.0], [14.0, 20.0], [6.0, 28.0]],
        dtype=np.float64,
    )
    original = features.copy()

    actual = predictor.predict(features)
    overridden = predictor.predict(features, batch_size=20)

    np.testing.assert_allclose(actual, _expected(features), rtol=1e-6)
    np.testing.assert_allclose(overridden, actual, rtol=1e-6)
    np.testing.assert_array_equal(features, original)
    assert actual.shape == (5, 2)
    assert actual.dtype == np.float32


def test_noncontiguous_direct_prediction_is_gathered_by_batch(
    predictor: ModelPredictor,
) -> None:
    storage = np.asarray(
        [
            [10.0, -1.0, 20.0, -1.0],
            [12.0, -1.0, 24.0, -1.0],
            [8.0, -1.0, 16.0, -1.0],
            [14.0, -1.0, 20.0, -1.0],
        ]
    )
    features = storage[:, ::2]
    assert not features.flags.c_contiguous

    actual = predictor.predict(features, batch_size=3)

    np.testing.assert_allclose(actual, _expected(features), rtol=1e-6)


def test_named_prediction_shares_simulation_features_without_repetition(
    predictor: ModelPredictor,
) -> None:
    shared = np.asarray([10, 14], dtype=np.int16)
    related = np.asarray([[20.0, 24.0, 16.0], [28.0, 20.0, 12.0]])
    features = {"related": related, "shared": shared}
    expanded = np.stack(
        (
            np.repeat(shared[:, None], related.shape[1], axis=1),
            related,
        ),
        axis=-1,
    )

    actual = predictor.predict(features, batch_size=4)

    np.testing.assert_allclose(actual, _expected(expanded.reshape(-1, 2)).reshape(2, 3, 2))
    assert actual.shape == (2, 3, 2)
    assert actual.dtype == np.float32


def test_noncontiguous_named_features_preserve_simulation_major_order(
    predictor: ModelPredictor,
) -> None:
    shared_storage = np.asarray([10.0, -1.0, 14.0, -1.0])
    shared = shared_storage[::2]
    related = np.asarray(
        [[20.0, 28.0], [24.0, 20.0], [16.0, 12.0]]
    ).T
    assert not shared.flags.c_contiguous
    assert not related.flags.c_contiguous
    expanded = np.stack(
        (
            np.repeat(shared[:, None], related.shape[1], axis=1),
            related,
        ),
        axis=-1,
    )

    actual = predictor.predict(
        {"related": related, "shared": shared},
        batch_size=2,
    )

    np.testing.assert_allclose(
        actual,
        _expected(expanded.reshape(-1, 2)).reshape(2, 3, 2),
    )


def test_prediction_borrows_named_arrays_only_for_the_call(
    predictor: ModelPredictor,
) -> None:
    shared = np.asarray([10.0, 12.0])
    related = np.asarray([20.0, 24.0])
    shared_before = shared.copy()
    related_before = related.copy()
    shared_reference = weakref.ref(shared)
    related_reference = weakref.ref(related)
    features = {"shared": shared, "related": related}

    predictor.predict(features)

    np.testing.assert_array_equal(shared, shared_before)
    np.testing.assert_array_equal(related, related_before)
    del features, shared, related
    gc.collect()
    assert shared_reference() is None
    assert related_reference() is None


def test_all_rank_one_named_features_return_simulation_target_matrix(
    predictor: ModelPredictor,
) -> None:
    features = {
        "related": np.asarray([20.0, 24.0]),
        "shared": np.asarray([10.0, 12.0]),
    }

    actual = predictor.predict(features)

    assert actual.shape == (2, 2)
    np.testing.assert_allclose(actual, _expected(np.asarray([[10, 20], [12, 24]])))


def test_predict_one_accepts_positional_and_named_values(
    predictor: ModelPredictor,
) -> None:
    positional = predictor.predict_one(np.asarray([12, 24], dtype=np.int64))
    named = predictor.predict_one({"related": np.float64(24), "shared": 12})

    np.testing.assert_allclose(positional, _expected(np.asarray([[12, 24]]))[0])
    np.testing.assert_array_equal(named, positional)
    assert positional.shape == (2,)
    assert positional.dtype == np.float32


def test_empty_prediction_restores_direct_and_structured_shapes(
    predictor: ModelPredictor,
) -> None:
    direct = predictor.predict(np.empty((0, 2), dtype=np.float32))
    structured = predictor.predict(
        {
            "shared": np.empty((0,), dtype=np.float32),
            "related": np.empty((0, 3), dtype=np.float32),
        }
    )

    assert direct.shape == (0, 2)
    assert structured.shape == (0, 3, 2)
    assert direct.dtype == structured.dtype == np.float32


def test_evaluate_pools_physical_relative_error_across_all_values(
    predictor: ModelPredictor,
) -> None:
    features = np.asarray([[10.0, 20.0], [12.0, 24.0], [8.0, 16.0]])
    prediction = _expected(features)
    targets = prediction * np.asarray([[1.1, 0.8], [0.9, 1.2], [1.05, 0.95]])
    squared = np.square(
        (prediction.astype(np.float32) - targets.astype(np.float32))
        / targets.astype(np.float32)
    )

    result = predictor.evaluate(features, targets, batch_size=2)
    one_batch = predictor.evaluate(features, targets, batch_size=100)

    assert isinstance(result, ModelEvaluationResult)
    assert result.example_count == 3
    assert result.relative_mse == pytest.approx(float(np.sum(squared)) / 6)
    assert result.relative_rmse == pytest.approx(math.sqrt(result.relative_mse))
    assert result.target_relative_rmse == {
        "first": pytest.approx(math.sqrt(float(np.sum(squared[:, 0])) / 3)),
        "second": pytest.approx(math.sqrt(float(np.sum(squared[:, 1])) / 3)),
    }
    assert one_batch.relative_mse == pytest.approx(result.relative_mse, rel=1e-6)
    assert one_batch.relative_rmse == pytest.approx(result.relative_rmse, rel=1e-6)
    with pytest.raises(TypeError):
        result.target_relative_rmse["first"] = 0.0  # type: ignore[index]


def test_evaluate_accepts_independently_named_structured_targets(
    predictor: ModelPredictor,
) -> None:
    features = {
        "shared": np.asarray([10.0, 12.0]),
        "related": np.asarray([[20.0, 24.0], [16.0, 28.0]]),
    }
    prediction = predictor.predict(features)
    targets = {
        "second": prediction[..., 1],
        "first": prediction[..., 0],
    }

    result = predictor.evaluate(features, targets)

    assert result.example_count == 4
    assert result.relative_mse == 0.0
    assert result.relative_rmse == 0.0
    assert result.target_relative_rmse == {"first": 0.0, "second": 0.0}


def test_evaluate_accepts_both_mixed_forms_and_noncontiguous_targets(
    predictor: ModelPredictor,
) -> None:
    direct_features = np.asarray([[10.0, 20.0], [12.0, 24.0]])
    direct_prediction = predictor.predict(direct_features)
    mapped_targets = {
        "second": direct_prediction[:, 1],
        "first": direct_prediction[:, 0],
    }

    direct_result = predictor.evaluate(direct_features, mapped_targets)

    mapped_features = {
        "shared": np.asarray([10.0, 12.0]),
        "related": np.asarray([[20.0, 24.0], [16.0, 28.0]]),
    }
    structured_prediction = predictor.predict(mapped_features)
    target_storage = np.empty((2, 2, 4), dtype=np.float64)
    direct_targets = target_storage[..., ::2]
    direct_targets[...] = structured_prediction
    assert not direct_targets.flags.c_contiguous

    structured_result = predictor.evaluate(
        mapped_features,
        direct_targets,
        batch_size=3,
    )

    assert direct_result.relative_mse == 0.0
    assert structured_result.relative_mse == 0.0


def test_evaluation_matches_trainer_validation_error_semantics(tmp_path: Path) -> None:
    features = np.asarray(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0], [5.0, 3.0]],
        dtype=np.float32,
    )
    targets = np.asarray(
        [[2.0, 4.0], [3.0, 3.0], [4.0, 5.0], [5.0, 7.0], [6.0, 8.0]],
        dtype=np.float32,
    )
    training = model_training_data_from_rows(
        features[:3],
        targets[:3],
        ("a", "b"),
        ("first", "second"),
    )
    validation = model_training_data_from_rows(
        features[3:],
        targets[3:],
        ("a", "b"),
        ("first", "second"),
    )
    model_path = tmp_path / "trained"
    trained = train_model(
        TrainModelRequest(
            model_path=model_path,
            training_data=training,
            validation_data=validation,
            hidden_widths=(3,),
            batch_size=2,
            validation_batch_size=1,
            training_seed=37,
            warmup_epochs=0,
            minimum_training_epochs=0,
            maximum_validation_checks=1,
        )
    )

    evaluated = load_model_predictor(model_path).evaluate(features[3:], targets[3:])

    assert evaluated.relative_rmse == pytest.approx(
        trained.best_model_validation_error_percent / 100.0,
        rel=1e-6,
    )


@pytest.mark.parametrize(
    ("features", "error", "match"),
    [
        ([[1.0, 2.0]], TypeError, "NumPy array or a mapping"),
        (np.ones((2, 3)), ValueError, "2 columns"),
        (np.ones((2, 2), dtype=np.bool_), ValueError, "non-Boolean"),
        (np.asarray([[1.0, np.nan]]), ValueError, "finite"),
        (
            {"shared": np.ones(2), "wrong": np.ones(2)},
            ValueError,
            "names must match",
        ),
        (
            {"shared": np.ones(2), "related": np.ones((3, 2))},
            ValueError,
            "simulation count",
        ),
    ],
)
def test_predict_rejects_invalid_inputs(
    predictor: ModelPredictor,
    features: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        predictor.predict(features)  # type: ignore[arg-type]


def test_single_and_evaluation_validation_are_strict(
    predictor: ModelPredictor,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        predictor.predict_one(np.asarray([]))
    with pytest.raises(TypeError, match="real scalar"):
        predictor.predict_one({"shared": 10.0, "related": np.asarray(20.0)})
    with pytest.raises(ValueError, match="positive"):
        predictor.evaluate(np.ones((1, 2)), np.asarray([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="positive float32"):
        predictor.evaluate(
            np.ones((1, 2)),
            np.full((1, 2), np.nextafter(0.0, 1.0), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="at least one"):
        predictor.evaluate(np.empty((0, 2)), np.empty((0, 2)))
    with pytest.raises(ValueError, match="shape"):
        predictor.evaluate(np.ones((2, 2)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="positive integer"):
        predictor.predict(np.ones((1, 2)), batch_size=0)
    with pytest.raises(ValueError, match="finite"):
        predictor.predict_one({"shared": 10.0, "related": 10**1000})


def test_loading_uses_suffix_syntax_and_reports_missing_or_malformed_packages(
    tmp_path: Path,
) -> None:
    exact = tmp_path / "missing.model.zip"
    with pytest.raises(FileNotFoundError) as missing:
        load_model_predictor(exact)
    assert missing.value.filename == str(exact)

    malformed = tmp_path / "malformed.model.zip"
    malformed.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="malformed"):
        load_model_predictor(malformed)

    wrong_members = tmp_path / "wrong.model.zip"
    with zipfile.ZipFile(wrong_members, "w") as archive:
        archive.writestr("other", b"content")
    with pytest.raises(ValueError, match="exactly"):
        load_model_predictor(wrong_members)

    appended_only = tmp_path / "appended.model.zip"
    _publish_known_model(appended_only)
    with pytest.raises(FileNotFoundError) as syntactic:
        load_model_predictor(appended_only)
    assert syntactic.value.filename == str(appended_only)


def test_loading_rejects_invalid_serialized_weights_and_scalers(tmp_path: Path) -> None:
    invalid_weights = _publish_known_model(tmp_path / "invalid-weights")
    _replace_package_member(invalid_weights, "weights.pt", b"not torch data")
    with pytest.raises(ValueError, match="weights.pt is invalid"):
        load_model_predictor(invalid_weights)

    invalid_scaler = _publish_known_model(tmp_path / "invalid-scaler")
    with zipfile.ZipFile(invalid_scaler) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    metadata["features"][0]["mean"] = 10**1000
    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    _replace_package_member(invalid_scaler, "metadata.json", metadata_bytes)
    with pytest.raises(ValueError, match="must be finite"):
        load_model_predictor(invalid_scaler)


def test_loading_enforces_package_version_and_member_integrity(tmp_path: Path) -> None:
    unsupported = _publish_known_model(tmp_path / "unsupported")
    with zipfile.ZipFile(unsupported) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["version"] = 999
    _replace_package_member(
        unsupported,
        "manifest.json",
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
    )
    with pytest.raises(ValueError, match="Unsupported model package version"):
        load_model_predictor(unsupported)

    changed = _publish_known_model(tmp_path / "changed")
    with zipfile.ZipFile(changed) as archive:
        metadata_bytes = archive.read("metadata.json")
    _replace_package_member(changed, "metadata.json", metadata_bytes + b" ")
    with zipfile.ZipFile(changed) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    manifest["members"]["metadata.json"]["sha256"] = "0" * 64
    _replace_package_member(
        changed,
        "manifest.json",
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
    )
    with pytest.raises(ValueError, match="checksum validation"):
        load_model_predictor(changed)


def test_runtime_controls_validate_device_threads_and_batch_size(tmp_path: Path) -> None:
    model_path = tmp_path / "known"
    _publish_known_model(model_path)
    previous_threads = torch.get_num_threads()
    try:
        predictor = load_model_predictor(model_path, cpu_threads=1, batch_size=1)
        assert predictor.device == "cpu"
        assert load_model_predictor(model_path).batch_size == 16_384
        assert torch.get_num_threads() == 1
        with pytest.raises(ValueError, match="positive integer"):
            load_model_predictor(model_path, cpu_threads=0)
        with pytest.raises(ValueError, match="CPU devices"):
            load_model_predictor(model_path, device="cpu:0")
        with pytest.raises(ValueError, match="only with a CPU"):
            load_model_predictor(model_path, device="cuda", cpu_threads=1)
        if torch.cuda.is_available():
            with pytest.raises(ValueError, match="index"):
                load_model_predictor(model_path, device="cuda:999999")
            cpu_prediction = predictor.predict(np.asarray([[12.0, 24.0]]))
            cuda_predictor = load_model_predictor(model_path, device="cuda")
            assert cuda_predictor.device.startswith("cuda:")
            cuda_prediction = cuda_predictor.predict(np.asarray([[12.0, 24.0]]))
            np.testing.assert_allclose(
                cuda_prediction,
                cpu_prediction,
                rtol=1e-5,
                atol=1e-6,
            )
        else:
            with pytest.raises(ValueError, match="unavailable"):
                load_model_predictor(model_path, device="cuda")
        if torch.backends.mps.is_available():
            cpu_prediction = predictor.predict(np.asarray([[12.0, 24.0]]))
            mps_prediction = load_model_predictor(
                model_path,
                device="mps",
            ).predict(np.asarray([[12.0, 24.0]]))
            np.testing.assert_allclose(
                mps_prediction,
                cpu_prediction,
                rtol=1e-5,
                atol=1e-6,
            )
        else:
            with pytest.raises(ValueError, match="unavailable"):
                load_model_predictor(model_path, device="mps")
    finally:
        torch.set_num_threads(previous_threads)
