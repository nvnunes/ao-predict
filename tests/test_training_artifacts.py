"""Tests for model-package, path, collision, and locking contracts."""

from __future__ import annotations

import importlib
import io
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from ao_predict import (
    ModelTrainingDataConfig,
    ModelTrainingValidationError,
    TrainingTerminationReason,
    TrainModelRequest,
    train_model,
)
from ao_predict.training.artifacts import (
    atomic_save_recovery,
    clean_temporary_paths,
    load_model_package,
    load_recovery,
    prepare_training_parent,
    publish_model_package,
    resolve_training_paths,
    training_path_lock,
)

training_artifacts = importlib.import_module("ao_predict.training.artifacts")
training_api = importlib.import_module("ao_predict.training.api")


def _training_data_from_rows(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
) -> ModelTrainingDataConfig:
    return ModelTrainingDataConfig(
        features={name: features[:, index] for index, name in enumerate(feature_names)},
        targets={name: targets[:, index] for index, name in enumerate(target_names)},
    )


def _request(model_path: Path, *, overwrite: bool = False) -> TrainModelRequest:
    features = np.asarray(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0]],
        dtype=np.float32,
    )
    targets = np.asarray([[2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    training = _training_data_from_rows(
        features[:3], targets[:3], ("a", "b"), ("metric",)
    )
    validation = _training_data_from_rows(
        features[3:], targets[3:], ("a", "b"), ("metric",)
    )
    return TrainModelRequest(
        model_path=model_path,
        training_data=training,
        validation_data=validation,
        hidden_widths=(3,),
        batch_size=2,
        overwrite=overwrite,
        training_seed=41,
        warmup_epochs=0,
        minimum_training_epochs=0,
        maximum_validation_checks=1,
    )


def test_training_publishes_exact_package_and_removes_recovery(tmp_path: Path) -> None:
    model_path = tmp_path / "model"

    result = train_model(_request(model_path))

    assert (
        result.termination_reason is TrainingTerminationReason.MAXIMUM_VALIDATION_CHECKS
    )
    assert result.model_path == model_path
    package_path = Path(f"{model_path}.model.zip")
    with zipfile.ZipFile(package_path) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "metadata.json",
            "weights.pt",
        }
        weights = torch.load(
            io.BytesIO(archive.read("weights.pt")),
            map_location="cpu",
            weights_only=True,
        )
    assert all(isinstance(value, torch.Tensor) for value in weights.values())
    loaded = load_model_package(package_path)
    assert loaded.metadata["model"]["hidden_widths"] == [3]
    assert loaded.metadata["features"][0]["name"] == "a"
    assert loaded.metadata["targets"][0]["name"] == "metric"
    assert not Path(f"{model_path}.recovery.pt").exists()
    log = Path(f"{model_path}.training.log").read_text(encoding="utf-8")
    assert "completed:" in log
    assert "model_package_sha256:" in log


def test_loader_rejects_member_changed_after_manifest_capture(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    train_model(_request(model_path))
    package_path = Path(f"{model_path}.model.zip")
    with zipfile.ZipFile(package_path) as archive:
        content = {name: archive.read(name) for name in archive.namelist()}
    content["metadata.json"] += b" "
    with zipfile.ZipFile(package_path, "w") as archive:
        for name in ("manifest.json", "metadata.json", "weights.pt"):
            archive.writestr(name, content[name])

    with pytest.raises(ValueError, match="wrong size"):
        load_model_package(package_path)


def test_existing_outputs_require_explicit_overwrite(tmp_path: Path) -> None:
    request = _request(tmp_path / "model")
    first = train_model(request)

    with pytest.raises(FileExistsError):
        train_model(request)
    second = train_model(replace(request, overwrite=True))

    assert second.model_path == first.model_path
    assert second.validation_history == first.validation_history


def test_missing_model_parent_is_created_without_run_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "nested" / "more" / "model"

    train_model(_request(model_path))

    assert Path(f"{model_path}.model.zip").is_file()
    assert not model_path.exists()


def test_simultaneous_model_path_use_is_rejected_even_for_overwrite(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path / "model", overwrite=True)
    paths = resolve_training_paths(request.model_path)
    prepare_training_parent(paths)

    with (
        training_path_lock(paths),
        pytest.raises(RuntimeError, match="already using model_path"),
    ):
        train_model(request)


def test_model_path_cannot_be_an_existing_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    with pytest.raises(IsADirectoryError):
        train_model(_request(model_path))


def test_model_path_string_cannot_end_at_a_directory_boundary(tmp_path: Path) -> None:
    request = replace(_request(tmp_path / "placeholder"), model_path=f"{tmp_path}/")

    with pytest.raises(ModelTrainingValidationError, match="non-empty final"):
        train_model(request)


def test_finished_model_loads_without_training_companions(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    train_model(_request(model_path))
    Path(f"{model_path}.training.log").unlink()

    loaded = load_model_package(Path(f"{model_path}.model.zip"))

    assert loaded.metadata["model"]["input_width"] == 2


def test_failed_recovery_replacement_preserves_previous_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = resolve_training_paths(tmp_path / "model")
    prepare_training_parent(paths)
    atomic_save_recovery(paths, {"marker": 1})

    with monkeypatch.context() as patch:
        patch.setattr(
            training_artifacts.os,
            "replace",
            lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
        )
        with pytest.raises(OSError, match="replace failed"):
            atomic_save_recovery(paths, {"marker": 2})

    assert load_recovery(paths) == {"marker": 1}
    clean_temporary_paths(paths)


def test_temporary_package_validation_failure_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source"
    train_model(_request(source_path))
    source = load_model_package(Path(f"{source_path}.model.zip"))
    target_paths = resolve_training_paths(tmp_path / "target")
    prepare_training_parent(target_paths)
    real_load = training_artifacts.load_model_package

    def reject_temporary(path: Path):  # type: ignore[no-untyped-def]
        if path == target_paths.package_temporary_path:
            raise ValueError("temporary validation failed")
        return real_load(path)

    with monkeypatch.context() as patch:
        patch.setattr(training_artifacts, "load_model_package", reject_temporary)
        with pytest.raises(ValueError, match="temporary validation failed"):
            publish_model_package(target_paths, source.metadata, source.weights)

    assert not target_paths.package_path.exists()
    clean_temporary_paths(target_paths)


def test_recovery_survives_until_log_finalization_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path / "model")
    real_write = training_api._write_log

    def fail_completion(handle, line=""):  # type: ignore[no-untyped-def]
        if line.startswith("completed:"):
            raise OSError("log finalization failed")
        return real_write(handle, line)

    with monkeypatch.context() as patch:
        patch.setattr(training_api, "_write_log", fail_completion)
        with pytest.raises(OSError, match="log finalization failed"):
            train_model(request)

    assert Path(f"{request.model_path}.model.zip").exists()
    assert Path(f"{request.model_path}.recovery.pt").exists()

    result = train_model(request)

    assert result.validation_checks == 1
    assert not Path(f"{request.model_path}.recovery.pt").exists()


def test_overwrite_cleans_owned_temporary_siblings(tmp_path: Path) -> None:
    request = _request(tmp_path / "model")
    train_model(request)
    paths = resolve_training_paths(request.model_path)
    paths.package_temporary_path.write_bytes(b"stale")
    paths.recovery_temporary_path.write_bytes(b"stale")

    train_model(replace(request, overwrite=True))

    assert not paths.package_temporary_path.exists()
    assert not paths.recovery_temporary_path.exists()
