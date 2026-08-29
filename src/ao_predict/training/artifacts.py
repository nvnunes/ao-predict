"""Private model-package, recovery, path, lock, and publication contracts."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import BinaryIO

import torch

from .._model_package import (
    MODEL_PACKAGE_KIND,
    MODEL_PACKAGE_VERSION,
    _LoadedModelPackage,
    load_model_package,
    sha256_bytes,
)

RECOVERY_KIND = "ao_predict_model_training_recovery"
RECOVERY_VERSION = 1


def producer_version() -> str:
    """Return the installed AO Predict producer version."""

    try:
        return importlib_metadata.version("ao-predict")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.1"


@dataclass(frozen=True)
class _TrainingPaths:
    """Caller path stem and every AO Predict-owned derived path."""

    model_path: Path
    package_path: Path
    log_path: Path
    recovery_path: Path
    lock_path: Path
    package_temporary_path: Path
    recovery_temporary_path: Path


def resolve_training_paths(value: str | Path) -> _TrainingPaths:
    """Normalize a caller path stem and derive the stable companion paths."""

    if not isinstance(value, (str, Path)):
        raise TypeError("model_path must be a string or pathlib.Path.")
    raw = os.fspath(value)
    if not raw or not raw.strip():
        raise ValueError("model_path must not be empty.")
    separators = (os.sep,) if os.altsep is None else (os.sep, os.altsep)
    if isinstance(value, str) and raw.endswith(separators):
        raise ValueError("model_path must include a non-empty final path component.")
    model_path = Path(os.path.normpath(raw))
    if model_path.name in {"", ".", ".."}:
        raise ValueError("model_path must include a non-empty final path component.")
    package_path = Path(f"{model_path}.model.zip")
    recovery_path = Path(f"{model_path}.recovery.pt")
    return _TrainingPaths(
        model_path=model_path,
        package_path=package_path,
        log_path=Path(f"{model_path}.training.log"),
        recovery_path=recovery_path,
        lock_path=model_path.with_name(f".{model_path.name}.training.lock"),
        package_temporary_path=package_path.with_name(f".{package_path.name}.tmp"),
        recovery_temporary_path=recovery_path.with_name(f".{recovery_path.name}.tmp"),
    )


def prepare_training_parent(paths: _TrainingPaths) -> None:
    """Create the requested parent and reject invalid path components."""

    paths.model_path.parent.mkdir(parents=True, exist_ok=True)
    if paths.model_path.exists() and paths.model_path.is_dir():
        raise IsADirectoryError(
            f"model_path is an existing directory: {paths.model_path}"
        )


@contextmanager
def training_path_lock(paths: _TrainingPaths) -> Iterator[None]:
    """Hold one private non-blocking exclusive lock for the complete lifecycle."""

    handle = paths.lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another training operation is already using model_path {paths.model_path}."
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def clean_temporary_paths(paths: _TrainingPaths) -> None:
    """Remove only known AO Predict-owned temporary siblings."""

    paths.package_temporary_path.unlink(missing_ok=True)
    paths.recovery_temporary_path.unlink(missing_ok=True)


def remove_derived_outputs(paths: _TrainingPaths) -> None:
    """Remove the caller-authorized stable derived output set."""

    paths.package_path.unlink(missing_ok=True)
    paths.log_path.unlink(missing_ok=True)
    paths.recovery_path.unlink(missing_ok=True)
    clean_temporary_paths(paths)


def _flush_and_sync(handle: BinaryIO) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def atomic_save_recovery(paths: _TrainingPaths, value: Mapping[str, object]) -> None:
    """Replace recovery while preserving the preceding valid checkpoint."""

    with paths.recovery_temporary_path.open("wb") as handle:
        torch.save(dict(value), handle)
        _flush_and_sync(handle)
    os.replace(paths.recovery_temporary_path, paths.recovery_path)


def load_recovery(paths: _TrainingPaths) -> dict[str, object]:
    """Load one constrained weights-only recovery mapping on CPU."""

    value = torch.load(paths.recovery_path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Training recovery must contain one string-keyed mapping.")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _weights_bytes(weights: Mapping[str, torch.Tensor]) -> bytes:
    output = io.BytesIO()
    torch.save(dict(weights), output)
    return output.getvalue()


def _manifest(metadata_bytes: bytes, weights_bytes: bytes) -> dict[str, object]:
    return {
        "kind": MODEL_PACKAGE_KIND,
        "version": MODEL_PACKAGE_VERSION,
        "producer_version": producer_version(),
        "required_members": ["manifest.json", "metadata.json", "weights.pt"],
        "members": {
            "metadata.json": {
                "size": len(metadata_bytes),
                "sha256": sha256_bytes(metadata_bytes),
            },
            "weights.pt": {
                "size": len(weights_bytes),
                "sha256": sha256_bytes(weights_bytes),
            },
        },
    }


def _same_weights(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> bool:
    return set(first) == set(second) and all(
        torch.equal(first[name].cpu(), second[name].cpu()) for name in first
    )


def publish_model_package(
    paths: _TrainingPaths,
    metadata: Mapping[str, object],
    weights: Mapping[str, torch.Tensor],
) -> _LoadedModelPackage:
    """Validate then atomically publish, or accept an identical terminal package."""

    metadata_value = dict(metadata)
    metadata_bytes = _json_bytes(metadata_value)
    weights_bytes = _weights_bytes(weights)
    if paths.package_path.exists():
        try:
            existing = load_model_package(paths.package_path)
        except (OSError, ValueError, zipfile.BadZipFile):
            existing = None
        if (
            existing is not None
            and existing.metadata == metadata_value
            and _same_weights(existing.weights, weights)
        ):
            return existing
    manifest_bytes = _json_bytes(_manifest(metadata_bytes, weights_bytes))
    with paths.package_temporary_path.open("wb") as raw:
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", manifest_bytes)
            archive.writestr("metadata.json", metadata_bytes)
            archive.writestr("weights.pt", weights_bytes)
        _flush_and_sync(raw)
    loaded = load_model_package(paths.package_temporary_path)
    if loaded.metadata != metadata_value or not _same_weights(loaded.weights, weights):
        raise ValueError(
            "Validated temporary model package changed its declared content."
        )
    os.replace(paths.package_temporary_path, paths.package_path)
    return load_model_package(paths.package_path)
