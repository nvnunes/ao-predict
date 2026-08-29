"""Private model-package, recovery, path, lock, and publication contracts."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import math
import os
import string
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import BinaryIO

import torch

from .model import build_dense_model

MODEL_PACKAGE_KIND = "ao_predict_dense_regression_model_package"
MODEL_PACKAGE_VERSION = 1
MODEL_METADATA_KIND = "ao_predict_dense_regression_model"
MODEL_METADATA_VERSION = 1
RECOVERY_KIND = "ao_predict_model_training_recovery"
RECOVERY_VERSION = 1
MODEL_PACKAGE_MEMBERS = frozenset({"manifest.json", "metadata.json", "weights.pt"})
_MANIFEST_KEYS = frozenset(
    {"kind", "version", "producer_version", "required_members", "members"}
)
_METADATA_KEYS = frozenset(
    {
        "kind",
        "version",
        "producer_version",
        "model",
        "features",
        "targets",
        "numerical",
        "training_seed",
    }
)
_MODEL_DEFINITION_KEYS = frozenset(
    {
        "input_width",
        "hidden_widths",
        "output_width",
        "hidden_activation",
        "output_activation",
        "bias",
    }
)
_COLUMN_DEFINITION_KEYS = frozenset({"name", "unit", "mean", "scale"})


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


@dataclass(frozen=True)
class _LoadedModelPackage:
    """Fully validated private package content."""

    manifest: dict[str, object]
    metadata: dict[str, object]
    weights: dict[str, torch.Tensor]


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_sha256(value: object, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 hexadecimal digest.")


def _validate_model_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        # A wrong persisted JSON shape is invalid data, not a caller type error.
        raise ValueError("Model metadata must be a JSON object.")  # noqa: TRY004
    if set(value) != _METADATA_KEYS:
        raise ValueError("Model metadata fields do not match format version 1.")
    if value.get("kind") != MODEL_METADATA_KIND:
        raise ValueError(f"Unsupported model metadata kind: {value.get('kind')!r}.")
    if value.get("version") != MODEL_METADATA_VERSION:
        raise ValueError(
            f"Unsupported model metadata version: {value.get('version')!r}."
        )
    if not _is_nonempty_string(value.get("producer_version")):
        raise ValueError("Model metadata producer_version must be a non-empty string.")
    training_seed = value.get("training_seed")
    if (
        not isinstance(training_seed, int)
        or isinstance(training_seed, bool)
        or training_seed < 0
    ):
        raise ValueError("Model metadata training_seed must be a non-negative integer.")
    model = value.get("model")
    if not isinstance(model, dict):
        # A wrong persisted JSON shape is invalid data, not a caller type error.
        raise ValueError(  # noqa: TRY004
            "Model metadata is missing the model definition."
        )
    if set(model) != _MODEL_DEFINITION_KEYS:
        raise ValueError("Model definition fields do not match format version 1.")
    input_width = model.get("input_width")
    output_width = model.get("output_width")
    hidden_widths = model.get("hidden_widths")
    if not _is_positive_integer(input_width):
        raise ValueError("Model metadata input_width must be positive.")
    if not _is_positive_integer(output_width):
        raise ValueError("Model metadata output_width must be positive.")
    if not isinstance(hidden_widths, list) or not all(
        _is_positive_integer(item) for item in hidden_widths
    ):
        raise ValueError("Model metadata hidden_widths must contain positive integers.")
    if (
        model.get("hidden_activation") != "relu"
        or model.get("output_activation") != "linear"
    ):
        raise ValueError("Model metadata declares unsupported activation semantics.")
    if model.get("bias") is not True:
        raise ValueError("Model metadata must declare biased dense layers.")
    for key, width in (("features", input_width), ("targets", output_width)):
        assert isinstance(width, int)
        definitions = value.get(key)
        if not isinstance(definitions, list) or len(definitions) != width:
            raise ValueError(f"Model metadata {key} must have width {width}.")
        for definition in definitions:
            if not isinstance(definition, dict):
                # A wrong persisted JSON shape is invalid data, not a caller type error.
                raise ValueError(  # noqa: TRY004
                    f"Model metadata {key} entries must be objects."
                )
            if set(definition) != _COLUMN_DEFINITION_KEYS:
                raise ValueError(
                    f"Model metadata {key} entry fields do not match format version 1."
                )
            if not _is_nonempty_string(definition.get("name")):
                raise ValueError(f"Model metadata {key} names must be non-empty.")
            unit = definition.get("unit")
            if unit is not None and not _is_nonempty_string(unit):
                raise ValueError(
                    f"Model metadata {key} units must be non-empty strings or null."
                )
            mean = definition.get("mean")
            scale = definition.get("scale")
            if not isinstance(mean, (int, float)) or not isinstance(
                scale, (int, float)
            ):
                # A wrong persisted JSON shape is invalid data, not a caller type error.
                raise ValueError(  # noqa: TRY004
                    f"Model metadata {key} scaler values must be numeric."
                )
            if isinstance(mean, bool) or isinstance(scale, bool):
                raise ValueError(  # noqa: TRY004
                    f"Model metadata {key} scaler values must be numeric."
                )
            if not math.isfinite(float(mean)) or not math.isfinite(float(scale)):
                raise ValueError(f"Model metadata {key} scaler values must be finite.")
            if float(scale) <= 0.0:
                raise ValueError(f"Model metadata {key} scales must be positive.")
        names = [definition["name"] for definition in definitions]
        if len(names) != len(set(names)):
            raise ValueError(f"Model metadata {key} names must be unique.")
    numerical = value.get("numerical")
    if numerical != {
        "model_dtype": "float32",
        "prediction_dtype": "float32",
        "standardization_variance": "population",
        "constant_scale": 1.0,
        "objective": "physical_relative_mse",
    }:
        raise ValueError("Model metadata declares unsupported numerical semantics.")
    return value


def load_model_package(path: Path) -> _LoadedModelPackage:
    """Validate and independently reconstruct one deployable model package."""

    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if (
            len(names) != len(MODEL_PACKAGE_MEMBERS)
            or set(names) != MODEL_PACKAGE_MEMBERS
        ):
            raise ValueError(
                "Model package must contain exactly manifest.json, metadata.json, and weights.pt."
            )
        manifest_bytes = archive.read("manifest.json")
        metadata_bytes = archive.read("metadata.json")
        weights_bytes = archive.read("weights.pt")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Model package manifest.json is invalid JSON.") from exc
    if not isinstance(manifest, dict):
        # A wrong persisted JSON shape is invalid data, not a caller type error.
        raise ValueError("Model package manifest must be a JSON object.")  # noqa: TRY004
    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError("Model package manifest fields do not match format version 1.")
    if manifest.get("kind") != MODEL_PACKAGE_KIND:
        raise ValueError(f"Unsupported model package kind: {manifest.get('kind')!r}.")
    if manifest.get("version") != MODEL_PACKAGE_VERSION:
        raise ValueError(
            f"Unsupported model package version: {manifest.get('version')!r}."
        )
    if not _is_nonempty_string(manifest.get("producer_version")):
        raise ValueError("Model package producer_version must be a non-empty string.")
    if manifest.get("required_members") != [
        "manifest.json",
        "metadata.json",
        "weights.pt",
    ]:
        raise ValueError("Model package required-members declaration is invalid.")
    members = manifest.get("members")
    if not isinstance(members, dict) or set(members) != {"metadata.json", "weights.pt"}:
        raise ValueError("Model package member integrity declarations are invalid.")
    for name, content in (
        ("metadata.json", metadata_bytes),
        ("weights.pt", weights_bytes),
    ):
        declaration = members.get(name)
        if not isinstance(declaration, dict):
            # A wrong persisted JSON shape is invalid data, not a caller type error.
            raise ValueError(  # noqa: TRY004
                f"Model package declaration for {name} is invalid."
            )
        if set(declaration) != {"size", "sha256"}:
            raise ValueError(f"Model package declaration for {name} is invalid.")
        if not isinstance(declaration.get("size"), int) or isinstance(
            declaration.get("size"), bool
        ):
            raise ValueError(  # noqa: TRY004
                f"Model package member {name} size is invalid."
            )
        _validate_sha256(
            declaration.get("sha256"),
            label=f"Model package member {name} checksum",
        )
        if declaration.get("size") != len(content):
            raise ValueError(f"Model package member {name} has the wrong size.")
        if declaration.get("sha256") != sha256_bytes(content):
            raise ValueError(f"Model package member {name} failed checksum validation.")
    try:
        metadata = json.loads(metadata_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Model package metadata.json is invalid JSON.") from exc
    metadata = _validate_model_metadata(metadata)
    if metadata["producer_version"] != manifest["producer_version"]:
        raise ValueError("Model package producer versions do not agree.")
    weights = torch.load(
        io.BytesIO(weights_bytes), map_location="cpu", weights_only=True
    )
    if not isinstance(weights, dict) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in weights.items()
    ):
        raise ValueError("Model package weights.pt must contain only named tensors.")
    if not all(
        tensor.dtype == torch.float32
        and tensor.layout == torch.strided
        and bool(torch.all(torch.isfinite(tensor)))
        for tensor in weights.values()
    ):
        raise ValueError(
            "Model package weights.pt tensors must be finite dense float32 values."
        )
    model_definition = metadata["model"]
    assert isinstance(model_definition, dict)
    model, _ = build_dense_model(
        int(model_definition["input_width"]),
        tuple(int(value) for value in model_definition["hidden_widths"]),
        int(model_definition["output_width"]),
        initialization_seed=0,
    )
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Model package weights do not match the model definition."
        ) from exc
    return _LoadedModelPackage(manifest=manifest, metadata=metadata, weights=weights)


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
