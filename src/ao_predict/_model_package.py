"""Private validation and reconstruction of AO Predict model packages."""

from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
import string
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ._model import build_dense_model

MODEL_PACKAGE_KIND = "ao_predict_dense_regression_model_package"
MODEL_PACKAGE_VERSION = 1
MODEL_METADATA_KIND = "ao_predict_dense_regression_model"
MODEL_METADATA_VERSION = 1
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


@dataclass(frozen=True)
class _LoadedModelPackage:
    """Fully validated private package content and reconstructed model."""

    manifest: dict[str, object]
    metadata: dict[str, object]
    weights: dict[str, torch.Tensor]
    model: nn.Module


def sha256_bytes(value: bytes) -> str:
    """Return the package contract's lowercase content digest."""

    return hashlib.sha256(value).hexdigest()


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


def _finite_float(value: float, *, label: str) -> float:
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite.")
    return converted


def _validate_float32_scaler(mean: float, scale: float, *, label: str) -> None:
    try:
        converted = torch.tensor([mean, scale], dtype=torch.float32)
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Model metadata {label} scaler values must be representable as float32."
        ) from exc
    if not bool(torch.all(torch.isfinite(converted))):
        raise ValueError(
            f"Model metadata {label} scaler values must be representable as finite float32."
        )
    if float(converted[1].item()) <= 0.0:
        raise ValueError(
            f"Model metadata {label} scale must remain positive as float32."
        )


def _validate_model_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
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
            if (
                not isinstance(mean, (int, float))
                or not isinstance(scale, (int, float))
                or isinstance(mean, bool)
                or isinstance(scale, bool)
            ):
                raise ValueError(  # noqa: TRY004
                    f"Model metadata {key} scaler values must be numeric."
                )
            mean_value = _finite_float(
                mean,
                label=f"Model metadata {key} mean",
            )
            scale_value = _finite_float(
                scale,
                label=f"Model metadata {key} scale",
            )
            if scale_value <= 0.0:
                raise ValueError(f"Model metadata {key} scales must be positive.")
            _validate_float32_scaler(mean_value, scale_value, label=key)
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
        if not isinstance(declaration, dict) or set(declaration) != {"size", "sha256"}:
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
    try:
        weights = torch.load(
            io.BytesIO(weights_bytes), map_location="cpu", weights_only=True
        )
    except (
        pickle.UnpicklingError,
        RuntimeError,
        EOFError,
        IndexError,
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise ValueError("Model package weights.pt is invalid.") from exc
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
    return _LoadedModelPackage(
        manifest=manifest,
        metadata=metadata,
        weights=weights,
        model=model,
    )
