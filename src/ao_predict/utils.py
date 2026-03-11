"""Generic utility helpers shared across ao-predict packages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def as_array(value: Any) -> np.ndarray:
    """Convert scalar/sequence-like values to numpy arrays.

    Args:
        value: Input value.

    Returns:
        Numpy array view/copy of the input value.
    """
    if isinstance(value, np.ndarray):
        return value
    if np.isscalar(value):
        return np.asarray(value)
    return np.asarray(value)


def as_float_scalar(value: object, *, label: str) -> float:
    """Coerce a value to a scalar ``float``.

    Args:
        value: Scalar-like input.
        label: Human-readable field label used in errors.

    Returns:
        Scalar float value.

    Raises:
        ValueError: If ``value`` cannot be interpreted as a scalar.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    flat = arr.reshape(-1)
    if flat.size != 1:
        raise ValueError(f"{label} must be scalar-compatible, got shape {arr.shape}.")
    return float(flat[0])


def as_float_vector(value: object, *, label: str, length: int | None = None) -> np.ndarray:
    """Coerce a value to a 1D float vector.

    Args:
        value: Vector-like input.
        label: Human-readable field label used in errors.
        length: Optional expected vector length.

    Returns:
        1D float numpy array.

    Raises:
        ValueError: If length validation fails.
    """
    vec = np.asarray(value, dtype=float).reshape(-1)
    if length is not None and vec.shape[0] != int(length):
        raise ValueError(f"{label} must have length {int(length)}, got {vec.shape[0]}.")
    return vec


def as_float_matrix(value: object, *, label: str, rows: int | None = None) -> np.ndarray:
    """Coerce a value to a 2D float matrix.

    Args:
        value: Matrix-like input.
        label: Human-readable field label used in errors.
        rows: Optional expected first-dimension size.

    Returns:
        2D float numpy array.

    Raises:
        ValueError: If dimensionality or row-count validation fails.
    """
    mat = np.asarray(value, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"{label} must be 2D, got ndim={mat.ndim}.")
    if rows is not None and mat.shape[0] != int(rows):
        raise ValueError(f"{label} first dimension must be {int(rows)}, got {mat.shape[0]}.")
    return mat


def require_finite_positive_scalar(value: object, *, label: str) -> float:
    """Coerce a value to float and require it to be finite and positive.

    Args:
        value: Scalar-like input.
        label: Human-readable field label used in errors.

    Returns:
        Validated scalar float value.

    Raises:
        ValueError: If the value is not finite or is ``<= 0``.
    """
    x = as_float_scalar(value, label=label)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"{label} must be finite and > 0.")
    return x


def as_array_dict(
    mapping: Mapping[Any, Any],
    *,
    key_transform: Callable[[Any], str] = str,
    copy_arrays: bool = True,
) -> dict[str, np.ndarray]:
    """Convert a mapping into ``dict[str, np.ndarray]``.

    Args:
        mapping: Input key/value mapping.
        key_transform: Callable used to normalize each key to ``str``.
        copy_arrays: If ``True``, copies each converted array.

    Returns:
        Mapping of normalized string keys to numpy arrays.
    """
    out: dict[str, np.ndarray] = {}
    for key, value in mapping.items():
        out_key = key_transform(key)
        arr = np.asarray(value)
        out[out_key] = arr.copy() if copy_arrays else arr
    return out


def require_keys(mapping: Mapping[str, Any], keys: tuple[str, ...], *, label: str) -> None:
    """Validate that all required keys exist in a mapping.

    Args:
        mapping: Mapping to inspect.
        keys: Required key names.
        label: Human-readable label used in error messages.

    Raises:
        ValueError: If one or more keys are missing.
    """
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"Missing required {label} keys: {', '.join(missing)}")


def require_lowercase_mapping_keys(mapping: Mapping[object, object], *, label: str) -> None:
    """Validate that all string keys in a mapping are lowercase.

    Args:
        mapping: Mapping to validate.
        label: Human-readable path used in error messages.

    Raises:
        ValueError: If any string key is not lowercase.
    """
    for raw_key in mapping.keys():
        if isinstance(raw_key, str) and raw_key != raw_key.lower():
            raise ValueError(f"{label} keys must be lowercase. Invalid key: '{raw_key}'.")
