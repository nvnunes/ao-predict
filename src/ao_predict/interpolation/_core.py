"""Shared interpolation artifact primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import pickle

import numpy as np
from scipy.interpolate import RBFInterpolator


DEFAULT_RBF_KERNEL = "thin_plate_spline"
DEFAULT_RBF_SMOOTHING = 0.05
DEFAULT_RBF_DEGREE = 1
_AXIS_ATOL = 1.0e-10


@dataclass(frozen=True)
class RbfInterpolationConfig:
    """Configuration for scaled SciPy RBF interpolation.

    Builders validate this object before fitting and persist the validated
    values with interpolation artifacts. The options are passed directly to
    ``scipy.interpolate.RBFInterpolator`` after coordinate scaling.

    Attributes:
        kernel: SciPy ``RBFInterpolator`` kernel name.
        smoothing: Non-negative RBF smoothing value.
        degree: Polynomial degree passed to ``RBFInterpolator``.
    """

    kernel: str = DEFAULT_RBF_KERNEL
    smoothing: float = DEFAULT_RBF_SMOOTHING
    degree: int = DEFAULT_RBF_DEGREE


def validate_rbf_config(config: RbfInterpolationConfig) -> RbfInterpolationConfig:
    """Return a validated RBF configuration."""

    if not isinstance(config, RbfInterpolationConfig):
        raise TypeError("interpolation_config must be a RbfInterpolationConfig instance.")
    kernel = str(config.kernel).strip()
    if not kernel:
        raise ValueError("interpolation_config.kernel must be non-empty.")
    smoothing = float(config.smoothing)
    if not np.isfinite(smoothing) or smoothing < 0.0:
        raise ValueError("interpolation_config.smoothing must be finite and >= 0.")
    degree = int(config.degree)
    if degree < -1:
        raise ValueError("interpolation_config.degree must be >= -1.")
    return RbfInterpolationConfig(kernel=kernel, smoothing=smoothing, degree=degree)


def zenith_angle_to_airmass(zenith_angle_deg: float | np.ndarray) -> np.ndarray:
    """Convert zenith angle in degrees to airmass using ``sec(z)``.

    Args:
        zenith_angle_deg: Scalar or array-like zenith angle values in degrees.

    Returns:
        A NumPy array of airmass values with the input shape.
    """

    return 1.0 / np.cos(np.deg2rad(np.asarray(zenith_angle_deg, dtype=float)))


def require_finite_vector(value: Any, *, label: str, length: int | None = None) -> np.ndarray:
    """Return a finite 1-D float vector."""

    vector = np.asarray(value, dtype=float).reshape(-1)
    if length is not None and vector.size != int(length):
        raise ValueError(f"{label} must have length {int(length)}, got {vector.size}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain only finite values.")
    return vector


def require_positive_vector(value: Any, *, label: str, length: int | None = None) -> np.ndarray:
    """Return a finite positive 1-D float vector."""

    vector = require_finite_vector(value, label=label, length=length)
    if np.any(vector <= 0.0):
        raise ValueError(f"{label} must contain only values > 0.")
    return vector


def require_positive_scalar(value: Any, *, label: str) -> float:
    """Return a finite positive scalar float."""

    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{label} must be scalar.")
    scalar = float(array)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{label} must be finite and > 0.")
    return scalar


def require_pupil(value: Any, *, label: str = "tel_pupil") -> np.ndarray:
    """Return a finite 2-D telescope pupil array."""

    pupil = np.asarray(value, dtype=np.float32)
    if pupil.ndim != 2:
        raise ValueError(f"{label} must be a 2-D array; got shape {pupil.shape}.")
    if not np.all(np.isfinite(pupil)):
        raise ValueError(f"{label} must contain only finite values.")
    return pupil


def normalize_psfs(psfs: np.ndarray) -> np.ndarray:
    """Clip negative PSF pixels and normalize each PSF to unit sum."""

    psf_cube = np.maximum(np.asarray(psfs, dtype=np.float32), 0.0)
    totals = np.sum(psf_cube, axis=(-2, -1), dtype=np.float32, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("Cannot normalize PSFs with non-positive total flux.")
    return (psf_cube / totals).astype(np.float32, copy=False)


def validate_psf_array(psfs: Any, *, label: str, ndim: int) -> np.ndarray:
    """Return a finite PSF array with the expected dimensionality."""

    array = np.asarray(psfs, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{label} must have ndim={ndim}, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values.")
    if np.any(array < 0.0):
        raise ValueError(f"{label} must not contain negative values.")
    return array


def field_coordinates(x_arcsec: Any, y_arcsec: Any) -> np.ndarray:
    """Return validated ``(x_arcsec, y_arcsec)`` field coordinates."""

    x = require_finite_vector(x_arcsec, label="x_arcsec")
    y = require_finite_vector(y_arcsec, label="y_arcsec", length=x.size)
    if x.size == 0:
        raise ValueError("At least one field coordinate is required.")
    return np.column_stack([x, y])


def make_scaled_rbf_model(
    coordinates: np.ndarray,
    values_by_name: Mapping[str, np.ndarray],
    config: RbfInterpolationConfig,
) -> dict[str, Any]:
    """Fit scaled RBF models, dropping coordinate dimensions with no variation."""

    config = validate_rbf_config(config)
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] == 0:
        raise ValueError("coordinates must be a non-empty 2-D array.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must contain only finite values.")

    coord_mean = np.nanmean(coordinates, axis=0)
    coord_std = np.nanstd(coordinates, axis=0)
    active_dims = coord_std > 0.0
    if not np.any(active_dims):
        raise ValueError("At least one interpolation coordinate dimension must vary.")

    active_coordinates = coordinates[:, active_dims]
    active_mean = coord_mean[active_dims]
    active_scale = coord_std[active_dims]
    scaled_coordinates = (active_coordinates - active_mean) / active_scale

    models: dict[str, RBFInterpolator] = {}
    for name, values in values_by_name.items():
        values = np.asarray(values, dtype=float)
        if values.shape[0] != coordinates.shape[0]:
            raise ValueError(f"{name} has leading dimension {values.shape[0]}; expected {coordinates.shape[0]}.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")
        models[str(name)] = RBFInterpolator(
            scaled_coordinates,
            values,
            kernel=config.kernel,
            smoothing=config.smoothing,
            degree=config.degree,
        )

    return {
        "coord_mean": coord_mean,
        "coord_scale": coord_std,
        "active_dims": active_dims,
        "config": config,
        "models": models,
    }


def evaluate_scaled_rbf_model(model: Mapping[str, Any], coordinates: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate a scaled RBF model produced by ``make_scaled_rbf_model``."""

    coordinates = np.asarray(coordinates, dtype=float)
    active_dims = np.asarray(model["active_dims"], dtype=bool)
    coord_mean = np.asarray(model["coord_mean"], dtype=float)
    coord_scale = np.asarray(model["coord_scale"], dtype=float)
    scaled = (coordinates[:, active_dims] - coord_mean[active_dims]) / coord_scale[active_dims]
    return {
        name: np.asarray(interpolator(scaled), dtype=float)
        for name, interpolator in dict(model["models"]).items()
    }


def unique_sorted(values: np.ndarray, *, label: str) -> np.ndarray:
    """Return unique sorted axis values with duplicate detection."""

    values = require_finite_vector(values, label=label)
    if values.size == 0:
        raise ValueError(f"{label} must not be empty.")
    rounded = np.round(values, decimals=10)
    unique = np.asarray(sorted(set(float(value) for value in rounded)), dtype=float)
    return unique


def axis_index(axis: np.ndarray, value: float, *, label: str) -> int:
    """Return the unique index matching ``value`` on ``axis``."""

    matches = np.where(np.isclose(axis, float(value), rtol=0.0, atol=_AXIS_ATOL))[0]
    if matches.size != 1:
        raise ValueError(f"Could not locate {label} value {value}.")
    return int(matches[0])


def interpolation_axis_weights(axis: np.ndarray, value: float, *, label: str) -> list[tuple[int, float]]:
    """Return linear interpolation weights, rejecting out-of-range values."""

    axis = np.asarray(axis, dtype=float).reshape(-1)
    value = float(value)
    if axis.size == 0:
        raise ValueError(f"{label} axis is empty.")
    if axis.size == 1:
        if np.isclose(value, axis[0], rtol=0.0, atol=_AXIS_ATOL):
            return [(0, 1.0)]
        raise ValueError(f"{label}={value} is outside the supported range [{axis[0]}, {axis[0]}].")
    if value < axis[0] and not np.isclose(value, axis[0], rtol=0.0, atol=_AXIS_ATOL):
        raise ValueError(f"{label}={value} is below the supported range minimum {axis[0]}.")
    if value > axis[-1] and not np.isclose(value, axis[-1], rtol=0.0, atol=_AXIS_ATOL):
        raise ValueError(f"{label}={value} is above the supported range maximum {axis[-1]}.")
    if np.isclose(value, axis[0], rtol=0.0, atol=_AXIS_ATOL):
        return [(0, 1.0)]
    if np.isclose(value, axis[-1], rtol=0.0, atol=_AXIS_ATOL):
        return [(axis.size - 1, 1.0)]
    upper = int(np.searchsorted(axis, value, side="right"))
    lower = upper - 1
    if np.isclose(axis[lower], value, rtol=0.0, atol=_AXIS_ATOL):
        return [(lower, 1.0)]
    if np.isclose(axis[upper], value, rtol=0.0, atol=_AXIS_ATOL):
        return [(upper, 1.0)]
    weight = float((value - axis[lower]) / (axis[upper] - axis[lower]))
    return [(lower, 1.0 - weight), (upper, weight)]


def save_payload(payload: Mapping[str, Any], path: Path, *, overwrite: bool) -> None:
    """Persist a structured artifact payload."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(dict(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_payload(path: Path) -> dict[str, Any]:
    """Load a structured artifact payload."""

    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain an ao-predict interpolation artifact payload.")
    return payload


def validate_payload_kind(payload: Mapping[str, Any], *, kind: str, version: int) -> None:
    """Validate common persisted artifact kind and version fields."""

    if payload.get("kind") != kind:
        raise ValueError(f"Unsupported artifact kind: {payload.get('kind')!r}.")
    if payload.get("version") != version:
        raise ValueError(f"Unsupported artifact version: {payload.get('version')!r}.")
