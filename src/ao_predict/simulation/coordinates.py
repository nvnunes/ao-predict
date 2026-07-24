"""Science-coordinate resolution for invariant setup grids and row offsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..utils import as_float_vector
from . import schema


@dataclass(frozen=True)
class ScienceCoordinates:
    """Resolved science coordinates for one simulation.

    Attributes:
        r_arcsec: Effective radial coordinates in arcseconds.
        theta_deg: Effective angular coordinates in degrees.
        x_arcsec: Effective Cartesian x-coordinates in arcseconds.
        y_arcsec: Effective Cartesian y-coordinates in arcseconds.
    """

    r_arcsec: np.ndarray
    theta_deg: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray


def polar_to_cartesian(r_arcsec: Any, theta_deg: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert matching polar field-coordinate vectors to Cartesian arcseconds."""
    r = as_float_vector(r_arcsec, label="r_arcsec")
    theta = as_float_vector(theta_deg, label="theta_deg")
    if r.shape != theta.shape:
        raise ValueError(f"Coordinate shapes differ: {r.shape} != {theta.shape}.")
    theta_rad = np.deg2rad(theta)
    return r * np.cos(theta_rad), r * np.sin(theta_rad)


def resolve_science_coordinates(
    setup: Mapping[str, Any] | object,
    options: Mapping[str, Any],
) -> ScienceCoordinates:
    """Return effective science coordinates for one simulation option row.

    The invariant polar field is read from setup. Optional Cartesian
    ``sci_dx_arcsec`` and ``sci_dy_arcsec`` option vectors are then applied
    elementwise. An absent offset axis means zero without allocating a
    replacement vector.

    Args:
        setup: Setup mapping or typed setup object containing the invariant
            ``sci_r_arcsec`` and ``sci_theta_deg`` vectors.
        options: One simulation's option mapping. Present science-offset
            fields must be finite one-dimensional vectors matching the setup
            science-point count.

    Returns:
        Effective polar and Cartesian science coordinates.

    Raises:
        ValueError: If coordinate or offset vectors are malformed, have
            inconsistent lengths, or contain non-finite values.
    """
    r = as_float_vector(_setup_value(setup, schema.KEY_SETUP_SCI_R_ARCSEC), label="setup.sci_r_arcsec")
    theta = as_float_vector(
        _setup_value(setup, schema.KEY_SETUP_SCI_THETA_DEG),
        label="setup.sci_theta_deg",
    )
    if r.shape != theta.shape:
        raise ValueError(f"Science coordinate shapes differ: {r.shape} != {theta.shape}.")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(theta)):
        raise ValueError("Setup science coordinates must be finite.")

    dx = _offset_vector(options, schema.KEY_OPTION_SCI_DX_ARCSEC, length=r.size)
    dy = _offset_vector(options, schema.KEY_OPTION_SCI_DY_ARCSEC, length=r.size)
    x, y = polar_to_cartesian(r, theta)
    if dx is None and dy is None:
        return ScienceCoordinates(r_arcsec=r, theta_deg=theta, x_arcsec=x, y_arcsec=y)

    if dx is not None:
        x = x + dx
    if dy is not None:
        y = y + dy
    effective_r = np.hypot(x, y)
    effective_theta = np.mod(np.rad2deg(np.arctan2(y, x)), 360.0)
    effective_theta[effective_r == 0.0] = 0.0
    return ScienceCoordinates(
        r_arcsec=effective_r,
        theta_deg=effective_theta,
        x_arcsec=x,
        y_arcsec=y,
    )


def _setup_value(setup: Mapping[str, Any] | object, key: str) -> Any:
    if isinstance(setup, Mapping):
        if key not in setup:
            raise ValueError(f"Setup is missing required science coordinate field '{key}'.")
        return setup[key]
    if not hasattr(setup, key):
        raise ValueError(f"Setup is missing required science coordinate field '{key}'.")
    return getattr(setup, key)


def _offset_vector(options: Mapping[str, Any], key: str, *, length: int) -> np.ndarray | None:
    if key not in options:
        return None
    value = np.asarray(options[key], dtype=float)
    if value.ndim != 1:
        raise ValueError(f"options['{key}'] must be a 1D science-offset vector.")
    value = as_float_vector(value, label=f"options['{key}']", length=length)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"options['{key}'] must be finite.")
    return value


__all__ = [
    "ScienceCoordinates",
    "polar_to_cartesian",
    "resolve_science_coordinates",
]
