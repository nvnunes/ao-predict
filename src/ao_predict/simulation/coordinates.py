"""Science-coordinate resolution for invariant setup grids and row offsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from astropy import units as u

from .._units import quantity_value
from . import schema


@dataclass(frozen=True)
class ScienceCoordinates:
    """Resolved science coordinates for one simulation.

    Attributes:
        r: Effective radial coordinates in arcseconds.
        theta: Effective angular coordinates in degrees.
        x: Effective Cartesian x-coordinates in arcseconds.
        y: Effective Cartesian y-coordinates in arcseconds.
    """

    r: u.Quantity
    theta: u.Quantity
    x: u.Quantity
    y: u.Quantity


def polar_to_cartesian(r: Any, theta: Any) -> tuple[u.Quantity, u.Quantity]:
    """Convert matching polar field-coordinate vectors to Cartesian arcseconds."""
    r_values = quantity_value(r, u.arcsec, label="r", dtype=float).reshape(-1)
    theta_values = quantity_value(theta, u.deg, label="theta", dtype=float).reshape(-1)
    if r_values.shape != theta_values.shape:
        raise ValueError(f"Coordinate shapes differ: {r_values.shape} != {theta_values.shape}.")
    theta_rad = np.deg2rad(theta_values)
    return r_values * np.cos(theta_rad) * u.arcsec, r_values * np.sin(theta_rad) * u.arcsec


def resolve_science_coordinates(
    setup: Mapping[str, Any] | object,
    options: Mapping[str, Any],
) -> ScienceCoordinates:
    """Return effective science coordinates for one simulation option row.

    The invariant polar field is read from setup. Optional Cartesian
    ``sci_dx`` and ``sci_dy`` option vectors are then applied
    elementwise. An absent offset axis means zero without allocating a
    replacement vector.

    Args:
        setup: Setup mapping or typed setup object containing the invariant
            ``sci_r`` and ``sci_theta`` vectors.
        options: One simulation's option mapping. Present science-offset
            fields must be finite one-dimensional vectors matching the setup
            science-point count.

    Returns:
        Effective polar and Cartesian science coordinates.

    Raises:
        ValueError: If coordinate or offset vectors are malformed, have
            inconsistent lengths, or contain non-finite values.
    """
    r = quantity_value(_setup_value(setup, schema.KEY_SETUP_SCI_R), u.arcsec, label="setup.sci_r", dtype=float).reshape(-1)
    theta = quantity_value(
        _setup_value(setup, schema.KEY_SETUP_SCI_THETA),
        u.deg,
        label="setup.sci_theta",
        dtype=float,
    ).reshape(-1)
    if r.shape != theta.shape:
        raise ValueError(f"Science coordinate shapes differ: {r.shape} != {theta.shape}.")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(theta)):
        raise ValueError("Setup science coordinates must be finite.")

    dx = _offset_vector(options, schema.KEY_OPTION_SCI_DX, length=r.size)
    dy = _offset_vector(options, schema.KEY_OPTION_SCI_DY, length=r.size)
    x, y = polar_to_cartesian(r * u.arcsec, theta * u.deg)
    if dx is None and dy is None:
        return ScienceCoordinates(r=r * u.arcsec, theta=theta * u.deg, x=x, y=y)

    if dx is not None:
        x = x + dx * u.arcsec
    if dy is not None:
        y = y + dy * u.arcsec
    effective_r = np.hypot(x.to_value(u.arcsec), y.to_value(u.arcsec))
    effective_theta = np.mod(np.rad2deg(np.arctan2(y.to_value(u.arcsec), x.to_value(u.arcsec))), 360.0)
    effective_theta[effective_r == 0.0] = 0.0
    return ScienceCoordinates(
        r=effective_r * u.arcsec,
        theta=effective_theta * u.deg,
        x=x,
        y=y,
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
    value = quantity_value(options[key], u.arcsec, label=f"options['{key}']", dtype=float)
    if value.ndim != 1:
        raise ValueError(f"options['{key}'] must be a 1D science-offset vector.")
    value = value.reshape(-1)
    if value.size != length:
        raise ValueError(f"options['{key}'] must have length {length}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"options['{key}'] must be finite.")
    return value


__all__ = [
    "ScienceCoordinates",
    "polar_to_cartesian",
    "resolve_science_coordinates",
]
