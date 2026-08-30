"""Shared Astropy unit primitives for AO Predict public and persisted data."""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u


UNITS_ATTRIBUTE = "units"


def parse_unit(value: Any, *, label: str) -> u.UnitBase:
    """Return one validated Astropy unit from a persisted or declared value."""
    try:
        return u.Unit(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a valid Astropy unit.") from exc


def unit_string(unit: u.UnitBase) -> str:
    """Return the canonical generic string persisted for one Astropy unit."""
    unit = u.Unit(unit)
    if unit == u.dimensionless_unscaled:
        return "1"
    return unit.to_string("generic")


def require_quantity(
    value: Any,
    unit: u.UnitBase,
    *,
    label: str,
) -> u.Quantity:
    """Require a quantity and convert it to one canonical equivalent unit."""
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{label} must be an Astropy Quantity.")
    try:
        return value.to(unit, copy=False)
    except u.UnitConversionError as exc:
        raise ValueError(f"{label} must have units equivalent to {unit_string(unit)!r}.") from exc


def quantity_value(
    value: Any,
    unit: u.UnitBase,
    *,
    label: str,
    dtype: Any | None = None,
) -> np.ndarray:
    """Return the unit-normalized numerical view of one required quantity."""
    quantity = require_quantity(value, unit, label=label)
    return np.asarray(quantity.value, dtype=dtype)


def quantity_from_value(value: Any, unit: u.UnitBase) -> u.Quantity:
    """Wrap numerical values in one unit without copying when NumPy permits."""
    return u.Quantity(np.asarray(value), unit=unit, copy=False)


def freeze_quantity(value: u.Quantity) -> u.Quantity:
    """Return a read-only quantity that retains its unit and numerical storage."""
    quantity = u.Quantity(value, copy=False)
    quantity.setflags(write=False)
    return quantity
