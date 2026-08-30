"""Simulation-domain utility helpers."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from astropy import units as u

from . import schema
from .interfaces import SimulationSetup
from .._units import quantity_value
from ..utils import as_array


# Setup helpers

_MISSING = object()


def select_mapping_value(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    key: str,
    *,
    default: Any = _MISSING,
) -> Any:
    """Select mapping value preferring ``primary``, then ``secondary``."""
    if key in primary:
        return primary[key]
    if key in secondary:
        return secondary[key]
    if default is not _MISSING:
        return default
    raise KeyError(key)

def get_num_sci(setup: Mapping[str, Any] | SimulationSetup) -> int:
    """Return the number of science points ``M`` from setup payload or object.

    Args:
        setup: Persisted setup payload mapping or typed setup object.

    Returns:
        Number of science points inferred from ``sci_r``.
    """
    sci_r = setup[schema.KEY_SETUP_SCI_R] if isinstance(setup, Mapping) else getattr(setup, schema.KEY_SETUP_SCI_R)
    return int(as_array(sci_r).shape[0])


def get_ee_apertures(setup: Mapping[str, Any] | SimulationSetup) -> u.Quantity:
    """Return EE aperture widths as a non-empty 1D float vector.

    Args:
        setup: Persisted setup payload mapping or typed setup object.

    Returns:
        1D float array of EE aperture widths.

    Raises:
        ValueError: If the EE aperture vector is empty.
    """
    ee_apertures = setup[schema.KEY_SETUP_EE_APERTURES] if isinstance(setup, Mapping) else getattr(setup, schema.KEY_SETUP_EE_APERTURES)
    ee = quantity_value(ee_apertures, u.mas, label=schema.KEY_SETUP_EE_APERTURES, dtype=float).reshape(-1)
    if ee.shape[0] == 0:
        raise ValueError(f"setup['{schema.KEY_SETUP_EE_APERTURES}'] must be a non-empty 1D array.")
    return ee * u.mas


def get_sr_method(setup: Mapping[str, Any] | SimulationSetup) -> str:
    """Return the dataset-level Strehl selector from setup."""
    sr_method = setup[schema.KEY_SETUP_SR_METHOD] if isinstance(setup, Mapping) else getattr(setup, schema.KEY_SETUP_SR_METHOD)
    value = str(sr_method).strip()
    if not value:
        raise ValueError(f"setup['{schema.KEY_SETUP_SR_METHOD}'] must be a non-empty string.")
    return value


def get_fwhm_summary(setup: Mapping[str, Any] | SimulationSetup) -> str:
    """Return the dataset-level FWHM summary selector from setup."""
    fwhm_summary = setup[schema.KEY_SETUP_FWHM_SUMMARY] if isinstance(setup, Mapping) else getattr(setup, schema.KEY_SETUP_FWHM_SUMMARY)
    value = str(fwhm_summary).strip()
    if not value:
        raise ValueError(f"setup['{schema.KEY_SETUP_FWHM_SUMMARY}'] must be a non-empty string.")
    return value


def get_ee_geometry(setup: Mapping[str, Any] | SimulationSetup) -> str:
    """Return the dataset-level EE aperture geometry selector from setup."""
    ee_geometry = setup[schema.KEY_SETUP_EE_GEOMETRY] if isinstance(setup, Mapping) else getattr(setup, schema.KEY_SETUP_EE_GEOMETRY)
    value = str(ee_geometry).strip()
    if not value:
        raise ValueError(f"setup['{schema.KEY_SETUP_EE_GEOMETRY}'] must be a non-empty string.")
    return value


# Stats preprocessing helpers


def clip_psf_non_negative(psfs: np.ndarray) -> np.ndarray:
    """Return PSFs after applying the shared non-negative clipping stage."""
    return np.maximum(psfs, 0.0)


def normalize_psf_pixel_sum(psfs: np.ndarray) -> np.ndarray:
    """Return PSFs normalized by per-PSF pixel sum when that sum is positive."""
    psf_cube = psfs.copy()
    denom = np.sum(psf_cube, axis=(-2, -1), dtype=np.float32)
    positive = denom > 0.0
    if np.any(positive):
        psf_cube[positive] /= denom[positive, np.newaxis, np.newaxis]
    return psf_cube


# Atmosphere helpers

def r0_to_seeing(r0: u.Quantity, wavelength: u.Quantity) -> u.Quantity:
    """Convert ``r0`` at wavelength into seeing."""
    r0_value = float(quantity_value(r0, u.m, label="r0").item())
    wavelength_value = float(quantity_value(wavelength, u.m, label="wavelength").item())
    if r0_value <= 0.0:
        raise ValueError("r0 must be > 0 for conversion to seeing.")
    if wavelength_value <= 0.0:
        raise ValueError("wavelength must be > 0 for conversion to seeing.")
    seeing_rad = 0.98 * wavelength_value / r0_value
    return float(seeing_rad * (648000.0 / math.pi)) * u.arcsec


def seeing_to_r0(seeing: u.Quantity, wavelength: u.Quantity) -> u.Quantity:
    """Convert seeing at wavelength into ``r0``."""
    seeing_value = float(quantity_value(seeing, u.arcsec, label="seeing").item())
    wavelength_value = float(quantity_value(wavelength, u.m, label="wavelength").item())
    if seeing_value <= 0.0:
        raise ValueError("seeing must be > 0 for conversion to r0.")
    if wavelength_value <= 0.0:
        raise ValueError("wavelength must be > 0 for conversion to r0.")
    seeing_rad = seeing_value * (math.pi / 648000.0)
    return float(0.98 * wavelength_value / seeing_rad) * u.m
