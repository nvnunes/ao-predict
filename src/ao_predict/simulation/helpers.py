"""Simulation-domain utility helpers."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np

from . import schema
from .interfaces import SimulationSetup
from ..utils import as_array, as_float_vector


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
        Number of science points inferred from ``sci_r_arcsec``.
    """
    sci_r_arcsec = setup[schema.KEY_SETUP_SCI_R_ARCSEC] if isinstance(setup, Mapping) else getattr(setup, schema.KEY_SETUP_SCI_R_ARCSEC)
    return int(as_array(sci_r_arcsec).shape[0])


def get_ee_apertures(setup: Mapping[str, Any] | SimulationSetup) -> np.ndarray:
    """Return EE aperture widths as a non-empty 1D float vector.

    Args:
        setup: Persisted setup payload mapping or typed setup object.

    Returns:
        1D float array of EE aperture widths.

    Raises:
        ValueError: If the EE aperture vector is empty.
    """
    ee_apertures_mas = setup[schema.KEY_SETUP_EE_APERTURES_MAS] if isinstance(setup, Mapping) else getattr(setup, schema.KEY_SETUP_EE_APERTURES_MAS)
    ee = as_float_vector(ee_apertures_mas, label=schema.KEY_SETUP_EE_APERTURES_MAS)
    if ee.shape[0] == 0:
        raise ValueError(f"setup['{schema.KEY_SETUP_EE_APERTURES_MAS}'] must be a non-empty 1D array.")
    return ee


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


# Atmosphere helpers

def r0_to_seeing_arcsec(r0_m: float, wavelength_m: float) -> float:
    """Convert ``r0`` at wavelength into seeing in arcseconds."""
    if r0_m <= 0.0:
        raise ValueError("r0_m must be > 0 for conversion to seeing.")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be > 0 for conversion to seeing.")
    seeing_rad = 0.98 * float(wavelength_m) / float(r0_m)
    return float(seeing_rad * (648000.0 / math.pi))


def seeing_arcsec_to_r0_m(seeing_arcsec: float, wavelength_m: float) -> float:
    """Convert seeing in arcseconds at wavelength into ``r0`` in meters."""
    if seeing_arcsec <= 0.0:
        raise ValueError("seeing_arcsec must be > 0 for conversion to r0_m.")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be > 0 for conversion to r0_m.")
    seeing_rad = float(seeing_arcsec) * (math.pi / 648000.0)
    return float(0.98 * float(wavelength_m) / seeing_rad)
