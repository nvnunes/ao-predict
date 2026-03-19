"""Shared statistics helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np

from . import schema
from .helpers import get_ee_apertures, get_fwhm_summary, get_sr_method
from .interfaces import SimulationSetup
from .interfaces import Simulation


StatValue: TypeAlias = np.ndarray | float


# Input validation and shaping


def _prepare_stats_inputs(
    psfs: np.ndarray,
    setup: Mapping[str, Any] | SimulationSetup,
    options: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> tuple[np.ndarray, bool, float, float, str, np.ndarray, str]:
    """Validate inputs and return normalized stats-computation prerequisites."""
    psf_cube = np.asarray(psfs, dtype=np.float32)
    scalar_output = False
    if psf_cube.ndim == 2:
        psf_cube = psf_cube[None, :, :]
        scalar_output = True
    elif psf_cube.ndim != 3:
        raise ValueError(f"PSFs must be [Ny, Nx] or [M, Ny, Nx], got shape {psf_cube.shape}")
    if not np.all(np.isfinite(psf_cube)):
        raise ValueError("PSFs contain non-finite values.")

    pixel_scale_mas = meta.get(schema.KEY_META_PIXEL_SCALE_MAS)
    if pixel_scale_mas is None:
        raise ValueError(f"meta['{schema.KEY_META_PIXEL_SCALE_MAS}'] is required for PSF stats computation.")
    pixel_scale_mas = float(pixel_scale_mas)
    if not np.isfinite(pixel_scale_mas) or pixel_scale_mas <= 0.0:
        raise ValueError(f"meta['{schema.KEY_META_PIXEL_SCALE_MAS}'] must be finite and > 0.")

    wavelength_um = options.get(schema.KEY_OPTION_WAVELENGTH_UM)
    if wavelength_um is None:
        raise ValueError(f"options['{schema.KEY_OPTION_WAVELENGTH_UM}'] is required for PSF stats computation.")
    wavelength_um = float(wavelength_um)
    if not np.isfinite(wavelength_um) or wavelength_um <= 0.0:
        raise ValueError(f"options['{schema.KEY_OPTION_WAVELENGTH_UM}'] must be finite and > 0.")

    try:
        ee_apertures_mas = get_ee_apertures(setup)
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"setup['{schema.KEY_SETUP_EE_APERTURES_MAS}'] is required for PSF stats computation.") from exc
    if ee_apertures_mas.size == 0:
        raise ValueError("ee_apertures_mas must be non-empty.")
    if not np.all(np.isfinite(ee_apertures_mas)) or np.any(ee_apertures_mas <= 0.0):
        raise ValueError("ee_apertures_mas must contain only finite values > 0.")

    try:
        sr_method = get_sr_method(setup)
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"setup['{schema.KEY_SETUP_SR_METHOD}'] is required for PSF stats computation.") from exc
    if sr_method not in schema.SETUP_STATS_SR_METHODS:
        raise ValueError(
            f"setup['{schema.KEY_SETUP_SR_METHOD}'] must be one of: {', '.join(schema.SETUP_STATS_SR_METHODS)}."
        )

    try:
        fwhm_summary = get_fwhm_summary(setup)
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"setup['{schema.KEY_SETUP_FWHM_SUMMARY}'] is required for PSF stats computation.") from exc
    if fwhm_summary not in schema.SETUP_STATS_FWHM_SUMMARIES:
        raise ValueError(
            f"setup['{schema.KEY_SETUP_FWHM_SUMMARY}'] must be one of: {', '.join(schema.SETUP_STATS_FWHM_SUMMARIES)}."
        )

    return psf_cube, scalar_output, pixel_scale_mas, wavelength_um, sr_method, ee_apertures_mas, fwhm_summary


# Placeholder metric stages


def _compute_strehl_pixel_fit(psfs: np.ndarray, pixel_scale_mas: float) -> tuple[np.ndarray, np.ndarray]:
    """Return placeholder Strehl values and fitted peak locations for ``pixel_fit``."""
    del pixel_scale_mas
    num_sci = int(psfs.shape[0])
    return (
        np.zeros((num_sci,), dtype=np.float32),
        np.zeros((num_sci, 2), dtype=np.float32),
    )


def _compute_strehl_pixel_max(psfs: np.ndarray, pixel_scale_mas: float) -> tuple[np.ndarray, np.ndarray]:
    """Return placeholder Strehl values and peak locations for ``pixel_max``."""
    del pixel_scale_mas
    num_sci = int(psfs.shape[0])
    return (
        np.zeros((num_sci,), dtype=np.float32),
        np.zeros((num_sci, 2), dtype=np.float32),
    )


def _compute_strehl(psfs: np.ndarray, sr_method: str, pixel_scale_mas: float) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the selected placeholder Strehl implementation."""
    if sr_method == schema.STATS_SR_METHOD_PIXEL_FIT:
        return _compute_strehl_pixel_fit(psfs, pixel_scale_mas)
    if sr_method == schema.STATS_SR_METHOD_PIXEL_MAX:
        return _compute_strehl_pixel_max(psfs, pixel_scale_mas)
    raise ValueError(f"Unsupported Strehl method: {sr_method}")


def _compute_ensquared_energy(
    psfs: np.ndarray,
    ee_apertures_mas: np.ndarray,
    pixel_scale_mas: float,
    peak_locations_xy: np.ndarray,
) -> np.ndarray:
    """Return placeholder EE values with the correct `[M, A]` shape."""
    del pixel_scale_mas, peak_locations_xy
    return np.zeros((int(psfs.shape[0]), int(ee_apertures_mas.shape[0])), dtype=np.float32)


def _measure_contour_fwhms(psfs: np.ndarray, pixel_scale_mas: float) -> tuple[np.ndarray, np.ndarray]:
    """Return placeholder contour-derived minimum and maximum FWHM vectors."""
    del pixel_scale_mas
    num_sci = int(psfs.shape[0])
    return (
        np.zeros((num_sci,), dtype=np.float32),
        np.zeros((num_sci,), dtype=np.float32),
    )


def _compute_fwhm_summary(fwhm_summary: str, fwhm_min: np.ndarray, fwhm_max: np.ndarray) -> np.ndarray:
    """Select one FWHM summary vector from contour-derived minimum/maximum widths."""
    if fwhm_summary == schema.STATS_FWHM_SUMMARY_GEOM:
        return np.sqrt(fwhm_max * fwhm_min, dtype=np.float32)
    if fwhm_summary == schema.STATS_FWHM_SUMMARY_MEAN:
        return 0.5 * (fwhm_max + fwhm_min)
    if fwhm_summary == schema.STATS_FWHM_SUMMARY_MAX:
        return fwhm_max
    if fwhm_summary == schema.STATS_FWHM_SUMMARY_MIN:
        return fwhm_min
    raise ValueError(f"Unsupported FWHM summary: {fwhm_summary}")


# Public entrypoint


def compute_psf_stats(
    psfs: np.ndarray,
    simulation: Simulation,
    setup: Mapping[str, Any] | SimulationSetup,
    options: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> tuple[StatValue, StatValue, StatValue]:
    """Return placeholder core PSF statistics through the staged Pass 2 flow.

    Args:
        psfs: PSF image or PSF cube.
        simulation: Bound simulation implementation providing preprocessing.
        setup: Setup payload or typed setup object.
        options: Per-simulation options mapping used by downstream stats algorithms.
        meta: Persisted PSF metadata mapping.

    Returns:
        Tuple ``(sr, ee, fwhm_mas)`` matching the shared core stats contract.
    """
    psf_cube, scalar_output, pixel_scale_mas, wavelength_um, sr_method, ee_apertures_mas, fwhm_summary = _prepare_stats_inputs(
        psfs,
        setup,
        options,
        meta,
    )
    del wavelength_um
    psf_cube = simulation.prepare_psfs_for_stats(psf_cube, setup, meta)

    sr, peak_locations_yx = _compute_strehl(psf_cube, sr_method, pixel_scale_mas)
    ee = _compute_ensquared_energy(psf_cube, ee_apertures_mas, pixel_scale_mas, peak_locations_yx)
    fwhm_min, fwhm_max = _measure_contour_fwhms(psf_cube, pixel_scale_mas)
    fwhm_mas = _compute_fwhm_summary(fwhm_summary, fwhm_min, fwhm_max)

    sr = np.asarray(sr, dtype=np.float32)
    ee = np.asarray(ee, dtype=np.float32)
    fwhm_mas = np.asarray(fwhm_mas, dtype=np.float32)

    if scalar_output:
        sr_value: StatValue = float(sr[0])
        ee_value: StatValue = ee[0].copy()
        if ee_value.shape[0] == 1:
            ee_value = float(ee_value[0])
        fwhm_value: StatValue = float(fwhm_mas[0])
        return sr_value, ee_value, fwhm_value

    return sr, ee, fwhm_mas
