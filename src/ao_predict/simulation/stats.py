"""Shared statistics helpers."""

from __future__ import annotations

from typing import Any, Mapping, TypeAlias

from .helpers import get_ee_apertures, get_fwhm_summary, get_sr_method
from .interfaces import SimulationSetup

import numpy as np

from . import schema


StatValue: TypeAlias = np.ndarray | float


def compute_psf_stats(
    psfs: np.ndarray,
    setup: Mapping[str, Any] | SimulationSetup,
    meta: Mapping[str, Any],
) -> tuple[StatValue, StatValue, StatValue]:
    """Return zero-valued placeholder core PSF statistics with the correct shape contract.

    Args:
        psfs: PSF image or PSF cube.
        setup: Setup payload or typed setup object.
        meta: Persisted PSF metadata mapping.

    Returns:
        Tuple ``(sr, ee, fwhm_mas)`` matching the shared core stats contract.
    """
    cube = np.asarray(psfs, dtype=np.float32)
    scalar_output = False
    if cube.ndim == 2:
        cube = cube[None, :, :]
        scalar_output = True
    elif cube.ndim != 3:
        raise ValueError(f"PSFs must be [Ny, Nx] or [M, Ny, Nx], got shape {cube.shape}")
    if not np.all(np.isfinite(cube)):
        raise ValueError("PSFs contain non-finite values.")

    pixel_scale_mas = meta.get(schema.KEY_META_PIXEL_SCALE_MAS)
    if pixel_scale_mas is None:
        raise ValueError(f"meta['{schema.KEY_META_PIXEL_SCALE_MAS}'] is required for PSF stats computation.")
    pixel_scale_mas = float(pixel_scale_mas)
    if not np.isfinite(pixel_scale_mas) or pixel_scale_mas <= 0.0:
        raise ValueError(f"meta['{schema.KEY_META_PIXEL_SCALE_MAS}'] must be finite and > 0.")

    try:
        apertures = get_ee_apertures(setup)
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"setup['{schema.KEY_SETUP_EE_APERTURES_MAS}'] is required for PSF stats computation.") from exc
    if apertures.size == 0:
        raise ValueError("ee_apertures_mas must be non-empty.")
    if not np.all(np.isfinite(apertures)) or np.any(apertures <= 0.0):
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

    num_sci = int(cube.shape[0])

    # Placeholder implementation until the shared PSF statistics code is added.
    sr = np.zeros((num_sci,), dtype=np.float32)
    ee = np.zeros((num_sci, apertures.size), dtype=np.float32)
    fwhm_mas = np.zeros((num_sci,), dtype=np.float32)

    if scalar_output:
        sr = float(sr[0])
        ee = ee[0].copy()
        if ee.shape[0] == 1:
            ee = float(ee[0])
        fwhm_mas = float(fwhm_mas[0])

    return sr, ee, fwhm_mas
