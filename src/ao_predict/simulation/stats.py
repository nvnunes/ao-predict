"""Shared statistics helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias
import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

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
) -> tuple[np.ndarray, bool, float, float, float, np.ndarray, str, np.ndarray, str]:
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

    wavelength_um = options.get(schema.KEY_OPTION_WAVELENGTH_UM)
    if wavelength_um is None:
        raise ValueError(f"options['{schema.KEY_OPTION_WAVELENGTH_UM}'] is required for PSF stats computation.")
    wavelength_um = float(wavelength_um)
    if not np.isfinite(wavelength_um) or wavelength_um <= 0.0:
        raise ValueError(f"options['{schema.KEY_OPTION_WAVELENGTH_UM}'] must be finite and > 0.")

    pixel_scale_mas = meta.get(schema.KEY_META_PIXEL_SCALE_MAS)
    if pixel_scale_mas is None:
        raise ValueError(f"meta['{schema.KEY_META_PIXEL_SCALE_MAS}'] is required for PSF stats computation.")
    pixel_scale_mas = float(pixel_scale_mas)
    if not np.isfinite(pixel_scale_mas) or pixel_scale_mas <= 0.0:
        raise ValueError(f"meta['{schema.KEY_META_PIXEL_SCALE_MAS}'] must be finite and > 0.")

    tel_diameter_m = meta.get(schema.KEY_META_TEL_DIAMETER_M)
    if tel_diameter_m is None:
        raise ValueError(f"meta['{schema.KEY_META_TEL_DIAMETER_M}'] is required for PSF stats computation.")
    tel_diameter_m = float(tel_diameter_m)
    if not np.isfinite(tel_diameter_m) or tel_diameter_m <= 0.0:
        raise ValueError(f"meta['{schema.KEY_META_TEL_DIAMETER_M}'] must be finite and > 0.")

    tel_pupil = meta.get(schema.KEY_META_TEL_PUPIL)
    if tel_pupil is None:
        raise ValueError(f"meta['{schema.KEY_META_TEL_PUPIL}'] is required for PSF stats computation.")
    tel_pupil = np.asarray(tel_pupil, dtype=np.float32)
    if tel_pupil.ndim != 2:
        raise ValueError(f"meta['{schema.KEY_META_TEL_PUPIL}'] must be 2D [Ny, Nx].")
    if not np.all(np.isfinite(tel_pupil)):
        raise ValueError(f"meta['{schema.KEY_META_TEL_PUPIL}'] must contain only finite values.")

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

    return psf_cube, scalar_output, wavelength_um, pixel_scale_mas, tel_diameter_m, tel_pupil, sr_method, ee_apertures_mas, fwhm_summary


# Strehl helpers


def _axis_indices_and_coords(length: int, nbox: int) -> tuple[np.ndarray, np.ndarray]:
    """Return centered patch indices and coordinates for one PSF axis."""
    nbox = min(int(nbox), int(length))
    if (length % 2) == 0:
        size = nbox + 1 if (nbox % 2) == 1 else nbox
    else:
        size = nbox + 1 if (nbox % 2) == 0 else nbox
    size = min(size, int(length))

    center = int(length // 2)
    half = int(size // 2)
    if (length % 2) == 1:
        start = center - half
        stop = center + half + 1
        idx = np.arange(start, stop, dtype=np.int64)
        coord = idx - center
    else:
        start = center - half
        stop = start + size
        idx = np.arange(start, stop, dtype=np.int64)
        coord = (idx - center) + 0.5
    return idx, np.asarray(coord, dtype=float)


def _extract_core_patch(psf: np.ndarray, nbox: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a parity-aware core patch centered on the optical center."""
    y_idx, y_coord = _axis_indices_and_coords(int(psf.shape[0]), nbox)
    x_idx, x_coord = _axis_indices_and_coords(int(psf.shape[1]), nbox)
    patch = psf[np.ix_(y_idx, x_idx)]
    grid_x, grid_y = np.meshgrid(x_coord, y_coord)
    return patch, grid_x, grid_y


def _gaussian2d(coords: tuple[np.ndarray, np.ndarray], amplitude: float, sigma_x: float, sigma_y: float, x0: float, y0: float, theta: float) -> np.ndarray:
    """Evaluate a rotated 2D Gaussian."""
    x, y = coords
    xr = x - x0
    yr = y - y0
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rot = cos_theta * xr + sin_theta * yr
    y_rot = -sin_theta * xr + cos_theta * yr
    return amplitude * np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))


def _fit_gaussian_core(
    psf: np.ndarray,
    nbox: int = 4,
    *,
    assume_centered: bool = False,
    assume_circular: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a core Gaussian and return parameters in pixel-center coordinates."""
    patch, grid_x, grid_y = _extract_core_patch(psf, nbox)
    coords = np.vstack([grid_x.ravel(), grid_y.ravel()])
    values = patch.ravel()

    amplitude0 = float(patch.max())
    sigma_x0 = max(1.0, float(np.std(grid_x[patch == patch.max()]))) if np.any(patch == patch.max()) else 1.0
    sigma_y0 = max(1.0, float(np.std(grid_y[patch == patch.max()]))) if np.any(patch == patch.max()) else 1.0
    p0 = [amplitude0, sigma_x0, sigma_y0, 0.0, 0.0, 0.0]

    x_min, x_max = float(grid_x.min()), float(grid_x.max())
    y_min, y_max = float(grid_y.min()), float(grid_y.max())
    if assume_centered:
        x_min = y_min = -1e-6
        x_max = y_max = 1e-6
    theta_min = -np.pi / 2
    theta_max = np.pi / 2
    if assume_circular:
        theta_min = -1e-6
        theta_max = 1e-6

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, pcov = curve_fit(
            _gaussian2d,
            coords,
            values,
            p0=p0,
            bounds=(
                [0.0, 0.1, 0.1, x_min, y_min, theta_min],
                [np.inf, 100.0, 100.0, x_max, y_max, theta_max],
            ),
        )

    ny, nx = psf.shape
    x_center = 0.5 * (nx - 1)
    y_center = 0.5 * (ny - 1)
    amplitude_fit, sigma_x_fit, sigma_y_fit, x0_fit, y0_fit, theta_fit = popt
    return (
        np.asarray(
            [
                amplitude_fit,
                sigma_x_fit,
                sigma_y_fit,
                y_center + y0_fit,
                x_center + x0_fit,
                theta_fit,
            ],
            dtype=float,
        ),
        np.asarray(pcov, dtype=float),
    )


def _get_diffraction_limited_psf(
    tel_diameter_m: float,
    tel_pupil: np.ndarray,
    wavelength_um: float,
    pixel_scale_mas: float,
    *,
    center_in_one_pix: bool,
) -> np.ndarray:
    """Compute the AO Predict diffraction-limited PSF from the pupil."""
    wavelength_m = float(wavelength_um) * 1e-6
    pixel_scale_rad = float(pixel_scale_mas) / 1000.0 / 3600.0 / 180.0 * np.pi
    sampling = wavelength_m / float(tel_diameter_m) / pixel_scale_rad

    pupil = np.abs(np.asarray(tel_pupil, dtype=float))
    nx, ny = pupil.shape
    pad_x = int((sampling - 1.0) * nx / 2.0)
    pad_y = int((sampling - 1.0) * ny / 2.0)
    pupil_pad = np.pad(pupil, [(pad_x, pad_x), (pad_y, pad_y)])

    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_pad)))
    psf = np.abs(field) ** 2

    n_x, n_y = psf.shape
    if (n_x % 2) == 0 and not center_in_one_pix:
        u_1d = np.fft.fftshift(np.fft.fftfreq(n_x))
        v_1d = np.fft.fftshift(np.fft.fftfreq(n_y))
        u_2d, v_2d = np.meshgrid(u_1d, v_1d, indexing="ij")
        otf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf)))
        otf *= np.exp(1j * np.pi * (u_2d + v_2d))
        psf = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(otf))))

    psf = np.clip(psf, 0.0, None)
    psf /= psf.sum()
    return psf.astype(np.float32, copy=False)


def _compute_strehl_pixel_fit(
    psfs: np.ndarray,
    pixel_scale_mas: float,
    wavelength_um: float,
    tel_diameter_m: float,
    tel_pupil: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AO Predict Strehl via fitted core amplitude."""
    peak_fits = np.asarray(
        [
            _fit_gaussian_core(np.asarray(psf, dtype=float), assume_centered=False)[0]
            for psf in np.asarray(psfs, dtype=float)
        ],
        dtype=float,
    )
    peak_amplitude = peak_fits[:, 0]
    peak_locations_yx = peak_fits[:, [3, 4]].astype(np.float32)

    psf_dl = _get_diffraction_limited_psf(
        tel_diameter_m,
        tel_pupil,
        wavelength_um,
        pixel_scale_mas,
        center_in_one_pix=False,
    )
    peak_amplitude_dl = float(_fit_gaussian_core(psf_dl, assume_centered=True, assume_circular=True)[0][0])
    sr = peak_amplitude / peak_amplitude_dl
    return np.asarray(sr, dtype=np.float32), peak_locations_yx


def _compute_strehl_pixel_max(
    psfs: np.ndarray,
    pixel_scale_mas: float,
    wavelength_um: float,
    tel_diameter_m: float,
    tel_pupil: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AO Predict Strehl via native peak pixel."""
    psfs = np.asarray(psfs, dtype=np.float32)
    num_sci = int(psfs.shape[0])
    peak_idx = np.argmax(psfs.reshape(num_sci, -1), axis=1)
    peak_amplitude = psfs.reshape(num_sci, -1)[np.arange(num_sci), peak_idx]
    peak_y, peak_x = np.unravel_index(peak_idx, psfs.shape[1:])
    peak_locations_yx = np.stack((peak_y, peak_x), axis=1).astype(np.float32)

    psf_dl = _get_diffraction_limited_psf(
        tel_diameter_m,
        tel_pupil,
        wavelength_um,
        pixel_scale_mas,
        center_in_one_pix=True,
    )
    peak_amplitude_dl = float(psf_dl.max())
    sr = peak_amplitude / peak_amplitude_dl
    return np.asarray(sr, dtype=np.float32), peak_locations_yx


def _compute_strehl(
    psfs: np.ndarray,
    sr_method: str,
    pixel_scale_mas: float,
    wavelength_um: float,
    tel_diameter_m: float,
    tel_pupil: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the selected AO Predict Strehl implementation."""
    if sr_method == schema.STATS_SR_METHOD_PIXEL_FIT:
        return _compute_strehl_pixel_fit(psfs, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil)
    if sr_method == schema.STATS_SR_METHOD_PIXEL_MAX:
        return _compute_strehl_pixel_max(psfs, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil)
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
    psf_cube, scalar_output, wavelength_um, pixel_scale_mas, tel_diameter_m, tel_pupil, sr_method, ee_apertures_mas, fwhm_summary = _prepare_stats_inputs(
        psfs,
        setup,
        options,
        meta,
    )
    psf_cube = simulation.prepare_psfs_for_stats(psf_cube, setup, meta)

    sr, peak_locations_yx = _compute_strehl(
        psf_cube,
        sr_method,
        pixel_scale_mas,
        wavelength_um,
        tel_diameter_m,
        tel_pupil,
    )
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
