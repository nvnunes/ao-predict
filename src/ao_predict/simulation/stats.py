"""Shared statistics helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias
import warnings

import contourpy
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import shift
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
    pixel_scale_rad = pixel_scale_mas / 1000.0 / 3600.0 / 180.0 * np.pi
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


# Ensquared Energy helpers

def _anchor_psfs_for_ee(
    psfs: np.ndarray,
    peak_locations_yx: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PSFs plus integer peak anchors for AO Predict EE."""
    psfs = np.asarray(psfs, dtype=np.float32)
    num_sci, ny, nx = psfs.shape

    if peak_locations_yx is not None:
        peak_locations_yx = np.asarray(peak_locations_yx)
        if np.issubdtype(peak_locations_yx.dtype, np.integer):
            peak_y = peak_locations_yx[:, 0].astype(np.int64, copy=False)
            peak_x = peak_locations_yx[:, 1].astype(np.int64, copy=False)
        elif np.issubdtype(peak_locations_yx.dtype, np.floating):
            eps = 1e-12
            anchor_yx = np.floor(peak_locations_yx + 0.5 - eps).astype(np.int64)
            shift_yx = anchor_yx - peak_locations_yx
            shifted_psfs = np.empty_like(psfs)
            for i in range(num_sci):
                shifted_psfs[i] = shift(
                    psfs[i],
                    shift=tuple(shift_yx[i]),
                    order=3,
                    mode="constant",
                    cval=0.0,
                    prefilter=True,
                )
            psfs = np.clip(shifted_psfs, 0.0, None)
            peak_y = anchor_yx[:, 0]
            peak_x = anchor_yx[:, 1]
        else:
            raise TypeError("peak_locations_yx must have an integer or floating dtype")
    else:
        peak_idx = np.argmax(psfs.reshape(num_sci, -1), axis=1)
        peak_y, peak_x = np.divmod(peak_idx, nx)

    return psfs, peak_y, peak_x


def _measure_peak_centered_ee_curves(
    psfs: np.ndarray,
    peak_y: np.ndarray,
    peak_x: np.ndarray,
    max_ee_radius_mas: float,
    pixel_scale_mas: float,
    *,
    extra_box_radii: int = 3,
) -> np.ndarray:
    """Measure cumulative odd-sized square-box EE curves about integer anchors."""
    num_sci, ny, nx = psfs.shape
    integral = np.pad(
        psfs.cumsum(axis=1).cumsum(axis=2),
        ((0, 0), (1, 0), (1, 0)),
        mode="constant",
    )

    def _rect_sum(idx: np.ndarray, y0: np.ndarray, y1: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return (
            integral[idx, y1, x1]
            - integral[idx, y0, x1]
            - integral[idx, y1, x0]
            + integral[idx, y0, x0]
        )

    edge_limited_radius = int(np.min(np.stack([peak_y, ny - 1 - peak_y, peak_x, nx - 1 - peak_x])))
    required_radius = int(np.ceil((max_ee_radius_mas / pixel_scale_mas - 1.0) * 0.5))
    max_radius = min(edge_limited_radius, max(0, required_radius) + int(extra_box_radii))
    radii = np.arange(max_radius + 1, dtype=np.int64)
    ee_curves = np.zeros((num_sci, radii.size), dtype=psfs.dtype)
    idx = np.arange(num_sci, dtype=np.int64)

    for j, radius in enumerate(radii):
        y0 = peak_y - radius
        y1 = peak_y + radius + 1
        x0 = peak_x - radius
        x1 = peak_x + radius + 1
        ee_curves[:, j] = _rect_sum(idx, y0, y1, x0, x1)
    return ee_curves


def _interpolate_ee_at_radii(
    ee_curves: np.ndarray,
    pixel_scale_mas: float,
    ee_radii: np.ndarray,
) -> np.ndarray:
    """Interpolate cumulative EE curves onto requested physical half-widths."""
    rr = np.arange(1, ee_curves.shape[1] * 2, 2, dtype=float) * pixel_scale_mas * 0.5
    ee_at_radius = np.empty((ee_curves.shape[0], ee_radii.size), dtype=float)
    for i, values in enumerate(ee_curves):
        if rr.size == 1:
            interpolated = np.full(ee_radii.shape, float(values[0]), dtype=float)
        else:
            interp_kind = "cubic" if rr.size >= 4 else "linear"
            interpolated = interp1d(rr, values, kind=interp_kind, bounds_error=False, fill_value="extrapolate")(ee_radii)
        ee_at_radius[i, :] = np.clip(interpolated, 0.0, 1.0)
    return ee_at_radius


def _compute_ensquared_energy(
    psfs: np.ndarray,
    ee_apertures_mas: np.ndarray,
    pixel_scale_mas: float,
    peak_locations_yx: np.ndarray | None
) -> np.ndarray:
    """Compute AO Predict EE using peak-centered odd-sized square apertures."""
    ee_apertures_mas = np.asarray(ee_apertures_mas, dtype=float)
    ee_radii = ee_apertures_mas / 2.0

    psfs, peak_y, peak_x = _anchor_psfs_for_ee(psfs, peak_locations_yx)
    ee_curves = _measure_peak_centered_ee_curves(
        psfs,
        peak_y,
        peak_x,
        ee_radii.max(),
        pixel_scale_mas,
    )
    ee_at_radius = _interpolate_ee_at_radii(ee_curves, pixel_scale_mas, ee_radii)
    return np.asarray(ee_at_radius, dtype=np.float32)


# FWHM helpers

def _find_contours(psf: np.ndarray, level: float) -> list[np.ndarray]:
    """Return contour vertices in `[N, 2]` arrays with `(x, y)` columns."""
    generator = contourpy.contour_generator(z=np.asarray(psf, dtype=float))
    return [np.asarray(line, dtype=float) for line in generator.lines(float(level))]

def _measure_contour_fwhms(psfs: np.ndarray, pixel_scale_mas: float) -> tuple[np.ndarray, np.ndarray]:
    """Measure AO Predict contour-derived minimum and maximum FWHM vectors."""
    psfs = np.asarray(psfs, dtype=np.float32)
    num_sci, ny, nx = psfs.shape
    fwhm_min = np.full((num_sci,), np.nan, dtype=np.float32)
    fwhm_max = np.full((num_sci,), np.nan, dtype=np.float32)
    edge_eps = 1e-6

    for i, psf in enumerate(psfs):
        # Non-finite PSFs do not define a usable half-max contour.
        if not np.all(np.isfinite(psf)):
            continue

        max_val = float(np.max(psf))
        min_val = float(np.min(psf))
        half_max = 0.5 * max_val
        # Non-positive peaks have no scientifically meaningful FWHM threshold.
        if max_val <= 0.0:
            continue
        # If the image never drops below half max, no contour crossing exists.
        if min_val >= half_max:
            continue

        contours = _find_contours(psf, half_max)
        # No extracted contour means the half-max geometry is unrecoverable.
        if not contours:
            continue
        contour_points = max(contours, key=len)
        # At least three vertices are required to define a 2D contour shape.
        if len(contour_points) < 3:
            continue

        x_coords = contour_points[:, 0]
        y_coords = contour_points[:, 1]
        # Edge-touching contours are treated as truncated and therefore invalid.
        if x_coords.min() <= edge_eps or x_coords.max() >= (nx - 1) - edge_eps:
            continue
        if y_coords.min() <= edge_eps or y_coords.max() >= (ny - 1) - edge_eps:
            continue

        x_span = x_coords.max() - x_coords.min()
        y_span = y_coords.max() - y_coords.min()
        # Collapsed spans indicate degenerate contour geometry on the native grid.
        if x_span <= edge_eps or y_span <= edge_eps:
            continue
        rounded_points = np.round(contour_points, decimals=6)
        # Too few distinct vertices after rounding means the contour is numerically degenerate.
        if np.unique(rounded_points, axis=0).shape[0] < 3:
            continue

        contour_center = 0.5 * np.array(
            [
                x_coords.max() + x_coords.min(),
                y_coords.max() + y_coords.min(),
            ],
            dtype=float,
        )
        radial_distances = np.hypot(
            x_coords - contour_center[0],
            y_coords - contour_center[1],
        ) * pixel_scale_mas
        radial_max = float(radial_distances.max())
        radial_min = float(radial_distances.min())
        # Non-positive radial widths indicate collapsed contour-derived FWHM.
        if radial_max <= 0.0 or radial_min <= 0.0:
            continue

        fwhm_max[i] = np.float32(2.0 * radial_max)
        fwhm_min[i] = np.float32(2.0 * radial_min)

    return fwhm_min, fwhm_max


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
    """Compute the core AO Predict PSF statistics for one PSF image or cube.

    Args:
        psfs: PSF image or PSF cube.
        simulation: Bound simulation implementation providing preprocessing.
        setup: Setup payload or typed setup object containing the dataset-level
            stats selectors and EE apertures.
        options: Per-simulation options mapping. ``wavelength_um`` is required
            for the diffraction-limited Strehl reference PSF.
        meta: Persisted PSF metadata mapping.

    Returns:
        Tuple ``(sr, ee, fwhm_mas)`` matching the shared core stats contract.
        The implemented metric family is:
        - Strehl: image-domain `pixel_fit` or `pixel_max`
        - EE: fixed peak-centered image-domain square-box accumulation
        - FWHM: fixed native contour measurement summarized by
          ``setup['fwhm_summary']``

        Successful results may return ``NaN`` in ``fwhm_mas`` when the
        contour-based FWHM is not scientifically recoverable.
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
