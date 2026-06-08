"""Generic plotting helpers for loaded analysis simulations.

This module owns generic plotting for immutable ``AnalysisSimulation``-style
objects. PSF plots use persisted pixel-scale metadata, PSF-core plots use a
centered native-pixel crop, and metric-field plots use persisted science
coordinates with SciPy interpolation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TypeAlias

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import Rbf, griddata

from .analysis import AnalysisSimulation
from .simulation import schema


_MIN_LOG_VALUE = 1.0e-30
_DEFAULT_FIGSIZE = (5.0, 4.0)
_LABEL_ARCSEC_SKY = '["/Sky]'
_LABEL_MAS_SKY = "[mas/Sky]"
_PlotMapping: TypeAlias = Mapping[str, object]
_MetricInterpolation: TypeAlias = str


def _figure_and_axis(ax: Axes | None) -> tuple[Figure, Axes]:
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE, constrained_layout=True)
    return fig, ax


def _add_matched_colorbar(fig: Figure, ax: Axes, im: object, label: str) -> None:
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label)
    plt.sca(ax)


def _require_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer.")
    return value


def _require_positive_int(value: int, label: str) -> int:
    value = _require_int(value, label)
    if value <= 0:
        raise ValueError(f"{label} must be positive.")
    return value


def _require_metric_interpolation(value: str) -> _MetricInterpolation:
    if value not in {"rbf", "nearest"}:
        raise ValueError("interpolation must be 'rbf' or 'nearest'.")
    return value


def _require_positive_scalar(mapping: _PlotMapping, key: str) -> float:
    if key not in mapping:
        raise ValueError(f"Missing required plotting field '{key}'.")
    value = np.asarray(mapping[key], dtype=float)
    if value.size != 1:
        raise ValueError(f"Plotting field '{key}' must be a scalar.")
    scalar = float(value.reshape(-1)[0])
    if scalar <= 0.0:
        raise ValueError(f"Plotting field '{key}' must be positive.")
    return scalar


def _require_1d_array(mapping: _PlotMapping, key: str) -> np.ndarray:
    if key not in mapping:
        raise ValueError(f"Missing required plotting field '{key}'.")
    value = np.asarray(mapping[key], dtype=np.float32)
    if value.ndim != 1:
        raise ValueError(f"Plotting field '{key}' must be a 1D array.")
    return value


def _select_psf(simulation: AnalysisSimulation, psf_index: int) -> np.ndarray:
    psf_index = _require_int(psf_index, "psf_index")
    psfs = np.asarray(simulation.psfs)
    if psfs.ndim != 3:
        raise ValueError("AnalysisSimulation.psfs must be a 3D cube with shape (N, Y, X).")
    if psf_index < 0 or psf_index >= psfs.shape[0]:
        raise ValueError(f"psf_index {psf_index} is out of range for {psfs.shape[0]} PSFs.")
    return np.asarray(psfs[psf_index], dtype=np.float32)


def _select_metric_values(
    simulation: AnalysisSimulation,
    metric_name: str,
    value_index: int,
) -> np.ndarray:
    if metric_name not in simulation.stats:
        raise ValueError(f"Metric '{metric_name}' is not available in simulation.stats.")
    values = np.asarray(simulation.stats[metric_name], dtype=np.float32)
    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim > 1:
        value_index = _require_int(value_index, "value_index")
        values = values.reshape(values.shape[0], -1)
        if value_index < 0 or value_index >= values.shape[1]:
            raise ValueError(
                f"value_index {value_index} is out of range for metric '{metric_name}' "
                f"with {values.shape[1]} values per science point."
            )
        values = values[:, value_index]
    return values


def _psf_extent(psf: np.ndarray, pixel_scale_mas: float) -> tuple[float, float, float, float]:
    ny, nx = psf.shape
    return (
        -0.5 * nx * pixel_scale_mas,
        0.5 * nx * pixel_scale_mas,
        -0.5 * ny * pixel_scale_mas,
        0.5 * ny * pixel_scale_mas,
    )


def _center_crop(psf: np.ndarray, size_px: int) -> np.ndarray:
    ny, nx = psf.shape
    cy = int(round((ny - 1) / 2.0))
    cx = int(round((nx - 1) / 2.0))
    half = size_px // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(ny, y0 + size_px)
    x1 = min(nx, x0 + size_px)
    return psf[y0:y1, x0:x1]


def _default_core_size_px(psf: np.ndarray) -> int:
    if psf.shape[1] % 2 == 0:
        return 6
    return 5


def _core_extent(core: np.ndarray) -> tuple[float, float, float, float]:
    ny, nx = core.shape
    return (-0.5 * nx, 0.5 * nx, -0.5 * ny, 0.5 * ny)


def _default_metric_cmap(metric_name: str) -> str:
    if metric_name == schema.KEY_STATS_FWHM_MAS:
        return "plasma_r"
    return "plasma"


def _metric_colorbar_label(metric_name: str) -> str:
    if metric_name == schema.KEY_STATS_SR:
        return "Strehl Ratio"
    if metric_name == schema.KEY_STATS_EE:
        return "EE"
    if metric_name == schema.KEY_STATS_FWHM_MAS:
        return "FWHM [mas]"
    return metric_name


def _field_radius(r_arcsec: np.ndarray) -> float:
    if r_arcsec.size == 0:
        raise ValueError("Science coordinate arrays must contain at least one point.")
    radius = float(np.nanmax(r_arcsec))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Science coordinates must span a positive field radius.")
    return radius


def _field_grid(
    radius: float,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid_size = _require_positive_int(grid_size, "grid_size")
    xi = np.linspace(-radius, radius, grid_size)
    yi = np.linspace(-radius, radius, grid_size)
    x_grid, y_grid = np.meshgrid(xi, yi)
    mask = np.ones(x_grid.shape, dtype=np.float32)
    mask[np.sqrt(x_grid**2 + y_grid**2) > radius] = np.nan
    return xi, yi, x_grid, y_grid, mask


def _interpolate_metric_field(
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
    metric_values: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    interpolation: _MetricInterpolation,
) -> np.ndarray:
    if x_arcsec.size < 3:
        raise ValueError("Metric field interpolation requires at least three science points.")
    if interpolation == "nearest":
        return griddata((x_arcsec, y_arcsec), metric_values, (x_grid, y_grid), method="nearest")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rbf = Rbf(x_arcsec, y_arcsec, metric_values, function="cubic")
        return rbf(x_grid, y_grid)


def _require_science_field_coordinates(simulation: AnalysisSimulation) -> tuple[np.ndarray, np.ndarray]:
    setup = simulation.config["setup"]
    r_arcsec = _require_1d_array(setup, schema.KEY_SETUP_SCI_R_ARCSEC)
    theta_deg = _require_1d_array(setup, schema.KEY_SETUP_SCI_THETA_DEG)
    if r_arcsec.shape != theta_deg.shape:
        raise ValueError("Science coordinate arrays must have matching shapes.")
    return r_arcsec, theta_deg


def _require_metric_field_values(
    simulation: AnalysisSimulation,
    metric_name: str,
    value_index: int,
    *,
    num_science_points: int,
) -> np.ndarray:
    metric_values = _select_metric_values(simulation, metric_name, value_index)
    if metric_values.shape[0] != num_science_points:
        raise ValueError(
            f"Metric '{metric_name}' length {metric_values.shape[0]} does not match "
            f"science coordinate length {num_science_points}."
        )
    return metric_values


def plot_psf(
    simulation: AnalysisSimulation,
    psf_index: int = 0,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    log10: bool = True,
    cmap: str = "hot",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    colorbar_label: str | None = None,
    add_colorbar: bool = True,
) -> Figure:
    """Plot one science PSF from a loaded analysis simulation.

    The plot reads ``simulation.psfs[psf_index]`` and uses the persisted
    native pixel scale from ``simulation.meta["pixel_scale_mas"]`` to label
    both axes in milliarcseconds. By default the displayed image is
    ``log10`` intensity clipped at a small positive floor. The helper returns
    the owning, unshown Matplotlib figure.

    Args:
        simulation: Loaded analysis simulation with a 3D PSF cube and
            ``pixel_scale_mas`` metadata.
        psf_index: Zero-based PSF index within ``simulation.psfs``.
        ax: Optional target axes. When omitted, a new figure and axes are
            created.
        title: Optional axes title. When omitted, a one-based PSF-index title
            is used.
        log10: Whether to display clipped ``log10`` intensity instead of
            native intensity.
        cmap: Matplotlib colormap name passed to ``imshow``. The default
            matches the legacy PSF plot.
        vmin: Optional lower color-limit passed to ``imshow``.
        vmax: Optional upper color-limit passed to ``imshow``.
        norm: Optional Matplotlib normalization passed to ``imshow``. Do not
            provide ``vmin`` or ``vmax`` with ``norm``.
        colorbar_label: Optional colorbar label. When omitted, a legacy-style
            intensity label is used.
        add_colorbar: Whether to add a colorbar to the returned figure.

    Returns:
        The Matplotlib figure that owns the plotted axes.

    Raises:
        TypeError: If ``psf_index`` is not an integer.
        ValueError: If PSFs are unavailable, the PSF cube has the wrong shape,
            ``psf_index`` is out of range, or ``pixel_scale_mas`` is missing
            or invalid.
    """

    psf = _select_psf(simulation, psf_index)
    pixel_scale_mas = _require_positive_scalar(simulation.meta, schema.KEY_META_PIXEL_SCALE_MAS)
    values = np.log10(np.clip(psf, _MIN_LOG_VALUE, None)) if log10 else psf
    fig, ax = _figure_and_axis(ax)
    im = ax.imshow(
        values,
        extent=_psf_extent(psf, pixel_scale_mas),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )
    ax.set_xlabel(_LABEL_MAS_SKY)
    ax.set_ylabel(_LABEL_MAS_SKY)
    ax.set_title(title if title is not None else f"PSF {psf_index + 1}")
    ax.set_aspect("equal")
    if add_colorbar:
        _add_matched_colorbar(
            fig,
            ax,
            im,
            colorbar_label if colorbar_label is not None else "Log Intensity" if log10 else "Intensity",
        )
    return fig


def plot_psf_core(
    simulation: AnalysisSimulation,
    psf_index: int = 0,
    *,
    size_px: int | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    colorbar_label: str | None = None,
    add_colorbar: bool = True,
) -> Figure:
    """Plot a centered core crop from one science PSF.

    The plot reads ``simulation.psfs[psf_index]`` and displays a centered
    square crop in native pixel coordinates. When ``size_px`` is omitted, the
    crop uses the legacy default of ``6`` pixels for even-width PSFs and ``5``
    pixels for odd-width PSFs. The crop center follows the legacy rounded
    ``(N - 1) / 2`` convention and is clipped at the PSF image bounds. The
    helper returns the owning, unshown Matplotlib figure.

    Args:
        simulation: Loaded analysis simulation with a 3D PSF cube.
        psf_index: Zero-based PSF index within ``simulation.psfs``.
        size_px: Requested square crop width in pixels. When omitted, use the
            legacy parity-based default.
        ax: Optional target axes. When omitted, a new figure and axes are
            created.
        title: Optional axes title. When omitted, a one-based PSF-index core
            title is used.
        cmap: Matplotlib colormap name passed to ``imshow``.
        vmin: Optional lower color-limit passed to ``imshow``.
        vmax: Optional upper color-limit passed to ``imshow``.
        norm: Optional Matplotlib normalization passed to ``imshow``. Do not
            provide ``vmin`` or ``vmax`` with ``norm``.
        colorbar_label: Optional colorbar label. When omitted, ``"Intensity"``
            is used.
        add_colorbar: Whether to add a colorbar to the returned figure.

    Returns:
        The Matplotlib figure that owns the plotted axes.

    Raises:
        TypeError: If ``psf_index`` is not an integer or ``size_px`` is
            provided and is not an integer.
        ValueError: If PSFs are unavailable, the PSF cube has the wrong shape,
            ``psf_index`` is out of range, or ``size_px`` is not positive.
    """

    psf = _select_psf(simulation, psf_index)
    size_px = _default_core_size_px(psf) if size_px is None else _require_positive_int(size_px, "size_px")
    core = _center_crop(psf, size_px)
    fig, ax = _figure_and_axis(ax)
    im = ax.imshow(
        core,
        extent=_core_extent(core),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )
    ax.set_title(title if title is not None else f"PSF {psf_index + 1} Core")
    ax.set_aspect("equal")
    if add_colorbar:
        _add_matched_colorbar(fig, ax, im, colorbar_label if colorbar_label is not None else "Intensity")
    return fig


def plot_metric_field(
    simulation: AnalysisSimulation,
    metric_name: str = schema.KEY_STATS_SR,
    *,
    value_index: int = 0,
    interpolation: str = "rbf",
    grid_size: int = 201,
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    colorbar_label: str | None = None,
    add_colorbar: bool = True,
) -> Figure:
    """Plot one metric over science field coordinates for a single simulation.

    Science coordinates are read from ``setup["sci_r_arcsec"]`` and
    ``setup["sci_theta_deg"]`` and converted to Cartesian arcsecond axes.
    Metric values are read from ``simulation.stats[metric_name]``. For
    multidimensional metric arrays, trailing dimensions are flattened and
    ``value_index`` selects one value per science point. The selected metric is
    interpolated onto a regular field grid using the same generic
    RBF/griddata approach as the legacy field plot. The helper returns the
    owning, unshown Matplotlib figure.

    Args:
        simulation: Loaded analysis simulation with science field coordinates
            in ``config["setup"]`` and metric values in ``stats``.
        metric_name: Name of the metric in ``simulation.stats``.
        value_index: Zero-based flattened trailing metric index for
            multidimensional metric arrays.
        interpolation: Interpolation method for the regular field grid.
            ``"rbf"`` uses cubic SciPy ``Rbf`` interpolation. ``"nearest"``
            uses SciPy ``griddata(..., method="nearest")``.
        grid_size: Number of samples along each axis in the regular field
            grid.
        ax: Optional target axes. When omitted, a new figure and axes are
            created.
        title: Optional axes title. When omitted, ``metric_name`` is used.
        cmap: Optional Matplotlib colormap name passed to ``imshow``. When
            omitted, a legacy-aligned metric-dependent default is used.
        vmin: Optional lower color-limit passed to ``imshow``.
        vmax: Optional upper color-limit passed to ``imshow``.
        norm: Optional Matplotlib normalization passed to ``imshow``. Do not
            provide ``vmin`` or ``vmax`` with ``norm``.
        colorbar_label: Optional colorbar label. When omitted, a
            metric-dependent default label is used.
        add_colorbar: Whether to add a colorbar to the returned figure.

    Returns:
        The Matplotlib figure that owns the plotted axes.

    Raises:
        TypeError: If ``value_index`` is not an integer for a multidimensional
            metric, or ``grid_size`` is not an integer.
        ValueError: If science coordinates are missing or malformed,
            ``metric_name`` is unavailable, ``value_index`` is out of range,
            ``interpolation`` is unsupported, the field radius is invalid,
            fewer than three science points are available, or coordinate and
            metric lengths do not match.
    """

    interpolation = _require_metric_interpolation(interpolation)
    r_arcsec, theta_deg = _require_science_field_coordinates(simulation)
    metric_values = _require_metric_field_values(
        simulation,
        metric_name,
        value_index,
        num_science_points=r_arcsec.shape[0],
    )
    theta_rad = np.deg2rad(theta_deg)
    x_arcsec = r_arcsec * np.cos(theta_rad)
    y_arcsec = r_arcsec * np.sin(theta_rad)
    radius = _field_radius(r_arcsec)
    xi, yi, x_grid, y_grid, field_mask = _field_grid(radius, grid_size)
    values_grid = _interpolate_metric_field(
        x_arcsec,
        y_arcsec,
        metric_values,
        x_grid,
        y_grid,
        interpolation,
    )
    values_grid *= field_mask

    fig, ax = _figure_and_axis(ax)
    im = ax.imshow(
        values_grid,
        extent=(xi[0], xi[-1], yi[0], yi[-1]),
        origin="lower",
        cmap=cmap if cmap is not None else _default_metric_cmap(metric_name),
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )
    ax.set_xlabel(_LABEL_ARCSEC_SKY)
    ax.set_ylabel(_LABEL_ARCSEC_SKY)
    ax.set_title(title if title is not None else metric_name)
    ax.set_aspect("equal")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    if add_colorbar:
        _add_matched_colorbar(
            fig,
            ax,
            im,
            colorbar_label if colorbar_label is not None else _metric_colorbar_label(metric_name),
        )
    return fig


__all__ = [
    "plot_metric_field",
    "plot_psf",
    "plot_psf_core",
]
