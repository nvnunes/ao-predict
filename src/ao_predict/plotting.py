"""Generic plotting helpers for loaded analysis simulations.

This module owns generic plotting for immutable ``AnalysisSimulation``-style
objects. PSF plots use persisted pixel-scale metadata, PSF-core plots use a
centered native-pixel crop, and metric-field plots use persisted science
coordinates with SciPy interpolation.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, TypeAlias

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import Rbf, griddata

from .analysis import AnalysisSimulation
from .simulation import schema
from .simulation.base import BaseSimulation


_MIN_LOG_VALUE = 1.0e-30
_DEFAULT_FIGSIZE = (5.0, 4.0)
_LABEL_ARCSEC_SKY = '"/Sky'
_LABEL_MAS_SKY = "[mas/Sky]"
_DEFAULT_SOURCE_MARKER_SIZE = 200.0
_NGS_MAG_LABEL_FONTSIZE = 8.0
_NGS_MAG_LABEL_OFFSET_POINTS = (5.0, 4.0)
_PANEL_COLORBAR_WIDTH = 0.04
_PANEL_COLORBAR_PAD = 0.06
_SHARED_PANEL_COLORBAR_PAD = 0.045
_PANEL_GRID_WSPACE = 0.06
_COMPARISON_GRID_WSPACE = 0.02
_COMPARISON_AUTO_GRID_WSPACE = 0.02
_COMPARISON_GRID_COMPARISON_SPACER_WIDTH = 0.24
_COMPARISON_AUTO_COMPARISON_SPACER_WIDTH = 0.24
_COMPARISON_TITLE_FONTSIZE = 12.0
_COMPARISON_AXIS_LABEL_FONTSIZE = 9.0
_COMPARISON_TICK_FONTSIZE = 9.0
_COMPARISON_COLORBAR_LABEL_FONTSIZE = 9.0
_COMPARISON_COLORBAR_TICK_FONTSIZE = 9.0
_DIFFERENCE_METRIC_NAME = "difference"
_RELATIVE_PERCENT_METRIC_NAME = "relative_percent"
_METRIC_NAME_LOCATION_TITLE = "title"
_METRIC_NAME_LOCATION_Y_AXIS = "y_axis"
_METRIC_NAME_LOCATION_COLORBAR = "colorbar"
_METRIC_NAME_LOCATION_NONE = "none"
MetricNameLocation: TypeAlias = Literal["title", "y_axis", "colorbar", "none"]
_PlotMapping: TypeAlias = Mapping[str, object]
_MetricInterpolation: TypeAlias = str
_ContourFormat: TypeAlias = str | Callable[[float], str]
MetricFieldPlotter: TypeAlias = Callable[..., Figure]
_METRIC_NAME_ALIASES = {
    "sr": schema.KEY_STATS_SR,
    "strehl": schema.KEY_STATS_SR,
    "strehl_ratio": schema.KEY_STATS_SR,
    "ee": schema.KEY_STATS_EE,
    "ensquared_energy": schema.KEY_STATS_EE,
    "fwhm": schema.KEY_STATS_FWHM,
}


class MetricComparison(StrEnum):
    """Supported comparison fields for :func:`plot_metric_field_comparison`.

    Attributes:
        DIFFERENCE: Plot the direct field difference ``right - left``.
        RELATIVE_PERCENT: Plot ``100 * (right - left) / left`` with percent
            formatting in colorbar ticks.
    """

    DIFFERENCE = "difference"
    RELATIVE_PERCENT = "relative_percent"


@dataclass(frozen=True)
class MetricFieldPanelRow:
    """Caller-owned layout row for :func:`plot_metric_field_panel`.

    Attributes:
        axes: Plot axes used for one metric-field panel row.
        colorbar_ax: Optional axis reserved for the row colorbar.
    """

    axes: tuple[Axes, ...]
    colorbar_ax: Axes | None = None


@dataclass(frozen=True)
class MetricFieldGrid:
    """Prepared metric-field panel layout.

    Attributes:
        figure: Matplotlib figure containing all panel rows.
        rows: Prepared rows that can be passed to
            :func:`plot_metric_field_panel`.
    """

    figure: Figure
    rows: tuple[MetricFieldPanelRow, ...]

    def __getitem__(self, index: int) -> MetricFieldPanelRow:
        """Return a prepared row by zero-based index."""

        return self.rows[index]


@dataclass(frozen=True)
class MetricFieldComparisonRow:
    """Caller-owned layout row for :func:`plot_metric_field_comparison`.

    Attributes:
        axes: The three plot axes used for left, right, and comparison fields.
        metric_colorbar_ax: Optional axis reserved for the shared metric
            colorbar.
        comparison_colorbar_ax: Optional axis reserved for the comparison
            colorbar.
    """

    axes: tuple[Axes, Axes, Axes]
    metric_colorbar_ax: Axes | None = None
    comparison_colorbar_ax: Axes | None = None


@dataclass(frozen=True)
class MetricFieldComparisonGrid:
    """Prepared metric-comparison panel layout.

    Attributes:
        figure: Matplotlib figure containing all comparison rows.
        rows: Prepared rows that can be passed to
            :func:`plot_metric_field_comparison`.
    """

    figure: Figure
    rows: tuple[MetricFieldComparisonRow, ...]

    def __getitem__(self, index: int) -> MetricFieldComparisonRow:
        """Return a prepared row by zero-based index."""

        return self.rows[index]


def prepare_metric_field_grid(
    nrows: int,
    ncols: int,
    *,
    figure_size: tuple[float, float] | None = None,
    add_colorbar: bool = True,
    colorbar_width: float = _PANEL_COLORBAR_WIDTH,
    left: float = 0.055,
    right: float = 0.955,
    bottom: float = 0.08,
    top: float = 0.93,
    wspace: float = _PANEL_GRID_WSPACE,
    hspace: float = 0.18,
) -> MetricFieldGrid:
    """Prepare a reusable metric-field panel grid.

    Args:
        nrows: Number of panel rows to allocate.
        ncols: Number of metric-field plot axes per row.
        figure_size: Optional Matplotlib figure size in inches. When omitted,
            a compact default is derived from ``nrows`` and ``ncols``.
        add_colorbar: Whether each row reserves a colorbar axis.
        colorbar_width: Grid width ratio for the optional colorbar column.
        left: Left margin passed to ``Figure.add_gridspec``.
        right: Right margin passed to ``Figure.add_gridspec``.
        bottom: Bottom margin passed to ``Figure.add_gridspec``.
        top: Top margin passed to ``Figure.add_gridspec``.
        wspace: Column spacing passed to ``Figure.add_gridspec``.
        hspace: Row spacing passed to ``Figure.add_gridspec``.

    Returns:
        A prepared figure plus one row object per requested row.

    Raises:
        ValueError: If row, column, or colorbar dimensions are invalid.
    """

    if nrows <= 0:
        raise ValueError("nrows must be positive.")
    if ncols <= 0:
        raise ValueError("ncols must be positive.")
    if colorbar_width <= 0.0:
        raise ValueError("colorbar_width must be positive.")
    if figure_size is None:
        figure_size = (3.25 * ncols + (0.45 if add_colorbar else 0.0), 3.6 * nrows)

    fig = plt.figure(figsize=figure_size)
    grid_ncols = ncols + 1 if add_colorbar else ncols
    width_ratios = [1.0] * ncols + ([colorbar_width] if add_colorbar else [])
    grid = fig.add_gridspec(
        nrows,
        grid_ncols,
        width_ratios=width_ratios,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )
    rows = []
    for row in range(nrows):
        axes = tuple(fig.add_subplot(grid[row, col]) for col in range(ncols))
        colorbar_ax = fig.add_subplot(grid[row, ncols]) if add_colorbar else None
        rows.append(MetricFieldPanelRow(axes=axes, colorbar_ax=colorbar_ax))
    return MetricFieldGrid(figure=fig, rows=tuple(rows))


def prepare_metric_field_comparison_grid(
    nrows: int,
    *,
    figure_size: tuple[float, float] | None = None,
    comparison_spacer_width: float = _COMPARISON_GRID_COMPARISON_SPACER_WIDTH,
    left: float = 0.055,
    right: float = 0.955,
    bottom: float = 0.08,
    top: float = 0.93,
    wspace: float = _COMPARISON_GRID_WSPACE,
    hspace: float = 0.18,
) -> MetricFieldComparisonGrid:
    """Prepare a reusable metric-field comparison grid.

    Each row contains three plot axes for left, right, and comparison fields.
    When colorbars are enabled, the row also reserves one shared metric
    colorbar axis and one comparison colorbar axis.

    Args:
        nrows: Number of comparison rows to allocate.
        figure_size: Optional Matplotlib figure size in inches. When omitted,
            a compact default is derived from ``nrows``.
        comparison_spacer_width: Width ratio for the spacer between the right
            metric field and the comparison field.
        left: Left margin passed to ``Figure.add_gridspec``.
        right: Right margin passed to ``Figure.add_gridspec``.
        bottom: Bottom margin passed to ``Figure.add_gridspec``.
        top: Top margin passed to ``Figure.add_gridspec``.
        wspace: Column spacing passed to ``Figure.add_gridspec``.
        hspace: Row spacing passed to ``Figure.add_gridspec``.

    Returns:
        A prepared figure plus one comparison row object per requested row.

    Raises:
        ValueError: If ``nrows`` is not positive.
    """

    if nrows <= 0:
        raise ValueError("nrows must be positive.")
    if comparison_spacer_width <= 0.0:
        raise ValueError("comparison_spacer_width must be positive.")
    if figure_size is None:
        figure_size = (12.0, 4.0 * nrows)

    fig = plt.figure(figsize=figure_size, constrained_layout=False)
    grid = fig.add_gridspec(
        nrows,
        4,
        width_ratios=[1.0, 1.0, comparison_spacer_width, 1.0],
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )

    rows = []
    for row in range(nrows):
        axes = (
            fig.add_subplot(grid[row, 0]),
            fig.add_subplot(grid[row, 1]),
            fig.add_subplot(grid[row, 3]),
        )
        metric_colorbar_ax = None
        comparison_colorbar_ax = None
        axes[1].sharex(axes[0])
        axes[1].sharey(axes[0])
        axes[2].sharex(axes[0])
        axes[2].sharey(axes[0])
        rows.append(
            MetricFieldComparisonRow(
                axes=axes,
                metric_colorbar_ax=metric_colorbar_ax,
                comparison_colorbar_ax=comparison_colorbar_ax,
            )
        )
    return MetricFieldComparisonGrid(figure=fig, rows=tuple(rows))


def _metric_lookup_name(metric_name: str) -> str:
    return metric_name.lower().replace(" ", "_").replace("-", "_")


def resolve_metric_name(
    metric_name: str,
    *,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """Resolve a display metric name to a persisted stats key.

    AO Predict owns aliases for its core stats: SR, EE, and FWHM. Callers may
    pass additional aliases for downstream extra stats. Unknown names are
    returned unchanged so downstream persisted stats remain usable without
    registering aliases upstream.

    Args:
        metric_name: User-facing metric name or persisted stats key.
        aliases: Optional extra alias map. Keys are matched case-insensitively
            after stripping whitespace and replacing spaces and hyphens with
            underscores.

    Returns:
        The resolved metric key for ``simulation.stats`` lookup.

    Raises:
        ValueError: If ``metric_name`` is empty.
    """

    metric_name = str(metric_name).strip()
    if not metric_name:
        raise ValueError("metric_name must be a non-empty string.")
    lookup_name = _metric_lookup_name(metric_name)
    if aliases is not None:
        normalized_aliases = {_metric_lookup_name(alias): value for alias, value in aliases.items()}
        if lookup_name in normalized_aliases:
            return normalized_aliases[lookup_name]
    return _METRIC_NAME_ALIASES.get(lookup_name, metric_name)


def _polar_to_xy(r: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta_rad = np.deg2rad(theta)
    return r * np.cos(theta_rad), r * np.sin(theta_rad)


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


def _require_metric_name_location(value: str) -> MetricNameLocation:
    if value in {
        _METRIC_NAME_LOCATION_TITLE,
        _METRIC_NAME_LOCATION_Y_AXIS,
        _METRIC_NAME_LOCATION_COLORBAR,
        _METRIC_NAME_LOCATION_NONE,
    }:
        return value  # type: ignore[return-value]
    raise ValueError("metric_name_location must be 'title', 'y_axis', 'colorbar', or 'none'.")


def _require_metric_comparison(comparison: MetricComparison | str) -> MetricComparison:
    try:
        return MetricComparison(comparison)
    except ValueError as exc:
        allowed = "', '".join(item.value for item in MetricComparison)
        raise ValueError(f"comparison must be one of '{allowed}'.") from exc


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


def _require_source_coordinate_arrays(
    mapping: _PlotMapping,
    r_key: str,
    theta_key: str,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    r = _require_1d_array(mapping, r_key)
    theta = _require_1d_array(mapping, theta_key)
    if r.shape != theta.shape:
        raise ValueError(f"{label} coordinate arrays must have matching shapes.")
    finite = np.isfinite(r) & np.isfinite(theta)
    return r[finite], theta[finite]


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


def _require_matching_science_coordinates(
    left: AnalysisSimulation,
    right: AnalysisSimulation,
) -> None:
    left_setup = left.config["setup"]
    right_setup = right.config["setup"]
    for key in (schema.KEY_SETUP_SCI_R, schema.KEY_SETUP_SCI_THETA):
        left_values = np.asarray(left_setup[key], dtype=float)
        right_values = np.asarray(right_setup[key], dtype=float)
        if left_values.shape != right_values.shape or not np.allclose(left_values, right_values):
            raise ValueError("Metric field comparisons require matching science coordinates.")


def _with_metric_values(
    simulation: AnalysisSimulation,
    metric_name: str,
    metric_values: np.ndarray,
) -> AnalysisSimulation:
    return AnalysisSimulation(
        _config={
            "setup": dict(simulation.config["setup"]),
            "options": dict(simulation.config["options"]),
        },
        _meta=dict(simulation.meta),
        _stats={metric_name: np.asarray(metric_values, dtype=np.float32)},
    )


def _finite_metric_values(values: Sequence[np.ndarray]) -> np.ndarray:
    finite_values = [value[np.isfinite(value)] for value in values]
    if not finite_values:
        return np.array([], dtype=float)
    return np.concatenate(finite_values)


def _shared_metric_range(
    simulations: Sequence[AnalysisSimulation],
    metric_name: str,
    value_index: int,
    *,
    vmin: float | None,
    vmax: float | None,
    norm: Normalize | None,
) -> tuple[float | None, float | None]:
    if norm is not None or vmin is not None or vmax is not None:
        return vmin, vmax
    values = [_select_metric_values(simulation, metric_name, value_index) for simulation in simulations]
    finite = _finite_metric_values(values)
    if finite.size:
        return float(np.min(finite)), float(np.max(finite))
    return vmin, vmax


def _psf_extent(psf: np.ndarray, pixel_scale: float) -> tuple[float, float, float, float]:
    ny, nx = psf.shape
    return (
        -0.5 * nx * pixel_scale,
        0.5 * nx * pixel_scale,
        -0.5 * ny * pixel_scale,
        0.5 * ny * pixel_scale,
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
    if metric_name == schema.KEY_STATS_FWHM:
        return "plasma_r"
    return "plasma"


def _metric_display_name(metric_name: str) -> str:
    if metric_name == schema.KEY_STATS_SR:
        return "SR"
    if metric_name == schema.KEY_STATS_EE:
        return "EE"
    if metric_name == schema.KEY_STATS_FWHM:
        return "FWHM"
    return metric_name


def _metric_display_title(
    simulation: AnalysisSimulation,
    metric_name: str,
    value_index: int,
    metric_label: str | None = None,
) -> str:
    if metric_label is not None:
        return metric_label
    if metric_name != schema.KEY_STATS_EE:
        return _metric_display_name(metric_name)

    apertures = np.asarray(
        simulation.config["setup"].get(schema.KEY_SETUP_EE_APERTURES, []),
        dtype=float,
    ).reshape(-1)
    if value_index < apertures.size and np.isfinite(apertures[value_index]):
        return f"EE{apertures[value_index]:g}"
    return _metric_display_name(metric_name)


def _set_metric_y_axis_label(ax: Axes, metric_label: str) -> None:
    coordinate_label = ax.get_ylabel()
    ax.set_ylabel(f"{metric_label}\n{coordinate_label}" if coordinate_label else metric_label)


def _metric_colorbar_label(metric_name: str) -> str:
    if metric_name == schema.KEY_STATS_SR:
        return "Strehl Ratio"
    if metric_name == schema.KEY_STATS_EE:
        return "EE"
    if metric_name == schema.KEY_STATS_FWHM:
        return "FWHM [mas]"
    return metric_name


def _metric_colorbar_unit_label(metric_name: str) -> str:
    if metric_name == schema.KEY_STATS_FWHM:
        return "[mas]"
    return ""


def _align_colorbar_axis_to_plot(
    cax: Axes,
    ax: Axes,
    *,
    pad: float = _PANEL_COLORBAR_PAD,
) -> None:
    ax.figure.canvas.draw()
    plot_position = ax.get_position()
    colorbar_position = cax.get_position()
    cax.set_position(
        [
            plot_position.x1 + pad * plot_position.width,
            plot_position.y0,
            colorbar_position.width,
            plot_position.height,
        ]
    )


def _add_panel_colorbar(
    ax: Axes,
    *,
    cax: Axes | None,
    label: str,
    pad: float,
    tick_suffix: str = "",
    tick_precision: int | None = None,
) -> None:
    if not ax.images:
        return
    if cax is None:
        ax.figure.canvas.draw()
        plot_position = ax.get_position()
        cax = ax.figure.add_axes(
            [
                plot_position.x1 + pad * plot_position.width,
                plot_position.y0,
                0.04 * plot_position.width,
                plot_position.height,
            ]
        )
        cbar = ax.figure.colorbar(ax.images[0], cax=cax)
    else:
        _align_colorbar_axis_to_plot(cax, ax, pad=pad)
        cbar = ax.figure.colorbar(ax.images[0], cax=cax)
    cbar.set_label(label, fontsize=_COMPARISON_COLORBAR_LABEL_FONTSIZE)
    if tick_suffix:
        if tick_precision is None:
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}{tick_suffix}"))
        else:
            cbar.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:.{tick_precision}f}{tick_suffix}")
            )
    cbar.ax.tick_params(labelsize=_COMPARISON_COLORBAR_TICK_FONTSIZE)
    plt.sca(ax)


def _add_title_label(ax: Axes, label: str) -> None:
    title = ax.get_title()
    if not title:
        ax.set_title(label)
        return
    ax.set_title(f"{label}: {title}")


def _set_panel_title_size(axes: Sequence[Axes]) -> None:
    for axis in axes:
        axis.title.set_fontsize(_COMPARISON_TITLE_FONTSIZE)


def _set_panel_axis_font_size(axes: Sequence[Axes]) -> None:
    for axis in axes:
        axis.xaxis.label.set_fontsize(_COMPARISON_AXIS_LABEL_FONTSIZE)
        axis.yaxis.label.set_fontsize(_COMPARISON_AXIS_LABEL_FONTSIZE)
        axis.tick_params(labelsize=_COMPARISON_TICK_FONTSIZE)


def _format_contour_label(value: float) -> str:
    text = f"{value:.2f}"
    if text.endswith("0"):
        text = f"{value:.1f}"
    return text


def _field_radius(r: np.ndarray) -> float:
    if r.size == 0:
        raise ValueError("Science coordinate arrays must contain at least one point.")
    radius = float(np.nanmax(r))
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
    x: np.ndarray,
    y: np.ndarray,
    metric_values: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    interpolation: _MetricInterpolation,
) -> np.ndarray:
    if x.size < 3:
        raise ValueError("Metric field interpolation requires at least three science points.")
    if interpolation == "nearest":
        return griddata((x, y), metric_values, (x_grid, y_grid), method="nearest")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rbf = Rbf(x, y, metric_values, function="cubic")
        return rbf(x_grid, y_grid)


def _add_metric_contours(
    ax: Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values_grid: np.ndarray,
    *,
    contour_levels: int | np.ndarray | None,
    contour_color: str,
    contour_linewidths: float,
    contour_labels: bool,
    contour_fmt: _ContourFormat,
) -> None:
    contour_set = ax.contour(
        x_grid,
        y_grid,
        values_grid,
        levels=contour_levels,
        colors=contour_color,
        linewidths=contour_linewidths,
    )
    if contour_labels:
        ax.clabel(contour_set, inline=True, fontsize=8, fmt=contour_fmt)


def _add_source_markers(
    ax: Axes,
    r: np.ndarray,
    theta: np.ndarray,
    *,
    marker: str | MarkerStyle,
    color: str,
    edgecolor: str,
    size: float,
    linewidth: float,
    label: str,
) -> None:
    if r.size == 0:
        return
    x, y = _polar_to_xy(r, theta)
    ax.scatter(
        x,
        y,
        marker=marker,
        s=size,
        facecolors=color,
        edgecolors=edgecolor,
        linewidths=linewidth,
        label=label,
        zorder=5,
    )


def _add_ngs_magnitude_labels(
    ax: Axes,
    r: np.ndarray,
    theta: np.ndarray,
    magnitudes: np.ndarray,
) -> None:
    if r.size == 0:
        return
    x, y = _polar_to_xy(r, theta)
    for x_value, y_value, magnitude in zip(x, y, magnitudes):
        ax.annotate(
            f"{magnitude:.1f}",
            (x_value, y_value),
            xytext=_NGS_MAG_LABEL_OFFSET_POINTS,
            textcoords="offset points",
            fontsize=_NGS_MAG_LABEL_FONTSIZE,
            ha="left",
            va="bottom",
            color="black",
            zorder=6,
        )


def _require_ngs_marker_arrays(
    options: _PlotMapping,
    *,
    include_magnitudes: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    ngs_r = _require_1d_array(options, schema.KEY_OPTION_NGS_R)
    ngs_theta = _require_1d_array(options, schema.KEY_OPTION_NGS_THETA)
    if ngs_r.shape != ngs_theta.shape:
        raise ValueError("NGS coordinate arrays must have matching shapes.")
    finite = np.isfinite(ngs_r) & np.isfinite(ngs_theta)
    ngs_magnitude = None
    if include_magnitudes:
        ngs_magnitude = _require_1d_array(options, schema.KEY_OPTION_NGS_MAGNITUDE)
        if ngs_magnitude.shape != ngs_r.shape:
            raise ValueError("NGS coordinate and magnitude arrays must have matching shapes.")
        finite &= np.isfinite(ngs_magnitude)
        ngs_magnitude = ngs_magnitude[finite]
    return ngs_r[finite], ngs_theta[finite], ngs_magnitude


def _require_science_field_coordinates(simulation: AnalysisSimulation) -> tuple[np.ndarray, np.ndarray]:
    setup = simulation.config["setup"]
    r = _require_1d_array(setup, schema.KEY_SETUP_SCI_R)
    theta = _require_1d_array(setup, schema.KEY_SETUP_SCI_THETA)
    if r.shape != theta.shape:
        raise ValueError("Science coordinate arrays must have matching shapes.")
    return r, theta


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
    native pixel scale from ``simulation.meta["pixel_scale"]`` to label
    both axes in milliarcseconds. By default the displayed image is
    ``log10`` intensity clipped at a small positive floor. The helper returns
    the owning, unshown Matplotlib figure.

    Args:
        simulation: Loaded analysis simulation with a 3D PSF cube and
            ``pixel_scale`` metadata.
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
            ``psf_index`` is out of range, or ``pixel_scale`` is missing
            or invalid.
    """

    psf = _select_psf(simulation, psf_index)
    pixel_scale = _require_positive_scalar(simulation.meta, schema.KEY_META_PIXEL_SCALE)
    values = np.log10(np.clip(psf, _MIN_LOG_VALUE, None)) if log10 else psf
    fig, ax = _figure_and_axis(ax)
    im = ax.imshow(
        values,
        extent=_psf_extent(psf, pixel_scale),
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
    metric_name_location: MetricNameLocation = _METRIC_NAME_LOCATION_TITLE,
    metric_label: str | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    colorbar_label: str | None = None,
    add_colorbar: bool = True,
    show_contours: bool = False,
    contour_levels: int | np.ndarray | None = None,
    contour_color: str = "black",
    contour_linewidths: float = 0.5,
    contour_labels: bool = True,
    contour_fmt: _ContourFormat = _format_contour_label,
    mask_contours: bool = True,
    show_ngs: bool = False,
    show_ngs_mags: bool = False,
    show_lgs: bool = False,
    ngs_marker: str | MarkerStyle = (5, 1),
    lgs_marker: str | MarkerStyle = (5, 1),
    ngs_color: str = "red",
    lgs_color: str = "yellow",
    ngs_marker_size: float = _DEFAULT_SOURCE_MARKER_SIZE,
    lgs_marker_size: float = _DEFAULT_SOURCE_MARKER_SIZE,
    source_marker_size: float | None = None,
    source_marker_edgecolor: str = "black",
    source_marker_linewidth: float = 0.5,
) -> Figure:
    """Plot one metric over science field coordinates for a single simulation.

    Science coordinates are read from ``setup["sci_r"]`` and
    ``setup["sci_theta"]`` and converted to Cartesian arcsecond axes.
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
        title: Optional axes title. When omitted, a metric-specific display
            label is used when ``metric_name_location="title"``.
        metric_name_location: Where to place the metric display label when no
            explicit ``title`` is provided. ``"title"`` places it in the axes
            title, ``"y_axis"`` prefixes the y-axis label, ``"colorbar"``
            uses the colorbar label, and ``"none"`` omits it.
        metric_label: Optional display label for the metric. When omitted,
            AO Predict uses its default display label, including EE aperture
            suffixes when configured.
        cmap: Optional Matplotlib colormap name passed to ``imshow``. When
            omitted, a legacy-aligned metric-dependent default is used.
        vmin: Optional lower color-limit passed to ``imshow``.
        vmax: Optional upper color-limit passed to ``imshow``.
        norm: Optional Matplotlib normalization passed to ``imshow``. Do not
            provide ``vmin`` or ``vmax`` with ``norm``.
        colorbar_label: Optional colorbar label. When omitted, a
            metric-dependent default label is used.
        add_colorbar: Whether to add a colorbar to the returned figure.
        show_contours: Whether to draw contour lines over the interpolated
            metric field.
        contour_levels: Optional Matplotlib contour levels. When omitted,
            Matplotlib chooses levels automatically.
        contour_color: Contour line color.
        contour_linewidths: Contour line width.
        contour_labels: Whether to label contour lines.
        contour_fmt: Matplotlib contour-label format string or callable.
        mask_contours: Whether contours should use the same circular field
            mask as the rendered image.
        show_ngs: Whether to draw NGS markers from persisted option
            coordinates. Missing NGS coordinate fields raise ``ValueError``
            only when this is enabled.
        show_ngs_mags: Whether to label NGS markers with persisted NGS
            magnitudes. This implies NGS marker plotting and raises
            ``ValueError`` when NGS magnitude fields are missing or malformed.
        show_lgs: Whether to draw LGS markers from persisted setup
            coordinates. Missing LGS coordinate fields raise ``ValueError``
            only when this is enabled.
        ngs_marker: Matplotlib marker for NGS positions.
        lgs_marker: Matplotlib marker for LGS positions.
        ngs_color: Marker face color for NGS positions.
        lgs_color: Marker face color for LGS positions.
        ngs_marker_size: Marker size for NGS positions.
        lgs_marker_size: Marker size for LGS positions.
        source_marker_size: Optional shared marker size for both NGS and LGS positions.
        source_marker_edgecolor: Marker edge color for NGS and LGS positions.
        source_marker_linewidth: Marker edge linewidth for NGS and LGS positions.

    Returns:
        The Matplotlib figure that owns the plotted axes.

    Raises:
        TypeError: If ``value_index`` is not an integer for a multidimensional
            metric, or ``grid_size`` is not an integer.
        ValueError: If science coordinates are missing or malformed,
            ``metric_name`` is unavailable, ``value_index`` is out of range,
            ``interpolation`` is unsupported, the field radius is invalid,
            fewer than three science points are available, coordinate and
            metric lengths do not match, or explicitly requested NGS/LGS marker
            coordinates or NGS magnitude labels are unavailable or malformed.
    """

    metric_name = resolve_metric_name(metric_name)
    metric_name_location = _require_metric_name_location(metric_name_location)
    interpolation = _require_metric_interpolation(interpolation)
    r, theta = _require_science_field_coordinates(simulation)
    metric_values = _require_metric_field_values(
        simulation,
        metric_name,
        value_index,
        num_science_points=r.shape[0],
    )
    x, y = _polar_to_xy(r, theta)
    radius = _field_radius(r)
    xi, yi, x_grid, y_grid, field_mask = _field_grid(radius, grid_size)
    values_grid = _interpolate_metric_field(
        x,
        y,
        metric_values,
        x_grid,
        y_grid,
        interpolation,
    )
    image_grid = values_grid * field_mask

    fig, ax = _figure_and_axis(ax)
    im = ax.imshow(
        image_grid,
        extent=(xi[0], xi[-1], yi[0], yi[-1]),
        origin="lower",
        cmap=cmap if cmap is not None else _default_metric_cmap(metric_name),
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )
    ax.set_xlabel(_LABEL_ARCSEC_SKY)
    ax.set_ylabel(_LABEL_ARCSEC_SKY)
    display_label = _metric_display_title(simulation, metric_name, value_index, metric_label)
    if title is not None:
        plot_title = title
    elif metric_name_location == _METRIC_NAME_LOCATION_TITLE:
        plot_title = display_label
    else:
        plot_title = ""
    ax.set_title(plot_title)
    ax.set_aspect("equal")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    if metric_name_location == _METRIC_NAME_LOCATION_Y_AXIS:
        _set_metric_y_axis_label(ax, display_label)
    if show_contours:
        _add_metric_contours(
            ax,
            x_grid,
            y_grid,
            values_grid * field_mask if mask_contours else values_grid,
            contour_levels=contour_levels,
            contour_color=contour_color,
            contour_linewidths=contour_linewidths,
            contour_labels=contour_labels,
            contour_fmt=contour_fmt,
        )
    if show_lgs:
        setup = simulation.config["setup"]
        lgs_r, lgs_theta = _require_source_coordinate_arrays(
            setup,
            BaseSimulation.KEY_SETUP_LGS_R,
            BaseSimulation.KEY_SETUP_LGS_THETA,
            label="LGS",
        )
        _add_source_markers(
            ax,
            lgs_r,
            lgs_theta,
            marker=lgs_marker,
            color=lgs_color,
            edgecolor=source_marker_edgecolor,
            size=lgs_marker_size if source_marker_size is None else source_marker_size,
            linewidth=source_marker_linewidth,
            label="LGS",
        )
    if show_ngs or show_ngs_mags:
        options = simulation.config["options"]
        ngs_r, ngs_theta, ngs_magnitude = _require_ngs_marker_arrays(
            options,
            include_magnitudes=show_ngs_mags,
        )
        _add_source_markers(
            ax,
            ngs_r,
            ngs_theta,
            marker=ngs_marker,
            color=ngs_color,
            edgecolor=source_marker_edgecolor,
            size=ngs_marker_size if source_marker_size is None else source_marker_size,
            linewidth=source_marker_linewidth,
            label="NGS",
        )
        if ngs_magnitude is not None:
            _add_ngs_magnitude_labels(ax, ngs_r, ngs_theta, ngs_magnitude)
    if add_colorbar:
        if colorbar_label is None and metric_name_location == _METRIC_NAME_LOCATION_COLORBAR:
            colorbar_label = display_label
        _add_matched_colorbar(
            fig,
            ax,
            im,
            colorbar_label if colorbar_label is not None else "",
        )
    return fig


def plot_metric_field_comparison(
    left: AnalysisSimulation,
    right: AnalysisSimulation,
    metric_name: str = schema.KEY_STATS_SR,
    *,
    value_index: int = 0,
    labels: tuple[str, str] = ("Left", "Right"),
    comparison: MetricComparison | str = MetricComparison.DIFFERENCE,
    metric_name_location: MetricNameLocation = _METRIC_NAME_LOCATION_TITLE,
    metric_label: str | None = None,
    field_plotter: MetricFieldPlotter | None = None,
    field_plotter_kwargs: Mapping[str, Any] | None = None,
    ax: Sequence[Axes] | None = None,
    panel: MetricFieldComparisonRow | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    comparison_vmin: float | None = None,
    comparison_vmax: float | None = None,
    comparison_norm: Normalize | None = None,
    comparison_cmap: str = "coolwarm",
    add_colorbar: bool = True,
    **plot_kwargs: Any,
) -> Figure:
    """Plot two metric fields and a generic difference-style comparison.

    The compared simulations must have matching science coordinates. The left
    and right panels share one value range and one metric colorbar. The third
    panel displays either ``right - left`` or ``100 * (right - left) / left``.
    ``field_plotter`` defaults to :func:`plot_metric_field`; downstream
    packages can pass a wrapper plus ``field_plotter_kwargs`` to apply their
    own field-presentation policy without reimplementing composition.

    Args:
        left: Left-hand analysis simulation.
        right: Right-hand analysis simulation.
        metric_name: Metric key or supported alias to compare.
        value_index: Trailing metric column to select for multi-value metrics.
        labels: Labels shown above the left and right field panels.
        comparison: Difference field to compute.
        metric_name_location: Where to display the metric name.
        metric_label: Optional display label replacing the default metric name.
        field_plotter: Optional metric-field plotting callable. When omitted,
            :func:`plot_metric_field` is used.
        field_plotter_kwargs: Extra keyword arguments passed to each
            ``field_plotter`` call.
        ax: Optional three-axis sequence for the left, right, and comparison
            fields.
        panel: Optional prepared comparison row from
            :func:`prepare_metric_field_comparison_grid`.
        title: Optional title for the comparison field.
        vmin: Optional shared metric-field lower color limit.
        vmax: Optional shared metric-field upper color limit.
        norm: Optional shared metric-field normalization.
        comparison_vmin: Optional comparison-field lower color limit.
        comparison_vmax: Optional comparison-field upper color limit.
        comparison_norm: Optional comparison-field normalization.
        comparison_cmap: Matplotlib colormap for the comparison field.
        add_colorbar: Whether to add shared metric and comparison colorbars.
        **plot_kwargs: Additional keyword arguments passed to ``field_plotter``.

    Returns:
        The Matplotlib figure containing the comparison.

    Raises:
        ValueError: If coordinates, metric shapes, labels, or axes are invalid.
    """

    metric_name = resolve_metric_name(metric_name)
    metric_name_location = _require_metric_name_location(metric_name_location)
    comparison = _require_metric_comparison(comparison)
    if panel is not None and ax is not None:
        raise ValueError("Pass either panel or ax, not both.")
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two entries.")
    _require_matching_science_coordinates(left, right)
    left_values = _select_metric_values(left, metric_name, value_index)
    right_values = _select_metric_values(right, metric_name, value_index)
    if left_values.shape != right_values.shape:
        raise ValueError("Metric field comparisons require matching metric shapes.")

    vmin, vmax = _shared_metric_range(
        [left, right],
        metric_name,
        value_index,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )

    if comparison is MetricComparison.DIFFERENCE:
        comparison_values = right_values - left_values
        comparison_metric_name = _DIFFERENCE_METRIC_NAME
        comparison_title = "Diff"
        comparison_value_suffix = ""
        comparison_precision = None
    elif comparison is MetricComparison.RELATIVE_PERCENT:
        with np.errstate(divide="ignore", invalid="ignore"):
            comparison_values = 100.0 * (right_values - left_values) / left_values
        comparison_metric_name = _RELATIVE_PERCENT_METRIC_NAME
        comparison_title = "Diff"
        comparison_value_suffix = "%"
        comparison_precision = 1

    if comparison_norm is None and comparison_vmin is None and comparison_vmax is None:
        finite_difference = comparison_values[np.isfinite(comparison_values)]
        diff_limit = float(np.max(np.abs(finite_difference))) if finite_difference.size else 1.0
        if diff_limit <= 0.0:
            diff_limit = 1.0
        comparison_vmin = -diff_limit
        comparison_vmax = diff_limit

    if panel is not None:
        axes = list(panel.axes)
        fig = axes[0].figure
        metric_cbar_ax = panel.metric_colorbar_ax
        comparison_cbar_ax = panel.comparison_colorbar_ax
    elif ax is None:
        grid = prepare_metric_field_comparison_grid(
            1,
            figure_size=(12.0, 4.0),
            comparison_spacer_width=_COMPARISON_AUTO_COMPARISON_SPACER_WIDTH,
            wspace=_COMPARISON_AUTO_GRID_WSPACE,
        )
        fig = grid.figure
        panel = grid[0]
        axes = list(panel.axes)
        metric_cbar_ax = panel.metric_colorbar_ax
        comparison_cbar_ax = panel.comparison_colorbar_ax
    else:
        if len(ax) != 3:
            raise ValueError("ax must contain exactly three axes.")
        axes = list(ax)
        fig = axes[0].figure
        metric_cbar_ax = None
        comparison_cbar_ax = None

    plot_one = plot_metric_field if field_plotter is None else field_plotter
    child_kwargs = dict(plot_kwargs)
    if field_plotter_kwargs is not None:
        child_kwargs.update(field_plotter_kwargs)
    child_metric_name_location = (
        _METRIC_NAME_LOCATION_NONE
        if metric_name_location == _METRIC_NAME_LOCATION_Y_AXIS
        else metric_name_location
    )
    plot_one(
        left,
        metric_name,
        value_index=value_index,
        ax=axes[0],
        title=None,
        metric_name_location=child_metric_name_location,
        metric_label=metric_label,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        add_colorbar=False,
        **child_kwargs,
    )
    _add_title_label(axes[0], labels[0])
    plot_one(
        right,
        metric_name,
        value_index=value_index,
        ax=axes[1],
        title=None,
        metric_name_location=child_metric_name_location,
        metric_label=metric_label,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        add_colorbar=False,
        **child_kwargs,
    )
    _add_title_label(axes[1], labels[1])
    if metric_name_location == _METRIC_NAME_LOCATION_Y_AXIS:
        _set_metric_y_axis_label(
            axes[0],
            _metric_display_title(left, metric_name, value_index, metric_label),
        )
    comparison_simulation = _with_metric_values(left, comparison_metric_name, comparison_values)
    plot_one(
        comparison_simulation,
        comparison_metric_name,
        ax=axes[2],
        title=comparison_title if title is None else title,
        vmin=comparison_vmin,
        vmax=comparison_vmax,
        norm=comparison_norm,
        cmap=comparison_cmap,
        add_colorbar=False,
        **child_kwargs,
    )
    for axis in axes[1:]:
        axis.set_ylabel("")
        axis.tick_params(labelleft=False)
    _set_panel_title_size(axes)
    _set_panel_axis_font_size(axes)
    if add_colorbar:
        metric_colorbar_label = (
            (metric_label if metric_label is not None else _metric_colorbar_label(metric_name))
            if metric_name_location == _METRIC_NAME_LOCATION_COLORBAR
            else ""
        )
        comparison_colorbar_label = (
            metric_colorbar_label
            if (
                metric_name_location == _METRIC_NAME_LOCATION_COLORBAR
                and comparison is not MetricComparison.RELATIVE_PERCENT
            )
            else ""
        )
        _add_panel_colorbar(
            axes[1],
            cax=metric_cbar_ax,
            label=metric_colorbar_label,
            pad=_PANEL_COLORBAR_PAD,
        )
        _add_panel_colorbar(
            axes[2],
            cax=comparison_cbar_ax,
            label=comparison_colorbar_label,
            pad=_PANEL_COLORBAR_PAD,
            tick_suffix=comparison_value_suffix,
            tick_precision=comparison_precision,
        )
    return fig


def plot_metric_field_panel(
    simulations: Sequence[AnalysisSimulation],
    metric_name: str = schema.KEY_STATS_SR,
    *,
    value_index: int = 0,
    labels: Sequence[str] | None = None,
    ncols: int = 3,
    metric_name_location: MetricNameLocation = _METRIC_NAME_LOCATION_COLORBAR,
    metric_label: str | None = None,
    field_plotter: MetricFieldPlotter | None = None,
    field_plotter_kwargs: Mapping[str, Any] | None = None,
    ax: Sequence[Axes] | None = None,
    panel: MetricFieldPanelRow | None = None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    add_colorbar: bool = True,
    **plot_kwargs: Any,
) -> Figure:
    """Plot a panel of metric fields from analysis simulations.

    Panels share one metric value range by default because they use one
    colorbar. ``field_plotter`` defaults to :func:`plot_metric_field`;
    downstream packages can pass a wrapper plus ``field_plotter_kwargs`` to
    apply their own field-presentation policy while reusing AO Predict's panel
    composition.

    Args:
        simulations: Analysis simulations to render.
        metric_name: Metric key or supported alias to plot.
        value_index: Trailing metric column to select for multi-value metrics.
        labels: Optional per-panel labels.
        ncols: Number of plot columns when this function creates the figure.
        metric_name_location: Where to display the metric name.
        metric_label: Optional display label replacing the default metric name.
        field_plotter: Optional metric-field plotting callable. When omitted,
            :func:`plot_metric_field` is used.
        field_plotter_kwargs: Extra keyword arguments passed to each
            ``field_plotter`` call.
        ax: Optional axes matching ``simulations``.
        panel: Optional prepared panel row from
            :func:`prepare_metric_field_grid`.
        title: Optional figure title.
        vmin: Optional shared metric-field lower color limit.
        vmax: Optional shared metric-field upper color limit.
        norm: Optional shared metric-field normalization.
        add_colorbar: Whether to add one shared panel colorbar.
        **plot_kwargs: Additional keyword arguments passed to ``field_plotter``.

    Returns:
        The Matplotlib figure containing the panel.

    Raises:
        ValueError: If inputs, labels, axes, or panel shape are invalid.
    """

    metric_name = resolve_metric_name(metric_name)
    metric_name_location = _require_metric_name_location(metric_name_location)
    if len(simulations) == 0:
        raise ValueError("simulations must contain at least one simulation.")
    if ncols <= 0:
        raise ValueError("ncols must be positive.")
    if labels is not None and len(labels) != len(simulations):
        raise ValueError("labels must match the number of simulations.")
    if panel is not None and ax is not None:
        raise ValueError("Pass either panel or ax, not both.")
    if panel is not None and len(panel.axes) != len(simulations):
        raise ValueError("panel axes must match the number of simulations.")

    vmin, vmax = _shared_metric_range(
        simulations,
        metric_name,
        value_index,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )

    if panel is not None:
        flat_axes = np.asarray(panel.axes, dtype=object)
        fig = flat_axes[0].figure
        panel_cbar_ax = panel.colorbar_ax
    elif ax is None:
        nrows = int(np.ceil(len(simulations) / ncols))
        display_ncols = min(ncols, len(simulations))
        figure_width = 3.25 * display_ncols + (0.45 if add_colorbar else 0.0)
        fig = plt.figure(
            figsize=(figure_width, 3.6 * nrows),
            constrained_layout=False,
        )
        grid_ncols = display_ncols + 1 if add_colorbar else display_ncols
        width_ratios = [1.0] * display_ncols + ([_PANEL_COLORBAR_WIDTH] if add_colorbar else [])
        grid = fig.add_gridspec(
            nrows,
            grid_ncols,
            width_ratios=width_ratios,
            wspace=_PANEL_GRID_WSPACE,
            hspace=0.18,
        )
        flat_axes = np.asarray(
            [
                fig.add_subplot(grid[row, col])
                for row in range(nrows)
                for col in range(display_ncols)
            ],
            dtype=object,
        )
        panel_cbar_ax = fig.add_subplot(grid[:, display_ncols]) if add_colorbar else None
        for axis in flat_axes[len(simulations) :]:
            axis.set_axis_off()
    else:
        if len(ax) != len(simulations):
            raise ValueError("ax must match the number of simulations.")
        flat_axes = np.asarray(ax, dtype=object)
        fig = flat_axes[0].figure
        panel_cbar_ax = None

    plot_one = plot_metric_field if field_plotter is None else field_plotter
    child_kwargs = dict(plot_kwargs)
    if field_plotter_kwargs is not None:
        child_kwargs.update(field_plotter_kwargs)
    active_axes = list(flat_axes[: len(simulations)])
    child_metric_name_location = (
        _METRIC_NAME_LOCATION_NONE
        if metric_name_location == _METRIC_NAME_LOCATION_Y_AXIS
        else metric_name_location
    )
    for index, simulation in enumerate(simulations):
        plot_one(
            simulation,
            metric_name,
            value_index=value_index,
            ax=active_axes[index],
            title=None,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            add_colorbar=False,
            metric_name_location=child_metric_name_location,
            metric_label=metric_label,
            **child_kwargs,
        )
        if labels is not None:
            _add_title_label(active_axes[index], labels[index])
    if metric_name_location == _METRIC_NAME_LOCATION_Y_AXIS:
        _set_metric_y_axis_label(
            active_axes[0],
            _metric_display_title(simulations[0], metric_name, value_index, metric_label),
        )
    for axis in active_axes[1:]:
        axis.set_ylabel("")
        axis.tick_params(labelleft=False)
    _set_panel_title_size(active_axes)
    _set_panel_axis_font_size(active_axes)
    if add_colorbar and active_axes and active_axes[-1].images:
        colorbar_axis = active_axes[-1]
        colorbar_label = (
            (metric_label if metric_label is not None else _metric_colorbar_label(metric_name))
            if metric_name_location == _METRIC_NAME_LOCATION_COLORBAR
            else ""
        )
        if panel_cbar_ax is None:
            _add_panel_colorbar(
                colorbar_axis,
                cax=None,
                label=colorbar_label,
                pad=_SHARED_PANEL_COLORBAR_PAD,
            )
        else:
            _align_colorbar_axis_to_plot(panel_cbar_ax, colorbar_axis, pad=_SHARED_PANEL_COLORBAR_PAD)
            cbar = fig.colorbar(colorbar_axis.images[0], cax=panel_cbar_ax)
            cbar.set_label(colorbar_label, fontsize=_COMPARISON_COLORBAR_LABEL_FONTSIZE)
            cbar.ax.tick_params(labelsize=_COMPARISON_COLORBAR_TICK_FONTSIZE)
    if title is not None:
        fig.suptitle(title)
    return fig


__all__ = [
    "MetricComparison",
    "MetricFieldPlotter",
    "MetricFieldComparisonGrid",
    "MetricFieldComparisonRow",
    "MetricFieldGrid",
    "MetricFieldPanelRow",
    "prepare_metric_field_comparison_grid",
    "prepare_metric_field_grid",
    "plot_metric_field_comparison",
    "plot_metric_field",
    "plot_metric_field_panel",
    "plot_psf",
    "plot_psf_core",
    "resolve_metric_name",
]
