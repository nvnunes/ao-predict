from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure

from ao_predict.analysis import AnalysisSimulation
from ao_predict.plotting import MetricComparison
from ao_predict.plotting import plot_metric_field
from ao_predict.plotting import plot_metric_field_comparison
from ao_predict.plotting import plot_metric_field_panel
from ao_predict.plotting import plot_psf, plot_psf_core
from ao_predict.plotting import prepare_metric_field_comparison_grid
from ao_predict.plotting import prepare_metric_field_grid
from ao_predict.plotting import resolve_metric_name
from ao_predict.simulation import schema
from ao_predict.simulation.base import BaseSimulation


def _make_simulation(
    *,
    psfs: np.ndarray | None = None,
    setup: dict[str, object] | None = None,
    options: dict[str, object] | None = None,
    meta: dict[str, object] | None = None,
    stats: dict[str, object] | None = None,
) -> AnalysisSimulation:
    if psfs is None:
        psfs = np.stack(
            [
                np.arange(25, dtype=np.float32).reshape(5, 5) + 1.0,
                np.full((5, 5), 2.0, dtype=np.float32),
            ]
        )
    if setup is None:
        setup = {
            schema.KEY_SETUP_SCI_R_ARCSEC: np.array([0.0, 5.0, 5.0], dtype=np.float32),
            schema.KEY_SETUP_SCI_THETA_DEG: np.array([0.0, 0.0, 90.0], dtype=np.float32),
        }
    if options is None:
        options = {}
    if meta is None:
        meta = {schema.KEY_META_PIXEL_SCALE_MAS: np.float32(4.0)}
    if stats is None:
        stats = {
            schema.KEY_STATS_SR: np.array([0.1, 0.2, 0.3], dtype=np.float32),
            schema.KEY_STATS_EE: np.array(
                [
                    [0.5, 0.6],
                    [0.6, 0.7],
                    [0.7, 0.8],
                ],
                dtype=np.float32,
            ),
            schema.KEY_STATS_FWHM_MAS: np.array([50.0, 45.0, 40.0], dtype=np.float32),
        }
    return AnalysisSimulation(
        _config={"setup": setup, "options": options},
        _meta=meta,
        _stats=stats,
        _extra_lazy_fields={"psfs": lambda: psfs},
    )


def test_plot_psf_returns_savable_figure() -> None:
    fig = plot_psf(_make_simulation(), psf_index=1)

    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        assert fig.axes[0].images[0].cmap.name == "hot"
        assert fig.axes[0].get_xlabel() == "[mas/Sky]"
        assert fig.axes[0].get_ylabel() == "[mas/Sky]"
        assert fig.axes[0].get_title() == "PSF 2"
        assert _saved_png_size(fig) > 0
    finally:
        plt.close(fig)


def test_plot_psf_applies_color_limits() -> None:
    fig = plot_psf(_make_simulation(), vmin=-2.0, vmax=2.0)

    try:
        assert fig.axes[0].images[0].get_clim() == (-2.0, 2.0)
    finally:
        plt.close(fig)


def test_plot_psf_uses_explicit_title() -> None:
    fig = plot_psf(_make_simulation(), title="Science PSF")

    try:
        assert fig.axes[0].get_title() == "Science PSF"
    finally:
        plt.close(fig)


def test_plot_psf_core_returns_savable_figure() -> None:
    fig = plot_psf_core(_make_simulation(), size_px=3)

    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        assert fig.axes[0].images[0].get_array().shape == (3, 3)
        assert fig.axes[0].images[0].cmap.name == "viridis"
        assert fig.axes[0].get_xlabel() == ""
        assert fig.axes[0].get_ylabel() == ""
        assert fig.axes[0].get_title() == "PSF 1 Core"
        assert _saved_png_size(fig) > 0
    finally:
        plt.close(fig)


def test_plot_psf_core_applies_color_limits() -> None:
    fig = plot_psf_core(_make_simulation(), vmin=8.0, vmax=14.0)

    try:
        assert fig.axes[0].images[0].get_clim() == (8.0, 14.0)
    finally:
        plt.close(fig)


def test_plot_psf_core_uses_explicit_title() -> None:
    fig = plot_psf_core(_make_simulation(), title="PSF Core")

    try:
        assert fig.axes[0].get_title() == "PSF Core"
    finally:
        plt.close(fig)


def test_plot_psf_core_uses_legacy_odd_width_default_crop() -> None:
    psf = np.arange(49, dtype=np.float32).reshape(7, 7)
    fig = plot_psf_core(_make_simulation(psfs=psf[np.newaxis, ...]))

    try:
        plotted = fig.axes[0].images[0].get_array()
        np.testing.assert_array_equal(plotted, psf[1:6, 1:6])
    finally:
        plt.close(fig)


def test_plot_psf_core_uses_legacy_even_width_default_crop() -> None:
    psf = np.arange(64, dtype=np.float32).reshape(8, 8)
    fig = plot_psf_core(_make_simulation(psfs=psf[np.newaxis, ...]))

    try:
        plotted = fig.axes[0].images[0].get_array()
        np.testing.assert_array_equal(plotted, psf[1:7, 1:7])
    finally:
        plt.close(fig)


def test_plot_metric_field_returns_savable_figure() -> None:
    fig = plot_metric_field(_make_simulation(), metric_name=schema.KEY_STATS_SR)

    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        assert len(fig.axes[0].images) == 1
        assert fig.axes[0].images[0].get_array().shape == (201, 201)
        assert fig.axes[0].images[0].cmap.name == "plasma"
        assert fig.axes[0].get_xlabel() == '"/Sky'
        assert fig.axes[0].get_ylabel() == '"/Sky'
        assert _saved_png_size(fig) > 0
    finally:
        plt.close(fig)


def test_resolve_metric_name_supports_core_aliases_and_extension_aliases() -> None:
    assert resolve_metric_name("FWHM") == schema.KEY_STATS_FWHM_MAS
    assert resolve_metric_name("Strehl Ratio") == schema.KEY_STATS_SR
    assert resolve_metric_name("extra_metric") == "extra_metric"
    assert resolve_metric_name("Jitter mas", aliases={"Jitter mas": "jitter"}) == "jitter"


def test_plot_metric_field_resolves_display_metric_alias() -> None:
    fig = plot_metric_field(
        _make_simulation(),
        metric_name="FWHM",
        interpolation="nearest",
        grid_size=21,
    )

    try:
        assert fig.axes[0].get_title() == "FWHM"
        assert fig.axes[0].images[0].get_array().shape == (21, 21)
    finally:
        plt.close(fig)


def test_plot_metric_field_applies_color_limits() -> None:
    fig = plot_metric_field(_make_simulation(), vmin=0.0, vmax=0.5)

    try:
        assert fig.axes[0].images[0].get_clim() == (0.0, 0.5)
    finally:
        plt.close(fig)


def test_plot_metric_field_selects_multidimensional_metric_value() -> None:
    setup = {
        schema.KEY_SETUP_SCI_R_ARCSEC: np.array([0.0, 5.0, 5.0], dtype=np.float32),
        schema.KEY_SETUP_SCI_THETA_DEG: np.array([0.0, 0.0, 90.0], dtype=np.float32),
        schema.KEY_SETUP_EE_APERTURES_MAS: np.array([50.0, 100.0], dtype=np.float32),
    }
    fig = plot_metric_field(
        _make_simulation(setup=setup),
        metric_name=schema.KEY_STATS_EE,
        value_index=1,
        interpolation="nearest",
        grid_size=21,
    )

    try:
        values = fig.axes[0].images[0].get_array()
        assert values.shape == (21, 21)
        assert np.nanmin(values) >= 0.6 - 1.0e-6
        assert np.nanmax(values) <= 0.8 + 1.0e-6
        assert fig.axes[0].get_title() == "EE100"
    finally:
        plt.close(fig)


def test_plot_metric_field_can_draw_contours() -> None:
    fig = plot_metric_field(
        _make_simulation(),
        show_contours=True,
        contour_levels=np.array([0.15, 0.25], dtype=np.float32),
        contour_labels=True,
        grid_size=21,
    )

    try:
        assert len(fig.axes[0].collections) > 0
        assert len(fig.axes[0].texts) > 0
        assert _saved_png_size(fig) > 0
    finally:
        plt.close(fig)


def test_plot_metric_field_can_draw_ngs_and_lgs_markers() -> None:
    setup = {
        schema.KEY_SETUP_SCI_R_ARCSEC: np.array([0.0, 5.0, 5.0], dtype=np.float32),
        schema.KEY_SETUP_SCI_THETA_DEG: np.array([0.0, 0.0, 90.0], dtype=np.float32),
        BaseSimulation.KEY_SETUP_LGS_R_ARCSEC: np.array([30.0, 30.0], dtype=np.float32),
        BaseSimulation.KEY_SETUP_LGS_THETA_DEG: np.array([45.0, 135.0], dtype=np.float32),
    }
    options = {
        schema.KEY_OPTION_NGS_R_ARCSEC: np.array([10.0, 12.0], dtype=np.float32),
        schema.KEY_OPTION_NGS_THETA_DEG: np.array([0.0, 90.0], dtype=np.float32),
        schema.KEY_OPTION_NGS_MAG: np.array([14.0, 15.5], dtype=np.float32),
    }
    fig = plot_metric_field(
        _make_simulation(setup=setup, options=options),
        show_contours=True,
        show_ngs=True,
        show_ngs_mags=True,
        show_lgs=True,
        grid_size=21,
    )

    try:
        marker_collections = [
            collection
            for collection in fig.axes[0].collections
            if isinstance(collection, PathCollection)
        ]
        assert len(marker_collections) == 2
        assert marker_collections[0].get_offsets().shape == (2, 2)
        assert marker_collections[1].get_offsets().shape == (2, 2)
        assert np.allclose(marker_collections[0].get_sizes(), [200.0])
        assert np.allclose(marker_collections[1].get_sizes(), [200.0])
        for collection in marker_collections:
            assert collection.get_zorder() > fig.axes[0].collections[0].get_zorder()
            assert np.allclose(collection.get_edgecolors()[0][:3], np.zeros(3))
            assert np.allclose(collection.get_linewidths(), [0.5])
        assert fig.axes[0].get_legend() is None
        assert [text.get_text() for text in fig.axes[0].texts[-2:]] == ["14.0", "15.5"]
        assert _saved_png_size(fig) > 0
    finally:
        plt.close(fig)


def test_plot_metric_field_can_move_metric_name_to_y_axis() -> None:
    setup = {
        schema.KEY_SETUP_SCI_R_ARCSEC: np.array([0.0, 5.0, 5.0], dtype=np.float32),
        schema.KEY_SETUP_SCI_THETA_DEG: np.array([0.0, 0.0, 90.0], dtype=np.float32),
        schema.KEY_SETUP_EE_APERTURES_MAS: np.array([50.0, 100.0], dtype=np.float32),
    }
    fig = plot_metric_field(
        _make_simulation(setup=setup),
        metric_name=schema.KEY_STATS_EE,
        metric_name_location="y_axis",
        value_index=1,
        interpolation="nearest",
        grid_size=21,
    )

    try:
        assert fig.axes[0].get_title() == ""
        assert fig.axes[0].get_ylabel() == 'EE100\n"/Sky'
    finally:
        plt.close(fig)


def test_plot_metric_field_panel_uses_delegate_plotter_kwargs() -> None:
    calls: list[tuple[AnalysisSimulation, str, dict[str, object]]] = []

    def field_plotter(
        simulation: AnalysisSimulation,
        metric_name: str,
        **kwargs: object,
    ) -> Figure:
        calls.append((simulation, metric_name, kwargs))
        return plot_metric_field(simulation, metric_name, **kwargs)

    fig = plot_metric_field_panel(
        [_make_simulation(), _make_simulation()],
        labels=("A", "B"),
        ncols=2,
        field_plotter=field_plotter,
        field_plotter_kwargs={"interpolation": "nearest", "grid_size": 21},
    )

    try:
        assert len(calls) == 2
        assert all(call[1] == schema.KEY_STATS_SR for call in calls)
        assert all(call[2]["interpolation"] == "nearest" for call in calls)
        assert all(call[2]["grid_size"] == 21 for call in calls)
        assert all(call[2]["add_colorbar"] is False for call in calls)
        assert fig.axes[0].get_title() == "A"
        assert fig.axes[1].get_title() == "B"
        assert len([axis for axis in fig.axes if axis.images]) == 2
        assert _saved_png_size(fig) > 0
    finally:
        plt.close(fig)


def test_plot_metric_field_panel_accepts_prepared_grid_row() -> None:
    grid = prepare_metric_field_grid(1, 2, figure_size=(6.0, 3.0))
    fig = plot_metric_field_panel(
        [_make_simulation(), _make_simulation()],
        panel=grid[0],
        interpolation="nearest",
        grid_size=21,
    )

    try:
        assert fig is grid.figure
        assert len([axis for axis in fig.axes if axis.images]) == 2
        assert grid[0].colorbar_ax in fig.axes
    finally:
        plt.close(fig)


def test_plot_metric_field_comparison_uses_delegate_plotter_kwargs() -> None:
    calls: list[tuple[AnalysisSimulation, str, dict[str, object]]] = []

    def field_plotter(
        simulation: AnalysisSimulation,
        metric_name: str,
        **kwargs: object,
    ) -> Figure:
        calls.append((simulation, metric_name, kwargs))
        return plot_metric_field(simulation, metric_name, **kwargs)

    fig = plot_metric_field_comparison(
        _make_simulation(),
        _make_simulation(stats={schema.KEY_STATS_SR: np.array([0.2, 0.3, 0.4], dtype=np.float32)}),
        labels=("Left", "Right"),
        comparison=MetricComparison.RELATIVE_PERCENT,
        metric_name_location="colorbar",
        field_plotter=field_plotter,
        field_plotter_kwargs={"interpolation": "nearest", "grid_size": 21},
    )

    try:
        assert [call[1] for call in calls] == [
            schema.KEY_STATS_SR,
            schema.KEY_STATS_SR,
            "relative_percent",
        ]
        assert all(call[2]["interpolation"] == "nearest" for call in calls)
        assert all(call[2]["grid_size"] == 21 for call in calls)
        plot_axes = [axis for axis in fig.axes if axis.images]
        assert len(plot_axes) == 3
        assert plot_axes[0].get_title() == "Left"
        assert plot_axes[1].get_title() == "Right"
        assert plot_axes[2].get_title() == "Diff"
        colorbar_axes = [axis for axis in fig.axes if not axis.images]
        assert colorbar_axes[0].get_ylabel() == "Strehl Ratio"
        assert colorbar_axes[1].get_ylabel() == ""
        fig.canvas.draw()
        assert any(label.get_text().endswith("%") for label in colorbar_axes[1].get_yticklabels())
    finally:
        plt.close(fig)


def test_plot_metric_field_comparison_accepts_prepared_grid_row() -> None:
    grid = prepare_metric_field_comparison_grid(1, figure_size=(8.0, 3.0))
    fig = plot_metric_field_comparison(
        _make_simulation(),
        _make_simulation(stats={schema.KEY_STATS_SR: np.array([0.2, 0.3, 0.4], dtype=np.float32)}),
        panel=grid[0],
        interpolation="nearest",
        grid_size=21,
    )

    try:
        assert fig is grid.figure
        assert len([axis for axis in fig.axes if axis.images]) == 3
        assert grid[0].metric_colorbar_ax is None
        assert grid[0].comparison_colorbar_ax is None
        assert len([axis for axis in fig.axes if not axis.images]) == 2
    finally:
        plt.close(fig)


def test_plot_metric_field_raises_for_missing_requested_ngs_markers() -> None:
    with pytest.raises(ValueError, match="Missing required plotting field 'ngs_r_arcsec'"):
        plot_metric_field(_make_simulation(), show_ngs=True)


def test_plot_metric_field_raises_for_missing_requested_ngs_magnitude_labels() -> None:
    options = {
        schema.KEY_OPTION_NGS_R_ARCSEC: np.array([10.0], dtype=np.float32),
        schema.KEY_OPTION_NGS_THETA_DEG: np.array([0.0], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="Missing required plotting field 'ngs_mag'"):
        plot_metric_field(_make_simulation(options=options), show_ngs_mags=True)


def test_plot_metric_field_raises_for_missing_metric() -> None:
    with pytest.raises(ValueError, match="Metric 'missing' is not available"):
        plot_metric_field(_make_simulation(), metric_name="missing")


def test_plot_psf_raises_for_invalid_index() -> None:
    with pytest.raises(ValueError, match="psf_index 2 is out of range"):
        plot_psf(_make_simulation(), psf_index=2)


def test_plot_psf_rejects_non_integer_index() -> None:
    with pytest.raises(TypeError, match="psf_index must be an integer"):
        plot_psf(_make_simulation(), psf_index=1.2)


def test_plot_psf_core_rejects_non_integer_size() -> None:
    with pytest.raises(TypeError, match="size_px must be an integer"):
        plot_psf_core(_make_simulation(), size_px="3")


def test_plot_metric_field_rejects_non_integer_value_index() -> None:
    with pytest.raises(TypeError, match="value_index must be an integer"):
        plot_metric_field(_make_simulation(), metric_name=schema.KEY_STATS_EE, value_index=1.2)


def test_plot_metric_field_rejects_non_integer_grid_size() -> None:
    with pytest.raises(TypeError, match="grid_size must be an integer"):
        plot_metric_field(_make_simulation(), grid_size=21.0)


def test_plot_metric_field_rejects_unknown_interpolation() -> None:
    with pytest.raises(ValueError, match="interpolation must be 'rbf' or 'nearest'"):
        plot_metric_field(_make_simulation(), interpolation="linear")


def test_plot_metric_field_raises_for_coordinate_metric_length_mismatch() -> None:
    sim = _make_simulation(
        stats={schema.KEY_STATS_SR: np.array([0.1, 0.2], dtype=np.float32)}
    )

    with pytest.raises(ValueError, match="does not match science coordinate length"):
        plot_metric_field(sim)


def _saved_png_size(fig: Figure) -> int:
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    return buffer.tell()
