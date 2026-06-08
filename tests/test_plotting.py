from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from ao_predict.analysis import AnalysisSimulation
from ao_predict.plotting import plot_metric_field, plot_psf, plot_psf_core
from ao_predict.simulation import schema


def _make_simulation(
    *,
    psfs: np.ndarray | None = None,
    setup: dict[str, object] | None = None,
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
        _config={"setup": setup, "options": {}},
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
        assert fig.axes[0].get_xlabel() == '["/Sky]'
        assert fig.axes[0].get_ylabel() == '["/Sky]'
        assert _saved_png_size(fig) > 0
    finally:
        plt.close(fig)


def test_plot_metric_field_applies_color_limits() -> None:
    fig = plot_metric_field(_make_simulation(), vmin=0.0, vmax=0.5)

    try:
        assert fig.axes[0].images[0].get_clim() == (0.0, 0.5)
    finally:
        plt.close(fig)


def test_plot_metric_field_selects_multidimensional_metric_value() -> None:
    fig = plot_metric_field(
        _make_simulation(),
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
    finally:
        plt.close(fig)


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
