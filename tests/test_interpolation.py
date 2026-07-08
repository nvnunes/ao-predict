from __future__ import annotations

import pickle

import numpy as np
import pytest

from ao_predict.interpolation import (
    NgsHoMetricInterpolator,
    NgsHoMetricSamples,
    NgsHoPsfSamples,
    RegularGridInterpolationConfig,
    RbfInterpolationConfig,
    ScienceHoPsfSamples,
    build_ngs_ho_metric_interpolator,
    build_ngs_ho_metric_interpolator_from_psfs,
    build_ngs_ho_metric_samples_from_psfs,
    build_science_ho_psf_interpolator,
    evaluate_ngs_ho_metric_interpolator,
    evaluate_science_ho_psf_interpolator,
    load_ngs_ho_metric_interpolator,
    load_science_ho_psf_interpolator,
    replay_ngs_ho_metric_interpolator,
    replay_science_ho_psf_interpolator,
    save_ngs_ho_metric_interpolator,
    save_science_ho_psf_interpolator,
    validate_ngs_ho_metric_interpolator,
    validate_ngs_ho_metric_query,
    validate_science_ho_psf_query,
    zenith_angle_to_airmass,
)
from ao_predict.interpolation._core import grid_field_values, rectangular_field_axes


def test_science_ho_psf_interpolator_round_trips_and_replays_sources(tmp_path) -> None:
    interpolator = build_science_ho_psf_interpolator(_science_samples())
    path = tmp_path / "science.pkl"

    save_science_ho_psf_interpolator(interpolator, path)
    loaded = load_science_ho_psf_interpolator(path)
    replay = replay_science_ho_psf_interpolator(loaded, _science_samples())
    prediction = evaluate_science_ho_psf_interpolator(
        loaded,
        zenith_angle_deg=20.0,
        wavelength_um=1.0,
        x_arcsec=_science_samples().x_arcsec,
        y_arcsec=_science_samples().y_arcsec,
    )
    expected_flux = np.sum(_science_samples().psfs[0], axis=(-2, -1))

    assert loaded.builder["interpolation_method"] == "regular_grid_linear_airmass_wavelength_y_x"
    assert loaded.coordinate_order == ("airmass", "wavelength_um", "y_arcsec", "x_arcsec")
    assert loaded.meta["norm_correction"] == pytest.approx(0.75)
    np.testing.assert_allclose(loaded.meta["plane_correction"], np.asarray([[1.0, 2.0], [3.0, 4.0]]))
    assert prediction.meta["norm_correction"] == pytest.approx(0.75)
    assert prediction.meta["plane_correction"] == pytest.approx(1.0)
    np.testing.assert_allclose(np.sum(prediction.psfs, axis=(-2, -1)), expected_flux, rtol=1.0e-6)
    assert not np.allclose(expected_flux, 1.0)
    assert replay.psf_nrms_max == pytest.approx(0.0, abs=1.0e-5)
    assert replay.pixel_scale_abs_max_mas == pytest.approx(0.0)
    assert replay.metric_max_abs["fwhm_mas"] == pytest.approx(0.0, abs=1.0e-5)
    assert replay.metric_max_abs["ee"] == pytest.approx(0.0, abs=1.0e-5)


def test_science_ho_psf_interpolator_uses_rectangular_grid() -> None:
    interpolator = build_science_ho_psf_interpolator(_science_samples())

    assert interpolator.psf_grid.shape == (2, 2, 2, 2, 9, 9)
    np.testing.assert_allclose(interpolator.x_arcsec, np.asarray([-1.0, 1.0]))
    np.testing.assert_allclose(interpolator.y_arcsec, np.asarray([-1.0, 1.0]))


def test_rectangular_field_helpers_fill_values_in_y_x_order() -> None:
    x = np.asarray([1.0, -1.0, 1.0, -1.0])
    y = np.asarray([1.0, -1.0, -1.0, 1.0])
    values = np.asarray([4.0, 1.0, 2.0, 3.0])

    x_axis, y_axis = rectangular_field_axes(x, y, label="test samples")
    grid = grid_field_values(x, y, values, x_axis, y_axis, label="test samples", dtype=float)

    np.testing.assert_allclose(x_axis, np.asarray([-1.0, 1.0]))
    np.testing.assert_allclose(y_axis, np.asarray([-1.0, 1.0]))
    np.testing.assert_allclose(grid, np.asarray([[1.0, 2.0], [3.0, 4.0]]))


def test_science_ho_psf_interpolator_interpolates_wavelength_zenith_and_pixel_scale() -> None:
    samples = _science_samples()
    interpolator = build_science_ho_psf_interpolator(samples)
    airmass_axis = zenith_angle_to_airmass(np.asarray([20.0, 30.0]))
    query_airmass = float(np.mean(airmass_axis))
    query_zenith = float(np.rad2deg(np.arccos(1.0 / query_airmass)))

    prediction = evaluate_science_ho_psf_interpolator(
        interpolator,
        zenith_angle_deg=query_zenith,
        wavelength_um=1.5,
        x_arcsec=np.asarray([0.0]),
        y_arcsec=np.asarray([0.0]),
    )

    assert prediction.psfs.shape == (1, 9, 9)
    assert float(np.sum(prediction.psfs)) > 1.0
    assert prediction.pixel_scale_mas == pytest.approx(6.375)
    assert prediction.metadata.pixel_scale_mas == pytest.approx(6.375)
    assert prediction.meta["norm_correction"] == pytest.approx(0.75)
    assert prediction.meta["plane_correction"] == pytest.approx(2.5)
    with pytest.raises(ValueError, match="x_arcsec"):
        evaluate_science_ho_psf_interpolator(
            interpolator,
            zenith_angle_deg=query_zenith,
            wavelength_um=1.5,
            x_arcsec=np.asarray([2.0]),
            y_arcsec=np.asarray([0.0]),
        )


def test_science_ho_psf_field_only_artifact_omits_physical_queries(tmp_path) -> None:
    source = _science_samples()
    samples = ScienceHoPsfSamples(
        x_arcsec=source.x_arcsec,
        y_arcsec=source.y_arcsec,
        psfs=source.psfs[0],
        wavelength_um=1.0,
        pixel_scale_mas=4.0,
        tel_diameter_m=source.tel_diameter_m,
        tel_pupil=source.tel_pupil,
        provenance=("field-only",),
    )

    interpolator = build_science_ho_psf_interpolator(samples)
    path = tmp_path / "science-field.pkl"
    save_science_ho_psf_interpolator(interpolator, path)
    loaded = load_science_ho_psf_interpolator(path)
    replay = replay_science_ho_psf_interpolator(loaded, samples)
    prediction = evaluate_science_ho_psf_interpolator(
        loaded,
        x_arcsec=np.asarray([0.0]),
        y_arcsec=np.asarray([0.0]),
    )

    assert loaded.coordinate_order == ("y_arcsec", "x_arcsec")
    assert loaded.psf_grid.shape == (2, 2, 9, 9)
    assert prediction.metadata.wavelength_um == pytest.approx(1.0)
    assert prediction.pixel_scale_mas == pytest.approx(4.0)
    assert replay.psf_nrms_max == pytest.approx(0.0, abs=1.0e-5)
    with pytest.raises(ValueError, match="wavelength_um"):
        evaluate_science_ho_psf_interpolator(
            loaded,
            wavelength_um=2.0,
            x_arcsec=np.asarray([0.0]),
            y_arcsec=np.asarray([0.0]),
        )


def test_science_ho_psf_supports_single_physical_axes() -> None:
    source = _science_samples()
    wavelength_samples = ScienceHoPsfSamples(
        zenith_angle_deg=20.0,
        wavelength_um=source.wavelength_um[:2],
        x_arcsec=source.x_arcsec,
        y_arcsec=source.y_arcsec,
        psfs=source.psfs[:2],
        pixel_scale_mas=source.pixel_scale_mas[:2],
        tel_diameter_m=source.tel_diameter_m,
        tel_pupil=source.tel_pupil,
    )
    wavelength_artifact = build_science_ho_psf_interpolator(wavelength_samples)

    assert wavelength_artifact.coordinate_order == ("wavelength_um", "y_arcsec", "x_arcsec")
    prediction = evaluate_science_ho_psf_interpolator(
        wavelength_artifact,
        wavelength_um=1.5,
        x_arcsec=np.asarray([0.0]),
        y_arcsec=np.asarray([0.0]),
    )
    assert prediction.pixel_scale_mas == pytest.approx(6.0)

    airmass_samples = ScienceHoPsfSamples(
        zenith_angle_deg=np.asarray([20.0, 30.0]),
        wavelength_um=1.0,
        x_arcsec=source.x_arcsec,
        y_arcsec=source.y_arcsec,
        psfs=source.psfs[[0, 2]],
        pixel_scale_mas=source.pixel_scale_mas[[0, 2]],
        tel_diameter_m=source.tel_diameter_m,
        tel_pupil=source.tel_pupil,
    )
    airmass_artifact = build_science_ho_psf_interpolator(airmass_samples)

    assert airmass_artifact.coordinate_order == ("airmass", "y_arcsec", "x_arcsec")
    with pytest.raises(ValueError, match="zenith_angle_deg is required"):
        evaluate_science_ho_psf_interpolator(
            airmass_artifact,
            x_arcsec=np.asarray([0.0]),
            y_arcsec=np.asarray([0.0]),
        )


def test_science_ho_psf_requires_only_active_physical_queries() -> None:
    interpolator = build_science_ho_psf_interpolator(_science_samples())

    with pytest.raises(ValueError, match="zenith_angle_deg is required"):
        evaluate_science_ho_psf_interpolator(
            interpolator,
            wavelength_um=1.0,
            x_arcsec=np.asarray([0.0]),
            y_arcsec=np.asarray([0.0]),
        )
    with pytest.raises(ValueError, match="wavelength_um is required"):
        evaluate_science_ho_psf_interpolator(
            interpolator,
            zenith_angle_deg=20.0,
            x_arcsec=np.asarray([0.0]),
            y_arcsec=np.asarray([0.0]),
        )


def test_science_ho_psf_query_validation_rejects_out_of_range_before_runtime() -> None:
    interpolator = build_science_ho_psf_interpolator(_science_samples())

    with pytest.raises(ValueError, match="wavelength_um"):
        validate_science_ho_psf_query(interpolator, zenith_angle_deg=20.0, wavelength_um=3.0)
    with pytest.raises(ValueError, match="airmass"):
        validate_science_ho_psf_query(interpolator, zenith_angle_deg=40.0, wavelength_um=1.5)


def test_science_ho_psf_interpolator_rejects_malformed_grids() -> None:
    samples = _science_samples()

    with pytest.raises(ValueError, match="complete active physical-coordinate grid"):
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
                zenith_angle_deg=samples.zenith_angle_deg[:-1],
                wavelength_um=samples.wavelength_um[:-1],
                x_arcsec=samples.x_arcsec,
                y_arcsec=samples.y_arcsec,
                psfs=samples.psfs[:-1],
                pixel_scale_mas=samples.pixel_scale_mas[:-1],
                tel_diameter_m=samples.tel_diameter_m,
                tel_pupil=samples.tel_pupil,
            )
        )

    with pytest.raises(ValueError, match="pixel_scale_mas"):
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
                zenith_angle_deg=samples.zenith_angle_deg,
                wavelength_um=samples.wavelength_um,
                x_arcsec=samples.x_arcsec,
                y_arcsec=samples.y_arcsec,
                psfs=samples.psfs,
                pixel_scale_mas=np.array([4.0, -1.0, 4.5, 9.0]),
                tel_diameter_m=samples.tel_diameter_m,
                tel_pupil=samples.tel_pupil,
            )
        )

    zero_flux = np.array(samples.psfs, copy=True)
    zero_flux[0, 0] = 0.0
    with pytest.raises(ValueError, match="strictly positive per-PSF total flux"):
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
                zenith_angle_deg=samples.zenith_angle_deg,
                wavelength_um=samples.wavelength_um,
                x_arcsec=samples.x_arcsec,
                y_arcsec=samples.y_arcsec,
                psfs=zero_flux,
                pixel_scale_mas=samples.pixel_scale_mas,
                tel_diameter_m=samples.tel_diameter_m,
                tel_pupil=samples.tel_pupil,
            )
        )

    with pytest.raises(ValueError, match="meta\\['bad'\\]"):
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
                zenith_angle_deg=samples.zenith_angle_deg,
                wavelength_um=samples.wavelength_um,
                x_arcsec=samples.x_arcsec,
                y_arcsec=samples.y_arcsec,
                psfs=samples.psfs,
                pixel_scale_mas=samples.pixel_scale_mas,
                tel_diameter_m=samples.tel_diameter_m,
                tel_pupil=samples.tel_pupil,
                meta={"bad": np.array([1.0, 2.0])},
            )
        )

    with pytest.raises(ValueError, match="complete rectangular"):
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
                zenith_angle_deg=samples.zenith_angle_deg,
                wavelength_um=samples.wavelength_um,
                x_arcsec=samples.x_arcsec[:-1],
                y_arcsec=samples.y_arcsec[:-1],
                psfs=samples.psfs[:, :-1],
                pixel_scale_mas=samples.pixel_scale_mas,
                tel_diameter_m=samples.tel_diameter_m,
                tel_pupil=samples.tel_pupil,
            )
        )

    with pytest.raises(ValueError, match="must be finite"):
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
                zenith_angle_deg=samples.zenith_angle_deg,
                wavelength_um=samples.wavelength_um,
                x_arcsec=samples.x_arcsec,
                y_arcsec=samples.y_arcsec,
                psfs=samples.psfs,
                pixel_scale_mas=samples.pixel_scale_mas,
                tel_diameter_m=samples.tel_diameter_m,
                tel_pupil=samples.tel_pupil,
                meta={"bad": np.nan},
            )
        )

    with pytest.raises(ValueError, match="collides with a core /meta field"):
        build_science_ho_psf_interpolator(
            ScienceHoPsfSamples(
                zenith_angle_deg=samples.zenith_angle_deg,
                wavelength_um=samples.wavelength_um,
                x_arcsec=samples.x_arcsec,
                y_arcsec=samples.y_arcsec,
                psfs=samples.psfs,
                pixel_scale_mas=samples.pixel_scale_mas,
                tel_diameter_m=samples.tel_diameter_m,
                tel_pupil=samples.tel_pupil,
                meta={"pixel_scale_mas": 1.0},
            )
        )


def test_science_ho_psf_interpolator_defaults_missing_payload_meta(tmp_path) -> None:
    interpolator = build_science_ho_psf_interpolator(_science_samples())
    path = tmp_path / "missing-meta.pkl"
    payload = {
        "kind": "ao_predict_science_ho_psf_interpolator",
        "version": 1,
        "builder": dict(interpolator.builder),
        "interpolation": {"coordinate_order": interpolator.coordinate_order},
        "metadata": {
            "zenith_angle_deg_axis": interpolator.zenith_angle_deg_axis,
            "airmass_axis": interpolator.airmass_axis,
            "wavelength_um_axis": interpolator.wavelength_um_axis,
            "x_arcsec": interpolator.x_arcsec,
            "y_arcsec": interpolator.y_arcsec,
            "psf_shape": interpolator.psf_shape,
            "pixel_scale_mas_grid": interpolator.pixel_scale_mas_grid,
            "tel_diameter_m": interpolator.tel_diameter_m,
            "tel_pupil": interpolator.tel_pupil,
            "provenance": tuple(interpolator.provenance),
        },
        "model": {
            "psf_grid": interpolator.psf_grid,
        },
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    loaded = load_science_ho_psf_interpolator(path)
    prediction = evaluate_science_ho_psf_interpolator(
        loaded,
        zenith_angle_deg=20.0,
        wavelength_um=1.0,
        x_arcsec=_science_samples().x_arcsec,
        y_arcsec=_science_samples().y_arcsec,
    )

    assert loaded.meta == {}
    assert prediction.meta == {}


def test_ngs_ho_metric_interpolator_round_trips_and_replays_sources(tmp_path) -> None:
    samples = _ngs_metric_samples()
    interpolator = build_ngs_ho_metric_interpolator(samples, interpolation_config=RbfInterpolationConfig(smoothing=0.0))
    path = tmp_path / "ngs.pkl"

    save_ngs_ho_metric_interpolator(interpolator, path)
    loaded = load_ngs_ho_metric_interpolator(path)
    replay = replay_ngs_ho_metric_interpolator(loaded, samples)
    prediction = evaluate_ngs_ho_metric_interpolator(
        loaded,
        zenith_angle_deg=np.repeat(samples.zenith_angle_deg, samples.x_arcsec.size),
        x_arcsec=np.tile(samples.x_arcsec, samples.zenith_angle_deg.size),
        y_arcsec=np.tile(samples.y_arcsec, samples.zenith_angle_deg.size),
    )

    assert loaded.interpolation_config.smoothing == pytest.approx(0.0)
    assert set(loaded.metric_names) == {"ee", "fwhm_mas", "sr"}
    assert replay.metric_max_abs["ee"] == pytest.approx(0.0, abs=1.0e-8)
    assert replay.metric_max_abs["fwhm_mas"] == pytest.approx(0.0, abs=1.0e-8)
    assert replay.metric_max_abs["sr"] == pytest.approx(0.0, abs=1.0e-8)
    assert prediction.sr is not None


def test_ngs_ho_metric_interpolator_uses_regular_grid_by_default() -> None:
    interpolator = build_ngs_ho_metric_interpolator(_ngs_metric_samples())

    assert isinstance(interpolator.interpolation_config, RegularGridInterpolationConfig)
    assert interpolator.interpolation_config.method == "linear"
    assert interpolator.builder["interpolation_strategy"] == "regular_grid"
    assert set(interpolator.model["metric_grids"]) == {"ee", "fwhm_mas", "sr"}


def test_ngs_ho_metric_regular_grid_round_trips_replays_and_interpolates(tmp_path) -> None:
    samples = _ngs_metric_samples()
    interpolator = build_ngs_ho_metric_interpolator(samples)
    path = tmp_path / "ngs-grid.pkl"

    save_ngs_ho_metric_interpolator(interpolator, path)
    loaded = load_ngs_ho_metric_interpolator(path)
    replay = replay_ngs_ho_metric_interpolator(loaded, samples)
    prediction = evaluate_ngs_ho_metric_interpolator(
        loaded,
        zenith_angle_deg=20.0,
        x_arcsec=np.asarray([0.0]),
        y_arcsec=np.asarray([0.0]),
    )

    assert isinstance(loaded.interpolation_config, RegularGridInterpolationConfig)
    assert replay.metric_max_abs["ee"] == pytest.approx(0.0, abs=1.0e-12)
    assert replay.metric_max_abs["fwhm_mas"] == pytest.approx(0.0, abs=1.0e-12)
    assert replay.metric_max_abs["sr"] == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(prediction.ee, np.asarray([0.26]))
    np.testing.assert_allclose(prediction.fwhm_mas, np.asarray([83.0]))
    np.testing.assert_allclose(prediction.sr, np.asarray([0.115]))


def test_ngs_ho_metric_field_only_regular_grid_and_rbf_omit_zenith() -> None:
    source = _ngs_metric_samples()
    samples = NgsHoMetricSamples(
        zenith_angle_deg=20.0,
        x_arcsec=source.x_arcsec,
        y_arcsec=source.y_arcsec,
        ee=source.ee[0],
        fwhm_mas=source.fwhm_mas[0],
        sr=source.sr[0],
    )

    regular = build_ngs_ho_metric_interpolator(samples)
    rbf = build_ngs_ho_metric_interpolator(samples, interpolation_config=RbfInterpolationConfig(smoothing=0.0))

    assert regular.coordinate_order == ("y_arcsec", "x_arcsec")
    assert rbf.coordinate_order == ("x_arcsec", "y_arcsec")
    assert "wavelength_um" not in regular.coordinate_order
    assert "wavelength_um" not in rbf.coordinate_order
    regular_prediction = evaluate_ngs_ho_metric_interpolator(
        regular,
        x_arcsec=np.asarray([0.0]),
        y_arcsec=np.asarray([0.0]),
    )
    rbf_prediction = evaluate_ngs_ho_metric_interpolator(
        rbf,
        x_arcsec=np.asarray([-1.0]),
        y_arcsec=np.asarray([-1.0]),
    )
    np.testing.assert_allclose(regular_prediction.ee, np.asarray([0.26]))
    np.testing.assert_allclose(rbf_prediction.ee, np.asarray([0.20]), atol=1.0e-8)
    with pytest.raises(ValueError, match="fixed artifact value"):
        evaluate_ngs_ho_metric_interpolator(
            regular,
            zenith_angle_deg=30.0,
            x_arcsec=np.asarray([0.0]),
            y_arcsec=np.asarray([0.0]),
        )


def test_ngs_ho_metric_active_airmass_requires_zenith() -> None:
    regular = build_ngs_ho_metric_interpolator(_ngs_metric_samples())
    rbf = build_ngs_ho_metric_interpolator(
        _ngs_metric_samples(),
        interpolation_config=RbfInterpolationConfig(smoothing=0.0),
    )

    assert regular.coordinate_order == ("airmass", "y_arcsec", "x_arcsec")
    assert rbf.coordinate_order == ("airmass", "x_arcsec", "y_arcsec")
    with pytest.raises(ValueError, match="zenith_angle_deg is required"):
        evaluate_ngs_ho_metric_interpolator(
            regular,
            x_arcsec=np.asarray([0.0]),
            y_arcsec=np.asarray([0.0]),
        )
    with pytest.raises(ValueError, match="zenith_angle_deg is required"):
        evaluate_ngs_ho_metric_interpolator(
            rbf,
            x_arcsec=np.asarray([0.0]),
            y_arcsec=np.asarray([0.0]),
        )


def test_ngs_ho_metric_interpolator_uses_explicit_rbf_default_smoothing() -> None:
    interpolator = build_ngs_ho_metric_interpolator(
        _ngs_metric_samples(),
        interpolation_config=RbfInterpolationConfig(),
    )

    assert interpolator.interpolation_config.smoothing == pytest.approx(0.05)
    assert interpolator.builder["interpolation_strategy"] == "rbf"


def test_ngs_ho_metric_interpolator_broadcasts_scalar_zenith() -> None:
    samples = _ngs_metric_samples()
    interpolator = build_ngs_ho_metric_interpolator(samples, interpolation_config=RbfInterpolationConfig(smoothing=0.0))

    prediction = evaluate_ngs_ho_metric_interpolator(
        interpolator,
        zenith_angle_deg=20.0,
        x_arcsec=np.asarray([-1.0, 1.0]),
        y_arcsec=np.asarray([-1.0, 1.0]),
    )

    assert prediction.ee.shape == (2,)
    assert prediction.fwhm_mas.shape == (2,)
    assert prediction.sr.shape == (2,)


def test_ngs_ho_metric_query_validation_rejects_unsupported_queries_before_evaluation() -> None:
    interpolator = build_ngs_ho_metric_interpolator(
        _ngs_metric_samples(),
        interpolation_config=RbfInterpolationConfig(smoothing=0.0),
    )

    validate_ngs_ho_metric_query(
        interpolator,
        zenith_angle_deg=20.0,
        x_arcsec=np.asarray([0.0]),
        y_arcsec=np.asarray([0.0]),
    )
    with pytest.raises(ValueError, match="airmass"):
        validate_ngs_ho_metric_query(
            interpolator,
            zenith_angle_deg=40.0,
            x_arcsec=np.asarray([0.0]),
            y_arcsec=np.asarray([0.0]),
        )
    with pytest.raises(ValueError, match="field"):
        evaluate_ngs_ho_metric_interpolator(
            interpolator,
            zenith_angle_deg=20.0,
            x_arcsec=np.asarray([2.0]),
            y_arcsec=np.asarray([0.0]),
        )


def test_ngs_ho_metric_regular_grid_rejects_unsupported_queries_before_evaluation() -> None:
    interpolator = build_ngs_ho_metric_interpolator(_ngs_metric_samples())

    validate_ngs_ho_metric_query(
        interpolator,
        zenith_angle_deg=20.0,
        x_arcsec=np.asarray([0.0]),
        y_arcsec=np.asarray([0.0]),
    )
    with pytest.raises(ValueError, match="airmass"):
        validate_ngs_ho_metric_query(
            interpolator,
            zenith_angle_deg=40.0,
            x_arcsec=np.asarray([0.0]),
            y_arcsec=np.asarray([0.0]),
        )
    with pytest.raises(ValueError, match="x_arcsec"):
        evaluate_ngs_ho_metric_interpolator(
            interpolator,
            zenith_angle_deg=20.0,
            x_arcsec=np.asarray([2.0]),
            y_arcsec=np.asarray([0.0]),
        )


def test_ngs_ho_metric_interpolator_rejects_bad_metric_values() -> None:
    samples = _ngs_metric_samples()

    with pytest.raises(ValueError, match="ee"):
        build_ngs_ho_metric_interpolator(
            NgsHoMetricSamples(
                zenith_angle_deg=samples.zenith_angle_deg,
                x_arcsec=samples.x_arcsec,
                y_arcsec=samples.y_arcsec,
                ee=np.full(samples.ee.shape, 1.2),
                fwhm_mas=samples.fwhm_mas,
                sr=samples.sr,
            )
        )
    with pytest.raises(ValueError, match="fwhm_mas"):
        build_ngs_ho_metric_interpolator(
            NgsHoMetricSamples(
                zenith_angle_deg=samples.zenith_angle_deg,
                x_arcsec=samples.x_arcsec,
                y_arcsec=samples.y_arcsec,
                ee=samples.ee,
                fwhm_mas=np.full(samples.fwhm_mas.shape, -1.0),
                sr=samples.sr,
            )
        )
    with pytest.raises(ValueError, match="sr"):
        build_ngs_ho_metric_interpolator(
            NgsHoMetricSamples(
                zenith_angle_deg=samples.zenith_angle_deg,
                x_arcsec=samples.x_arcsec,
                y_arcsec=samples.y_arcsec,
                ee=samples.ee,
                fwhm_mas=samples.fwhm_mas,
                sr=np.full(samples.sr.shape, -1.0),
            )
        )


def test_ngs_ho_metric_regular_grid_rejects_malformed_field_grids() -> None:
    samples = _ngs_metric_samples()

    with pytest.raises(ValueError, match="complete rectangular"):
        build_ngs_ho_metric_interpolator(
            NgsHoMetricSamples(
                zenith_angle_deg=samples.zenith_angle_deg,
                x_arcsec=samples.x_arcsec[:-1],
                y_arcsec=samples.y_arcsec[:-1],
                ee=samples.ee[:, :-1],
                fwhm_mas=samples.fwhm_mas[:, :-1],
                sr=samples.sr[:, :-1],
            )
        )

    with pytest.raises(ValueError, match="Duplicate NGS HO metric samples field point"):
        build_ngs_ho_metric_interpolator(
            NgsHoMetricSamples(
                zenith_angle_deg=samples.zenith_angle_deg,
                x_arcsec=np.asarray([-1.0, -1.0, 1.0, 1.0]),
                y_arcsec=np.asarray([-1.0, -1.0, 1.0, 1.0]),
                ee=samples.ee,
                fwhm_mas=samples.fwhm_mas,
                sr=samples.sr,
            )
        )


def test_ngs_ho_metric_interpolator_rejects_model_metric_mismatch() -> None:
    interpolator = build_ngs_ho_metric_interpolator(
        _ngs_metric_samples(),
        interpolation_config=RbfInterpolationConfig(smoothing=0.0),
    )
    model = dict(interpolator.model)
    models = dict(model["models"])
    del models["sr"]
    model["models"] = models
    bad = NgsHoMetricInterpolator(
        coordinate_order=interpolator.coordinate_order,
        zenith_angle_deg_axis=interpolator.zenith_angle_deg_axis,
        airmass_axis=interpolator.airmass_axis,
        x_arcsec=interpolator.x_arcsec,
        y_arcsec=interpolator.y_arcsec,
        metric_names=interpolator.metric_names,
        interpolation_config=interpolator.interpolation_config,
        model=model,
        provenance=interpolator.provenance,
        builder=interpolator.builder,
    )

    with pytest.raises(ValueError, match="model does not match metric_names"):
        validate_ngs_ho_metric_interpolator(bad)


def test_ngs_ho_psfs_feed_metric_samples_then_metric_interpolator() -> None:
    psf_samples = _ngs_psf_samples()
    metric_samples = build_ngs_ho_metric_samples_from_psfs(psf_samples)
    direct = build_ngs_ho_metric_interpolator(metric_samples, interpolation_config=RbfInterpolationConfig(smoothing=0.0))
    from_psfs = build_ngs_ho_metric_interpolator_from_psfs(
        psf_samples,
        interpolation_config=RbfInterpolationConfig(smoothing=0.0),
    )

    assert metric_samples.sr is not None
    assert metric_samples.ee.shape == (2, 4)
    assert metric_samples.fwhm_mas.shape == (2, 4)
    assert metric_samples.sr.shape == (2, 4)
    assert set(direct.metric_names) == {"ee", "fwhm_mas", "sr"}
    assert set(from_psfs.metric_names) == {"ee", "fwhm_mas", "sr"}
    assert np.all(np.isfinite(metric_samples.ee))
    assert np.all(np.isfinite(metric_samples.fwhm_mas))
    assert np.all(np.isfinite(metric_samples.sr))


def test_ngs_ho_psf_samples_require_full_psf_metadata() -> None:
    samples = _ngs_psf_samples()

    with pytest.raises(ValueError, match="tel_pupil"):
        build_ngs_ho_metric_samples_from_psfs(
            NgsHoPsfSamples(
                zenith_angle_deg=samples.zenith_angle_deg,
                x_arcsec=samples.x_arcsec,
                y_arcsec=samples.y_arcsec,
                psfs=samples.psfs,
                wavelength_um=samples.wavelength_um,
                pixel_scale_mas=samples.pixel_scale_mas,
                tel_diameter_m=samples.tel_diameter_m,
                tel_pupil=np.ones(4),
            )
        )


def _science_samples() -> ScienceHoPsfSamples:
    zenith = np.asarray([20.0, 20.0, 30.0, 30.0])
    wavelength = np.asarray([1.0, 2.0, 1.0, 2.0])
    x = np.asarray([-1.0, 1.0, -1.0, 1.0])
    y = np.asarray([-1.0, -1.0, 1.0, 1.0])
    psfs = np.empty((4, 4, 9, 9), dtype=np.float32)
    for plane in range(4):
        for point, (px, py) in enumerate(zip(x, y, strict=True)):
            psfs[plane, point] = (
                (2.0 + 0.5 * plane + 0.25 * point)
                * _gaussian_psf(9, sigma=1.0 + 0.05 * plane, dx=0.1 * px, dy=0.1 * py)
            )
    return ScienceHoPsfSamples(
        zenith_angle_deg=zenith,
        wavelength_um=wavelength,
        x_arcsec=x,
        y_arcsec=y,
        psfs=psfs,
        pixel_scale_mas=np.asarray([4.0, 8.0, 4.5, 9.0]),
        tel_diameter_m=30.0,
        tel_pupil=np.ones((4, 4), dtype=np.float32),
        meta={
            "norm_correction": 0.75,
            "plane_correction": np.asarray([1.0, 2.0, 3.0, 4.0]),
        },
        provenance=("synthetic",),
    )


def _ngs_metric_samples() -> NgsHoMetricSamples:
    zenith = np.asarray([20.0, 30.0])
    x = np.asarray([-1.0, 1.0, -1.0, 1.0])
    y = np.asarray([-1.0, -1.0, 1.0, 1.0])
    ee = np.asarray(
        [
            [0.20, 0.24, 0.28, 0.32],
            [0.30, 0.34, 0.38, 0.42],
        ],
        dtype=float,
    )
    fwhm_mas = np.asarray(
        [
            [80.0, 82.0, 84.0, 86.0],
            [90.0, 92.0, 94.0, 96.0],
        ],
        dtype=float,
    )
    sr = np.asarray(
        [
            [0.10, 0.11, 0.12, 0.13],
            [0.15, 0.16, 0.17, 0.18],
        ],
        dtype=float,
    )
    return NgsHoMetricSamples(
        zenith_angle_deg=zenith,
        x_arcsec=x,
        y_arcsec=y,
        ee=ee,
        fwhm_mas=fwhm_mas,
        sr=sr,
        provenance=("synthetic",),
    )


def _ngs_psf_samples() -> NgsHoPsfSamples:
    zenith = np.asarray([20.0, 30.0])
    x = np.asarray([-1.0, 1.0, -1.0, 1.0])
    y = np.asarray([-1.0, -1.0, 1.0, 1.0])
    psfs = np.empty((2, 4, 21, 21), dtype=np.float32)
    for plane in range(2):
        for point, (px, py) in enumerate(zip(x, y, strict=True)):
            psfs[plane, point] = _gaussian_psf(
                21,
                sigma=1.2 + 0.05 * plane + 0.03 * point,
                dx=0.1 * px,
                dy=0.1 * py,
            )
    return NgsHoPsfSamples(
        zenith_angle_deg=zenith,
        x_arcsec=x,
        y_arcsec=y,
        psfs=psfs,
        wavelength_um=np.asarray([1.65, 1.65]),
        pixel_scale_mas=np.asarray([5.0, 5.0]),
        tel_diameter_m=30.0,
        tel_pupil=np.ones((8, 8), dtype=np.float32),
        provenance=("synthetic",),
    )


def _gaussian_psf(size: int, *, sigma: float, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    coords = np.arange(size, dtype=float) - (size - 1) / 2.0
    xx, yy = np.meshgrid(coords - dx, coords - dy)
    psf = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    return (psf / psf.sum()).astype(np.float32)
