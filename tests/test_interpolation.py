from __future__ import annotations

import csv
import pickle
import sys

import numpy as np
import pytest

from ao_predict.cli import main as cli_main
from ao_predict.interpolation import (
    NgsHoMetricInterpolator,
    NgsHoMetricSamples,
    NgsHoPsfSamples,
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
    save_ngs_ho_metric_inputs,
    save_ngs_ho_metric_interpolator,
    save_ngs_ho_psf_inputs,
    save_science_ho_psf_inputs,
    save_science_ho_psf_interpolator,
    validate_ngs_ho_metric_interpolator,
    validate_ngs_ho_metric_query,
    validate_science_ho_psf_query,
    zenith_angle_to_airmass,
)


def test_science_ho_psf_interpolator_round_trips_and_replays_sources(tmp_path) -> None:
    config = RbfInterpolationConfig(smoothing=0.0, degree=1)
    interpolator = build_science_ho_psf_interpolator(_science_samples(), interpolation_config=config)
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

    assert loaded.interpolation_config.smoothing == pytest.approx(0.0)
    assert loaded.interpolation_config.degree == 1
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


def test_science_ho_psf_interpolator_default_smoothing_matches_validated_baseline() -> None:
    interpolator = build_science_ho_psf_interpolator(_science_samples())

    assert interpolator.interpolation_config.smoothing == pytest.approx(0.03)


def test_science_ho_psf_interpolator_interpolates_wavelength_zenith_and_pixel_scale() -> None:
    samples = _science_samples()
    interpolator = build_science_ho_psf_interpolator(samples, interpolation_config=RbfInterpolationConfig(smoothing=0.0))
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


def test_science_ho_psf_query_validation_rejects_out_of_range_before_runtime() -> None:
    interpolator = build_science_ho_psf_interpolator(_science_samples())

    with pytest.raises(ValueError, match="wavelength_um"):
        validate_science_ho_psf_query(interpolator, zenith_angle_deg=20.0, wavelength_um=3.0)
    with pytest.raises(ValueError, match="airmass"):
        validate_science_ho_psf_query(interpolator, zenith_angle_deg=40.0, wavelength_um=1.5)


def test_science_ho_psf_interpolator_rejects_malformed_grids() -> None:
    samples = _science_samples()

    with pytest.raises(ValueError, match="complete zenith_angle_deg x wavelength_um grid"):
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
        "interpolation_config": interpolator.interpolation_config,
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
            "plane_model_indices": interpolator.plane_model_indices,
            "plane_models": tuple(interpolator.plane_models),
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


def test_ngs_ho_metric_interpolator_uses_shared_default_smoothing() -> None:
    interpolator = build_ngs_ho_metric_interpolator(_ngs_metric_samples())

    assert interpolator.interpolation_config.smoothing == pytest.approx(0.05)


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


def test_interpolation_cli_builds_and_replays_science_inputs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = tmp_path / "science-inputs.pkl"
    artifact = tmp_path / "science-artifact.pkl"
    summary_csv = tmp_path / "science-summary.csv"
    save_science_ho_psf_inputs(_science_samples(), inputs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ao-predict",
            "interpolation",
            "build-science-ho-psf",
            "--inputs",
            str(inputs),
            "--output",
            str(artifact),
            "--smoothing",
            "0",
        ],
    )
    assert cli_main() == 0
    assert load_science_ho_psf_interpolator(artifact).interpolation_config.smoothing == pytest.approx(0.0)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ao-predict",
            "interpolation",
            "replay-science-ho-psf",
            "--inputs",
            str(inputs),
            "--artifact",
            str(artifact),
            "--summary-csv",
            str(summary_csv),
        ],
    )
    assert cli_main() == 0
    rows = _read_csv_rows(summary_csv)
    assert rows[0]["num_planes"] == "4"
    assert float(rows[0]["psf_nrms_max"]) == pytest.approx(0.0, abs=1.0e-5)
    assert float(rows[0]["fwhm_mas_max_abs"]) == pytest.approx(0.0, abs=1.0e-5)


def test_interpolation_cli_partial_rbf_overrides_keep_family_defaults(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = tmp_path / "science-inputs.pkl"
    artifact = tmp_path / "science-artifact.pkl"
    save_science_ho_psf_inputs(_science_samples(), inputs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ao-predict",
            "interpolation",
            "build-science-ho-psf",
            "--inputs",
            str(inputs),
            "--output",
            str(artifact),
            "--degree",
            "1",
        ],
    )
    assert cli_main() == 0
    loaded = load_science_ho_psf_interpolator(artifact)
    assert loaded.interpolation_config.degree == 1
    assert loaded.interpolation_config.smoothing == pytest.approx(0.03)


def test_interpolation_cli_builds_ngs_metric_from_psf_and_metric_inputs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    psf_inputs = tmp_path / "ngs-psf-inputs.pkl"
    metric_inputs = tmp_path / "ngs-metric-inputs.pkl"
    artifact_from_psfs = tmp_path / "ngs-from-psfs.pkl"
    artifact_from_metrics = tmp_path / "ngs-from-metrics.pkl"
    summary_csv = tmp_path / "ngs-summary.csv"
    psf_samples = _ngs_psf_samples()
    metric_samples = build_ngs_ho_metric_samples_from_psfs(psf_samples)
    save_ngs_ho_psf_inputs(psf_samples, psf_inputs)
    save_ngs_ho_metric_inputs(metric_samples, metric_inputs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ao-predict",
            "interpolation",
            "build-ngs-ho-metric-from-psfs",
            "--inputs",
            str(psf_inputs),
            "--output",
            str(artifact_from_psfs),
            "--smoothing",
            "0",
        ],
    )
    assert cli_main() == 0
    assert set(load_ngs_ho_metric_interpolator(artifact_from_psfs).metric_names) == {"ee", "fwhm_mas", "sr"}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ao-predict",
            "interpolation",
            "build-ngs-ho-metric",
            "--inputs",
            str(metric_inputs),
            "--output",
            str(artifact_from_metrics),
            "--smoothing",
            "0",
        ],
    )
    assert cli_main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ao-predict",
            "interpolation",
            "replay-ngs-ho-metric-from-psfs",
            "--inputs",
            str(psf_inputs),
            "--artifact",
            str(artifact_from_psfs),
            "--summary-csv",
            str(summary_csv),
        ],
    )
    assert cli_main() == 0
    rows = _read_csv_rows(summary_csv)
    assert rows[0]["num_planes"] == "2"
    assert float(rows[0]["ee_max_abs"]) == pytest.approx(0.0, abs=1.0e-8)
    assert float(rows[0]["sr_max_abs"]) == pytest.approx(0.0, abs=1.0e-8)


def test_interpolation_cli_rejects_malformed_input_package(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = tmp_path / "bad-inputs.pkl"
    artifact = tmp_path / "artifact.pkl"
    with inputs.open("wb") as handle:
        pickle.dump({"kind": "not_an_ao_predict_input", "version": 1}, handle)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ao-predict",
            "interpolation",
            "build-science-ho-psf",
            "--inputs",
            str(inputs),
            "--output",
            str(artifact),
        ],
    )
    with pytest.raises(ValueError, match="Unsupported artifact kind"):
        cli_main()


def test_interpolation_cli_accepts_science_input_missing_meta(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = _science_samples()
    inputs = tmp_path / "missing-meta-inputs.pkl"
    artifact = tmp_path / "artifact.pkl"
    with inputs.open("wb") as handle:
        pickle.dump(
            {
                "kind": "ao_predict_science_ho_psf_inputs",
                "version": 1,
                "samples": {
                    "zenith_angle_deg": samples.zenith_angle_deg,
                    "wavelength_um": samples.wavelength_um,
                    "x_arcsec": samples.x_arcsec,
                    "y_arcsec": samples.y_arcsec,
                    "psfs": samples.psfs,
                    "pixel_scale_mas": samples.pixel_scale_mas,
                    "tel_diameter_m": samples.tel_diameter_m,
                    "tel_pupil": samples.tel_pupil,
                    "provenance": tuple(samples.provenance),
                },
            },
            handle,
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ao-predict",
            "interpolation",
            "build-science-ho-psf",
            "--inputs",
            str(inputs),
            "--output",
            str(artifact),
        ],
    )
    cli_main()
    loaded = load_science_ho_psf_interpolator(artifact)
    assert loaded.meta == {}


def _read_csv_rows(path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
