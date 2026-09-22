from __future__ import annotations

from configparser import ConfigParser
from dataclasses import replace
from pathlib import Path

import h5py
import hybrid_ao_psf
import numpy as np
import pytest
from ao_stats import PsfMetadata
from astropy import units as u
from hybrid_ao_psf import (
    HybridResult,
    NgsHoMetricSamples,
    RbfInterpolationConfig,
    ScienceHoPsfPrediction,
    ScienceHoPsfSamples,
    build_ngs_ho_metric_interpolator,
    build_science_ho_psf_interpolator,
    save_ngs_ho_metric_interpolator,
    save_science_ho_psf_interpolator,
)

import ao_predict
from ao_predict import (
    InitDatasetRequest,
    OptionsConfig,
    SetupConfig,
    SimulationConfig,
    run_simulations_by_state,
)
from ao_predict.analysis import load_analysis_dataset
from ao_predict.persistence import SimulationStore
from ao_predict.simulation import schema
from ao_predict.simulation.hybrid import (
    HybridResolvedInputs,
    HybridSetup,
    HybridSimulation,
)
from ao_predict.simulation.runner import (
    _populate_result_stats,
    create_simulation_from_config,
)


def _ini_text() -> str:
    return (
        "[telescope]\nZenithAngle=20\nTelescopeDiameter=8.0\n"
        "[atmosphere]\nWavelength=500e-9\nr0_Value=0.16\nSeeing=0.6\nL0=25.0\n"
        "Cn2Heights=[0,5000]\nCn2Weights=[0.6,0.4]\nWindSpeed=[5,10]\nWindDirection=[0,90]\n"
        "[RTC]\nSensorFrameRate_LO=500.0\n"
        "[sensor_LO]\nNumberLenslets=[16]\nNumberPhotons=[100]\n"
        "[sources_LO]\nWavelength=[710e-9]\nZenith=[0.0]\nAzimuth=[0.0]\n"
        "[sources_science]\nWavelength=[1.0e-06]\nZenith=[0.0,1.0]\nAzimuth=[0.0,0.0]\n"
    )


def _base_payload(sim: HybridSimulation) -> dict[str, object]:
    return {
        "name": sim.name,
        "version": sim.version,
        "extra_stat_fields": {name: unit.to_string() for name, unit in sim.extra_stat_fields.items()},
        "ngs_mag_standard": sim.ngs_mag_standard,
    }


def _write_hybrid_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    ini_path = tmp_path / "mastsel.ini"
    ini_path.write_text(_ini_text(), encoding="utf-8")
    science_path = tmp_path / "science.pkl"
    ngs_path = tmp_path / "ngs.pkl"
    save_science_ho_psf_interpolator(
        build_science_ho_psf_interpolator(_science_samples()),
        science_path,
    )
    save_ngs_ho_metric_interpolator(
        build_ngs_ho_metric_interpolator(_ngs_samples(), interpolation_config=RbfInterpolationConfig(smoothing=0.0)),
        ngs_path,
    )
    return ini_path, science_path, ngs_path


def _simulation_payload(tmp_path: Path, *, diagnostics_level: str | None = None) -> dict[str, object]:
    sim = HybridSimulation()
    ini_path, science_path, ngs_path = _write_hybrid_inputs(tmp_path)
    simulation_cfg = {
        "base_path": str(tmp_path),
        "config_path": ini_path.name,
        "science_ho_psf_interpolator_path": science_path.name,
        "ngs_ho_metric_interpolator_path": ngs_path.name,
    }
    if diagnostics_level is not None:
        simulation_cfg["diagnostics_level"] = diagnostics_level
    return dict(
        sim.prepare_simulation_payload(
            _base_payload(sim),
            simulation_cfg,
        )
    )


def test_hybrid_exports_and_short_name_resolution() -> None:
    from ao_predict import simulation

    assert ao_predict.HybridSimulation is HybridSimulation
    assert simulation.HybridSimulation is HybridSimulation


def test_hybrid_short_name_resolves_through_public_config_path(tmp_path: Path) -> None:
    ini_path, science_path, ngs_path = _write_hybrid_inputs(tmp_path)

    sim, payload = create_simulation_from_config(
        {
            "name": "Hybrid",
            "base_path": str(tmp_path),
            "config_path": ini_path.name,
            "science_ho_psf_interpolator_path": science_path.name,
            "ngs_ho_metric_interpolator_path": ngs_path.name,
        }
    )

    assert isinstance(sim, HybridSimulation)
    assert payload["name"] == sim.name


def test_hybrid_payload_lifecycle_resolves_and_loads_interpolators(tmp_path: Path) -> None:
    sim = HybridSimulation()
    payload = _simulation_payload(tmp_path)

    assert payload["base_config"] == _ini_text()
    assert payload["diagnostics_level"] == "none"
    assert "diagnostic_fields" not in payload
    assert schema.KEY_SIMULATION_META_FIELDS not in payload
    assert Path(str(payload["science_ho_psf_interpolator_path"])).is_absolute()
    assert Path(str(payload["ngs_ho_metric_interpolator_path"])).is_absolute()

    sim.validate_simulation_payload(payload)
    sim.load_simulation_payload(payload)

    assert sim.science_ho_psf_interpolator.psf_shape == (5, 5)
    assert set(sim.ngs_ho_metric_interpolator.metric_names) == {"ee", "fwhm", "sr"}


def test_hybrid_payload_persists_non_psd_policy_and_defaults_legacy_payload(
    tmp_path: Path,
) -> None:
    payload = _simulation_payload(tmp_path)
    assert payload[HybridSimulation.KEY_NON_PSD_POLICY] == "error"

    legacy_payload = dict(payload)
    del legacy_payload[HybridSimulation.KEY_NON_PSD_POLICY]
    sim = HybridSimulation()
    sim.load_simulation_payload(legacy_payload)
    assert sim._non_psd_policy.value == "error"

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    ini_path, _, _ = _write_hybrid_inputs(invalid_root)
    with pytest.raises(ValueError, match="non_psd_policy"):
        HybridSimulation().prepare_simulation_payload(
            _base_payload(HybridSimulation()),
            {
                "base_path": str(invalid_root),
                "config_path": ini_path.name,
                "science_ho_psf_interpolator_path": "science.pkl",
                "ngs_ho_metric_interpolator_path": "ngs.pkl",
                "non_psd_policy": "unsupported",
            },
        )


def test_hybrid_payload_canonicalizes_dimensionless_field_units(
    tmp_path: Path,
) -> None:
    sim = HybridSimulation()
    ini_path, science_path, ngs_path = _write_hybrid_inputs(tmp_path)
    science_samples = replace(
        _science_samples(),
        meta={"normalization": 0.75 * u.one},
    )
    save_science_ho_psf_interpolator(
        build_science_ho_psf_interpolator(science_samples),
        science_path,
        overwrite=True,
    )

    payload = dict(
        sim.prepare_simulation_payload(
            _base_payload(sim),
            {
                "base_path": str(tmp_path),
                "config_path": ini_path.name,
                "science_ho_psf_interpolator_path": science_path.name,
                "ngs_ho_metric_interpolator_path": ngs_path.name,
                "diagnostics_level": "validation",
            },
        )
    )

    assert payload[schema.KEY_SIMULATION_META_FIELDS] == {"normalization": "1"}
    assert payload["diagnostic_fields"]["hybrid/psd_valid_fraction"]["unit"] == "1"
    sim.validate_simulation_payload(payload)


def test_hybrid_load_simulation_payload_defers_interpolator_loading(tmp_path: Path) -> None:
    sim = HybridSimulation()
    sim.load_simulation_payload(_simulation_payload(tmp_path))

    assert sim._science_ho_psf_interpolator is None
    assert sim._ngs_ho_metric_interpolator is None

    assert sim.science_ho_psf_interpolator.psf_shape == (5, 5)
    assert sim._science_ho_psf_interpolator is sim.science_ho_psf_interpolator
    assert set(sim.ngs_ho_metric_interpolator.metric_names) == {"ee", "fwhm", "sr"}
    assert sim._ngs_ho_metric_interpolator is sim.ngs_ho_metric_interpolator


def test_hybrid_runtime_getters_reuse_process_cached_interpolators_across_instances(tmp_path: Path) -> None:
    payload = _simulation_payload(tmp_path)
    first = HybridSimulation()
    second = HybridSimulation()

    first.load_simulation_payload(payload)
    second.load_simulation_payload(payload)

    assert second.science_ho_psf_interpolator is first.science_ho_psf_interpolator
    assert second.ngs_ho_metric_interpolator is first.ngs_ho_metric_interpolator


def test_hybrid_validate_simulation_payload_does_not_rebind_interpolators(tmp_path: Path) -> None:
    sim = HybridSimulation()
    payload = _simulation_payload(tmp_path)
    sim.load_simulation_payload(payload)
    original = sim.science_ho_psf_interpolator

    invalid = dict(payload)
    invalid["science_ho_psf_interpolator_path"] = str(tmp_path / "missing.pkl")
    with pytest.raises(FileNotFoundError, match="science_ho_psf_interpolator_path"):
        sim.validate_simulation_payload(invalid)

    assert sim.science_ho_psf_interpolator is original


def test_hybrid_load_simulation_payload_failure_does_not_partially_bind(tmp_path: Path) -> None:
    sim = HybridSimulation()
    payload = _simulation_payload(tmp_path)
    payload["science_ho_psf_interpolator_path"] = str(tmp_path / "missing.pkl")

    with pytest.raises(FileNotFoundError, match="science_ho_psf_interpolator_path"):
        sim.load_simulation_payload(payload)

    with pytest.raises(TypeError, match="base config is not configured"):
        _ = sim.base_config


def test_hybrid_provider_uses_artifact_pixel_scale_and_preserves_flux(tmp_path: Path) -> None:
    sim = HybridSimulation()
    payload = _simulation_payload(tmp_path)
    sim.load_simulation_payload(payload)
    sim.load_setup_payload(_setup_payload())
    resolved = sim._resolve_hybrid_inputs(sim.create(0, _options()))
    result = resolved.science_provider.predict(
        zenith_angle=resolved.request.zenith_angle,
        wavelength=resolved.request.wavelength,
        x=resolved.request.science_x,
        y=resolved.request.science_y,
    )

    assert result.pixel_scale.to_value(u.mas) == pytest.approx(4.0)
    assert result.meta == {}
    assert isinstance(result.metadata, PsfMetadata)
    assert result.metadata.wavelength.to_value(u.um) == pytest.approx(1.0)
    np.testing.assert_allclose(
        np.sum(result.psfs, axis=(-2, -1)),
        np.sum(_science_samples().psfs[0], axis=(-2, -1)),
        rtol=1.0e-6,
    )


def test_hybrid_create_applies_science_offsets_to_runtime_setup(tmp_path: Path) -> None:
    sim = HybridSimulation()
    payload = _simulation_payload(tmp_path)
    payload["base_config"] = str(payload["base_config"]).replace(
        "ZenithAngle=20", "ZenithAngle=19"
    )
    sim.load_simulation_payload(payload)
    sim.load_setup_payload(_setup_payload())
    options = {
        **_options(),
        schema.KEY_OPTION_SCI_DX: np.array([0.25, -0.25], dtype=np.float32) * u.arcsec,
        schema.KEY_OPTION_SCI_DY: np.array([0.5, 0.5], dtype=np.float32) * u.arcsec,
    }

    context = sim.create(0, options)

    assert context.setup is sim.setup
    np.testing.assert_allclose(context.setup.sci_r.to_value(u.arcsec), np.array([0.0, 1.0]))
    np.testing.assert_allclose(context.setup.sci_theta.to_value(u.deg), np.array([0.0, 0.0]))
    np.testing.assert_allclose(
        context.resolved_sci_r.to_value(u.arcsec),
        np.hypot([0.25, 0.75], [0.5, 0.5]),
    )
    np.testing.assert_allclose(
        context.resolved_sci_theta.to_value(u.deg),
        np.mod(np.rad2deg(np.arctan2([0.5, 0.5], [0.25, 0.75])), 360.0),
    )
    request = sim._resolve_hybrid_inputs(context).request
    np.testing.assert_allclose(
        request.science_x.to_value(u.arcsec),
        np.array([0.25, 0.75]),
    )
    np.testing.assert_allclose(
        request.science_y.to_value(u.arcsec),
        np.array([0.5, 0.5]),
    )
    pre_engine_ini = ConfigParser()
    pre_engine_ini.read_string(request.mastsel_ini)
    assert pre_engine_ini["sources_science"]["Zenith"] == "[0.0,1.0]"
    assert pre_engine_ini["sources_LO"]["Zenith"] == "[0.0]"
    assert pre_engine_ini["sensor_LO"]["NumberLenslets"] == "[16]"
    assert pre_engine_ini["telescope"]["ZenithAngle"] == "19"
    assert request.zenith_angle.to_value(u.deg) == pytest.approx(20.0)


def test_hybrid_subclass_can_replace_resolved_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    class ScienceProvider:
        def predict(self, *, zenith_angle, wavelength, x, y):
            del zenith_angle
            resolved_fields["science"] = (
                x.to_value(u.arcsec),
                y.to_value(u.arcsec),
            )
            return ScienceHoPsfPrediction(
                psfs=np.full((2, 5, 5), 2.0, dtype=np.float32),
                pixel_scale=7.0 * u.mas,
                metadata=PsfMetadata(
                    wavelength=wavelength,
                    pixel_scale=7.0 * u.mas,
                    tel_diameter=9.0 * u.m,
                    tel_pupil=np.ones((3, 3), dtype=np.float32) * u.one,
                ),
            )

    class CustomHybrid(HybridSimulation):
        def _resolve_hybrid_inputs(self, context):
            resolved = super()._resolve_hybrid_inputs(context)
            resolved_fields["request"] = (
                resolved.request.science_x.to_value(u.arcsec),
                resolved.request.science_y.to_value(u.arcsec),
            )
            return HybridResolvedInputs(
                request=resolved.request,
                science_provider=ScienceProvider(),
                ngs_provider=resolved.ngs_provider,
                ngs_used=resolved.ngs_used,
            )

    monkeypatch.setattr(
        "hybrid_ao_psf.engine._load_mavis_lo",
        lambda: _fake_mavis_lo(np.zeros((2, 2, 2), dtype=float)),
    )

    sim = CustomHybrid()
    sim.load_simulation_payload(_simulation_payload(tmp_path))
    sim.load_setup_payload(_setup_payload())
    options = {
        **_options(),
        schema.KEY_OPTION_SCI_DX: np.array([0.25, -0.25], dtype=np.float32) * u.arcsec,
        schema.KEY_OPTION_SCI_DY: np.array([0.5, 0.5], dtype=np.float32) * u.arcsec,
    }
    context = sim.create(0, options)
    assert context.runtime["effective_parser"] is not sim.base_config.parser
    sim.run(context)
    sim.finalize(context)

    expected_x = np.array([0.25, 0.75])
    expected_y = np.array([0.5, 0.5])
    assert set(resolved_fields) == {"science", "request"}
    for resolved_x, resolved_y in resolved_fields.values():
        np.testing.assert_allclose(resolved_x, expected_x)
        np.testing.assert_allclose(resolved_y, expected_y)
    assert context.result is not None
    assert context.result.meta["pixel_scale"].to_value(u.mas) == np.float32(7.0)
    assert context.result.meta["tel_diameter"].to_value(u.m) == np.float32(9.0)
    np.testing.assert_allclose(context.result.psfs, np.full((2, 5, 5), 2.0, dtype=np.float32))


def test_hybrid_run_calls_mastsel_with_metrics_and_converts_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sim = HybridSimulation()
    sim.load_simulation_payload(_simulation_payload(tmp_path))
    sim.load_setup_payload(_setup_payload())
    calls: dict[str, object] = {}

    class FakeMavisLO:
        def __init__(self, path2param, parameters_file, verbose=False):
            calls["init"] = (path2param, parameters_file, verbose)
            parser = ConfigParser()
            parser.optionxform = str
            parser.read(Path(path2param) / f"{parameters_file}.ini")
            calls["parser"] = parser
            self.error = False
            self.mas2nm = 2.0

        def computeTotalResidualMatrix(
            self,
            science_coords,
            ngs_coords,
            ngs_flux,
            ngs_frequency,
            star_sr,
            star_ee,
            star_fwhm,
            *,
            aNGS_FWHM_DL_mas=None,
            doAll=True,
        ):
            calls["ctot"] = (
                science_coords.copy(),
                ngs_coords.copy(),
                ngs_flux.copy(),
                ngs_frequency.copy(),
                star_sr.copy(),
                star_ee.copy(),
                star_fwhm.copy(),
                aNGS_FWHM_DL_mas,
                doAll,
            )
            return np.stack([np.eye(2), 4.0 * np.eye(2)])

    monkeypatch.setattr("hybrid_ao_psf.engine._load_mavis_lo", lambda: FakeMavisLO)

    context = sim.create(0, _options())
    sim.run(context)
    sim.finalize(context)
    extra = sim.build_extra_stats(context)

    science_coords, ngs_coords, ngs_flux, ngs_frequency, sr, ee, fwhm, dl, do_all = calls["ctot"]
    parser = calls["parser"]
    assert isinstance(parser, ConfigParser)
    assert parser["telescope"]["ZenithAngle"] == "20"
    np.testing.assert_allclose(
        np.fromstring(parser["sources_science"]["Zenith"].strip("[]"), sep=","),
        np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(
        np.fromstring(parser["sources_science"]["Azimuth"].strip("[]"), sep=","),
        np.array([0.0, 0.0]),
    )
    np.testing.assert_allclose(
        np.fromstring(parser["sources_science"]["Wavelength"].strip("[]"), sep=","),
        np.array([1.0e-6]),
    )
    assert parser["sources_LO"]["Zenith"] == "[0]"
    assert parser["sources_LO"]["Azimuth"] == "[0]"
    assert parser["sensor_LO"]["NumberLenslets"] == "[16]"
    assert parser["atmosphere"]["r0_Value"] == "0.16"
    assert "Seeing" not in parser["atmosphere"]
    assert parser["atmosphere"]["Cn2Heights"] == "[0,5000]"
    assert parser["atmosphere"]["Cn2Weights"] == "[0.6,0.4]"
    assert parser["atmosphere"]["WindSpeed"] == "[5,10]"
    assert parser["atmosphere"]["WindDirection"] == "[0,90]"
    np.testing.assert_allclose(science_coords, np.array([[0.0, 0.0], [1.0, 0.0]]), atol=1.0e-12)
    np.testing.assert_allclose(ngs_coords, np.array([[0.0, 0.0]]), atol=1.0e-12)
    assert np.all(ngs_flux > 0.0)
    np.testing.assert_allclose(ngs_frequency, np.array([500.0]))
    np.testing.assert_allclose(sr, np.array([0.1]))
    np.testing.assert_allclose(ee, np.array([0.3]))
    np.testing.assert_allclose(fwhm, np.array([80.0]))
    assert dl is None
    assert do_all is True
    np.testing.assert_allclose(
        extra["jitter"].to_value(u.mas),
        np.sqrt(np.array([0.5, 2.0])),
        rtol=1.0e-6,
    )
    assert context.result is not None
    assert context.result.stats == {}


def test_hybrid_run_calls_upstream_engine_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "hybrid_ao_psf.engine._load_mavis_lo",
        lambda: _fake_mavis_lo(np.zeros((2, 2, 2), dtype=float)),
    )
    calls: list[tuple[object, object, object]] = []

    def observed_simulate(request, science_provider, ngs_provider):
        calls.append((request, science_provider, ngs_provider))
        return hybrid_ao_psf.simulate(request, science_provider, ngs_provider)

    monkeypatch.setattr("ao_predict.simulation.hybrid.simulate", observed_simulate)
    sim = HybridSimulation()
    sim.load_simulation_payload(_simulation_payload(tmp_path))
    sim.load_setup_payload(_setup_payload())
    context = sim.create(0, _options())

    sim.run(context)

    assert len(calls) == 1
    request, science_provider, ngs_provider = calls[0]
    assert request.non_psd_policy.value == "error"
    assert request.diagnostics_level.value == "none"
    assert science_provider is sim.science_ho_psf_interpolator
    assert ngs_provider is sim.ngs_ho_metric_interpolator
    assert isinstance(context.runtime[HybridSimulation.KEY_RUNTIME_RESULT], HybridResult)


def test_hybrid_adapter_runs_real_mastsel(tmp_path: Path) -> None:
    pytest.importorskip("mastsel")
    ini_path, science_path, ngs_path = _write_hybrid_inputs(tmp_path)
    ini_path.write_text(
        _ini_text()
        .replace("TelescopeDiameter=8.0", "TelescopeDiameter=8.0\nTechnicalFoV=121")
        .replace(
            "[RTC]\n",
            "[DM]\nDmHeights=[0.0]\n[RTC]\n"
            "LoopGain_HO=0.5\nSensorFrameRate_HO=500.0\n"
            "LoopDelaySteps_HO=2\nLoopDelaySteps_LO=2\n",
        )
        .replace(
            "NumberLenslets=[16]\n",
            "NumberLenslets=[16]\nPixelScale=278.1\n"
            "WindowRadiusWCoG=2.75\nSigmaRON=0.5\n"
            "ExcessNoiseFactor=1.0\nDark=0.0\nSkyBackground=0.0\n"
            "ThresholdWCoG=0.5\nNewValueThrPix=0.0\n",
        ),
        encoding="utf-8",
    )
    sim = HybridSimulation()
    sim.load_simulation_payload(
        sim.prepare_simulation_payload(
            _base_payload(sim),
            {
                "base_path": str(tmp_path),
                "config_path": ini_path.name,
                "science_ho_psf_interpolator_path": science_path.name,
                "ngs_ho_metric_interpolator_path": ngs_path.name,
            },
        )
    )
    sim.load_setup_payload(_setup_payload())
    context = sim.create(0, _options())

    sim.run(context)
    sim.finalize(context)

    assert context.result is not None
    assert context.result.psfs.shape == (2, 5, 5)
    assert np.all(np.isfinite(context.result.psfs))
    assert np.all(sim.build_extra_stats(context)["jitter"] >= 0 * u.mas)


def test_hybrid_run_persists_jitter_through_public_dataset_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ini_path, science_path, ngs_path = _write_hybrid_inputs(tmp_path)
    dataset_path = tmp_path / "hybrid.h5"

    class FakeMavisLO:
        def __init__(self, path2param, parameters_file, verbose=False):
            del path2param, parameters_file, verbose
            self.error = False
            self.mas2nm = 2.0

        def computeTotalResidualMatrix(self, *args, **kwargs):
            del args, kwargs
            return np.stack([np.eye(2), 4.0 * np.eye(2)])

    monkeypatch.setattr("hybrid_ao_psf.engine._load_mavis_lo", lambda: FakeMavisLO)

    ao_predict.init_dataset(
        InitDatasetRequest(
            dataset_path=dataset_path,
            simulation=SimulationConfig(
                name="Hybrid",
                base_path=str(tmp_path),
                specific_fields={
                    "config_path": ini_path.name,
                    "science_ho_psf_interpolator_path": science_path.name,
                    "ngs_ho_metric_interpolator_path": ngs_path.name,
                },
            ),
            setup=SetupConfig(
                ee_apertures=np.array([50.0]) * u.mas,
                specific_fields={
                    "atm_wavelength": 0.5 * u.um,
                    "atm_profiles": _setup_payload()["atm_profiles"],
                    "lgs_r": np.array([]) * u.arcsec,
                    "lgs_theta": np.array([]) * u.deg,
                    "ngs_magnitude_zeropoint": 3.0e10 * u.photon / u.s,
                    "sci_r": np.array([0.0, 1.0]) * u.arcsec,
                    "sci_theta": np.array([0.0, 0.0]) * u.deg,
                },
            ),
            options=OptionsConfig(
                option_arrays={
                    "wavelength": np.array([1.0]) * u.um,
                    "zenith_angle": np.array([20.0]) * u.deg,
                    "atm_profile_id": np.array([0]),
                    "r0": np.array([0.16]) * u.m,
                    "ngs_r": np.array([[0.0]]) * u.arcsec,
                    "ngs_theta": np.array([[0.0]]) * u.deg,
                    "ngs_magnitude": np.array([[14.0]]) * u.mag,
                }
            ),
            save_psfs=True,
        )
    )

    summary = run_simulations_by_state(dataset_path, num_workers=1)

    assert summary.succeeded == 1
    store = SimulationStore(dataset_path)
    stats = store.read_simulation_stats(0)
    meta = store.read_simulation_meta(0)
    np.testing.assert_allclose(
        stats["jitter"].to_value(u.mas),
        np.sqrt(np.array([0.5, 2.0])),
        rtol=1.0e-6,
    )
    assert "norm_correction" not in meta
    assert set(meta) == {"pixel_scale", "tel_diameter", "tel_pupil"}
    assert store.read_analysis_diagnostics() == {}
    assert store.read_simulation_diagnostics(0) == {}

    # Model a completed pre-extraction dataset with no new policy field and
    # references to interpolation files no longer present on disk.
    with h5py.File(dataset_path, "r+") as handle:
        del handle["simulation"][HybridSimulation.KEY_NON_PSD_POLICY]
    science_path.unlink()
    ngs_path.unlink()
    historical = load_analysis_dataset(dataset_path)
    assert historical.simulation_payload["name"] == (
        "ao_predict.simulation.hybrid:HybridSimulation"
    )
    assert historical.simulation_payload["version"] == "0.0.1"
    assert HybridSimulation.KEY_NON_PSD_POLICY not in historical.simulation_payload
    np.testing.assert_allclose(
        historical.sim(0).stats["jitter"].to_value(u.mas),
        stats["jitter"].to_value(u.mas),
    )
    assert historical.sim(0).psfs.shape == (2, 5, 5)


def test_hybrid_stats_preprocessing_receives_psf_metadata_without_source_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeMavisLO:
        def __init__(self, path2param, parameters_file, verbose=False):
            del path2param, parameters_file, verbose
            self.error = False
            self.mas2nm = 2.0

        def computeTotalResidualMatrix(self, *args, **kwargs):
            del args, kwargs
            return np.stack([np.eye(2), 4.0 * np.eye(2)])

    observed_meta: list[tuple[str, ...]] = []

    class ObservingHybrid(HybridSimulation):
        def prepare_psfs_for_stats(self, psfs, setup, meta):
            observed_meta.append(tuple(sorted(meta)))
            return super().prepare_psfs_for_stats(psfs, setup, meta)

    monkeypatch.setattr("hybrid_ao_psf.engine._load_mavis_lo", lambda: FakeMavisLO)
    sim = ObservingHybrid()
    sim.load_simulation_payload(_simulation_payload(tmp_path))
    sim.load_setup_payload(_setup_payload())
    context = sim.create(0, _options())
    context.runtime["extra_stat_fields"] = dict(sim.extra_stat_fields)

    sim.run(context)
    sim.finalize(context)
    _populate_result_stats(sim, context)

    assert observed_meta == [("pixel_scale", "tel_diameter", "tel_pupil")]
    assert context.result is not None
    assert "norm_correction" not in context.result.meta
    assert set(context.result.meta) == {"pixel_scale", "tel_diameter", "tel_pupil"}


@pytest.mark.parametrize("non_psd_policy", ["error", "clip"])
def test_hybrid_validation_diagnostics_are_persisted_and_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    non_psd_policy: str,
) -> None:
    ini_path, science_path, ngs_path = _write_hybrid_inputs(tmp_path)
    dataset_path = tmp_path / "hybrid_validation.h5"

    class FakeMavisLO:
        def __init__(self, path2param, parameters_file, verbose=False):
            del path2param, parameters_file, verbose
            self.error = False
            self.mas2nm = 2.0

        def computeTotalResidualMatrix(self, *args, **kwargs):
            del args, kwargs
            second = 4.0 * np.eye(2)
            if non_psd_policy == "clip":
                second[1, 1] = -1.0
            return np.stack([np.eye(2), second])

    monkeypatch.setattr("hybrid_ao_psf.engine._load_mavis_lo", lambda: FakeMavisLO)

    ao_predict.init_dataset(
        InitDatasetRequest(
            dataset_path=dataset_path,
            simulation=SimulationConfig(
                name="Hybrid",
                base_path=str(tmp_path),
                specific_fields={
                    "config_path": ini_path.name,
                    "science_ho_psf_interpolator_path": science_path.name,
                    "ngs_ho_metric_interpolator_path": ngs_path.name,
                    "diagnostics_level": "validation",
                    "non_psd_policy": non_psd_policy,
                },
            ),
            setup=SetupConfig(
                ee_apertures=np.array([50.0]) * u.mas,
                specific_fields={
                    "atm_wavelength": 0.5 * u.um,
                    "atm_profiles": _setup_payload()["atm_profiles"],
                    "lgs_r": np.array([]) * u.arcsec,
                    "lgs_theta": np.array([]) * u.deg,
                    "ngs_magnitude_zeropoint": 3.0e10 * u.photon / u.s,
                    "sci_r": np.array([0.0, 1.0]) * u.arcsec,
                    "sci_theta": np.array([0.0, 0.0]) * u.deg,
                },
            ),
            options=OptionsConfig(
                option_arrays={
                    "wavelength": np.array([1.0]) * u.um,
                    "zenith_angle": np.array([20.0]) * u.deg,
                    "atm_profile_id": np.array([0]),
                    "r0": np.array([0.16]) * u.m,
                    "ngs_r": np.array([[0.0, np.nan]]) * u.arcsec,
                    "ngs_theta": np.array([[0.0, np.nan]]) * u.deg,
                    "ngs_magnitude": np.array([[14.0, np.nan]]) * u.mag,
                }
            ),
        )
    )

    store = SimulationStore(dataset_path)
    summary = run_simulations_by_state(dataset_path, num_workers=1)

    assert summary.succeeded == 1
    store.validate_schema()
    diagnostics = store.read_simulation_diagnostics(0)
    all_diagnostics = store.read_analysis_diagnostics()
    assert "diagnostics" not in store.read_simulation_stats(0)
    assert diagnostics["hybrid"]["angle_to_wavefront_scale"].to_value(
        u.nm / u.mas
    ) == pytest.approx(2.0)
    assert diagnostics["hybrid"]["psd_valid_count"] == 2
    assert diagnostics["hybrid"]["psd_valid_fraction"].to_value(u.one) == pytest.approx(1.0)
    np.testing.assert_array_equal(diagnostics["hybrid"]["psd_valid_mask"], np.array([True, True]))
    np.testing.assert_array_equal(
        diagnostics["hybrid"]["psd_clipped"],
        np.array([False, non_psd_policy == "clip"]),
    )
    np.testing.assert_array_equal(diagnostics["hybrid"]["ngs_used"], np.array([True, False]))
    np.testing.assert_allclose(
        diagnostics["hybrid"]["ngs"]["ee"].to_value(u.one),
        np.array([0.3, np.nan]),
        equal_nan=True,
    )
    assert "ctot_wavefront" not in diagnostics["hybrid"]
    assert "runtime_ini_text" not in diagnostics["hybrid"]
    assert all_diagnostics["hybrid"]["angle_to_wavefront_scale"].shape == (1,)


def test_hybrid_debug_string_diagnostics_read_as_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ini_path, science_path, ngs_path = _write_hybrid_inputs(tmp_path)
    dataset_path = tmp_path / "hybrid_debug.h5"

    class FakeMavisLO:
        def __init__(self, path2param, parameters_file, verbose=False):
            del path2param, parameters_file, verbose
            self.error = False
            self.mas2nm = 2.0

        def computeTotalResidualMatrix(self, *args, **kwargs):
            del args, kwargs
            return np.stack([np.eye(2), 4.0 * np.eye(2)])

    monkeypatch.setattr("hybrid_ao_psf.engine._load_mavis_lo", lambda: FakeMavisLO)

    ao_predict.init_dataset(
        InitDatasetRequest(
            dataset_path=dataset_path,
            simulation=SimulationConfig(
                name="Hybrid",
                base_path=str(tmp_path),
                specific_fields={
                    "config_path": ini_path.name,
                    "science_ho_psf_interpolator_path": science_path.name,
                    "ngs_ho_metric_interpolator_path": ngs_path.name,
                    "diagnostics_level": "debug",
                },
            ),
            setup=SetupConfig(
                ee_apertures=np.array([50.0]) * u.mas,
                specific_fields={
                    "atm_wavelength": 0.5 * u.um,
                    "atm_profiles": _setup_payload()["atm_profiles"],
                    "lgs_r": np.array([]) * u.arcsec,
                    "lgs_theta": np.array([]) * u.deg,
                    "ngs_magnitude_zeropoint": 3.0e10 * u.photon / u.s,
                    "sci_r": np.array([0.0, 1.0]) * u.arcsec,
                    "sci_theta": np.array([0.0, 0.0]) * u.deg,
                },
            ),
            options=OptionsConfig(
                option_arrays={
                    "wavelength": np.array([1.0]) * u.um,
                    "zenith_angle": np.array([20.0]) * u.deg,
                    "atm_profile_id": np.array([0]),
                    "r0": np.array([0.16]) * u.m,
                    "ngs_r": np.array([[0.0, np.nan]]) * u.arcsec,
                    "ngs_theta": np.array([[0.0, np.nan]]) * u.deg,
                    "ngs_magnitude": np.array([[14.0, np.nan]]) * u.mag,
                }
            ),
        )
    )

    summary = run_simulations_by_state(dataset_path, num_workers=1)

    assert summary.succeeded == 1
    runtime_ini_text = SimulationStore(dataset_path).read_simulation_diagnostics(0)["hybrid"]["runtime_ini_text"]
    assert isinstance(runtime_ini_text, str)
    assert "[telescope]" in runtime_ini_text


def test_hybrid_debug_diagnostics_include_full_ctot_and_runtime_ini(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctot_nm2 = np.stack([np.eye(2), 2.0 * np.eye(2)])
    monkeypatch.setattr(
        "hybrid_ao_psf.engine._load_mavis_lo",
        lambda: _fake_mavis_lo(ctot_nm2),
    )
    sim = HybridSimulation()
    sim.load_simulation_payload(_simulation_payload(tmp_path, diagnostics_level="debug"))
    sim.load_setup_payload(_setup_payload())
    context = sim.create(0, _options())
    sim.run(context)
    sim.finalize(context)

    assert context.result is not None
    diagnostics = context.result.diagnostics
    assert "[telescope]" in diagnostics["hybrid/runtime_ini_text"]
    np.testing.assert_allclose(
        diagnostics["hybrid/ctot_wavefront"].to_value(u.nm**2),
        np.stack([np.eye(2), 2.0 * np.eye(2)]),
    )
    np.testing.assert_allclose(
        diagnostics["hybrid/ctot_angle"].to_value(u.mas**2),
        np.stack([np.eye(2), 2.0 * np.eye(2)]) / 4.0,
    )


def test_hybrid_load_rejects_changed_diagnostic_field_specs(tmp_path: Path) -> None:
    payload = _simulation_payload(tmp_path, diagnostics_level="validation")
    fields = dict(payload["diagnostic_fields"])
    fields["hybrid/angle_to_wavefront_scale"] = {
        **fields["hybrid/angle_to_wavefront_scale"],
        "dtype": "float64",
    }
    payload["diagnostic_fields"] = fields

    with pytest.raises(
        ValueError,
        match="changed specs: hybrid/angle_to_wavefront_scale",
    ):
        HybridSimulation().load_simulation_payload(payload)


def test_hybrid_diagnostic_extension_fields_cannot_collide(tmp_path: Path) -> None:
    class CollidingHybrid(HybridSimulation):
        def _extend_hybrid_diagnostic_field_specs(self, diagnostics_level):
            del diagnostics_level
            return {
                "hybrid/angle_to_wavefront_scale": {
                    "dtype": "float32",
                    "shape": (),
                }
            }

    sim = CollidingHybrid()
    with pytest.raises(ValueError, match="collide"):
        sim.prepare_simulation_payload(
            _base_payload(sim),
            {
                "base_path": str(tmp_path),
                "config_path": _write_hybrid_inputs(tmp_path)[0].name,
                "science_ho_psf_interpolator_path": "science.pkl",
                "ngs_ho_metric_interpolator_path": "ngs.pkl",
                "diagnostics_level": "validation",
            },
        )


def test_hybrid_diagnostic_extension_fields_are_appended(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExtendedHybrid(HybridSimulation):
        def _extend_hybrid_diagnostic_field_specs(self, diagnostics_level):
            del diagnostics_level
            return {"project/custom_scalar": {"dtype": "float32", "shape": ()}}

        def _extend_hybrid_diagnostics(self, diagnostics_context):
            del diagnostics_context
            return {"project/custom_scalar": np.float32(12.5)}

    monkeypatch.setattr(
        "hybrid_ao_psf.engine._load_mavis_lo",
        lambda: _fake_mavis_lo(np.stack([np.eye(2), 2.0 * np.eye(2)])),
    )
    sim = ExtendedHybrid()
    payload = sim.prepare_simulation_payload(
        _base_payload(sim),
        {
            "base_path": str(tmp_path),
            "config_path": _write_hybrid_inputs(tmp_path)[0].name,
            "science_ho_psf_interpolator_path": "science.pkl",
            "ngs_ho_metric_interpolator_path": "ngs.pkl",
            "diagnostics_level": "validation",
        },
    )
    assert "project/custom_scalar" in payload["diagnostic_fields"]
    sim.load_simulation_payload(payload)
    sim.load_setup_payload(_setup_payload())
    context = sim.create(0, _options())
    sim.run(context)
    sim.finalize(context)

    assert context.result is not None
    assert context.result.diagnostics["project/custom_scalar"] == pytest.approx(12.5)
    assert "hybrid/angle_to_wavefront_scale" in context.result.diagnostics


def test_hybrid_rejects_missing_ngs() -> None:
    sim = HybridSimulation()
    with pytest.raises(ValueError, match="at least one active NGS"):
        sim._active_ngs_from_options(
            {
                schema.KEY_OPTION_NGS_R: np.array([0.0]) * u.arcsec,
                schema.KEY_OPTION_NGS_THETA: np.array([0.0]) * u.deg,
                schema.KEY_OPTION_NGS_MAGNITUDE: np.array([14.0]) * u.mag,
                schema.KEY_OPTION_NGS_USED: np.array([False]),
            }
        )


def _fake_mavis_lo(ctot_nm2: np.ndarray) -> type:
    class FakeMavisLO:
        def __init__(self, path2param, parameters_file, verbose=False):
            del path2param, parameters_file, verbose
            self.error = False
            self.mas2nm = 2.0

        def computeTotalResidualMatrix(self, *args, **kwargs):
            del args, kwargs
            return np.asarray(ctot_nm2, dtype=float)

    return FakeMavisLO


def _setup_payload() -> dict[str, object]:
    return {
        "ee_apertures": np.array([50.0]) * u.mas,
        "peak_method": "gaussian_fit",
        "fwhm_summary": "geom",
        "ee_geometry": "ensquared",
        "atm_wavelength": 0.5 * u.um,
        "atm_profiles": {
            "0": {
                "name": "default",
                "r0": 0.16 * u.m,
                "L0": 25.0 * u.m,
                "cn2_heights": np.array([0.0, 5000.0]) * u.m,
                "cn2_weights": np.array([0.6, 0.4]) * u.one,
                "wind_speed": np.array([5.0, 10.0]) * u.m / u.s,
                "wind_direction": np.array([0.0, 90.0]) * u.deg,
            }
        },
        "lgs_r": np.array([], dtype=float) * u.arcsec,
        "lgs_theta": np.array([], dtype=float) * u.deg,
        "ngs_magnitude_zeropoint": 3.0e10 * u.photon / u.s,
        "sci_r": np.array([0.0, 1.0]) * u.arcsec,
        "sci_theta": np.array([0.0, 0.0]) * u.deg,
    }


def _setup() -> HybridSetup:
    sim = HybridSimulation()
    return sim._parse_setup_payload(_setup_payload())


def _options() -> dict[str, object]:
    return {
        "wavelength": 1.0 * u.um,
        "zenith_angle": 20.0 * u.deg,
        "atm_profile_id": 0,
        "r0": 0.16 * u.m,
        "ngs_r": np.array([0.0]) * u.arcsec,
        "ngs_theta": np.array([0.0]) * u.deg,
        "ngs_magnitude": np.array([14.0]) * u.mag,
        "ngs_used": np.array([True]),
    }


def _science_samples() -> ScienceHoPsfSamples:
    psfs = np.zeros((1, 2, 5, 5), dtype=np.float32)
    psfs[0, 0, 2, 2] = 0.5
    psfs[0, 1, 2, 2] = 0.75
    return ScienceHoPsfSamples(
        zenith_angle=np.array([20.0]) * u.deg,
        wavelength=np.array([1.0]) * u.um,
        x=np.array([0.0, 1.0]) * u.arcsec,
        y=np.array([0.0, 0.0]) * u.arcsec,
        psfs=psfs,
        pixel_scale=np.array([4.0]) * u.mas,
        tel_diameter=8.0 * u.m,
        tel_pupil=np.ones((5, 5), dtype=np.float32) * u.one,
    )


def _ngs_samples() -> NgsHoMetricSamples:
    return NgsHoMetricSamples(
        zenith_angle=np.array([20.0, 30.0]) * u.deg,
        x=np.array([0.0, 1.0, 0.0]) * u.arcsec,
        y=np.array([0.0, 0.0, 1.0]) * u.arcsec,
        ee=np.array([[0.3, 0.31, 0.32], [0.33, 0.34, 0.35]]) * u.one,
        fwhm=np.array([[80.0, 81.0, 82.0], [83.0, 84.0, 85.0]]) * u.mas,
        sr=np.array([[0.1, 0.11, 0.12], [0.13, 0.14, 0.15]]) * u.one,
    )
