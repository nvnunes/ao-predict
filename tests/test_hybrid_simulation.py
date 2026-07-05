from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path

import ao_predict
import numpy as np
import pytest

from ao_predict import (
    InitDatasetRequest,
    OptionsConfig,
    SetupConfig,
    SimulationConfig,
    run_simulations_by_state,
)
from ao_predict.interpolation import (
    NgsHoMetricSamples,
    RbfInterpolationConfig,
    ScienceHoPsfSamples,
    build_ngs_ho_metric_interpolator,
    build_science_ho_psf_interpolator,
    save_ngs_ho_metric_interpolator,
    save_science_ho_psf_interpolator,
)
from ao_predict.persistence import SimulationStore
from ao_predict.simulation import SimulationContext
from ao_predict.simulation import schema
from ao_predict.simulation.hybrid import (
    HybridCtotResult,
    HybridSetup,
    HybridSimulation,
    NgsMetricProviderResult,
    SciencePsfProviderResult,
    apply_ctot_blur,
    jitter_mas_from_ctot,
)
from ao_predict.simulation.runner import _populate_result_stats, create_simulation_from_config
from ao_predict.simulation.stats import PsfMetadata


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
        "extra_stat_names": np.asarray(sim.extra_stat_names, dtype=str),
    }


def _write_hybrid_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    ini_path = tmp_path / "mastsel.ini"
    ini_path.write_text(_ini_text(), encoding="utf-8")
    science_path = tmp_path / "science.pkl"
    ngs_path = tmp_path / "ngs.pkl"
    save_science_ho_psf_interpolator(
        build_science_ho_psf_interpolator(_science_samples(), interpolation_config=RbfInterpolationConfig(smoothing=0.0)),
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
    import ao_predict.simulation as simulation

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
    np.testing.assert_array_equal(payload[schema.KEY_SIMULATION_META_FIELDS], np.asarray(["norm_correction"]))
    assert Path(str(payload["science_ho_psf_interpolator_path"])).is_absolute()
    assert Path(str(payload["ngs_ho_metric_interpolator_path"])).is_absolute()

    sim.validate_simulation_payload(payload)
    sim.load_simulation_payload(payload)

    assert sim.science_ho_psf_interpolator.psf_shape == (5, 5)
    assert set(sim.ngs_ho_metric_interpolator.metric_names) == {"ee", "fwhm_mas", "sr"}


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
    setup = _setup()
    result = sim._predict_science_psfs(setup, _options())

    assert result.pixel_scale_mas == pytest.approx(4.0)
    assert result.meta["norm_correction"] == pytest.approx(0.75)
    assert isinstance(result.metadata, PsfMetadata)
    assert result.metadata.wavelength_um == pytest.approx(1.0)
    np.testing.assert_allclose(
        np.sum(result.psfs, axis=(-2, -1)),
        np.sum(_science_samples().psfs[0], axis=(-2, -1)),
        rtol=1.0e-6,
    )


def test_hybrid_subclass_can_override_science_provider(tmp_path: Path) -> None:
    class CustomHybrid(HybridSimulation):
        def _predict_science_psfs(self, setup, options):
            del setup, options
            return SciencePsfProviderResult(
                psfs=np.full((2, 5, 5), 2.0, dtype=np.float32),
                metadata=PsfMetadata(
                    wavelength_um=1.0,
                    pixel_scale_mas=7.0,
                    tel_diameter_m=9.0,
                    tel_pupil=np.ones((3, 3), dtype=np.float32),
                ),
            )

        def _predict_ngs_metrics(self, active_ngs, options):
            del active_ngs, options
            return NgsMetricProviderResult(
                ee=np.array([0.3]),
                fwhm_mas=np.array([80.0]),
                sr=np.array([0.1]),
            )

        def _compute_mastsel_ctot(self, parser, setup, active_ngs, metrics):
            del parser, setup, active_ngs, metrics
            return HybridCtotResult(
                ctot_nm2=np.zeros((2, 2, 2), dtype=float),
                ctot_mas2=np.zeros((2, 2, 2), dtype=float),
                mas2nm=2.0,
            )

    sim = CustomHybrid()
    sim.load_simulation_payload(_simulation_payload(tmp_path))
    sim.load_setup_payload(_setup_payload())
    context = sim.create(0, _options())
    assert context.runtime["effective_parser"] is not sim.base_config.parser
    sim.run(context)
    sim.finalize(context)

    assert context.result is not None
    assert context.result.meta["pixel_scale_mas"] == np.float32(7.0)
    assert context.result.meta["tel_diameter_m"] == np.float32(9.0)
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

    monkeypatch.setattr("ao_predict.simulation.hybrid._load_mavis_lo", lambda: FakeMavisLO)

    context = sim.create(0, _options())
    sim.run(context)
    sim.finalize(context)
    extra = sim.build_extra_stats(context)

    science_coords, ngs_coords, ngs_flux, ngs_frequency, sr, ee, fwhm, dl, do_all = calls["ctot"]
    parser = calls["parser"]
    assert isinstance(parser, ConfigParser)
    assert parser["telescope"]["ZenithAngle"] == "20"
    assert parser["sources_science"]["Zenith"] == "[0,1]"
    assert parser["sources_science"]["Azimuth"] == "[0,0]"
    assert parser["sources_science"]["Wavelength"] == "[1.000000e-06]"
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
    np.testing.assert_allclose(extra["jitter"], np.sqrt(np.array([0.5, 2.0])), rtol=1.0e-6)
    assert context.result is not None
    assert context.result.stats == {}


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

    monkeypatch.setattr("ao_predict.simulation.hybrid._load_mavis_lo", lambda: FakeMavisLO)

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
                ee_apertures_mas=[50.0],
                specific_fields={
                    "atm_wavelength_um": 0.5,
                    "atm_profiles": _setup_payload()["atm_profiles"],
                    "lgs_r_arcsec": [],
                    "lgs_theta_deg": [],
                    "ngs_mag_zeropoint": 3.0e10,
                    "sci_r_arcsec": [0.0, 1.0],
                    "sci_theta_deg": [0.0, 0.0],
                },
            ),
            options=OptionsConfig(
                option_arrays={
                    "wavelength_um": np.array([1.0]),
                    "zenith_angle_deg": np.array([20.0]),
                    "atm_profile_id": np.array([0]),
                    "r0_m": np.array([0.16]),
                    "ngs_r_arcsec": np.array([[0.0]]),
                    "ngs_theta_deg": np.array([[0.0]]),
                    "ngs_mag": np.array([[14.0]]),
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
    np.testing.assert_allclose(stats["jitter"], np.sqrt(np.array([0.5, 2.0])), rtol=1.0e-6)
    assert meta["norm_correction"] == pytest.approx(0.75)
    assert store.read_analysis_diagnostics() == {}
    assert store.read_simulation_diagnostics(0) == {}


def test_hybrid_source_meta_is_available_during_stats_preprocessing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMavisLO:
        def __init__(self, path2param, parameters_file, verbose=False):
            del path2param, parameters_file, verbose
            self.error = False
            self.mas2nm = 2.0

        def computeTotalResidualMatrix(self, *args, **kwargs):
            del args, kwargs
            return np.stack([np.eye(2), 4.0 * np.eye(2)])

    observed_meta: list[float] = []

    class ObservingHybrid(HybridSimulation):
        def prepare_psfs_for_stats(self, psfs, setup, meta):
            observed_meta.append(float(meta["norm_correction"]))
            return super().prepare_psfs_for_stats(psfs, setup, meta)

    monkeypatch.setattr("ao_predict.simulation.hybrid._load_mavis_lo", lambda: FakeMavisLO)
    sim = ObservingHybrid()
    sim.load_simulation_payload(_simulation_payload(tmp_path))
    sim.load_setup_payload(_setup_payload())
    context = sim.create(0, _options())
    context.runtime["extra_stat_names"] = sim.extra_stat_names

    sim.run(context)
    sim.finalize(context)
    _populate_result_stats(sim, context)

    assert observed_meta == [pytest.approx(0.75)]
    assert context.result is not None
    assert context.result.meta["norm_correction"] == pytest.approx(0.75)


def test_hybrid_validation_diagnostics_are_persisted_and_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ini_path, science_path, ngs_path = _write_hybrid_inputs(tmp_path)
    dataset_path = tmp_path / "hybrid_validation.h5"

    class FakeMavisLO:
        def __init__(self, path2param, parameters_file, verbose=False):
            del path2param, parameters_file, verbose
            self.error = False
            self.mas2nm = 2.0

        def computeTotalResidualMatrix(self, *args, **kwargs):
            del args, kwargs
            return np.stack([np.eye(2), 4.0 * np.eye(2)])

    monkeypatch.setattr("ao_predict.simulation.hybrid._load_mavis_lo", lambda: FakeMavisLO)

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
                },
            ),
            setup=SetupConfig(
                ee_apertures_mas=[50.0],
                specific_fields={
                    "atm_wavelength_um": 0.5,
                    "atm_profiles": _setup_payload()["atm_profiles"],
                    "lgs_r_arcsec": [],
                    "lgs_theta_deg": [],
                    "ngs_mag_zeropoint": 3.0e10,
                    "sci_r_arcsec": [0.0, 1.0],
                    "sci_theta_deg": [0.0, 0.0],
                },
            ),
            options=OptionsConfig(
                option_arrays={
                    "wavelength_um": np.array([1.0]),
                    "zenith_angle_deg": np.array([20.0]),
                    "atm_profile_id": np.array([0]),
                    "r0_m": np.array([0.16]),
                    "ngs_r_arcsec": np.array([[0.0, np.nan]]),
                    "ngs_theta_deg": np.array([[0.0, np.nan]]),
                    "ngs_mag": np.array([[14.0, np.nan]]),
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
    assert diagnostics["hybrid"]["mas2nm"] == pytest.approx(2.0)
    assert diagnostics["hybrid"]["psd_valid_count"] == 2
    assert diagnostics["hybrid"]["psd_valid_fraction"] == pytest.approx(1.0)
    np.testing.assert_array_equal(diagnostics["hybrid"]["psd_valid_mask"], np.array([True, True]))
    np.testing.assert_array_equal(diagnostics["hybrid"]["ngs_used"], np.array([True, False]))
    np.testing.assert_allclose(diagnostics["hybrid"]["ngs"]["ee"], np.array([0.3, np.nan]), equal_nan=True)
    assert "ctot_nm2" not in diagnostics["hybrid"]
    assert "runtime_ini_text" not in diagnostics["hybrid"]
    assert all_diagnostics["hybrid"]["mas2nm"].shape == (1,)


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

    monkeypatch.setattr("ao_predict.simulation.hybrid._load_mavis_lo", lambda: FakeMavisLO)

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
                ee_apertures_mas=[50.0],
                specific_fields={
                    "atm_wavelength_um": 0.5,
                    "atm_profiles": _setup_payload()["atm_profiles"],
                    "lgs_r_arcsec": [],
                    "lgs_theta_deg": [],
                    "ngs_mag_zeropoint": 3.0e10,
                    "sci_r_arcsec": [0.0, 1.0],
                    "sci_theta_deg": [0.0, 0.0],
                },
            ),
            options=OptionsConfig(
                option_arrays={
                    "wavelength_um": np.array([1.0]),
                    "zenith_angle_deg": np.array([20.0]),
                    "atm_profile_id": np.array([0]),
                    "r0_m": np.array([0.16]),
                    "ngs_r_arcsec": np.array([[0.0, np.nan]]),
                    "ngs_theta_deg": np.array([[0.0, np.nan]]),
                    "ngs_mag": np.array([[14.0, np.nan]]),
                }
            ),
        )
    )

    summary = run_simulations_by_state(dataset_path, num_workers=1)

    assert summary.succeeded == 1
    runtime_ini_text = SimulationStore(dataset_path).read_simulation_diagnostics(0)["hybrid"]["runtime_ini_text"]
    assert isinstance(runtime_ini_text, str)
    assert "[telescope]" in runtime_ini_text


def test_hybrid_debug_diagnostics_include_full_ctot_and_runtime_ini(tmp_path: Path) -> None:
    class CustomHybrid(HybridSimulation):
        def _predict_science_psfs(self, setup, options):
            del setup, options
            return SciencePsfProviderResult(
                psfs=np.full((2, 5, 5), 1.0, dtype=np.float32),
                metadata=PsfMetadata(
                    wavelength_um=1.0,
                    pixel_scale_mas=4.0,
                    tel_diameter_m=8.0,
                    tel_pupil=np.ones((5, 5), dtype=np.float32),
                ),
            )

        def _predict_ngs_metrics(self, active_ngs, options):
            del active_ngs, options
            return NgsMetricProviderResult(ee=np.array([0.3]), fwhm_mas=np.array([80.0]), sr=np.array([0.1]))

        def _compute_mastsel_ctot(self, parser, setup, active_ngs, metrics):
            del parser, setup, active_ngs, metrics
            ctot_nm2 = np.stack([np.eye(2), 2.0 * np.eye(2)])
            return HybridCtotResult(
                ctot_nm2=ctot_nm2,
                ctot_mas2=ctot_nm2 / 4.0,
                mas2nm=2.0,
                ngs_flux=np.array([10.0]),
                ngs_frequency=np.array([500.0]),
                runtime_ini_text="[runtime]\n",
            )

    sim = CustomHybrid()
    sim.load_simulation_payload(_simulation_payload(tmp_path, diagnostics_level="debug"))
    sim.load_setup_payload(_setup_payload())
    context = sim.create(0, _options())
    sim.run(context)
    sim.finalize(context)

    assert context.result is not None
    diagnostics = context.result.diagnostics
    assert diagnostics["hybrid/runtime_ini_text"] == "[runtime]\n"
    np.testing.assert_allclose(diagnostics["hybrid/ctot_nm2"], np.stack([np.eye(2), 2.0 * np.eye(2)]))
    np.testing.assert_allclose(diagnostics["hybrid/ctot_mas2"], np.stack([np.eye(2), 2.0 * np.eye(2)]) / 4.0)


def test_hybrid_load_rejects_changed_diagnostic_field_specs(tmp_path: Path) -> None:
    payload = _simulation_payload(tmp_path, diagnostics_level="validation")
    fields = dict(payload["diagnostic_fields"])
    fields["hybrid/mas2nm"] = {**fields["hybrid/mas2nm"], "dtype": "float64"}
    payload["diagnostic_fields"] = fields

    with pytest.raises(ValueError, match="changed specs: hybrid/mas2nm"):
        HybridSimulation().load_simulation_payload(payload)


def test_hybrid_diagnostic_extension_fields_cannot_collide(tmp_path: Path) -> None:
    class CollidingHybrid(HybridSimulation):
        def _extend_hybrid_diagnostic_field_specs(self, diagnostics_level):
            del diagnostics_level
            return {"hybrid/mas2nm": {"dtype": "float32", "shape": ()}}

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


def test_hybrid_diagnostic_extension_fields_are_appended(tmp_path: Path) -> None:
    class ExtendedHybrid(HybridSimulation):
        def _extend_hybrid_diagnostic_field_specs(self, diagnostics_level):
            del diagnostics_level
            return {"project/custom_scalar": {"dtype": "float32", "shape": ()}}

        def _extend_hybrid_diagnostics(self, diagnostics_context):
            del diagnostics_context
            return {"project/custom_scalar": np.float32(12.5)}

    class CustomHybrid(ExtendedHybrid):
        def _predict_science_psfs(self, setup, options):
            del setup, options
            return SciencePsfProviderResult(
                psfs=np.full((2, 5, 5), 1.0, dtype=np.float32),
                metadata=PsfMetadata(
                    wavelength_um=1.0,
                    pixel_scale_mas=4.0,
                    tel_diameter_m=8.0,
                    tel_pupil=np.ones((5, 5), dtype=np.float32),
                ),
            )

        def _predict_ngs_metrics(self, active_ngs, options):
            del active_ngs, options
            return NgsMetricProviderResult(ee=np.array([0.3]), fwhm_mas=np.array([80.0]), sr=np.array([0.1]))

        def _compute_mastsel_ctot(self, parser, setup, active_ngs, metrics):
            del parser, setup, active_ngs, metrics
            ctot_nm2 = np.stack([np.eye(2), 2.0 * np.eye(2)])
            return HybridCtotResult(
                ctot_nm2=ctot_nm2,
                ctot_mas2=ctot_nm2 / 4.0,
                mas2nm=2.0,
                ngs_flux=np.array([10.0]),
                ngs_frequency=np.array([500.0]),
                runtime_ini_text="[runtime]\n",
            )

    sim = CustomHybrid()
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
    assert "hybrid/mas2nm" in context.result.diagnostics


def test_ctot_blur_preserves_zero_ctot_and_flux() -> None:
    psfs = np.zeros((2, 9, 9), dtype=np.float32)
    psfs[:, 4, 4] = np.array([2.0, 5.0], dtype=np.float32)
    original = psfs.copy()

    apply_ctot_blur(psfs, np.zeros((2, 2, 2), dtype=float), pixel_scale_mas=4.0, mas2nm=2.0)
    np.testing.assert_allclose(psfs, original)

    apply_ctot_blur(psfs, np.stack([np.eye(2), 2.0 * np.eye(2)]), pixel_scale_mas=4.0, mas2nm=2.0)
    np.testing.assert_allclose(np.sum(psfs, axis=(-2, -1)), np.array([2.0, 5.0]), rtol=1.0e-6)
    assert np.all(psfs >= 0.0)


def test_hybrid_rejects_bad_ctot_and_missing_ngs() -> None:
    with pytest.raises(ValueError, match="must have shape"):
        jitter_mas_from_ctot(np.ones((2, 2)))

    sim = HybridSimulation()
    with pytest.raises(ValueError, match="at least one active NGS"):
        sim._active_ngs_from_options(
            {
                schema.KEY_OPTION_NGS_R_ARCSEC: np.array([0.0]),
                schema.KEY_OPTION_NGS_THETA_DEG: np.array([0.0]),
                schema.KEY_OPTION_NGS_MAG: np.array([14.0]),
                schema.KEY_OPTION_NGS_USED: np.array([False]),
            }
        )


def _setup_payload() -> dict[str, object]:
    return {
        "ee_apertures_mas": np.array([50.0]),
        "sr_method": "pixel_fit",
        "fwhm_summary": "geom",
        "ee_geometry": "ensquared",
        "atm_wavelength_um": 0.5,
        "atm_profiles": {
            "0": {
                "name": "default",
                "r0_m": 0.16,
                "L0_m": 25.0,
                "cn2_heights_m": np.array([0.0, 5000.0]),
                "cn2_weights": np.array([0.6, 0.4]),
                "wind_speed_mps": np.array([5.0, 10.0]),
                "wind_direction_deg": np.array([0.0, 90.0]),
            }
        },
        "lgs_r_arcsec": np.array([], dtype=float),
        "lgs_theta_deg": np.array([], dtype=float),
        "ngs_mag_zeropoint": 3.0e10,
        "sci_r_arcsec": np.array([0.0, 1.0]),
        "sci_theta_deg": np.array([0.0, 0.0]),
    }


def _setup() -> HybridSetup:
    sim = HybridSimulation()
    return sim._parse_setup_payload(_setup_payload())


def _options() -> dict[str, object]:
    return {
        "wavelength_um": 1.0,
        "zenith_angle_deg": 20.0,
        "atm_profile_id": 0,
        "r0_m": 0.16,
        "ngs_r_arcsec": np.array([0.0]),
        "ngs_theta_deg": np.array([0.0]),
        "ngs_mag": np.array([14.0]),
        "ngs_used": np.array([True]),
    }


def _science_samples() -> ScienceHoPsfSamples:
    psfs = np.zeros((1, 2, 5, 5), dtype=np.float32)
    psfs[0, 0, 2, 2] = 2.0
    psfs[0, 1, 2, 2] = 3.0
    return ScienceHoPsfSamples(
        zenith_angle_deg=np.array([20.0]),
        wavelength_um=np.array([1.0]),
        x_arcsec=np.array([0.0, 1.0]),
        y_arcsec=np.array([0.0, 0.0]),
        psfs=psfs,
        pixel_scale_mas=np.array([4.0]),
        tel_diameter_m=8.0,
        tel_pupil=np.ones((5, 5), dtype=np.float32),
        meta={"norm_correction": 0.75},
    )


def _ngs_samples() -> NgsHoMetricSamples:
    return NgsHoMetricSamples(
        zenith_angle_deg=np.array([20.0, 30.0]),
        x_arcsec=np.array([0.0, 1.0, 0.0]),
        y_arcsec=np.array([0.0, 0.0, 1.0]),
        ee=np.array([[0.3, 0.31, 0.32], [0.33, 0.34, 0.35]]),
        fwhm_mas=np.array([[80.0, 81.0, 82.0], [83.0, 84.0, 85.0]]),
        sr=np.array([[0.1, 0.11, 0.12], [0.13, 0.14, 0.15]]),
    )
