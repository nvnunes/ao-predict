from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Mapping
from pathlib import Path

import contourpy
import h5py
import numpy as np
import pytest

from ao_predict.persistence import SimulationStore
from ao_predict.simulation.helpers import normalize_psf_pixel_sum
from ao_predict.simulation import (
    Simulation,
    SimulationContext,
    SimulationResult,
    SimulationSetup,
    SimulationState,
    schema,
)
from ao_predict.simulation.runner import _populate_result_stats
from ao_predict.simulation.runner import (
    RunSummary,
    create_simulation_from_config,
    create_simulation_from_payload,
    run_pending_simulations,
    run_simulations_by_state,
)
import ao_predict.simulation.stats as stats_module
from ao_predict.simulation.stats import PsfMetadata, compute_psf_ee, compute_psf_fwhm, compute_psf_sr, compute_psf_stats
from ao_predict.simulation.validation import validate_successful_result
from helpers import run_pending_with_callback
from mock_simulation import (
    ExtraStatsMockSimulation,
    FailingWarmupMockSimulation,
    FailOnceMockSimulation,
    MockSimulation,
    WarmupMockSimulation,
)

_GIRMOS_AOSTATS = None


def _simulation(*, extra_stat_names: tuple[str, ...] = (), meta_field_names: tuple[str, ...] = ()) -> dict:
    return {
        "name": "ao_predict.simulation.tiptop:TiptopSimulation",
        "version": "x.y",
        "extra_stat_names": np.asarray(extra_stat_names, dtype=str),
        schema.KEY_SIMULATION_NGS_MAG_STANDARD: schema.DEFAULT_NGS_MAG_STANDARD,
        **(
            {schema.KEY_SIMULATION_META_FIELDS: np.asarray(meta_field_names, dtype=str)}
            if meta_field_names
            else {}
        ),
        "base_config": "[section]\nvalue=1\n",
    }


def _mock_simulation(
    simulation_cls: type[MockSimulation] = MockSimulation,
    *,
    extra_stat_names: tuple[str, ...] = (),
    specific_fields: Mapping[str, object] | None = None,
) -> dict:
    payload = {
        "name": f"{simulation_cls.__module__}:{simulation_cls.__name__}",
        "version": simulation_cls._VERSION,
        "extra_stat_names": np.asarray(extra_stat_names, dtype=str),
        schema.KEY_SIMULATION_NGS_MAG_STANDARD: simulation_cls().ngs_mag_standard,
    }
    payload.update(dict(specific_fields or {}))
    return payload


def _setup() -> dict:
    return {
        "ee_apertures_mas": np.array([50.0, 100.0], dtype=float),
        "sr_method": schema.DEFAULT_SETUP_SR_METHOD,
        "fwhm_summary": schema.DEFAULT_SETUP_FWHM_SUMMARY,
        "ee_geometry": schema.DEFAULT_SETUP_EE_GEOMETRY,
        "atm_wavelength_um": 0.5,
        "ngs_mag_zeropoint": 3.0e10,
        "sci_r_arcsec": np.array([0.0, 10.0, 20.0], dtype=float),
        "sci_theta_deg": np.array([0.0, 90.0, 180.0], dtype=float),
        "lgs_r_arcsec": np.array([30.0, 30.0, 30.0, 30.0], dtype=float),
        "lgs_theta_deg": np.array([45.0, 135.0, 225.0, 315.0], dtype=float),
        "atm_profiles": {
            "0": {
                "name": "default",
                "r0_m": 0.16,
                "L0_m": 25.0,
                "cn2_heights_m": np.array([0.0, 5000.0], dtype=float),
                "cn2_weights": np.array([0.6, 0.4], dtype=float),
                "wind_speed_mps": np.array([5.0, 10.0], dtype=float),
                "wind_direction_deg": np.array([0.0, 90.0], dtype=float),
            }
        },
    }


def _options(num_sims: int = 3, max_ngs: int = 3) -> dict:
    return {
        "wavelength_um": np.full((num_sims,), 1.65, dtype=float),
        "atm_profile_id": np.zeros((num_sims,), dtype=np.int32),
        "zenith_angle_deg": np.full((num_sims,), 20.0, dtype=float),
        "r0_m": np.full((num_sims,), 0.16, dtype=float),
        "ngs_r_arcsec": np.ones((num_sims, max_ngs), dtype=float),
        "ngs_theta_deg": np.zeros((num_sims, max_ngs), dtype=float),
        "ngs_mag": np.full((num_sims, max_ngs), 15.0, dtype=float),
    }


def _options_row(index: int = 0) -> dict:
    options = _options()
    return {key: np.asarray(value)[index].copy() for key, value in options.items()}


def _stats_meta(pixel_scale_mas: float = 4.0) -> dict:
    return {
        schema.KEY_META_PIXEL_SCALE_MAS: float(pixel_scale_mas),
        schema.KEY_META_TEL_DIAMETER_M: 8.0,
        schema.KEY_META_TEL_PUPIL: np.ones((6, 6), dtype=np.float32),
    }


def _psf_metadata(
    *,
    wavelength_um: float | np.ndarray = 1.65,
    pixel_scale_mas: float | np.ndarray = 4.0,
    tel_diameter_m: float | np.ndarray = 8.0,
    tel_pupil: np.ndarray | None = None,
) -> PsfMetadata:
    return PsfMetadata(
        wavelength_um=wavelength_um,
        pixel_scale_mas=pixel_scale_mas,
        tel_diameter_m=tel_diameter_m,
        tel_pupil=np.ones((6, 6), dtype=np.float32) if tel_pupil is None else tel_pupil,
    )


def _stats_ee_apertures() -> np.ndarray:
    return np.array([50.0, 100.0], dtype=np.float32)


def _success_result(
    m: int = 3,
    a: int = 2,
    ny: int = 4,
    nx: int = 4,
    *,
    populate_stats: bool = True,
    extra_stats: dict[str, np.ndarray] | None = None,
    meta: dict[str, float] | None = None,
) -> SimulationResult:
    stats: dict[str, np.ndarray] = {}
    if populate_stats:
        stats = {
            "sr": np.linspace(0.1, 0.3, m, dtype=np.float32),
            "ee": np.full((m, a), 0.5, dtype=np.float32),
            "fwhm_mas": np.full((m,), 60.0, dtype=np.float32),
        }
        if extra_stats:
            stats.update(extra_stats)
    result_meta = {
        "pixel_scale_mas": 4.0,
        "tel_diameter_m": 8.0,
        "tel_pupil": np.ones((6, 6), dtype=np.float32),
    }
    result_meta.update(dict(meta or {}))
    return SimulationResult(
        state=SimulationState.SUCCEEDED,
        stats=stats,
        meta=result_meta,
        psfs=np.full((m, ny, nx), 0.1, dtype=np.float32),
    )


def _success_result_missing_required_outputs(m: int = 3, a: int = 2) -> SimulationResult:
    return SimulationResult(
        state=SimulationState.SUCCEEDED,
        stats={
            "sr": np.linspace(0.1, 0.3, m, dtype=np.float32),
            "ee": np.full((m, a), 0.5, dtype=np.float32),
            "fwhm_mas": np.full((m,), 60.0, dtype=np.float32),
        },
        meta={
            "pixel_scale_mas": 4.0,
            "tel_diameter_m": 8.0,
            "tel_pupil": np.ones((6, 6), dtype=np.float32),
        },
        psfs=None,
    )


def _setup_obj() -> SimulationSetup:
    setup = _setup()
    return SimulationSetup(
        ee_apertures_mas=np.asarray(setup["ee_apertures_mas"], dtype=float).reshape(-1),
        sr_method=str(setup["sr_method"]),
        fwhm_summary=str(setup["fwhm_summary"]),
        ee_geometry=str(setup["ee_geometry"]),
        atm_wavelength_um=float(setup["atm_wavelength_um"]),
        atm_profiles=dict(setup["atm_profiles"]),
        lgs_r_arcsec=np.asarray(setup["lgs_r_arcsec"], dtype=float).reshape(-1),
        lgs_theta_deg=np.asarray(setup["lgs_theta_deg"], dtype=float).reshape(-1),
        sci_r_arcsec=np.asarray(setup["sci_r_arcsec"], dtype=float).reshape(-1),
        sci_theta_deg=np.asarray(setup["sci_theta_deg"], dtype=float).reshape(-1),
    )


def _stub_unavailable_girmos_dependencies() -> None:
    """Install test-only stubs for upstream imports not needed by AO Predict paths."""
    if "skimage.measure" not in sys.modules:
        skimage_module = types.ModuleType("skimage")
        skimage_measure = types.ModuleType("skimage.measure")

        def _find_contours(z: np.ndarray, level: float) -> list[np.ndarray]:
            generator = contourpy.contour_generator(z=np.asarray(z, dtype=float))
            return [line[:, [1, 0]] for line in generator.lines(float(level))]

        skimage_measure.find_contours = _find_contours
        skimage_module.measure = skimage_measure
        sys.modules["skimage"] = skimage_module
        sys.modules["skimage.measure"] = skimage_measure

    if "mastsel.mavisPsf" not in sys.modules:
        mastsel_module = types.ModuleType("mastsel")
        mastsel_mavis = types.ModuleType("mastsel.mavisPsf")

        def _unused(*args, **kwargs):
            raise RuntimeError("Legacy-only upstream dependency should not be used in AO Predict regression tests.")

        mastsel_mavis.Field = object
        mastsel_mavis.convolve = _unused
        mastsel_mavis.residualToSpectrum = _unused
        mastsel_module.mavisPsf = mastsel_mavis
        sys.modules["mastsel"] = mastsel_module
        sys.modules["mastsel.mavisPsf"] = mastsel_mavis

    if "p3.aoSystem.FourierUtils" not in sys.modules:
        p3_module = types.ModuleType("p3")
        p3_aosystem = types.ModuleType("p3.aoSystem")
        p3_fourier = types.ModuleType("p3.aoSystem.FourierUtils")

        def _unused(*args, **kwargs):
            raise RuntimeError("Legacy-only upstream dependency should not be used in AO Predict regression tests.")

        p3_fourier.otf2psf = _unused
        p3_fourier.telescopeOtf = _unused
        p3_fourier.find_contour_points = _unused
        p3_fourier.fwhm_1d = _unused
        p3_aosystem.FourierUtils = p3_fourier
        p3_module.aoSystem = p3_aosystem
        sys.modules["p3"] = p3_module
        sys.modules["p3.aoSystem"] = p3_aosystem
        sys.modules["p3.aoSystem.FourierUtils"] = p3_fourier

    if "ao_tools" not in sys.modules:
        ao_tools_module = types.ModuleType("ao_tools")
        ao_tools_module.simulate = types.SimpleNamespace()
        sys.modules["ao_tools"] = ao_tools_module


def _load_girmos_aostats_for_regression():
    """Load the locked downstream AO stats module for regression comparisons."""
    global _GIRMOS_AOSTATS
    if _GIRMOS_AOSTATS is not None:
        return _GIRMOS_AOSTATS

    path = Path("/Users/nelsonnunes/Library/CloudStorage/Dropbox/Projects/girmos-aosims/ao_tools/aostats.py")
    if not path.exists():
        pytest.skip(f"Downstream regression source not available: {path}")

    _stub_unavailable_girmos_dependencies()
    spec = importlib.util.spec_from_file_location("girmos_aostats_regression", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _GIRMOS_AOSTATS = module
    return module


def _gaussian_psf(
    ny: int,
    nx: int,
    center_y: float,
    center_x: float,
    sigma_y: float,
    sigma_x: float,
) -> np.ndarray:
    y, x = np.indices((ny, nx), dtype=np.float32)
    return np.exp(
        -0.5 * (((y - center_y) / sigma_y) ** 2 + ((x - center_x) / sigma_x) ** 2),
        dtype=np.float32,
    )


class _ExtraStatsSimulation(Simulation):
    ngs_mag_standard = "R"

    _NAME = "ao_predict.simulation.tiptop:TiptopSimulation"
    _VERSION = "x.y"

    def __init__(self, extra_stats: Mapping[str, object], extra_stat_names: tuple[str, ...] = ()):
        self._extra_stats = dict(extra_stats)
        self._extra_stat_names = tuple(extra_stat_names)

    @property
    def extra_stat_names(self) -> tuple[str, ...]:
        return self._extra_stat_names

    def prepare_simulation_payload(self, base_simulation_payload, simulation_cfg):
        del simulation_cfg
        return dict(base_simulation_payload)

    def load_simulation_payload(self, simulation_payload):
        del simulation_payload

    def validate_simulation_payload(self, simulation_payload):
        del simulation_payload

    def prepare_setup_payload(self, base_setup_payload, setup_cfg):
        del base_setup_payload, setup_cfg
        raise NotImplementedError

    def validate_setup_payload(self, setup_payload):
        del setup_payload
        raise NotImplementedError

    def load_setup_payload(self, setup_payload):
        del setup_payload
        raise NotImplementedError

    def prepare_options_payload(self, num_sims, setup_payload, base_options_payload):
        del num_sims, setup_payload, base_options_payload
        raise NotImplementedError

    def create(self, index: int, options):
        del index, options
        raise NotImplementedError

    def run(self, context: SimulationContext) -> None:
        del context
        raise NotImplementedError

    def finalize(self, context: SimulationContext) -> None:
        del context
        raise NotImplementedError

    def prepare_psfs_for_stats(self, psfs, setup, meta):
        del setup, meta
        return np.asarray(psfs, dtype=np.float32)

    def build_extra_stats(self, context: SimulationContext):
        del context
        return dict(self._extra_stats)


def test_validate_success_result_accepts_valid_success_result():
    validate_successful_result(_success_result(), 3, 2, require_psfs=True)


def test_validate_success_result_accepts_nan_fwhm():
    result = _success_result()
    result.stats[schema.KEY_STATS_FWHM_MAS] = np.full((3,), np.nan, dtype=np.float32)
    validate_successful_result(result, 3, 2, require_psfs=True)


def test_validate_success_result_requires_declared_extra_stats():
    result = _success_result()
    with pytest.raises(ValueError, match="missing declared extra stats: halo_mas"):
        validate_successful_result(result, 3, 2, extra_stat_names=("halo_mas",), require_psfs=True)


def test_validate_success_result_rejects_psf_science_dimension_mismatch():
    result = _success_result()
    result.psfs = np.full((2, 4, 4), 0.1, dtype=np.float32)
    with pytest.raises(ValueError, match="result.psfs science dimension mismatch"):
        validate_successful_result(result, 3, 2, require_psfs=True)


def test_validate_success_result_rejects_missing_tel_pupil():
    result = _success_result()
    result.meta.pop(schema.KEY_META_TEL_PUPIL)
    with pytest.raises(ValueError, match="result.meta must include pixel_scale_mas, tel_diameter_m, and tel_pupil"):
        validate_successful_result(result, 3, 2, require_psfs=True)


def test_validate_success_result_rejects_non_2d_tel_pupil():
    result = _success_result()
    result.meta[schema.KEY_META_TEL_PUPIL] = np.ones((6,), dtype=np.float32)
    with pytest.raises(ValueError, match=r"result\.meta\.tel_pupil must be 2D \[Ny, Nx\]\."):
        validate_successful_result(result, 3, 2, require_psfs=True)


def test_populate_result_stats_rejects_simulation_provided_core_stats():
    context = SimulationContext(index=0, setup=_setup_obj(), options=_options_row())
    context.runtime["extra_stat_names"] = ()
    context.result = SimulationResult(
        state=SimulationState.SUCCEEDED,
        psfs=np.full((3, 4, 4), 0.1, dtype=np.float32),
        meta={
            "pixel_scale_mas": 4.0,
            "tel_diameter_m": 8.0,
            "tel_pupil": np.ones((6, 6), dtype=np.float32),
        },
    )
    simulation = _ExtraStatsSimulation(
        {schema.KEY_STATS_SR: np.full((3,), 0.2, dtype=np.float32)},
    )

    with pytest.raises(
        ValueError,
        match=r"Simulation built core stats in build_extra_stats\(\): sr\. Core stats are owned by ao-predict and must not be provided by the simulation\.",
    ):
        _populate_result_stats(simulation, context)


def test_populate_result_stats_rejects_direct_result_stats_population():
    context = SimulationContext(index=0, setup=_setup_obj(), options=_options_row())
    context.runtime["extra_stat_names"] = ()
    context.result = SimulationResult(
        state=SimulationState.SUCCEEDED,
        psfs=np.full((3, 4, 4), 0.1, dtype=np.float32),
        meta={
            "pixel_scale_mas": 4.0,
            "tel_diameter_m": 8.0,
            "tel_pupil": np.ones((6, 6), dtype=np.float32),
        },
        stats={"halo_mas": np.full((3,), 0.2, dtype=np.float32)},
    )
    simulation = _ExtraStatsSimulation({})

    with pytest.raises(
        ValueError,
        match=r"Successful simulations must not populate result\.stats directly\. Declared extra stats must be returned from build_extra_stats\(\.\.\.\)\.",
    ):
        _populate_result_stats(simulation, context)


def test_populate_result_stats_rejects_undeclared_extra_stats():
    context = SimulationContext(index=0, setup=_setup_obj(), options=_options_row())
    context.runtime["extra_stat_names"] = ()
    context.result = SimulationResult(
        state=SimulationState.SUCCEEDED,
        psfs=np.full((3, 4, 4), 0.1, dtype=np.float32),
        meta={
            "pixel_scale_mas": 4.0,
            "tel_diameter_m": 8.0,
            "tel_pupil": np.ones((6, 6), dtype=np.float32),
        },
    )
    simulation = _ExtraStatsSimulation({"halo_mas": np.full((3,), 0.2, dtype=np.float32)})

    with pytest.raises(ValueError, match=r"Simulation built undeclared extra stats in build_extra_stats\(\): halo_mas"):
        _populate_result_stats(simulation, context)


def test_populate_result_stats_passes_runtime_options_to_stats(monkeypatch):
    observed_options: list[dict[str, object]] = []
    context = SimulationContext(index=0, setup=_setup_obj(), options=_options_row())
    context.runtime["extra_stat_names"] = ()
    context.result = SimulationResult(
        state=SimulationState.SUCCEEDED,
        psfs=np.full((3, 4, 4), 0.1, dtype=np.float32),
        meta={
            "pixel_scale_mas": 4.0,
            "tel_diameter_m": 8.0,
            "tel_pupil": np.ones((6, 6), dtype=np.float32),
        },
    )

    def _compute(
        psfs,
        metadata,
        *,
        ee_apertures_mas,
        sr_method,
        fwhm_summary,
        ee_geometry,
        preprocess=None,
        **kwargs,
    ):
        del psfs, ee_apertures_mas, sr_method, fwhm_summary, preprocess, kwargs
        observed_options.append(
            {
                schema.KEY_OPTION_WAVELENGTH_UM: metadata.wavelength_um,
                schema.KEY_SETUP_EE_GEOMETRY: ee_geometry,
            }
        )
        return (
            np.zeros((3,), dtype=np.float32),
            np.zeros((3, 2), dtype=np.float32),
            np.zeros((3,), dtype=np.float32),
        )

    monkeypatch.setattr("ao_predict.simulation.runner.compute_psf_stats", _compute)

    _populate_result_stats(_ExtraStatsSimulation({}), context)

    assert len(observed_options) == 1
    assert float(observed_options[0][schema.KEY_OPTION_WAVELENGTH_UM]) == pytest.approx(1.65)
    assert observed_options[0][schema.KEY_SETUP_EE_GEOMETRY] == schema.DEFAULT_SETUP_EE_GEOMETRY


def test_compute_psf_stats_rejects_missing_ee_apertures():
    with pytest.raises(ValueError, match="ee_apertures_mas is required when computing EE"):
        compute_psf_stats(
            np.zeros((3, 4, 4), dtype=np.float32),
            _psf_metadata(),
        )


def test_compute_psf_stats_fwhm_metric_does_not_require_ee_apertures_or_compute_sr_ee(monkeypatch):
    def _compute_strehl(*args, **kwargs):
        raise AssertionError("SR should not be computed")

    def _compute_enclosed_energy(*args, **kwargs):
        raise AssertionError("EE should not be computed")

    def _measure(psfs, pixel_scale_mas):
        del pixel_scale_mas
        return (
            np.full((psfs.shape[0],), 4.0, dtype=np.float32),
            np.full((psfs.shape[0],), 9.0, dtype=np.float32),
        )

    monkeypatch.setattr(stats_module, "_compute_strehl", _compute_strehl)
    monkeypatch.setattr(stats_module, "_compute_enclosed_energy", _compute_enclosed_energy)
    monkeypatch.setattr(stats_module, "_measure_contour_fwhms", _measure)

    result = compute_psf_stats(
        np.zeros((2, 4, 4), dtype=np.float32),
        _psf_metadata(),
        metrics=("fwhm_mas",),
    )

    assert len(result) == 1
    np.testing.assert_allclose(result[0], np.full((2,), 6.0, dtype=np.float32))


def test_compute_psf_stats_ee_metric_uses_peak_locations_without_computing_sr(monkeypatch):
    def _compute_strehl(*args, **kwargs):
        raise AssertionError("SR should not be computed")

    def _peak_locations(psfs, sr_method):
        del sr_method
        return np.full((psfs.shape[0], 2), 1.0, dtype=np.float32)

    def _ee(psfs, ee_apertures_mas, pixel_scale_mas, peak_locations_yx=None, *, ee_geometry="ensquared"):
        del pixel_scale_mas, ee_geometry
        np.testing.assert_allclose(peak_locations_yx, np.full((psfs.shape[0], 2), 1.0, dtype=np.float32))
        return np.full((psfs.shape[0], ee_apertures_mas.shape[0]), 0.25, dtype=np.float32)

    monkeypatch.setattr(stats_module, "_compute_strehl", _compute_strehl)
    monkeypatch.setattr(stats_module, "_compute_peak_locations", _peak_locations)
    monkeypatch.setattr(stats_module, "_compute_enclosed_energy", _ee)

    result = compute_psf_stats(
        np.zeros((2, 4, 4), dtype=np.float32),
        _psf_metadata(),
        ee_apertures_mas=_stats_ee_apertures(),
        metrics=("ee",),
    )

    assert len(result) == 1
    np.testing.assert_allclose(result[0], np.full((2, 2), 0.25, dtype=np.float32))


def test_compute_psf_stats_rejects_invalid_metric_names():
    with pytest.raises(ValueError, match="metrics contains unsupported names"):
        compute_psf_stats(
            np.zeros((2, 4, 4), dtype=np.float32),
            _psf_metadata(),
            metrics=("fwhm",),
        )


def test_compute_psf_metric_wrappers_return_single_metric(monkeypatch):
    def _compute_strehl(psfs, sr_method, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil):
        del sr_method, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil
        return np.full((psfs.shape[0],), 0.5, dtype=np.float32), np.zeros((psfs.shape[0], 2), dtype=np.float32)

    def _peak_locations(psfs, sr_method):
        del sr_method
        return np.zeros((psfs.shape[0], 2), dtype=np.float32)

    def _ee(psfs, ee_apertures_mas, pixel_scale_mas, peak_locations_yx=None, *, ee_geometry="ensquared"):
        del pixel_scale_mas, peak_locations_yx, ee_geometry
        return np.full((psfs.shape[0], ee_apertures_mas.shape[0]), 0.25, dtype=np.float32)

    def _measure(psfs, pixel_scale_mas):
        del pixel_scale_mas
        return (
            np.full((psfs.shape[0],), 4.0, dtype=np.float32),
            np.full((psfs.shape[0],), 9.0, dtype=np.float32),
        )

    monkeypatch.setattr(stats_module, "_compute_strehl", _compute_strehl)
    monkeypatch.setattr(stats_module, "_compute_peak_locations", _peak_locations)
    monkeypatch.setattr(stats_module, "_compute_enclosed_energy", _ee)
    monkeypatch.setattr(stats_module, "_measure_contour_fwhms", _measure)

    psfs = np.zeros((2, 4, 4), dtype=np.float32)
    np.testing.assert_allclose(compute_psf_sr(psfs, _psf_metadata()), np.full((2,), 0.5, dtype=np.float32))
    np.testing.assert_allclose(
        compute_psf_ee(psfs, _psf_metadata(), ee_apertures_mas=_stats_ee_apertures()),
        np.full((2, 2), 0.25, dtype=np.float32),
    )
    np.testing.assert_allclose(compute_psf_fwhm(psfs, _psf_metadata()), np.full((2,), 6.0, dtype=np.float32))


def test_compute_psf_stats_rejects_invalid_sr_method():
    with pytest.raises(ValueError, match="sr_method must be one of"):
        compute_psf_stats(
            np.zeros((3, 4, 4), dtype=np.float32),
            _psf_metadata(),
            ee_apertures_mas=_stats_ee_apertures(),
            sr_method="invalid",
        )


def test_compute_psf_stats_rejects_invalid_fwhm_summary():
    with pytest.raises(ValueError, match="fwhm_summary must be one of"):
        compute_psf_stats(
            np.zeros((3, 4, 4), dtype=np.float32),
            _psf_metadata(),
            ee_apertures_mas=_stats_ee_apertures(),
            fwhm_summary="invalid",
        )


def test_compute_psf_stats_rejects_invalid_per_psf_wavelength_length():
    with pytest.raises(ValueError, match="metadata.wavelength_um per-PSF vector length must match the PSF cube length 3"):
        compute_psf_stats(
            np.zeros((3, 4, 4), dtype=np.float32),
            _psf_metadata(wavelength_um=np.array([1.6, 1.7], dtype=np.float32)),
            ee_apertures_mas=_stats_ee_apertures(),
        )


def test_compute_psf_stats_rejects_invalid_per_psf_pixel_scale_length():
    with pytest.raises(ValueError, match="metadata.pixel_scale_mas per-PSF vector length must match the PSF cube length 3"):
        compute_psf_stats(
            np.zeros((3, 4, 4), dtype=np.float32),
            _psf_metadata(pixel_scale_mas=np.array([4.0, 4.1], dtype=np.float32)),
            ee_apertures_mas=_stats_ee_apertures(),
        )


def test_compute_psf_stats_rejects_non_scalar_telescope_diameter():
    with pytest.raises(ValueError, match="metadata.tel_diameter_m must be a scalar"):
        compute_psf_stats(
            np.zeros((3, 4, 4), dtype=np.float32),
            _psf_metadata(tel_diameter_m=np.array([8.0, 8.1], dtype=np.float32)),
            ee_apertures_mas=_stats_ee_apertures(),
        )


def test_compute_psf_stats_rejects_non_2d_telescope_pupil():
    with pytest.raises(ValueError, match=r"metadata\.tel_pupil must be 2D"):
        compute_psf_stats(
            np.zeros((3, 4, 4), dtype=np.float32),
            _psf_metadata(tel_pupil=np.ones((3, 6, 6), dtype=np.float32)),
            ee_apertures_mas=_stats_ee_apertures(),
        )


def test_compute_psf_stats_rejects_invalid_per_psf_ee_aperture_length():
    with pytest.raises(ValueError, match="ee_apertures_mas per-PSF leading dimension must match the PSF cube length 3"):
        compute_psf_stats(
            np.zeros((3, 4, 4), dtype=np.float32),
            _psf_metadata(),
            ee_apertures_mas=np.ones((2, 1), dtype=np.float32),
        )


def test_compute_psf_stats_dispatches_selected_strehl_method(monkeypatch):
    calls: list[str] = []

    def _pixel_fit(psfs, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil):
        del pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil
        calls.append(schema.STATS_SR_METHOD_PIXEL_FIT)
        return (
            np.zeros((psfs.shape[0],), dtype=np.float32),
            np.zeros((psfs.shape[0], 2), dtype=np.float32),
        )

    def _pixel_max(psfs, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil):
        del pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil
        calls.append(schema.STATS_SR_METHOD_PIXEL_MAX)
        return (
            np.zeros((psfs.shape[0],), dtype=np.float32),
            np.zeros((psfs.shape[0], 2), dtype=np.float32),
        )

    monkeypatch.setattr(stats_module, "_compute_strehl_pixel_fit", _pixel_fit)
    monkeypatch.setattr(stats_module, "_compute_strehl_pixel_max", _pixel_max)

    compute_psf_stats(
        np.zeros((3, 4, 4), dtype=np.float32),
        _psf_metadata(),
        ee_apertures_mas=_stats_ee_apertures(),
        sr_method=schema.STATS_SR_METHOD_PIXEL_MAX,
    )

    assert calls == [schema.STATS_SR_METHOD_PIXEL_MAX]
    calls.clear()

    compute_psf_stats(
        np.zeros((2, 4, 4), dtype=np.float32),
        _psf_metadata(),
        ee_apertures_mas=_stats_ee_apertures(),
        sr_method=schema.STATS_SR_METHOD_PIXEL_FIT,
    )

    assert calls == [schema.STATS_SR_METHOD_PIXEL_FIT]


def test_compute_psf_stats_reuses_fit_peak_locations_for_ee(monkeypatch):
    ee_peak_locations: list[np.ndarray | None] = []
    ee_geometries: list[str] = []

    def _pixel_fit(psfs, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil):
        del pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil
        return (
            np.zeros((psfs.shape[0],), dtype=np.float32),
            np.full((psfs.shape[0], 2), 1.5, dtype=np.float32),
        )

    def _pixel_max(psfs, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil):
        del pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil
        return (
            np.zeros((psfs.shape[0],), dtype=np.float32),
            np.full((psfs.shape[0], 2), 2.5, dtype=np.float32),
        )

    def _ee(
        psfs,
        ee_apertures_mas,
        pixel_scale_mas,
        peak_locations_xy=None,
        *,
        ee_geometry="ensquared",
    ):
        del pixel_scale_mas
        ee_peak_locations.append(None if peak_locations_xy is None else np.asarray(peak_locations_xy, dtype=np.float32))
        ee_geometries.append(ee_geometry)
        return np.zeros((psfs.shape[0], ee_apertures_mas.shape[0]), dtype=np.float32)

    monkeypatch.setattr(stats_module, "_compute_strehl_pixel_fit", _pixel_fit)
    monkeypatch.setattr(stats_module, "_compute_strehl_pixel_max", _pixel_max)
    monkeypatch.setattr(stats_module, "_compute_enclosed_energy", _ee)

    compute_psf_stats(
        np.zeros((2, 4, 4), dtype=np.float32),
        _psf_metadata(),
        ee_apertures_mas=_stats_ee_apertures(),
        sr_method=schema.STATS_SR_METHOD_PIXEL_FIT,
    )

    assert len(ee_peak_locations) == 1
    np.testing.assert_allclose(ee_peak_locations[0], np.full((2, 2), 1.5, dtype=np.float32))
    assert ee_geometries == ["ensquared"]
    ee_peak_locations.clear()
    ee_geometries.clear()

    compute_psf_stats(
        np.zeros((2, 4, 4), dtype=np.float32),
        _psf_metadata(),
        ee_apertures_mas=_stats_ee_apertures(),
        sr_method=schema.STATS_SR_METHOD_PIXEL_MAX,
    )

    assert len(ee_peak_locations) == 1
    np.testing.assert_allclose(ee_peak_locations[0], np.full((2, 2), 2.5, dtype=np.float32))
    assert ee_geometries == ["ensquared"]


def test_compute_psf_stats_passes_requested_ee_geometry(monkeypatch):
    requested: list[str] = []

    def _ee(
        psfs,
        ee_apertures_mas,
        pixel_scale_mas,
        peak_locations_yx=None,
        *,
        ee_geometry="ensquared",
    ):
        del pixel_scale_mas, peak_locations_yx
        requested.append(ee_geometry)
        return np.zeros((psfs.shape[0], ee_apertures_mas.shape[0]), dtype=np.float32)

    monkeypatch.setattr(stats_module, "_compute_enclosed_energy", _ee)

    compute_psf_stats(
        np.zeros((2, 4, 4), dtype=np.float32),
        _psf_metadata(),
        ee_apertures_mas=_stats_ee_apertures(),
        ee_geometry="encircled",
    )

    assert requested == ["encircled"]


def test_compute_psf_stats_rejects_invalid_ee_geometry(monkeypatch):
    def _compute_strehl(psfs, sr_method, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil):
        del sr_method, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil
        return np.zeros((psfs.shape[0],), dtype=np.float32), np.full((psfs.shape[0], 2), 2, dtype=np.float32)

    monkeypatch.setattr(stats_module, "_compute_strehl", _compute_strehl)

    with pytest.raises(ValueError, match="ee_geometry must be one of"):
        compute_psf_stats(
            np.zeros((2, 4, 4), dtype=np.float32),
            _psf_metadata(),
            ee_apertures_mas=_stats_ee_apertures(),
            ee_geometry="triangle",
        )


def test_compute_psf_stats_computes_encircled_energy(monkeypatch):
    def _compute_strehl(psfs, sr_method, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil):
        del sr_method, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil
        return np.zeros((psfs.shape[0],), dtype=np.float32), np.full((psfs.shape[0], 2), 2, dtype=np.float32)

    monkeypatch.setattr(stats_module, "_compute_strehl", _compute_strehl)

    psfs = np.zeros((2, 5, 5), dtype=np.float32)
    psfs[:, 2, 2] = 1.0

    _sr, ee, _fwhm = compute_psf_stats(
        psfs,
        _psf_metadata(),
        ee_apertures_mas=_stats_ee_apertures(),
        ee_geometry="encircled",
    )

    assert ee.shape == (2, 2)
    np.testing.assert_allclose(ee, np.ones((2, 2), dtype=np.float32), atol=1e-6)


def test_compute_psf_stats_selects_requested_fwhm_summary(monkeypatch):
    requested: list[str] = []
    select_impl = stats_module._compute_fwhm_summary

    def _measure(psfs, pixel_scale_mas):
        del pixel_scale_mas
        return (
            np.full((psfs.shape[0],), 4.0, dtype=np.float32),
            np.full((psfs.shape[0],), 3.0, dtype=np.float32),
        )

    def _select(fwhm_summary, fwhm_min, fwhm_max):
        requested.append(fwhm_summary)
        return select_impl(fwhm_summary, fwhm_min, fwhm_max)

    monkeypatch.setattr(stats_module, "_measure_contour_fwhms", _measure)
    monkeypatch.setattr(stats_module, "_compute_fwhm_summary", _select)

    _sr, _ee, fwhm = compute_psf_stats(
        np.zeros((2, 4, 4), dtype=np.float32),
        _psf_metadata(),
        ee_apertures_mas=_stats_ee_apertures(),
        fwhm_summary=schema.STATS_FWHM_SUMMARY_MAX,
    )

    assert requested == [schema.STATS_FWHM_SUMMARY_MAX]
    np.testing.assert_allclose(fwhm, np.full((2,), 3.0, dtype=np.float32))


def test_compute_psf_stats_uses_per_psf_wavelength(monkeypatch):
    observed_wavelengths: list[float] = []

    def _compute_strehl(psfs, sr_method, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil):
        del sr_method, pixel_scale_mas, tel_diameter_m, tel_pupil
        observed_wavelengths.append(float(wavelength_um))
        return np.zeros((psfs.shape[0],), dtype=np.float32), np.zeros((psfs.shape[0], 2), dtype=np.float32)

    monkeypatch.setattr(stats_module, "_compute_strehl", _compute_strehl)

    compute_psf_stats(
        np.zeros((2, 4, 4), dtype=np.float32),
        _psf_metadata(wavelength_um=np.array([1.2, 1.8], dtype=np.float32)),
        ee_apertures_mas=_stats_ee_apertures(),
    )

    np.testing.assert_allclose(observed_wavelengths, [1.2, 1.8])


def test_compute_psf_stats_uses_per_psf_pixel_scale(monkeypatch):
    observed_pixel_scales: list[float] = []

    def _compute_strehl(psfs, sr_method, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil):
        del sr_method, wavelength_um, tel_diameter_m, tel_pupil
        observed_pixel_scales.append(float(pixel_scale_mas))
        return np.zeros((psfs.shape[0],), dtype=np.float32), np.zeros((psfs.shape[0], 2), dtype=np.float32)

    monkeypatch.setattr(stats_module, "_compute_strehl", _compute_strehl)

    compute_psf_stats(
        np.zeros((2, 4, 4), dtype=np.float32),
        _psf_metadata(pixel_scale_mas=np.array([3.5, 4.5], dtype=np.float32)),
        ee_apertures_mas=_stats_ee_apertures(),
    )

    np.testing.assert_allclose(observed_pixel_scales, [3.5, 4.5])


def test_compute_psf_stats_uses_per_psf_ee_apertures(monkeypatch):
    observed_apertures: list[np.ndarray] = []

    def _ee(psfs, ee_apertures_mas, pixel_scale_mas, peak_locations_yx=None, *, ee_geometry="ensquared"):
        del pixel_scale_mas, peak_locations_yx, ee_geometry
        observed_apertures.append(np.asarray(ee_apertures_mas, dtype=np.float32))
        return np.zeros((psfs.shape[0], ee_apertures_mas.shape[0]), dtype=np.float32)

    monkeypatch.setattr(stats_module, "_compute_enclosed_energy", _ee)

    _sr, ee, _fwhm = compute_psf_stats(
        np.zeros((2, 4, 4), dtype=np.float32),
        _psf_metadata(),
        ee_apertures_mas=np.array([[10.0], [20.0]], dtype=np.float32),
    )

    assert ee.shape == (2, 1)
    assert len(observed_apertures) == 2
    np.testing.assert_allclose(observed_apertures[0], [10.0])
    np.testing.assert_allclose(observed_apertures[1], [20.0])


def test_compute_psf_stats_default_preprocess_shortcut_clips_and_normalizes(monkeypatch):
    observed_psfs: list[np.ndarray] = []

    def _compute_strehl(psfs, sr_method, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil):
        del sr_method, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil
        observed_psfs.append(np.asarray(psfs, dtype=np.float32))
        return np.zeros((psfs.shape[0],), dtype=np.float32), np.zeros((psfs.shape[0], 2), dtype=np.float32)

    monkeypatch.setattr(stats_module, "_compute_strehl", _compute_strehl)

    compute_psf_stats(
        np.array([[[-1.0, 3.0], [1.0, 0.0]]], dtype=np.float32),
        _psf_metadata(),
        ee_apertures_mas=np.array([50.0], dtype=np.float32),
        preprocess="default",
    )

    assert len(observed_psfs) == 1
    np.testing.assert_allclose(
        observed_psfs[0],
        np.array([[[0.0, 0.75], [0.25, 0.0]]], dtype=np.float32),
    )


def test_compute_psf_stats_rejects_unknown_preprocess_string():
    with pytest.raises(ValueError, match="preprocess must be None, 'default', or a callable"):
        compute_psf_stats(
            np.zeros((2, 4, 4), dtype=np.float32),
            _psf_metadata(),
            ee_apertures_mas=_stats_ee_apertures(),
            preprocess="runner",
        )


def test_compute_psf_stats_uses_preprocess_callable(monkeypatch):
    observed_psfs: list[np.ndarray] = []

    def _preprocess(psfs):
        return np.asarray(psfs, dtype=np.float32) + 2.0

    def _compute_strehl(psfs, sr_method, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil):
        del sr_method, pixel_scale_mas, wavelength_um, tel_diameter_m, tel_pupil
        observed_psfs.append(np.asarray(psfs, dtype=np.float32))
        return np.zeros((psfs.shape[0],), dtype=np.float32), np.zeros((psfs.shape[0], 2), dtype=np.float32)

    monkeypatch.setattr(stats_module, "_compute_strehl", _compute_strehl)

    compute_psf_stats(
        np.zeros((2, 4, 4), dtype=np.float32),
        _psf_metadata(),
        ee_apertures_mas=_stats_ee_apertures(),
        preprocess=_preprocess,
    )

    assert len(observed_psfs) == 1
    np.testing.assert_allclose(observed_psfs[0], np.full((2, 4, 4), 2.0, dtype=np.float32))


def test_compute_strehl_pixel_max_matches_diffraction_limited_peak():
    tel_pupil = np.ones((6, 6), dtype=np.float32)
    psf_dl = stats_module._get_diffraction_limited_psf(
        8.0,
        tel_pupil,
        1.65,
        4.0,
        center_in_one_pix=True,
    )

    sr, peak_locations_yx = stats_module._compute_strehl_pixel_max(
        psf_dl[None, :, :],
        4.0,
        1.65,
        8.0,
        tel_pupil,
    )
    expected_peak = np.array(np.unravel_index(np.argmax(psf_dl), psf_dl.shape), dtype=np.float32)[None, :]

    np.testing.assert_allclose(sr, np.array([1.0], dtype=np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(peak_locations_yx, expected_peak, atol=0.0)


def test_compute_strehl_pixel_fit_matches_diffraction_limited_peak():
    tel_pupil = np.ones((6, 6), dtype=np.float32)
    psf_dl = stats_module._get_diffraction_limited_psf(
        8.0,
        tel_pupil,
        1.65,
        4.0,
        center_in_one_pix=False,
    )

    sr, peak_locations_yx = stats_module._compute_strehl_pixel_fit(
        psf_dl[None, :, :],
        4.0,
        1.65,
        8.0,
        tel_pupil,
    )

    np.testing.assert_allclose(sr, np.array([1.0], dtype=np.float32), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(peak_locations_yx, np.array([[psf_dl.shape[0] / 2.0 - 0.5, psf_dl.shape[1] / 2.0 - 0.5]], dtype=np.float32), atol=0.1)


def test_compute_encircled_energy_uses_exact_aperture_weights():
    psf = np.zeros((1, 5, 5), dtype=np.float32)
    psf[0, 2, 2] = 1.0

    ee = stats_module._compute_enclosed_energy(
        psf,
        np.array([2.0, 6.0], dtype=float),
        2.0,
        np.array([[2, 2]], dtype=np.int64),
        ee_geometry="encircled",
    )

    np.testing.assert_allclose(ee, np.array([[np.pi / 4.0, 1.0]], dtype=np.float32), atol=1e-6)


def test_compute_ensquared_energy_shifts_subpixel_peak_locations():
    psf = np.zeros((1, 5, 5), dtype=np.float32)
    psf[0, 2, 2] = 0.25
    psf[0, 2, 3] = 0.25
    psf[0, 3, 2] = 0.25
    psf[0, 3, 3] = 0.25

    ee_subpixel = stats_module._compute_enclosed_energy(
        psf,
        np.array([2.0], dtype=float),
        2.0,
        np.array([[2.5, 2.5]], dtype=np.float32),
    )
    ee_argmax = stats_module._compute_enclosed_energy(
        psf,
        np.array([2.0], dtype=float),
        2.0,
        None,
    )

    assert np.all(np.isfinite(ee_subpixel))
    assert ee_subpixel.shape == (1, 1)
    assert float(ee_subpixel[0, 0]) >= float(ee_argmax[0, 0])


def test_compute_ensquared_energy_single_aperture_returns_matrix():
    psf = np.zeros((2, 5, 5), dtype=np.float32)
    psf[:, 2, 2] = 1.0

    ee = stats_module._compute_enclosed_energy(
        psf,
        np.array([2.0], dtype=float),
        2.0,
        np.array([[2, 2], [2, 2]], dtype=np.int64),
    )

    assert ee.shape == (2, 1)
    np.testing.assert_allclose(ee, np.array([[1.0], [1.0]], dtype=np.float32), atol=1e-6)


def test_compute_ensquared_energy_uses_integer_peak_locations():
    psf = np.zeros((1, 5, 5), dtype=np.float32)
    psf[0, 2, 2] = 1.0

    ee = stats_module._compute_enclosed_energy(
        psf,
        np.array([2.0, 6.0], dtype=float),
        2.0,
        np.array([[2, 2]], dtype=np.int64),
    )

    np.testing.assert_allclose(ee, np.array([[1.0, 1.0]], dtype=np.float32), atol=1e-6)


def test_compute_encircled_energy_is_stable_for_extra_apertures():
    psf = _gaussian_psf(9, 9, 4.0, 4.0, 1.5, 1.5)[None, :, :]
    psf /= psf.sum()

    ee_single = stats_module._compute_enclosed_energy(
        psf,
        np.array([4.0], dtype=float),
        1.0,
        np.array([[4, 4]], dtype=np.int64),
        ee_geometry="encircled",
    )
    ee_multi = stats_module._compute_enclosed_energy(
        psf,
        np.array([4.0, 8.0], dtype=float),
        1.0,
        np.array([[4, 4]], dtype=np.int64),
        ee_geometry="encircled",
    )

    np.testing.assert_allclose(ee_single[:, 0], ee_multi[:, 0], atol=1e-7)


def test_compute_encircled_energy_is_no_larger_than_ensquared_energy():
    psf = _gaussian_psf(9, 9, 4.0, 4.0, 1.5, 1.5)[None, :, :]
    psf /= psf.sum()
    apertures = np.array([4.0, 6.0], dtype=float)
    peak_locations = np.array([[4, 4]], dtype=np.int64)

    encircled = stats_module._compute_enclosed_energy(psf, apertures, 1.0, peak_locations, ee_geometry="encircled")
    ensquared = stats_module._compute_enclosed_energy(psf, apertures, 1.0, peak_locations)

    assert np.all(encircled <= ensquared + 1e-6)


def test_measure_peak_centered_ensquared_energy_curves_applies_radius_factor():
    psf = np.zeros((1, 21, 21), dtype=np.float32)
    psf[0, 10, 10] = 1.0

    curves, curve_radii_mas = stats_module._measure_peak_centered_ensquared_energy_curves(
        psf,
        np.array([10], dtype=np.int64),
        np.array([10], dtype=np.int64),
        max_radius=4,
        pixel_scale_mas=1.0,
    )

    assert curves.shape == (1, 5)
    np.testing.assert_allclose(curve_radii_mas, np.array([0.5, 1.5, 2.5, 3.5, 4.5]))


def test_measure_contour_fwhms_returns_expected_widths_for_square_ring():
    psf = np.zeros((1, 7, 7), dtype=np.float32)
    psf[0, 1:6, 1:6] = 1.0
    psf[0, 2:5, 2:5] = 0.0

    fwhm_min, fwhm_max = stats_module._measure_contour_fwhms(psf, 2.0)

    np.testing.assert_allclose(fwhm_min, np.array([10.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(fwhm_max, np.array([12.806249], dtype=np.float32), atol=1e-6)


def test_compute_fwhm_summary_selects_geom_mean_max_min():
    fwhm_min = np.array([4.0], dtype=np.float32)
    fwhm_max = np.array([9.0], dtype=np.float32)

    np.testing.assert_allclose(
        stats_module._compute_fwhm_summary(schema.STATS_FWHM_SUMMARY_GEOM, fwhm_min, fwhm_max),
        np.array([6.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        stats_module._compute_fwhm_summary(schema.STATS_FWHM_SUMMARY_MEAN, fwhm_min, fwhm_max),
        np.array([6.5], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        stats_module._compute_fwhm_summary(schema.STATS_FWHM_SUMMARY_MAX, fwhm_min, fwhm_max),
        np.array([9.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        stats_module._compute_fwhm_summary(schema.STATS_FWHM_SUMMARY_MIN, fwhm_min, fwhm_max),
        np.array([4.0], dtype=np.float32),
        atol=1e-6,
    )


def test_measure_contour_fwhms_returns_nan_on_invalid_cases():
    truncated = np.zeros((1, 5, 5), dtype=np.float32)
    truncated[0, :, :3] = 1.0

    non_crossing = np.full((1, 4, 4), 1.0, dtype=np.float32)

    non_finite = np.zeros((1, 5, 5), dtype=np.float32)
    non_finite[0, 2, 2] = np.nan

    for psf in (truncated, non_crossing, non_finite):
        fwhm_min, fwhm_max = stats_module._measure_contour_fwhms(psf, 2.0)
        assert np.isnan(fwhm_min[0])
        assert np.isnan(fwhm_max[0])


def test_measure_contour_fwhms_returns_nan_for_zero_peak_threshold():
    psf = np.zeros((1, 5, 5), dtype=np.float32)

    fwhm_min, fwhm_max = stats_module._measure_contour_fwhms(psf, 2.0)

    assert np.isnan(fwhm_min[0])
    assert np.isnan(fwhm_max[0])


def test_measure_contour_fwhms_returns_nan_for_collapsed_contour_geometry(monkeypatch):
    psf = np.zeros((1, 7, 7), dtype=np.float32)
    psf[0, 3, 3] = 1.0

    monkeypatch.setattr(
        stats_module,
        "_find_contours",
        lambda _psf, _level: [np.array([[3.0, 2.0], [3.0, 3.0], [3.0, 4.0]], dtype=float)],
    )

    fwhm_min, fwhm_max = stats_module._measure_contour_fwhms(psf, 2.0)

    assert np.isnan(fwhm_min[0])
    assert np.isnan(fwhm_max[0])


@pytest.mark.parametrize(
    ("sr_method", "fwhm_summary"),
    [
        (schema.STATS_SR_METHOD_PIXEL_FIT, schema.STATS_FWHM_SUMMARY_GEOM),
        (schema.STATS_SR_METHOD_PIXEL_FIT, schema.STATS_FWHM_SUMMARY_MEAN),
        (schema.STATS_SR_METHOD_PIXEL_MAX, schema.STATS_FWHM_SUMMARY_MAX),
        (schema.STATS_SR_METHOD_PIXEL_MAX, schema.STATS_FWHM_SUMMARY_MIN),
    ],
)
def test_compute_psf_stats_matches_girmos_aopredict_regression(sr_method, fwhm_summary):
    upstream = _load_girmos_aostats_for_regression()
    simulation = MockSimulation()

    psfs = np.stack(
        [
            _gaussian_psf(31, 31, 15.2, 14.7, 2.1, 1.7),
            _gaussian_psf(31, 31, 13.8, 16.1, 2.5, 2.2),
        ],
        axis=0,
    ).astype(np.float32)
    options = {schema.KEY_OPTION_WAVELENGTH_UM: np.float32(1.65)}
    meta = _stats_meta(pixel_scale_mas=4.0)
    setup = {
        schema.KEY_SETUP_EE_APERTURES_MAS: np.array([12.0, 28.0, 44.0], dtype=float),
        schema.KEY_SETUP_SR_METHOD: sr_method,
        schema.KEY_SETUP_FWHM_SUMMARY: fwhm_summary,
    }

    sr, ee, fwhm = compute_psf_stats(
        psfs,
        PsfMetadata(
            wavelength_um=options[schema.KEY_OPTION_WAVELENGTH_UM],
            pixel_scale_mas=meta[schema.KEY_META_PIXEL_SCALE_MAS],
            tel_diameter_m=meta[schema.KEY_META_TEL_DIAMETER_M],
            tel_pupil=meta[schema.KEY_META_TEL_PUPIL],
        ),
        ee_apertures_mas=setup[schema.KEY_SETUP_EE_APERTURES_MAS],
        sr_method=sr_method,
        fwhm_summary=fwhm_summary,
        preprocess=lambda cube: simulation.prepare_psfs_for_stats(cube, setup, meta),
    )

    sr_method_upstream = {
        schema.STATS_SR_METHOD_PIXEL_FIT: upstream.SRMethod.PixelFit,
        schema.STATS_SR_METHOD_PIXEL_MAX: upstream.SRMethod.PixelMax,
    }[sr_method]
    fwhm_summary_upstream = {
        schema.STATS_FWHM_SUMMARY_GEOM: upstream.FWHMSummary.GeoM,
        schema.STATS_FWHM_SUMMARY_MEAN: upstream.FWHMSummary.Mean,
        schema.STATS_FWHM_SUMMARY_MAX: upstream.FWHMSummary.Max,
        schema.STATS_FWHM_SUMMARY_MIN: upstream.FWHMSummary.Min,
    }[fwhm_summary]

    sr_expected, fwhm_expected, ee_expected = upstream._compute_psf_stats(
        None,
        psfs,
        4.0,
        1.65e-6,
        8.0,
        meta[schema.KEY_META_TEL_PUPIL],
        method=upstream.StatsMethod.AOPredict,
        sr_method=sr_method_upstream,
        ee_apertures_mas=setup[schema.KEY_SETUP_EE_APERTURES_MAS],
        fwhm_summary=fwhm_summary_upstream,
    )

    np.testing.assert_allclose(sr, np.asarray(sr_expected, dtype=np.float32), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ee, np.asarray(ee_expected, dtype=np.float32), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(fwhm, np.asarray(fwhm_expected, dtype=np.float32), rtol=1e-5, atol=1e-6)


def test_store_create_and_row_writes(tmp_path):
    data_path = tmp_path / "sim_data.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=True)

    pending = store.pending_indices()
    assert pending.tolist() == [0, 1, 2]

    store.write_simulation_success(0, _success_result())
    store.write_simulation_failure(1)

    with h5py.File(data_path, "r") as f:
        expected_groups = [
            schema.KEY_META_SECTION,
            schema.KEY_OPTION_SECTION,
            schema.KEY_PSFS_SECTION,
            schema.KEY_SETUP_SECTION,
            schema.KEY_SIMULATION_SECTION,
            schema.KEY_STATS_SECTION,
            schema.KEY_STATUS_SECTION,
        ]
        assert list(f.keys()) == expected_groups
        status_path = f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"
        np.testing.assert_array_equal(
            f[status_path][:],
            np.array(
                [
                    int(SimulationState.SUCCEEDED),
                    int(SimulationState.FAILED),
                    int(SimulationState.PENDING),
                ],
                dtype=np.uint8,
            ),
        )

        assert f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_SR}"].shape == (3, 3)
        assert f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_EE}"].shape == (3, 3, 2)
        assert f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_FWHM_MAS}"].shape == (3, 3)

        assert np.all(np.isfinite(f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_SR}"][0]))
        assert np.all(np.isnan(f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_SR}"][1]))

        assert f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_PIXEL_SCALE_MAS}"][0] == np.float32(4.0)
        assert f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_DIAMETER_M}"][()] == np.float32(8.0)
        assert f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_PUPIL}"].shape == (6, 6)

        assert f[f"{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"].shape == (3, 3, 4, 4)
        assert np.all(np.isfinite(f[f"{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"][0]))
        assert np.all(np.isnan(f[f"{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"][1]))


def test_store_create_preallocates_empty_tel_pupil_dataset(tmp_path):
    data_path = tmp_path / "sim_data.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)

    with h5py.File(data_path, "r") as f:
        assert f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_PUPIL}"].shape == (0, 0)


def test_store_write_success_rejects_mismatched_dataset_level_telescope_meta(tmp_path):
    data_path = tmp_path / "sim_data_telescope_meta_mismatch.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)

    store.write_simulation_success(0, _success_result())

    bad_result = _success_result()
    bad_result.meta[schema.KEY_META_TEL_DIAMETER_M] = np.float32(10.0)
    with pytest.raises(ValueError, match=r"result\.meta\.tel_diameter_m does not match dataset-level /meta/tel_diameter_m\."):
        store.write_simulation_success(1, bad_result)

    bad_result = _success_result()
    bad_result.meta[schema.KEY_META_TEL_PUPIL] = np.full((6, 6), 2.0, dtype=np.float32)
    with pytest.raises(ValueError, match=r"result\.meta\.tel_pupil does not match dataset-level /meta/tel_pupil\."):
        store.write_simulation_success(1, bad_result)


def test_store_create_preallocates_declared_extra_stats(tmp_path):
    data_path = tmp_path / "sim_data_extra_stats.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(extra_stat_names=("halo_mas", "encircled_bg")), _setup(), _options(), save_psfs=False)

    with h5py.File(data_path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{schema.KEY_SIMULATION_SECTION}/{schema.KEY_SIMULATION_EXTRA_STAT_NAMES}"][:].astype(str),
            np.array(["halo_mas", "encircled_bg"]),
        )
        assert f[f"{schema.KEY_STATS_SECTION}/halo_mas"].shape == (3, 3)
        assert f[f"{schema.KEY_STATS_SECTION}/encircled_bg"].shape == (3, 3)


def test_store_create_preallocates_declared_meta_fields(tmp_path):
    data_path = tmp_path / "sim_data_extra_meta.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(meta_field_names=("norm_correction",)), _setup(), _options(), save_psfs=False)

    with h5py.File(data_path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{schema.KEY_SIMULATION_SECTION}/{schema.KEY_SIMULATION_META_FIELDS}"][:].astype(str),
            np.array(["norm_correction"]),
        )
        assert f[f"{schema.KEY_META_SECTION}/norm_correction"].shape == (3,)
        assert np.all(np.isnan(f[f"{schema.KEY_META_SECTION}/norm_correction"][...]))


def test_store_write_success_persists_declared_meta_fields(tmp_path):
    data_path = tmp_path / "sim_data_write_extra_meta.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(meta_field_names=("norm_correction",)), _setup(), _options(), save_psfs=False)

    store.write_simulation_success(0, _success_result(meta={"norm_correction": 0.75}))

    with h5py.File(data_path, "r") as f:
        assert f[f"{schema.KEY_META_SECTION}/norm_correction"][0] == np.float32(0.75)
        assert np.isnan(f[f"{schema.KEY_META_SECTION}/norm_correction"][1])

    analysis_meta = store.read_analysis_meta()
    simulation_meta = store.read_simulation_meta(0)
    np.testing.assert_allclose(analysis_meta["norm_correction"], np.asarray([0.75, np.nan, np.nan], dtype=np.float32))
    assert simulation_meta["norm_correction"] == np.float32(0.75)


def test_store_write_failure_clears_declared_meta_fields(tmp_path):
    data_path = tmp_path / "sim_data_clear_extra_meta.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(meta_field_names=("norm_correction",)), _setup(), _options(), save_psfs=False)

    store.write_simulation_success(0, _success_result(meta={"norm_correction": 0.75}))
    store.reset_to_pending(indexes=[0])
    store.write_simulation_failure(0)

    with h5py.File(data_path, "r") as f:
        assert np.isnan(f[f"{schema.KEY_META_SECTION}/norm_correction"][0])


def test_store_write_success_rejects_bad_meta_fields(tmp_path):
    data_path = tmp_path / "sim_data_bad_extra_meta.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(meta_field_names=("norm_correction",)), _setup(), _options(), save_psfs=False)

    with pytest.raises(ValueError, match="missing declared meta fields"):
        store.write_simulation_success(0, _success_result())

    with pytest.raises(ValueError, match="contains undeclared fields"):
        store.write_simulation_success(0, _success_result(meta={"norm_correction": 1.0, "other": 1.0}))

    with pytest.raises(ValueError, match="must contain only finite values"):
        store.write_simulation_success(0, _success_result(meta={"norm_correction": np.nan}))

    with pytest.raises(ValueError, match="must be a scalar"):
        store.write_simulation_success(0, _success_result(meta={"norm_correction": np.asarray([1.0])}))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (schema.KEY_SIMULATION_NAME, "broken.name", "Simulation payload name mismatch"),
        (schema.KEY_SIMULATION_VERSION, "broken.version", "Simulation payload version mismatch"),
        (
            schema.KEY_SIMULATION_EXTRA_STAT_NAMES,
            np.asarray(["broken_stat"], dtype=str),
            "Simulation payload extra stat registry mismatch",
        ),
        (
            schema.KEY_SIMULATION_NGS_MAG_STANDARD,
            "G_RP",
            "Simulation payload NGS magnitude standard mismatch",
        ),
    ],
)
def test_create_simulation_from_config_rejects_simulation_payload_core_field_overrides(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    match: str,
):
    original_prepare = MockSimulation.prepare_simulation_payload

    def _override_prepare(self, base_simulation_payload, simulation_cfg):
        payload = dict(original_prepare(self, base_simulation_payload, simulation_cfg))
        payload[field] = value
        return payload

    monkeypatch.setattr(MockSimulation, "prepare_simulation_payload", _override_prepare)

    with pytest.raises(ValueError, match=match):
        create_simulation_from_config({"name": "mock_simulation:MockSimulation"})


def test_create_simulation_from_config_rejects_removed_ngs_mag_standard(
    monkeypatch: pytest.MonkeyPatch,
):
    original_prepare = MockSimulation.prepare_simulation_payload

    def _remove_standard(self, base_simulation_payload, simulation_cfg):
        payload = dict(original_prepare(self, base_simulation_payload, simulation_cfg))
        del payload[schema.KEY_SIMULATION_NGS_MAG_STANDARD]
        return payload

    monkeypatch.setattr(MockSimulation, "prepare_simulation_payload", _remove_standard)

    with pytest.raises(ValueError, match="Missing required simulation keys: ngs_mag_standard"):
        create_simulation_from_config({"name": "mock_simulation:MockSimulation"})


def test_store_create_rejects_payload_without_ngs_mag_standard(tmp_path):
    payload = _simulation()
    del payload[schema.KEY_SIMULATION_NGS_MAG_STANDARD]

    with pytest.raises(ValueError, match="Missing required simulation keys: ngs_mag_standard"):
        SimulationStore(tmp_path / "missing_standard.h5").create(
            payload,
            _setup(),
            _options(),
        )


@pytest.mark.parametrize("standard", ["R", "G_RP", "future_standard"])
def test_create_simulation_from_config_accepts_stable_ngs_mag_standard(
    monkeypatch: pytest.MonkeyPatch,
    standard: str,
):
    monkeypatch.setattr(MockSimulation, "ngs_mag_standard", standard)

    _, payload = create_simulation_from_config({"name": "mock_simulation:MockSimulation"})

    assert payload[schema.KEY_SIMULATION_NGS_MAG_STANDARD] == standard


@pytest.mark.parametrize("standard", ["", "   ", " R ", 3, None])
def test_create_simulation_from_config_rejects_invalid_ngs_mag_standard(
    monkeypatch: pytest.MonkeyPatch,
    standard: object,
):
    monkeypatch.setattr(MockSimulation, "ngs_mag_standard", standard)

    with pytest.raises((TypeError, ValueError), match="ngs_mag_standard"):
        create_simulation_from_config({"name": "mock_simulation:MockSimulation"})


def test_create_simulation_from_payload_accepts_legacy_payload_without_ngs_mag_standard():
    payload = _mock_simulation()
    del payload[schema.KEY_SIMULATION_NGS_MAG_STANDARD]

    simulation = create_simulation_from_payload(payload)

    assert isinstance(simulation, MockSimulation)


def test_create_simulation_from_payload_rejects_other_invalid_legacy_fields():
    payload = _mock_simulation()
    del payload[schema.KEY_SIMULATION_NGS_MAG_STANDARD]
    payload[schema.KEY_SIMULATION_VERSION] = "broken.version"

    with pytest.raises(ValueError, match="Simulation payload version mismatch"):
        create_simulation_from_payload(payload)


def test_upstream_simulations_inherit_default_ngs_mag_standard():
    from ao_predict.simulation import HybridSimulation, TiptopSimulation

    assert TiptopSimulation().ngs_mag_standard == "R"
    assert HybridSimulation().ngs_mag_standard == "R"
    assert MockSimulation().ngs_mag_standard == "R"


def test_runner_resume_behavior(tmp_path):
    data_path = tmp_path / "sim_data.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)

    def run_one(idx: int) -> SimulationResult:
        if idx == 1:
            raise RuntimeError("boom")
        return _success_result(ny=2, nx=2)

    summary1 = run_pending_with_callback(store, run_one)
    assert summary1.attempted == 3
    assert summary1.succeeded == 2
    assert summary1.failed == 1

    summary2 = run_pending_with_callback(store, run_one)
    assert summary2.attempted == 0
    assert summary2.succeeded == 0
    assert summary2.failed == 0

    with h5py.File(data_path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:],
            np.array(
                [
                    int(SimulationState.SUCCEEDED),
                    int(SimulationState.FAILED),
                    int(SimulationState.SUCCEEDED),
                ],
                dtype=np.uint8,
            ),
        )
        assert schema.KEY_PSFS_SECTION not in f


def test_runner_with_simulation_interface(tmp_path):
    data_path = tmp_path / "sim_data.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)

    class TiptopSimulation(Simulation):
        _NAME = "ao_predict.simulation.tiptop:TiptopSimulation"
        _VERSION = "x.y"
        ngs_mag_standard = "R"

        def prepare_simulation_payload(self, base_simulation_payload, simulation_cfg):
            del simulation_cfg
            return {
                **dict(base_simulation_payload),
                "base_config": "[section]\\nvalue=1\\n",
            }

        def load_simulation_payload(self, simulation_payload):
            self._base_config = simulation_payload.get("base_config")

        def validate_simulation_payload(self, simulation_payload):
            _ = simulation_payload["base_config"]

        def prepare_setup_payload(self, base_setup_payload, setup_cfg):
            merged = dict(setup_cfg)
            merged.update(dict(base_setup_payload))
            return merged

        def prepare_options_payload(self, num_sims, setup_payload, base_options_payload):
            del num_sims
            del setup_payload
            return dict(base_options_payload)

        def validate_options_payload(self, num_sims, options_payload):
            del num_sims, options_payload

        def load_setup_payload(self, setup_payload):
            self._setup = SimulationSetup(
                ee_apertures_mas=np.asarray(setup_payload["ee_apertures_mas"], dtype=float).reshape(-1),
                sr_method=str(setup_payload["sr_method"]),
                fwhm_summary=str(setup_payload["fwhm_summary"]),
                ee_geometry=str(setup_payload["ee_geometry"]),
                atm_wavelength_um=float(setup_payload["atm_wavelength_um"]),
                atm_profiles=dict(setup_payload["atm_profiles"]),
                lgs_r_arcsec=np.asarray(setup_payload["lgs_r_arcsec"], dtype=float).reshape(-1),
                lgs_theta_deg=np.asarray(setup_payload["lgs_theta_deg"], dtype=float).reshape(-1),
                sci_r_arcsec=np.asarray(setup_payload["sci_r_arcsec"], dtype=float).reshape(-1),
                sci_theta_deg=np.asarray(setup_payload["sci_theta_deg"], dtype=float).reshape(-1),
            )

        def validate_setup_payload(self, setup_payload):
            _ = SimulationSetup(
                ee_apertures_mas=np.asarray(setup_payload["ee_apertures_mas"], dtype=float).reshape(-1),
                sr_method=str(setup_payload["sr_method"]),
                fwhm_summary=str(setup_payload["fwhm_summary"]),
                ee_geometry=str(setup_payload["ee_geometry"]),
                atm_wavelength_um=float(setup_payload["atm_wavelength_um"]),
                atm_profiles=dict(setup_payload["atm_profiles"]),
                lgs_r_arcsec=np.asarray(setup_payload["lgs_r_arcsec"], dtype=float).reshape(-1),
                lgs_theta_deg=np.asarray(setup_payload["lgs_theta_deg"], dtype=float).reshape(-1),
                sci_r_arcsec=np.asarray(setup_payload["sci_r_arcsec"], dtype=float).reshape(-1),
                sci_theta_deg=np.asarray(setup_payload["sci_theta_deg"], dtype=float).reshape(-1),
            )

        def create(self, index: int, options):
            context = SimulationContext(index=index, options=dict(options), setup=self._setup)
            context.runtime["created"] = True
            return context

        def run(self, context: SimulationContext) -> None:
            if context.index == 2:
                raise RuntimeError("intentional failure")
            context.runtime["ran"] = True

        def finalize(self, context: SimulationContext) -> None:
            context.result = _success_result(ny=2, nx=2, populate_stats=False, extra_stats=None)

        def prepare_psfs_for_stats(self, psfs, setup, meta):
            del setup, meta
            return normalize_psf_pixel_sum(np.asarray(psfs, dtype=np.float32))

    sim = TiptopSimulation()
    simulation_payload = store.read_simulation()
    sim.load_simulation_payload(simulation_payload)
    sim.load_setup_payload(store.read_setup())

    summary = run_pending_simulations(store, sim)
    assert summary.attempted == 3
    assert summary.succeeded == 2
    assert summary.failed == 1

    with h5py.File(data_path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:],
            np.array(
                [
                    int(SimulationState.SUCCEEDED),
                    int(SimulationState.SUCCEEDED),
                    int(SimulationState.FAILED),
                ],
                dtype=np.uint8,
            ),
        )


def _load_bound_mock_simulation(store: SimulationStore, simulation_cls: type[MockSimulation] = MockSimulation) -> MockSimulation:
    simulation = simulation_cls()
    simulation.load_simulation_payload(store.read_simulation())
    simulation.load_setup_payload(store.read_setup())
    return simulation


def test_runner_parallel_matches_serial_outputs(tmp_path):
    serial_path = tmp_path / "serial_data.h5"
    parallel_path = tmp_path / "parallel_data.h5"

    serial_store = SimulationStore(serial_path)
    serial_store.create(_mock_simulation(), _setup(), _options(num_sims=4), save_psfs=True)
    serial_summary = run_pending_simulations(
        serial_store,
        _load_bound_mock_simulation(serial_store),
        num_workers=1,
        chunk_multiple=1,
    )

    parallel_store = SimulationStore(parallel_path)
    parallel_store.create(_mock_simulation(), _setup(), _options(num_sims=4), save_psfs=True)
    parallel_summary = run_pending_simulations(
        parallel_store,
        _load_bound_mock_simulation(parallel_store),
        num_workers=2,
        chunk_multiple=1,
    )

    assert parallel_summary == serial_summary == RunSummary(attempted=4, succeeded=4, failed=0)
    with h5py.File(serial_path, "r") as serial, h5py.File(parallel_path, "r") as parallel:
        for path in (
            f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}",
            f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_SR}",
            f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_EE}",
            f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_FWHM_MAS}",
            f"{schema.KEY_META_SECTION}/{schema.KEY_META_PIXEL_SCALE_MAS}",
            f"{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}",
        ):
            np.testing.assert_allclose(parallel[path][...], serial[path][...], equal_nan=True)


def test_runner_parallel_reconstructs_simulation_from_persisted_payload(tmp_path):
    data_path = tmp_path / "parallel_reconstructs.h5"
    store = SimulationStore(data_path)
    store.create(_mock_simulation(), _setup(), _options(), save_psfs=False)

    summary = run_pending_simulations(
        store,
        MockSimulation(),
        num_workers=2,
        chunk_multiple=1,
    )

    assert summary == RunSummary(attempted=3, succeeded=3, failed=0)


def test_runner_parallel_calls_worker_warmup(tmp_path):
    data_path = tmp_path / "parallel_warmup.h5"
    store = SimulationStore(data_path)
    store.create(_mock_simulation(WarmupMockSimulation), _setup(), _options(), save_psfs=False)

    summary = run_pending_simulations(
        store,
        MockSimulation(),
        num_workers=2,
        chunk_multiple=1,
    )

    assert summary == RunSummary(attempted=3, succeeded=3, failed=0)


def test_runner_parallel_marks_chunk_failed_when_worker_warmup_fails(tmp_path):
    data_path = tmp_path / "parallel_warmup_failure.h5"
    store = SimulationStore(data_path)
    store.create(_mock_simulation(FailingWarmupMockSimulation), _setup(), _options(), save_psfs=False)

    summary = run_pending_simulations(
        store,
        MockSimulation(),
        verbose=True,
        num_workers=2,
        chunk_multiple=2,
    )

    assert summary == RunSummary(attempted=3, succeeded=0, failed=3)
    with h5py.File(data_path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:],
            np.array([2, 2, 2], dtype=np.uint8),
        )


def test_runner_parallel_persists_declared_extra_stats(tmp_path):
    data_path = tmp_path / "parallel_extra_stats.h5"
    store = SimulationStore(data_path)
    store.create(
        _mock_simulation(ExtraStatsMockSimulation, extra_stat_names=("halo_mas",)),
        _setup(),
        _options(),
        save_psfs=False,
    )

    summary = run_pending_simulations(
        store,
        MockSimulation(),
        num_workers=2,
        chunk_multiple=1,
    )

    assert summary == RunSummary(attempted=3, succeeded=3, failed=0)
    with h5py.File(data_path, "r") as f:
        np.testing.assert_allclose(
            f[f"{schema.KEY_STATS_SECTION}/halo_mas"][:],
            np.array(
                [
                    [10.0, 10.0, 10.0],
                    [11.0, 11.0, 11.0],
                    [12.0, 12.0, 12.0],
                ],
                dtype=np.float32,
            ),
        )


def test_runner_parallel_failure_isolation_and_retry(tmp_path):
    data_path = tmp_path / "parallel_retry.h5"
    marker_path = tmp_path / "failed_once.marker"
    store = SimulationStore(data_path)
    store.create(
        _mock_simulation(
            FailOnceMockSimulation,
            specific_fields={"marker_path": str(marker_path)},
        ),
        _setup(),
        _options(),
        save_psfs=False,
    )

    first = run_pending_simulations(
        store,
        MockSimulation(),
        verbose=True,
        num_workers=2,
        chunk_multiple=1,
    )
    assert first == RunSummary(attempted=3, succeeded=2, failed=1)
    with h5py.File(data_path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:],
            np.array([1, 2, 1], dtype=np.uint8),
        )

    second = run_simulations_by_state(
        store,
        MockSimulation(),
        SimulationState.FAILED,
        num_workers=2,
        chunk_multiple=1,
    )
    assert second == RunSummary(attempted=1, succeeded=1, failed=0)
    with h5py.File(data_path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:],
            np.array([1, 1, 1], dtype=np.uint8),
        )


def test_runner_parallel_filters_pending_and_failed_indexes(tmp_path):
    pending_path = tmp_path / "parallel_pending_subset.h5"
    pending_store = SimulationStore(pending_path)
    pending_store.create(_mock_simulation(), _setup(), _options(), save_psfs=False)

    pending_summary = run_pending_simulations(
        pending_store,
        MockSimulation(),
        indexes=[1],
        num_workers=2,
        chunk_multiple=1,
    )
    assert pending_summary == RunSummary(attempted=1, succeeded=1, failed=0)
    with h5py.File(pending_path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:],
            np.array([0, 1, 0], dtype=np.uint8),
        )

    failed_path = tmp_path / "parallel_failed_subset.h5"
    failed_store = SimulationStore(failed_path)
    failed_store.create(_mock_simulation(), _setup(), _options(), save_psfs=False)
    failed_store.write_simulation_failure(0)
    failed_store.write_simulation_failure(1)
    failed_store.write_simulation_failure(2)

    failed_summary = run_simulations_by_state(
        failed_store,
        MockSimulation(),
        SimulationState.FAILED,
        indexes=[2],
        num_workers=2,
        chunk_multiple=1,
    )
    assert failed_summary == RunSummary(attempted=1, succeeded=1, failed=0)
    with h5py.File(failed_path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:],
            np.array([2, 2, 1], dtype=np.uint8),
        )


def test_runner_rejects_invalid_parallel_controls(tmp_path):
    data_path = tmp_path / "parallel_invalid_controls.h5"
    store = SimulationStore(data_path)
    store.create(_mock_simulation(), _setup(), _options(), save_psfs=False)

    with pytest.raises(ValueError, match="num_workers must be >= 1"):
        run_pending_simulations(store, MockSimulation(), num_workers=0)
    with pytest.raises(ValueError, match="chunk_multiple must be >= 1"):
        run_pending_simulations(store, MockSimulation(), chunk_multiple=0)


def test_runner_with_simulation_interface_filtered_indexes(tmp_path):
    data_path = tmp_path / "sim_data.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)

    class TiptopSimulation(Simulation):
        _NAME = "ao_predict.simulation.tiptop:TiptopSimulation"
        _VERSION = "x.y"
        ngs_mag_standard = "R"

        def prepare_simulation_payload(self, base_simulation_payload, simulation_cfg):
            del simulation_cfg
            return {**dict(base_simulation_payload), "base_config": "[section]\\nvalue=1\\n"}

        def load_simulation_payload(self, simulation_payload):
            self._base_config = simulation_payload.get("base_config")

        def validate_simulation_payload(self, simulation_payload):
            _ = simulation_payload["base_config"]

        def prepare_setup_payload(self, base_setup_payload, setup_cfg):
            merged = dict(setup_cfg)
            merged.update(dict(base_setup_payload))
            return merged

        def prepare_options_payload(self, num_sims, setup_payload, base_options_payload):
            del num_sims
            del setup_payload
            return dict(base_options_payload)

        def validate_options_payload(self, num_sims, options_payload):
            del num_sims, options_payload

        def load_setup_payload(self, setup_payload):
            self._setup = SimulationSetup(
                ee_apertures_mas=np.asarray(setup_payload["ee_apertures_mas"], dtype=float).reshape(-1),
                sr_method=str(setup_payload["sr_method"]),
                fwhm_summary=str(setup_payload["fwhm_summary"]),
                ee_geometry=str(setup_payload["ee_geometry"]),
                atm_wavelength_um=float(setup_payload["atm_wavelength_um"]),
                atm_profiles=dict(setup_payload["atm_profiles"]),
                lgs_r_arcsec=np.asarray(setup_payload["lgs_r_arcsec"], dtype=float).reshape(-1),
                lgs_theta_deg=np.asarray(setup_payload["lgs_theta_deg"], dtype=float).reshape(-1),
                sci_r_arcsec=np.asarray(setup_payload["sci_r_arcsec"], dtype=float).reshape(-1),
                sci_theta_deg=np.asarray(setup_payload["sci_theta_deg"], dtype=float).reshape(-1),
            )

        def validate_setup_payload(self, setup_payload):
            _ = setup_payload["ee_apertures_mas"]

        def create(self, index: int, options):
            return SimulationContext(index=index, options=dict(options), setup=self._setup)

        def run(self, context: SimulationContext) -> None:
            _ = context

        def finalize(self, context: SimulationContext) -> None:
            context.result = _success_result(ny=2, nx=2, populate_stats=False, extra_stats=None)

        def prepare_psfs_for_stats(self, psfs, setup, meta):
            del setup, meta
            return normalize_psf_pixel_sum(np.asarray(psfs, dtype=np.float32))

    sim = TiptopSimulation()
    sim.load_simulation_payload(store.read_simulation())
    sim.load_setup_payload(store.read_setup())

    summary = run_pending_simulations(store, sim, indexes=[1])
    assert summary.attempted == 1
    assert summary.succeeded == 1
    assert summary.failed == 0

    with h5py.File(data_path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:],
            np.array(
                [
                    int(SimulationState.PENDING),
                    int(SimulationState.SUCCEEDED),
                    int(SimulationState.PENDING),
                ],
                dtype=np.uint8,
            ),
        )


def test_runner_persists_declared_extra_stats(tmp_path):
    data_path = tmp_path / "sim_data_declared_extra.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(extra_stat_names=("halo_mas",)), _setup(), _options(), save_psfs=False)

    class TiptopSimulation(Simulation):
        _NAME = "ao_predict.simulation.tiptop:TiptopSimulation"
        _VERSION = "x.y"
        ngs_mag_standard = "R"

        @property
        def extra_stat_names(self) -> tuple[str, ...]:
            return ("halo_mas",)

        def prepare_simulation_payload(self, base_simulation_payload, simulation_cfg):
            del simulation_cfg
            return {**dict(base_simulation_payload), "base_config": "[section]\\nvalue=1\\n"}

        def load_simulation_payload(self, simulation_payload):
            self._base_config = simulation_payload.get("base_config")

        def validate_simulation_payload(self, simulation_payload):
            _ = simulation_payload["base_config"]

        def prepare_setup_payload(self, base_setup_payload, setup_cfg):
            merged = dict(setup_cfg)
            merged.update(dict(base_setup_payload))
            return merged

        def prepare_options_payload(self, num_sims, setup_payload, base_options_payload):
            del num_sims, setup_payload
            return dict(base_options_payload)

        def validate_options_payload(self, num_sims, options_payload):
            del num_sims, options_payload

        def load_setup_payload(self, setup_payload):
            self._setup = SimulationSetup(
                ee_apertures_mas=np.asarray(setup_payload["ee_apertures_mas"], dtype=float).reshape(-1),
                sr_method=str(setup_payload["sr_method"]),
                fwhm_summary=str(setup_payload["fwhm_summary"]),
                ee_geometry=str(setup_payload["ee_geometry"]),
                atm_wavelength_um=float(setup_payload["atm_wavelength_um"]),
                atm_profiles=dict(setup_payload["atm_profiles"]),
                lgs_r_arcsec=np.asarray(setup_payload["lgs_r_arcsec"], dtype=float).reshape(-1),
                lgs_theta_deg=np.asarray(setup_payload["lgs_theta_deg"], dtype=float).reshape(-1),
                sci_r_arcsec=np.asarray(setup_payload["sci_r_arcsec"], dtype=float).reshape(-1),
                sci_theta_deg=np.asarray(setup_payload["sci_theta_deg"], dtype=float).reshape(-1),
            )

        def validate_setup_payload(self, setup_payload):
            _ = setup_payload["ee_apertures_mas"]

        def create(self, index: int, options):
            return SimulationContext(index=index, options=dict(options), setup=self._setup)

        def run(self, context: SimulationContext) -> None:
            del context

        def finalize(self, context: SimulationContext) -> None:
            context.result = _success_result(ny=2, nx=2, populate_stats=False, extra_stats=None)

        def build_extra_stats(self, context: SimulationContext):
            del context
            return {"halo_mas": np.full((3,), 7.0, dtype=np.float32)}

        def prepare_psfs_for_stats(self, psfs, setup, meta):
            del setup, meta
            return normalize_psf_pixel_sum(np.asarray(psfs, dtype=np.float32))

    sim = TiptopSimulation()
    sim.load_simulation_payload(store.read_simulation())
    sim.load_setup_payload(store.read_setup())

    summary = run_pending_simulations(store, sim)
    assert summary.attempted == 3
    assert summary.succeeded == 3
    assert summary.failed == 0

    with h5py.File(data_path, "r") as f:
        np.testing.assert_allclose(
            f[f"{schema.KEY_STATS_SECTION}/halo_mas"][:],
            np.full((3, 3), 7.0, dtype=np.float32),
        )


def test_store_validate_and_reset_failed(tmp_path):
    data_path = tmp_path / "sim_data.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)
    store.validate_schema()

    store.write_simulation_success(0, _success_result(ny=2, nx=2))
    store.write_simulation_failure(1)

    failed = store.failed_indices()
    assert failed.tolist() == [1]

    reset_count = store.reset_failed_to_pending()
    assert reset_count == 1
    assert store.pending_indices().tolist() == [1, 2]


def test_store_reset_all_to_pending(tmp_path):
    data_path = tmp_path / "sim_data.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)

    store.write_simulation_success(0, _success_result(ny=2, nx=2))
    store.write_simulation_failure(1)
    changed = store.reset_all_to_pending()
    assert changed == 2
    assert store.pending_indices().tolist() == [0, 1, 2]


def test_store_reset_selected_to_pending(tmp_path):
    data_path = tmp_path / "sim_data.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)

    store.write_simulation_success(0, _success_result(ny=2, nx=2))
    store.write_simulation_failure(1)
    changed = store.reset_to_pending(indexes=[1])
    assert changed == 1

    with h5py.File(data_path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:],
            np.array(
                [
                    int(SimulationState.SUCCEEDED),
                    int(SimulationState.PENDING),
                    int(SimulationState.PENDING),
                ],
                dtype=np.uint8,
            ),
        )


def test_store_create_rejects_partial_nan_ngs_triplets(tmp_path):
    data_path = tmp_path / "sim_data_partial_nan.h5"
    store = SimulationStore(data_path)
    options = _options(num_sims=2, max_ngs=3)
    options["ngs_r_arcsec"][0, 1] = np.nan
    # theta/mag for [0,1] remain finite -> invalid partial NaN state
    with np.testing.assert_raises(ValueError):
        store.create(_simulation(), _setup(), options, save_psfs=False)


def test_store_create_rejects_options_without_ngs_triplet(tmp_path):
    data_path = tmp_path / "sim_data_no_ngs.h5"
    store = SimulationStore(data_path)
    options = _options()
    for key in ("ngs_r_arcsec", "ngs_theta_deg", "ngs_mag"):
        options.pop(key)

    with np.testing.assert_raises(ValueError):
        store.create(_simulation(), _setup(), options, save_psfs=False)


def test_store_create_rejects_missing_required_option_keys(tmp_path):
    data_path = tmp_path / "sim_data_missing_options.h5"
    store = SimulationStore(data_path)
    options = {
        "wavelength_um": np.full((3,), 1.65, dtype=float),
    }
    with np.testing.assert_raises(ValueError):
        store.create(_simulation(), _setup(), options, save_psfs=False)


def test_store_create_rejects_unknown_option_keys(tmp_path):
    data_path = tmp_path / "sim_data_bad_options.h5"
    store = SimulationStore(data_path)
    options = {
        "wavelength_um": np.full((2,), 1.65, dtype=float),
        "bad_option": np.ones((2,), dtype=float),
    }
    with np.testing.assert_raises(ValueError):
        store.create(_simulation(), _setup(), options, save_psfs=False)


def test_store_schema_reports_invalid_state_values(tmp_path):
    data_path = tmp_path / "sim_data_bad_state.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)

    with h5py.File(data_path, "r+") as f:
        f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][1] = np.uint8(9)

    issues = store.collect_schema_issues()
    assert any("invalid values" in issue for issue in issues)


def test_store_write_success_clears_optional_outputs_on_rerun(tmp_path):
    data_path = tmp_path / "sim_data_optional_clear.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=True)

    store.write_simulation_success(0, _success_result())
    changed = store.reset_to_pending(indexes=[0])
    assert changed == 1
    with np.testing.assert_raises(ValueError):
        store.write_simulation_success(0, _success_result_missing_required_outputs())

    with h5py.File(data_path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:],
            np.array(
                [
                    int(SimulationState.PENDING),
                    int(SimulationState.PENDING),
                    int(SimulationState.PENDING),
                ],
                dtype=np.uint8,
            ),
        )
        assert np.all(np.isfinite(f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_FWHM_MAS}"][0]))
        assert np.all(np.isfinite(f[f"{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"][0]))


def test_store_write_success_accepts_nan_fwhm(tmp_path):
    data_path = tmp_path / "sim_data_nan_fwhm.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=True)

    result = _success_result()
    result.stats[schema.KEY_STATS_FWHM_MAS][:] = np.nan
    store.write_simulation_success(0, result)

    with h5py.File(data_path, "r") as f:
        np.testing.assert_allclose(
            f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_SR}"][0],
            result.stats[schema.KEY_STATS_SR],
        )
        np.testing.assert_allclose(
            f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_EE}"][0],
            result.stats[schema.KEY_STATS_EE],
        )
        assert np.all(np.isnan(f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_FWHM_MAS}"][0]))


def test_store_write_success_rejects_psf_science_dimension_mismatch(tmp_path):
    data_path = tmp_path / "sim_data_bad_psf_m.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=True)

    bad_result = _success_result()
    bad_result.psfs = np.full((2, 4, 4), 0.1, dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        store.write_simulation_success(0, bad_result)


def test_store_write_failure_clears_outputs(tmp_path):
    data_path = tmp_path / "sim_data_failure_clears.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=True)
    store.write_simulation_success(0, _success_result())
    changed = store.reset_to_pending(indexes=[0])
    assert changed == 1
    store.write_simulation_failure(0)

    with h5py.File(data_path, "r") as f:
        np.testing.assert_array_equal(
            f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:],
            np.array(
                [
                    int(SimulationState.FAILED),
                    int(SimulationState.PENDING),
                    int(SimulationState.PENDING),
                ],
                dtype=np.uint8,
            ),
        )
        assert np.all(np.isnan(f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_SR}"][0]))
        assert np.all(np.isnan(f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_FWHM_MAS}"][0]))
        assert np.isnan(f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_PIXEL_SCALE_MAS}"][0])
        assert f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_DIAMETER_M}"][()] == np.float32(8.0)
        assert np.all(np.isfinite(f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_PUPIL}"][...]))
        assert np.all(np.isnan(f[f"{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"][0]))


def test_store_rejects_negative_simulation_indexes(tmp_path):
    data_path = tmp_path / "sim_data_negative_index.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)

    with np.testing.assert_raises(IndexError):
        store.read_sim_options(-1)
    with np.testing.assert_raises(IndexError):
        store.write_simulation_failure(-1)
    with np.testing.assert_raises(IndexError):
        store.write_simulation_success(-1, _success_result(ny=2, nx=2))


def test_store_read_extra_stat_names(tmp_path):
    data_path = tmp_path / "sim_data_read_extra_stats.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(extra_stat_names=("halo_mas", "encircled_bg")), _setup(), _options(), save_psfs=False)

    assert store.read_extra_stat_names() == ("halo_mas", "encircled_bg")


def test_store_read_simulation_meta_includes_dataset_level_telescope_metadata(tmp_path):
    data_path = tmp_path / "sim_data_read_meta.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)
    store.write_simulation_success(0, _success_result())

    meta = store.read_simulation_meta(0)

    assert meta[schema.KEY_META_PIXEL_SCALE_MAS] == np.float32(4.0)
    assert meta[schema.KEY_META_TEL_DIAMETER_M] == np.float32(8.0)
    np.testing.assert_allclose(meta[schema.KEY_META_TEL_PUPIL], np.ones((6, 6), dtype=np.float32))


def test_store_read_simulation_stats_without_declared_extra_stats(tmp_path):
    data_path = tmp_path / "sim_data_read_stats_core.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)
    result = _success_result()
    store.write_simulation_success(0, result)

    stats = store.read_simulation_stats(0)

    assert tuple(stats.keys()) == schema.CORE_STATS_KEYS
    np.testing.assert_allclose(stats[schema.KEY_STATS_SR], result.stats[schema.KEY_STATS_SR])
    np.testing.assert_allclose(stats[schema.KEY_STATS_EE], result.stats[schema.KEY_STATS_EE])
    np.testing.assert_allclose(stats[schema.KEY_STATS_FWHM_MAS], result.stats[schema.KEY_STATS_FWHM_MAS])


def test_store_read_simulation_stats_with_declared_extra_stats(tmp_path):
    data_path = tmp_path / "sim_data_read_stats_extra.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(extra_stat_names=("halo_mas",)), _setup(), _options(), save_psfs=False)
    result = _success_result(extra_stats={"halo_mas": np.full((3,), 7.0, dtype=np.float32)})
    store.write_simulation_success(0, result)

    stats = store.read_simulation_stats(0)

    assert tuple(stats.keys()) == schema.CORE_STATS_KEYS + ("halo_mas",)
    np.testing.assert_allclose(stats["halo_mas"], np.full((3,), 7.0, dtype=np.float32))


def test_store_read_simulation_psfs(tmp_path):
    data_path = tmp_path / "sim_data_read_psfs.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=True)
    result = _success_result()
    store.write_simulation_success(0, result)

    psfs = store.read_simulation_psfs(0)

    np.testing.assert_allclose(psfs, result.psfs)


def test_store_read_simulation_psfs_rejects_missing_psf_dataset(tmp_path):
    data_path = tmp_path / "sim_data_missing_psfs.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)

    with pytest.raises(ValueError, match=r"Missing required dataset '/psfs/data'\."):
        store.read_simulation_psfs(0)


def test_store_read_simulation_stats_rejects_missing_declared_extra_stat_dataset(tmp_path):
    data_path = tmp_path / "sim_data_missing_declared_stat.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(extra_stat_names=("halo_mas",)), _setup(), _options(), save_psfs=False)

    with h5py.File(data_path, "r+") as f:
        del f[f"{schema.KEY_STATS_SECTION}/halo_mas"]

    with pytest.raises(ValueError, match=r"Missing required dataset '/stats/halo_mas'\."):
        store.read_simulation_stats(0)


@pytest.mark.parametrize(
    "reader",
    [
        lambda store: store.read_sim_options(99),
        lambda store: store.read_simulation_meta(99),
        lambda store: store.read_simulation_stats(99),
        lambda store: store.read_simulation_psfs(99),
    ],
)
def test_store_read_methods_validate_out_of_range_indexes(tmp_path, reader):
    data_path = tmp_path / "sim_data_read_index_oob.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=True)
    store.write_simulation_success(0, _success_result())

    with pytest.raises(IndexError, match=r"sim_idx 99 out of range"):
        reader(store)
