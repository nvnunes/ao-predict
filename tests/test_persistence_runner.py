from __future__ import annotations

from collections.abc import Mapping

import h5py
import numpy as np
import pytest

from ao_predict.persistence import SimulationStore
from ao_predict.simulation import (
    Simulation,
    SimulationContext,
    SimulationResult,
    SimulationSetup,
    SimulationState,
    schema,
)
from ao_predict.simulation.runner import _populate_result_stats
from ao_predict.simulation.runner import create_simulation_from_config, run_pending_simulations
from ao_predict.simulation.stats import compute_psf_stats
from ao_predict.simulation.validation import validate_successful_result
from helpers import run_pending_with_callback
from mock_simulation import MockSimulation


def _simulation(*, extra_stat_names: tuple[str, ...] = ()) -> dict:
    return {
        "name": "ao_predict.simulation.tiptop:TiptopSimulation",
        "version": "x.y",
        "extra_stat_names": np.asarray(extra_stat_names, dtype=str),
        "base_config": "[section]\nvalue=1\n",
    }


def _setup() -> dict:
    return {
        "ee_apertures_mas": np.array([50.0, 100.0], dtype=float),
        "sr_method": schema.DEFAULT_SETUP_SR_METHOD,
        "fwhm_summary": schema.DEFAULT_SETUP_FWHM_SUMMARY,
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


def _success_result(
    m: int = 3,
    a: int = 2,
    ny: int = 4,
    nx: int = 4,
    *,
    populate_stats: bool = True,
    extra_stats: dict[str, np.ndarray] | None = None,
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
    return SimulationResult(
        state=SimulationState.SUCCEEDED,
        stats=stats,
        meta={
            "pixel_scale_mas": 4.0,
            "tel_diameter_m": 8.0,
            "tel_pupil": np.ones((6, 6), dtype=np.float32),
        },
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
        atm_wavelength_um=float(setup["atm_wavelength_um"]),
        atm_profiles=dict(setup["atm_profiles"]),
        lgs_r_arcsec=np.asarray(setup["lgs_r_arcsec"], dtype=float).reshape(-1),
        lgs_theta_deg=np.asarray(setup["lgs_theta_deg"], dtype=float).reshape(-1),
        sci_r_arcsec=np.asarray(setup["sci_r_arcsec"], dtype=float).reshape(-1),
        sci_theta_deg=np.asarray(setup["sci_theta_deg"], dtype=float).reshape(-1),
    )


class _ExtraStatsSimulation(Simulation):
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

    def build_extra_stats(self, context: SimulationContext):
        del context
        return dict(self._extra_stats)


def test_validate_success_result_accepts_valid_success_result():
    validate_successful_result(_success_result(), 3, 2, require_psfs=True)


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
    context = SimulationContext(index=0, setup=_setup_obj(), options={})
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
    context = SimulationContext(index=0, setup=_setup_obj(), options={})
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
    context = SimulationContext(index=0, setup=_setup_obj(), options={})
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


def test_compute_psf_stats_rejects_missing_ee_apertures():
    with pytest.raises(
        ValueError,
        match=r"setup\['ee_apertures_mas'\] is required for PSF stats computation\.",
    ):
        compute_psf_stats(
            np.zeros((3, 4, 4), dtype=np.float32),
            {},
            {schema.KEY_META_PIXEL_SCALE_MAS: 4.0},
        )


def test_compute_psf_stats_rejects_missing_sr_method():
    with pytest.raises(
        ValueError,
        match=r"setup\['sr_method'\] is required for PSF stats computation\.",
    ):
        compute_psf_stats(
            np.zeros((3, 4, 4), dtype=np.float32),
            {
                schema.KEY_SETUP_EE_APERTURES_MAS: np.array([50.0], dtype=float),
                schema.KEY_SETUP_FWHM_SUMMARY: schema.DEFAULT_SETUP_FWHM_SUMMARY,
            },
            {schema.KEY_META_PIXEL_SCALE_MAS: 4.0},
        )


def test_compute_psf_stats_rejects_missing_fwhm_summary():
    with pytest.raises(
        ValueError,
        match=r"setup\['fwhm_summary'\] is required for PSF stats computation\.",
    ):
        compute_psf_stats(
            np.zeros((3, 4, 4), dtype=np.float32),
            {
                schema.KEY_SETUP_EE_APERTURES_MAS: np.array([50.0], dtype=float),
                schema.KEY_SETUP_SR_METHOD: schema.DEFAULT_SETUP_SR_METHOD,
            },
            {schema.KEY_META_PIXEL_SCALE_MAS: 4.0},
        )


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
        assert f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_DIAMETER_M}"][0] == np.float32(8.0)
        assert f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_PUPIL}"].shape == (3, 6, 6)

        assert f[f"{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"].shape == (3, 3, 4, 4)
        assert np.all(np.isfinite(f[f"{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"][0]))
        assert np.all(np.isnan(f[f"{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"][1]))


def test_store_create_preallocates_empty_tel_pupil_dataset(tmp_path):
    data_path = tmp_path / "sim_data.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)

    with h5py.File(data_path, "r") as f:
        assert f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_PUPIL}"].shape == (3, 0, 0)


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


def test_runner_with_simulation_interface_filtered_indexes(tmp_path):
    data_path = tmp_path / "sim_data.h5"
    store = SimulationStore(data_path)
    store.create(_simulation(), _setup(), _options(), save_psfs=False)

    class TiptopSimulation(Simulation):
        _NAME = "ao_predict.simulation.tiptop:TiptopSimulation"
        _VERSION = "x.y"

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
        assert np.isnan(f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_DIAMETER_M}"][0])
        assert np.all(np.isnan(f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_PUPIL}"][0]))
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
