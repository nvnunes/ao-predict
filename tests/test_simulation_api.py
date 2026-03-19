from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pytest

import ao_predict.simulation.api as sim_api
from ao_predict.simulation.helpers import normalize_psf_pixel_sum
from ao_predict.simulation import (
    Simulation,
    SimulationContext,
    SimulationResult,
    SimulationSetup,
    SimulationState,
    schema,
)
from ao_predict.simulation.api import InitDatasetRequest, OptionsConfig, SetupConfig, SimulationConfig, TableOptionsConfig

TIPTOP_INI_TEXT = (
    "[main]\nvalue=1\n"
    "[telescope]\nZenithAngle=20\nTelescopeDiameter=8.0\n"
    "[sources_LO]\nWavelength=[710e-9]\nZenith=[1,2]\nAzimuth=[0,180]\n"
    "[sources_HO]\nZenith=[30,30,30,30]\nAzimuth=[45,135,225,315]\n"
    "[sources_science]\nWavelength=[1.65e-06]\nZenith=[0,10,20]\nAzimuth=[0,90,180]\n"
    "[atmosphere]\nWavelength=500e-9\nr0_Value=0.16\nL0=25\nCn2Heights=[0,5000]\nCn2Weights=[0.6,0.4]\nWindSpeed=[5,10]\nWindDirection=[0,90]\n"
)


def _write_ini(tmp_path: Path) -> Path:
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(TIPTOP_INI_TEXT, encoding="utf-8")
    return ini_path


def _base_request(tmp_path: Path) -> InitDatasetRequest:
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "data.h5"
    return InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="Tiptop", base_path=str(Path(ini_path).parent), specific_fields={"config_path": str(ini_path)}),
        setup=SetupConfig(
            ee_apertures_mas=[50.0, 100.0],
            sr_method=schema.DEFAULT_SETUP_SR_METHOD,
            fwhm_summary=schema.DEFAULT_SETUP_FWHM_SUMMARY,
            specific_fields={"ngs_mag_zeropoint": 1.1e13 / 368.0},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength_um": np.array([1.65, 1.65, 1.65], dtype=float),
                "atm_profile_id": np.array([0, 0, 0], dtype=np.int32),
                "zenith_angle_deg": np.array([20.0, 25.0, 30.0], dtype=float),
                "r0_m": np.array([0.16, 0.15, 0.14], dtype=float),
                "ngs_r_arcsec": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float),
                "ngs_theta_deg": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float),
                "ngs_mag": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float),
            }
        ),
    )


def _success_result(m: int = 3, *, with_stats: bool = True, with_psfs: bool = True) -> SimulationResult:
    result = SimulationResult(
        state=SimulationState.SUCCEEDED,
        meta={
            "pixel_scale_mas": 4.0,
            "tel_diameter_m": 8.0,
            "tel_pupil": np.ones((6, 6), dtype=np.float32),
        },
        psfs=np.zeros((m, 4, 4), dtype=np.float32) if with_psfs else None,
    )
    if with_stats:
        result.stats = {
            "sr": np.linspace(0.1, 0.3, m, dtype=np.float32),
            "ee": np.full((m, 2), 0.5, dtype=np.float32),
            "fwhm_mas": np.full((m,), 60.0, dtype=np.float32),
        }
    return result


class FakeSimulation(Simulation):
    _NAME = "ao_predict.simulation.tiptop:TiptopSimulation"
    _VERSION = "0.0.1"

    def __init__(self, fail_idx: int | None = None):
        self.fail_idx = fail_idx
        self.failed_once: set[int] = set()

    def prepare_simulation_payload(self, base_simulation_payload, simulation_cfg):
        return {
            **dict(base_simulation_payload),
            "base_config": str(simulation_cfg.get("config_path", "")),
        }

    def load_simulation_payload(self, simulation_payload):
        self._base_config = simulation_payload.get("base_config")

    def validate_simulation_payload(self, simulation_payload):
        _ = simulation_payload["base_config"]

    def prepare_setup_payload(self, base_setup_payload, setup_cfg):
        merged = dict(setup_cfg)
        merged.update(dict(base_setup_payload))
        merged["atm_wavelength_um"] = float(merged.get("atm_wavelength_um", 0.5))
        return merged

    def prepare_options_payload(self, num_sims, setup_payload, base_options_payload):
        _ = setup_payload
        out = {str(k): np.asarray(v).copy() for k, v in base_options_payload.items()}
        n = int(num_sims)
        out.setdefault("wavelength_um", np.full((n,), 1.65, dtype=float))
        out.setdefault("zenith_angle_deg", np.full((n,), 20.0, dtype=float))
        out.setdefault("atm_profile_id", np.zeros((n,), dtype=np.int32))
        out.setdefault("r0_m", np.full((n,), 0.16, dtype=float))
        if not any(key in out for key in ("ngs_r_arcsec", "ngs_theta_deg", "ngs_mag")):
            out["ngs_r_arcsec"] = np.full((n, 1), 1.0, dtype=float)
            out["ngs_theta_deg"] = np.full((n, 1), 0.0, dtype=float)
            out["ngs_mag"] = np.full((n, 1), 15.0, dtype=float)
        out["atm_profile_id"] = np.asarray(out["atm_profile_id"], dtype=np.int32).reshape(-1)
        return out

    def validate_options_payload(self, num_sims, options_payload):
        _ = num_sims
        _ = options_payload

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
        if self.fail_idx is not None and context.index == self.fail_idx and context.index not in self.failed_once:
            self.failed_once.add(context.index)
            raise RuntimeError("intentional failure")

    def finalize(self, context: SimulationContext) -> None:
        context.result = _success_result(with_stats=False)

    def prepare_psfs_for_stats(self, psfs, setup, meta):
        del setup, meta
        return normalize_psf_pixel_sum(np.asarray(psfs, dtype=np.float32))


def test_api_init_and_check(tmp_path: Path):
    request = _base_request(tmp_path)
    num_sims = sim_api.init_dataset(request)
    dataset_path = Path(request.dataset_path)
    assert num_sims == 3
    assert dataset_path.exists()

    status = sim_api.check_dataset(dataset_path)
    assert status.num_sims == 3
    assert status.num_pending == 3
    assert status.num_failed == 0
    assert status.ok is False
    with pytest.raises(sim_api.DatasetValidationError):
        sim_api.validate_dataset(dataset_path)


def test_api_full_pipeline_with_test_simulation(tmp_path: Path):
    dataset_path = tmp_path / "test_simulation.h5"
    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="mock_simulation:MockSimulation"),
        setup=SetupConfig(ee_apertures_mas=[50.0, 100.0]),
        options=OptionsConfig(
            option_arrays={
                "zenith_angle_deg": np.array([15.0, 25.0, 35.0], dtype=float),
            }
        ),
    )

    num_sims = sim_api.init_dataset(request)
    assert num_sims == 3

    summary = sim_api.run_simulations_by_state(dataset_path, state=SimulationState.PENDING)
    assert summary.attempted == 3
    assert summary.succeeded == 3
    assert summary.failed == 0

    status = sim_api.check_dataset(dataset_path)
    assert status.ok is True
    assert status.num_pending == 0
    assert status.num_failed == 0
    assert status.num_succeeded == 3

    with h5py.File(dataset_path, "r") as f:
        sr = np.asarray(f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_SR}"][:], dtype=float)
        assert np.all(np.isfinite(sr))
        np.testing.assert_allclose(sr[:, 0], np.full((3,), sr[0, 0], dtype=float), rtol=1e-6, atol=1e-6)
        assert f[f"{schema.KEY_SETUP_SECTION}/{schema.KEY_SETUP_SR_METHOD}"][()].decode("utf-8") == schema.DEFAULT_SETUP_SR_METHOD
        assert (
            f[f"{schema.KEY_SETUP_SECTION}/{schema.KEY_SETUP_FWHM_SUMMARY}"][()].decode("utf-8")
            == schema.DEFAULT_SETUP_FWHM_SUMMARY
        )


def test_api_init_persists_explicit_setup_stats_selectors(tmp_path: Path):
    request = _base_request(tmp_path)
    request = InitDatasetRequest(
        dataset_path=request.dataset_path,
        simulation=request.simulation,
        setup=SetupConfig(
            ee_apertures_mas=[50.0, 100.0],
            sr_method=schema.STATS_SR_METHOD_PIXEL_MAX,
            fwhm_summary=schema.STATS_FWHM_SUMMARY_MAX,
            specific_fields={"ngs_mag_zeropoint": 1.1e13 / 368.0},
        ),
        options=request.options,
    )

    sim_api.init_dataset(request)

    with h5py.File(request.dataset_path, "r") as f:
        assert f[f"{schema.KEY_SETUP_SECTION}/{schema.KEY_SETUP_SR_METHOD}"][()].decode("utf-8") == schema.STATS_SR_METHOD_PIXEL_MAX
        assert (
            f[f"{schema.KEY_SETUP_SECTION}/{schema.KEY_SETUP_FWHM_SUMMARY}"][()].decode("utf-8")
            == schema.STATS_FWHM_SUMMARY_MAX
        )


def test_api_init_rejects_invalid_setup_stats_selector(tmp_path: Path):
    request = _base_request(tmp_path)
    request = InitDatasetRequest(
        dataset_path=request.dataset_path,
        simulation=request.simulation,
        setup=SetupConfig(
            ee_apertures_mas=[50.0, 100.0],
            sr_method="bad_selector",
            specific_fields={"ngs_mag_zeropoint": 1.1e13 / 368.0},
        ),
        options=request.options,
    )

    with pytest.raises(ValueError, match="setup\\['sr_method'\\] must be one of: pixel_fit, pixel_max\\."):
        sim_api.init_dataset(request)


def test_api_run_and_retry(tmp_path: Path, monkeypatch):
    request = _base_request(tmp_path)
    num_sims = sim_api.init_dataset(request)
    dataset_path = Path(request.dataset_path)
    assert num_sims == 3

    sim = FakeSimulation(fail_idx=1)
    monkeypatch.setattr(sim_api, "create_simulation_from_payload", lambda _payload: sim)

    summary1 = sim_api.run_simulations_by_state(dataset_path, state=SimulationState.PENDING)
    assert summary1.attempted == 3
    assert summary1.succeeded == 2
    assert summary1.failed == 1

    status1 = sim_api.check_dataset(dataset_path)
    assert status1.num_pending == 0
    assert status1.num_failed == 1
    assert status1.ok is False

    summary2 = sim_api.run_simulations_by_state(dataset_path, state=SimulationState.FAILED)
    assert summary2.attempted == 1
    assert summary2.succeeded == 1
    assert summary2.failed == 0

    status2 = sim_api.check_dataset(dataset_path)
    assert status2.num_pending == 0
    assert status2.num_failed == 0
    assert status2.ok is True
    sim_api.validate_dataset(dataset_path)


def test_api_run_and_retry_with_indexes(tmp_path: Path, monkeypatch):
    request = _base_request(tmp_path)
    num_sims = sim_api.init_dataset(request)
    dataset_path = Path(request.dataset_path)
    assert num_sims == 3

    sim = FakeSimulation()
    monkeypatch.setattr(sim_api, "create_simulation_from_payload", lambda _payload: sim)

    summary1 = sim_api.run_simulations_by_state(dataset_path, state=SimulationState.PENDING, indexes=[1])
    assert summary1.attempted == 1
    assert summary1.succeeded == 1
    assert summary1.failed == 0

    store = sim_api.SimulationStore(dataset_path)
    store.write_simulation_failure(0)
    store.write_simulation_failure(2)

    summary2 = sim_api.run_simulations_by_state(dataset_path, state=SimulationState.FAILED, indexes=[2])
    assert summary2.attempted == 1
    assert summary2.succeeded == 1
    assert summary2.failed == 0

    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_array_equal(f["status/state"][:], np.array([2, 1, 1], dtype=np.uint8))


def test_api_reset(tmp_path: Path):
    request = _base_request(tmp_path)
    num_sims = sim_api.init_dataset(request)
    dataset_path = Path(request.dataset_path)
    assert num_sims == 3
    store = sim_api.SimulationStore(dataset_path)

    store.write_simulation_success(0, _success_result())
    store.write_simulation_failure(1)

    changed = sim_api.reset_simulations(dataset_path)
    assert changed == 2
    np.testing.assert_array_equal(store.pending_indices(), np.array([0, 1, 2], dtype=np.int64))


def test_api_reset_with_indexes(tmp_path: Path):
    request = _base_request(tmp_path)
    num_sims = sim_api.init_dataset(request)
    dataset_path = Path(request.dataset_path)
    assert num_sims == 3
    store = sim_api.SimulationStore(dataset_path)

    store.write_simulation_success(0, _success_result())
    store.write_simulation_failure(1)

    changed = sim_api.reset_simulations(dataset_path, indexes=[1])
    assert changed == 1
    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_array_equal(f["status/state"][:], np.array([1, 0, 0], dtype=np.uint8))


def _r0_from_seeing_arcsec(seeing_arcsec: np.ndarray, atm_wavelength_um: float = 0.5) -> np.ndarray:
    seeing_rad = np.asarray(seeing_arcsec, dtype=float) * (math.pi / 648000.0)
    return 0.98 * (float(atm_wavelength_um) * 1e-6) / seeing_rad


def test_api_init_accepts_seeing_alias_columns(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "seeing_columns.h5"
    seeing = np.array([0.70, 0.80, 0.90], dtype=float)

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="Tiptop", base_path=str(Path(ini_path).parent), specific_fields={"config_path": str(ini_path)}),
        setup=SetupConfig(
            ee_apertures_mas=[50.0, 100.0],
            specific_fields={"ngs_mag_zeropoint": 1.1e13 / 368.0},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength_um": np.array([1.65, 1.65, 1.65], dtype=float),
                "atm_profile_id": np.array([0, 0, 0], dtype=np.int32),
                "zenith_angle_deg": np.array([20.0, 25.0, 30.0], dtype=float),
                "seeing_arcsec": seeing,
                "ngs_r_arcsec": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float),
                "ngs_theta_deg": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float),
                "ngs_mag": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float),
            }
        ),
    )
    sim_api.init_dataset(request)

    with h5py.File(dataset_path, "r") as f:
        assert "seeing_arcsec" not in f["options"]
        np.testing.assert_allclose(
            np.asarray(f["options/r0_m"][:], dtype=float),
            _r0_from_seeing_arcsec(seeing),
            rtol=1e-6,
            atol=1e-8,
        )


def test_api_init_accepts_options_input_config_table(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "seeing_table_config.h5"
    seeing = np.array([0.70, 0.80, 0.90], dtype=float)

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="Tiptop", base_path=str(Path(ini_path).parent), specific_fields={"config_path": str(ini_path)}),
        setup=SetupConfig(
            ee_apertures_mas=[50.0, 100.0],
            specific_fields={"ngs_mag_zeropoint": 1.1e13 / 368.0},
        ),
        options=TableOptionsConfig(
            broadcast={},
            columns=[
                "wavelength_um",
                "atm_profile_id",
                "zenith_angle_deg",
                "seeing_arcsec",
                "ngs1_r_arcsec",
                "ngs1_theta_deg",
                "ngs1_mag",
            ],
            rows=[
                [1.65, 0, 20.0, float(seeing[0]), 1.0, 0.0, 14.0],
                [1.65, 0, 25.0, float(seeing[1]), 1.0, 0.0, 14.0],
                [1.65, 0, 30.0, float(seeing[2]), 1.0, 0.0, 14.0],
            ],
        ),
    )
    sim_api.init_dataset(request)

    with h5py.File(dataset_path, "r") as f:
        assert "seeing_arcsec" not in f["options"]
        np.testing.assert_allclose(
            np.asarray(f["options/r0_m"][:], dtype=float),
            _r0_from_seeing_arcsec(seeing),
            rtol=1e-6,
            atol=1e-8,
        )


def test_api_init_accepts_ragged_ngs_table_input(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "ragged_ngs_table.h5"

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="Tiptop", base_path=str(Path(ini_path).parent), specific_fields={"config_path": str(ini_path)}),
        setup=SetupConfig(
            ee_apertures_mas=[50.0, 100.0],
            specific_fields={"ngs_mag_zeropoint": 1.1e13 / 368.0},
        ),
        options=TableOptionsConfig(
            broadcast={},
            columns=[
                "wavelength_um",
                "atm_profile_id",
                "zenith_angle_deg",
                "r0_m",
                "ngs1_r_arcsec",
                "ngs1_theta_deg",
                "ngs1_mag",
                "ngs2_r_arcsec",
                "ngs2_theta_deg",
                "ngs2_mag",
            ],
            rows=[
                [1.65, 0, 20.0, 0.16, 1.0, 0.0, 14.0, 2.0, 180.0, 15.0],
                [1.65, 0, 25.0, 0.15, 1.5, 30.0, 14.5, None, None, None],
            ],
        ),
    )

    sim_api.init_dataset(request)

    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_allclose(
            np.asarray(f["options/ngs_r_arcsec"][:], dtype=float),
            np.array([[1.0, 2.0], [1.5, np.nan]], dtype=float),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(f["options/ngs_theta_deg"][:], dtype=float),
            np.array([[0.0, 180.0], [30.0, np.nan]], dtype=float),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(f["options/ngs_mag"][:], dtype=float),
            np.array([[14.0, 15.0], [14.5, np.nan]], dtype=float),
            equal_nan=True,
        )


def test_api_init_rejects_non_columnar_options_mapping(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "cli_normalized_options.h5"
    seeing = np.array([0.70, 0.80, 0.90], dtype=float)

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation={"name": "Tiptop", "config_path": str(ini_path)},
        setup={"ee_apertures_mas": [50.0, 100.0], "ngs_mag_zeropoint": 1.1e13 / 368.0},
        options={
            "broadcast": {},
            "columns": [
                "wavelength_um",
                "atm_profile_id",
                "zenith_angle_deg",
                "seeing_arcsec",
                "ngs1_r_arcsec",
                "ngs1_theta_deg",
                "ngs1_mag",
            ],
            "rows": [
                [1.65, 0, 20.0, float(seeing[0]), 1.0, 0.0, 14.0],
                [1.65, 0, 25.0, float(seeing[1]), 1.0, 0.0, 14.0],
                [1.65, 0, 30.0, float(seeing[2]), 1.0, 0.0, 14.0],
            ],
        },
    )
    with pytest.raises(ValueError, match="per-simulation"):
        sim_api.init_dataset(request)


def test_api_init_rejects_non_columnar_table_payload(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "table_row_width_mismatch.h5"

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation={"name": "Tiptop", "config_path": str(ini_path)},
        setup={"ee_apertures_mas": [50.0, 100.0], "ngs_mag_zeropoint": 1.1e13 / 368.0},
        options={
            "broadcast": {},
            "columns": [
                "wavelength_um",
                "atm_profile_id",
                "zenith_angle_deg",
                "r0_m",
                "ngs1_r_arcsec",
                "ngs1_theta_deg",
                "ngs1_mag",
            ],
            "rows": [
                [1.65, 0, 20.0, 0.12, 1.0, 0.0],
            ],
        },
    )

    with pytest.raises(ValueError, match="per-simulation"):
        sim_api.init_dataset(request)


def test_api_init_rejects_inconsistent_r0_and_seeing(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "seeing_conflict.h5"
    seeing = np.array([0.70, 0.80, 0.90], dtype=float)

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="Tiptop", base_path=str(Path(ini_path).parent), specific_fields={"config_path": str(ini_path)}),
        setup=SetupConfig(
            ee_apertures_mas=[50.0, 100.0],
            specific_fields={"ngs_mag_zeropoint": 1.1e13 / 368.0},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength_um": np.array([1.65, 1.65, 1.65], dtype=float),
                "atm_profile_id": np.array([0, 0, 0], dtype=np.int32),
                "zenith_angle_deg": np.array([20.0, 25.0, 30.0], dtype=float),
                "seeing_arcsec": seeing,
                "r0_m": np.array([0.16, 0.15, 0.14], dtype=float),
                "ngs_r_arcsec": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float),
                "ngs_theta_deg": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float),
                "ngs_mag": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float),
            }
        ),
    )

    with pytest.raises(ValueError, match="r0_m and seeing_arcsec"):
        sim_api.init_dataset(request)


def test_api_init_rejects_non_positive_seeing_values(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "seeing_non_positive.h5"

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="Tiptop", base_path=str(Path(ini_path).parent), specific_fields={"config_path": str(ini_path)}),
        setup=SetupConfig(
            ee_apertures_mas=[50.0, 100.0],
            specific_fields={"ngs_mag_zeropoint": 1.1e13 / 368.0},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength_um": np.array([1.65, 1.65], dtype=float),
                "atm_profile_id": np.array([0, 0], dtype=np.int32),
                "zenith_angle_deg": np.array([20.0, 25.0], dtype=float),
                "seeing_arcsec": np.array([0.0, -0.1], dtype=float),
                "ngs_r_arcsec": np.array([[1.0, 2.0], [1.0, 2.0]], dtype=float),
                "ngs_theta_deg": np.array([[0.0, 180.0], [0.0, 180.0]], dtype=float),
                "ngs_mag": np.array([[14.0, 15.0], [14.0, 15.0]], dtype=float),
            }
        ),
    )

    with pytest.raises(ValueError, match="seeing_arcsec values must be > 0"):
        sim_api.init_dataset(request)


def test_api_init_rejects_seeing_length_mismatch(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "seeing_length_mismatch.h5"

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="Tiptop", base_path=str(Path(ini_path).parent), specific_fields={"config_path": str(ini_path)}),
        setup=SetupConfig(
            ee_apertures_mas=[50.0, 100.0],
            specific_fields={"ngs_mag_zeropoint": 1.1e13 / 368.0},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength_um": np.array([1.65, 1.65, 1.65], dtype=float),
                "atm_profile_id": np.array([0, 0, 0], dtype=np.int32),
                "zenith_angle_deg": np.array([20.0, 25.0, 30.0], dtype=float),
                "seeing_arcsec": np.array([[0.7, 0.7], [0.8, 0.8], [0.9, 0.9]], dtype=float),
                "ngs_r_arcsec": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float),
                "ngs_theta_deg": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float),
                "ngs_mag": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float),
            }
        ),
    )

    with pytest.raises(ValueError, match="seeing_arcsec"):
        sim_api.init_dataset(request)


def test_api_init_accepts_partial_r0_with_seeing_fill(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "seeing_partial_r0_fill.h5"
    seeing = np.array([0.70, 0.80, 0.90], dtype=float)
    r0_partial = np.array([_r0_from_seeing_arcsec(np.array([seeing[0]])).item(), np.nan, np.nan], dtype=float)

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="Tiptop", base_path=str(Path(ini_path).parent), specific_fields={"config_path": str(ini_path)}),
        setup=SetupConfig(
            ee_apertures_mas=[50.0, 100.0],
            specific_fields={"ngs_mag_zeropoint": 1.1e13 / 368.0},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength_um": np.array([1.65, 1.65, 1.65], dtype=float),
                "atm_profile_id": np.array([0, 0, 0], dtype=np.int32),
                "zenith_angle_deg": np.array([20.0, 25.0, 30.0], dtype=float),
                "seeing_arcsec": seeing,
                "r0_m": r0_partial,
                "ngs_r_arcsec": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float),
                "ngs_theta_deg": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float),
                "ngs_mag": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float),
            }
        ),
    )

    sim_api.init_dataset(request)
    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_allclose(
            np.asarray(f["options/r0_m"][:], dtype=float),
            _r0_from_seeing_arcsec(seeing),
            rtol=1e-6,
            atol=1e-8,
        )


def test_api_init_rejects_scalar_column_value(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "scalar_column_value.h5"

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="Tiptop", base_path=str(Path(ini_path).parent), specific_fields={"config_path": str(ini_path)}),
        setup=SetupConfig(
            ee_apertures_mas=[50.0, 100.0],
            specific_fields={"ngs_mag_zeropoint": 1.1e13 / 368.0},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength_um": 1.65,
                "atm_profile_id": np.array([0, 0, 0], dtype=np.int32),
                "zenith_angle_deg": np.array([20.0, 25.0, 30.0], dtype=float),
                "r0_m": np.array([0.16, 0.15, 0.14], dtype=float),
                "ngs_r_arcsec": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float),
                "ngs_theta_deg": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float),
                "ngs_mag": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float),
            }
        ),
    )

    with pytest.raises(ValueError, match="must be per-simulation"):
        sim_api.init_dataset(request)


def test_api_init_rejects_first_dimension_mismatch(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "column_first_dim_mismatch.h5"

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="Tiptop", base_path=str(Path(ini_path).parent), specific_fields={"config_path": str(ini_path)}),
        setup=SetupConfig(
            ee_apertures_mas=[50.0, 100.0],
            specific_fields={"ngs_mag_zeropoint": 1.1e13 / 368.0},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength_um": np.array([1.65, 1.65, 1.65], dtype=float),
                "atm_profile_id": np.array([0, 0], dtype=np.int32),
                "zenith_angle_deg": np.array([20.0, 25.0, 30.0], dtype=float),
                "r0_m": np.array([0.16, 0.15, 0.14], dtype=float),
                "ngs_r_arcsec": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float),
                "ngs_theta_deg": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float),
                "ngs_mag": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float),
            }
        ),
    )

    with pytest.raises(ValueError, match="first dimension must match"):
        sim_api.init_dataset(request)


def test_api_init_rejects_non_lowercase_mapping_keys(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "bad_case.h5"

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation={"Name": "Tiptop", "config_path": str(ini_path)},
        setup={
            "ee_apertures_mas": [50.0, 100.0],
            "ngs_mag_zeropoint": 1.1e13 / 368.0,
        },
        options={
            "wavelength_um": np.array([1.65], dtype=float),
            "atm_profile_id": np.array([0], dtype=np.int32),
            "zenith_angle_deg": np.array([20.0], dtype=float),
            "r0_m": np.array([0.16], dtype=float),
            "ngs_r_arcsec": np.array([[1.0]], dtype=float),
            "ngs_theta_deg": np.array([[0.0]], dtype=float),
            "ngs_mag": np.array([[15.0]], dtype=float),
        },
    )

    with pytest.raises(ValueError, match="must be lowercase"):
        sim_api.init_dataset(request)
