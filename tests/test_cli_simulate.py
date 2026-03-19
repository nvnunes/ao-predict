from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

import ao_predict.cli as cli
import ao_predict.simulation.api as sim_api
from ao_predict import __version__
from ao_predict.simulation import (
    Simulation,
    SimulationContext,
    SimulationResult,
    SimulationSetup,
    SimulationState,
    schema,
)

TIPTOP_INI_TEXT = (
    "[main]\nvalue=1\n"
    "[telescope]\nTelescopeDiameter=8.0\nZenithAngle=20.0\n"
    "[RTC]\nSensorFrameRate_LO=500.0\n"
    "[sensor_LO]\nNumberLenslets=[1]\n"
    "[sources_LO]\nWavelength=[710e-9]\n"
    "[sources_HO]\nZenith=[30,30,30,30]\nAzimuth=[45,135,225,315]\n"
    "[sources_science]\nWavelength=[1.65e-06]\nZenith=[0,10,20]\nAzimuth=[0,90,180]\n"
    "[atmosphere]\nWavelength=500e-9\nr0_Value=0.16\nL0=25\nCn2Heights=[0,5000]\nCn2Weights=[0.6,0.4]\nWindSpeed=[5,10]\nWindDirection=[0,90]\n"
)


def _write_config_yaml(path: Path, ini_path: Path, *, options_cfg: dict[str, object] | None = None) -> None:
    if options_cfg is None:
        options_cfg = {
            "table": {
                "columns": [
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
                    "ngs3_r_arcsec",
                    "ngs3_theta_deg",
                    "ngs3_mag",
                ],
                "rows": [
                    [1.65, 0, 20.0, 0.16, 1.0, 0.0, 15.0, 1.0, 0.0, 15.0, 1.0, 0.0, 15.0],
                    [1.65, 0, 20.0, 0.16, 1.0, 0.0, 15.0, 1.0, 0.0, 15.0, 1.0, 0.0, 15.0],
                    [1.65, 0, 20.0, 0.16, 1.0, 0.0, 15.0, 1.0, 0.0, 15.0, 1.0, 0.0, 15.0],
                ],
            }
        }

    cfg = {
        "simulation": {
            "name": "ao_predict.simulation.tiptop:TiptopSimulation",
            "config_path": str(ini_path),
        },
        "setup": {
            "ee_apertures_mas": [50.0, 100.0],
            "sr_method": schema.DEFAULT_SETUP_SR_METHOD,
            "fwhm_summary": schema.DEFAULT_SETUP_FWHM_SUMMARY,
            "ngs_mag_zeropoint": 1.1e13 / 368.0,
            "sci_r_arcsec": [0.0, 10.0, 20.0],
            "sci_theta_deg": [0.0, 90.0, 180.0],
            "lgs_r_arcsec": [30.0, 30.0, 30.0, 30.0],
            "lgs_theta_deg": [45.0, 135.0, 225.0, 315.0],
            "atm_profiles": {
                "0": {
                    "name": "default",
                    "r0_m": 0.16,
                    "L0_m": 25.0,
                    "cn2_heights_m": [0.0, 5000.0],
                    "cn2_weights": [0.6, 0.4],
                    "wind_speed_mps": [5.0, 10.0],
                    "wind_direction_deg": [0.0, 90.0],
                }
            },
        },
        "options": options_cfg,
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _prepare_cli_paths(tmp_path: Path) -> tuple[Path, Path]:
    dataset_path = tmp_path / "sim_data.h5"
    config_yaml = tmp_path / "config.yaml"
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(TIPTOP_INI_TEXT, encoding="utf-8")
    _write_config_yaml(config_yaml, ini_path)
    return dataset_path, config_yaml


def _cli_init_dataset(monkeypatch, config_yaml: Path, dataset_path: Path) -> None:
    monkeypatch.setattr(
        sys, "argv", ["ao-predict", "simulate", "init", str(config_yaml), "--dataset", str(dataset_path)]
    )
    assert cli.main() == 0


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


class TiptopSimulation(Simulation):
    _NAME = "ao_predict.simulation.tiptop:TiptopSimulation"
    _VERSION = "0.0.1"

    def __init__(self, fail_idx: int | None = None):
        self.fail_idx = fail_idx
        self.failed_once: set[int] = set()

    def prepare_simulation_payload(self, base_simulation_payload, simulation_cfg):
        return {
            **dict(base_simulation_payload),
            "base_config": f"source_path={simulation_cfg.get('config_path')}",
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
        return SimulationContext(index=index, options=dict(options), setup=self._setup)

    def run(self, context: SimulationContext) -> None:
        if self.fail_idx is not None and context.index == self.fail_idx and context.index not in self.failed_once:
            self.failed_once.add(context.index)
            raise RuntimeError("intentional failure")

    def finalize(self, context: SimulationContext) -> None:
        context.result = _success_result(with_stats=False)


def test_cli_simulate_init_and_run(tmp_path: Path, monkeypatch):
    dataset_path, config_yaml = _prepare_cli_paths(tmp_path)
    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)

    sim = TiptopSimulation()
    monkeypatch.setattr(sim_api, "create_simulation_from_payload", lambda _payload: sim)

    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "run", str(dataset_path)])
    assert cli.main() == 0

    with h5py.File(dataset_path, "r") as f:
        assert float(f["setup/ngs_mag_zeropoint"][()]) > 0.0
        assert f["setup/sr_method"][()].decode("utf-8") == schema.DEFAULT_SETUP_SR_METHOD
        assert f["setup/fwhm_summary"][()].decode("utf-8") == schema.DEFAULT_SETUP_FWHM_SUMMARY
        np.testing.assert_array_equal(f["status/state"][:], np.array([1, 1, 1], dtype=np.uint8))


def test_cli_simulate_init_supports_nested_broadcast(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "sim_data.h5"
    config_yaml = tmp_path / "config.yaml"
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(TIPTOP_INI_TEXT, encoding="utf-8")
    _write_config_yaml(
        config_yaml,
        ini_path,
        options_cfg={
            "broadcast": {
                "zenith_angle_deg": 20.0,
            },
            "table": {
                "columns": [
                    "wavelength_um",
                    "atm_profile_id",
                    "r0_m",
                    "ngs1_r_arcsec",
                    "ngs1_theta_deg",
                    "ngs1_mag",
                ],
                "rows": [
                    [1.654, 0, 0.16, 1.0, 0.0, 15.0],
                    [2.179, 0, 0.16, 1.0, 0.0, 15.0],
                ],
            },
        },
    )

    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)

    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_allclose(f["options/wavelength_um"][:], np.array([1.654, 2.179], dtype=float))
        np.testing.assert_allclose(f["options/zenith_angle_deg"][:], np.array([20.0, 20.0], dtype=float))


def test_cli_version(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ao-predict", "--version"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == __version__


def test_cli_simulate_retry_failed(tmp_path: Path, monkeypatch, capsys):
    dataset_path, config_yaml = _prepare_cli_paths(tmp_path)
    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)

    sim = TiptopSimulation(fail_idx=1)
    monkeypatch.setattr(sim_api, "create_simulation_from_payload", lambda _payload: sim)

    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "run", str(dataset_path), "--verbose"])
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert "Simulation 1 failed: RuntimeError: intentional failure" in captured.out
    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_array_equal(f["status/state"][:], np.array([1, 2, 1], dtype=np.uint8))

    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "retry", str(dataset_path)])
    assert cli.main() == 0
    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_array_equal(f["status/state"][:], np.array([1, 1, 1], dtype=np.uint8))


def test_cli_simulate_run_with_sims(tmp_path: Path, monkeypatch):
    dataset_path, config_yaml = _prepare_cli_paths(tmp_path)
    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)

    sim = TiptopSimulation()
    monkeypatch.setattr(sim_api, "create_simulation_from_payload", lambda _payload: sim)
    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "run", str(dataset_path), "--sims", "2"])
    assert cli.main() == 0

    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_array_equal(f["status/state"][:], np.array([0, 1, 0], dtype=np.uint8))


def test_cli_simulate_retry_with_sims(tmp_path: Path, monkeypatch):
    dataset_path, config_yaml = _prepare_cli_paths(tmp_path)
    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)
    store = sim_api.SimulationStore(dataset_path)
    store.write_simulation_failure(0)
    store.write_simulation_failure(1)
    store.write_simulation_failure(2)

    sim = TiptopSimulation()
    monkeypatch.setattr(sim_api, "create_simulation_from_payload", lambda _payload: sim)
    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "retry", str(dataset_path), "--sims", "2"])
    assert cli.main() == 0

    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_array_equal(f["status/state"][:], np.array([2, 1, 2], dtype=np.uint8))


def test_cli_simulate_reset(tmp_path: Path, monkeypatch):
    dataset_path, config_yaml = _prepare_cli_paths(tmp_path)
    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)

    sim = TiptopSimulation()
    monkeypatch.setattr(sim_api, "create_simulation_from_payload", lambda _payload: sim)
    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "run", str(dataset_path)])
    assert cli.main() == 0

    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_array_equal(f["status/state"][:], np.array([1, 1, 1], dtype=np.uint8))

    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "reset", str(dataset_path)])
    assert cli.main() == 0
    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_array_equal(f["status/state"][:], np.array([0, 0, 0], dtype=np.uint8))


def test_cli_simulate_reset_with_sims(tmp_path: Path, monkeypatch):
    dataset_path, config_yaml = _prepare_cli_paths(tmp_path)
    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)

    sim = TiptopSimulation()
    monkeypatch.setattr(sim_api, "create_simulation_from_payload", lambda _payload: sim)
    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "run", str(dataset_path)])
    assert cli.main() == 0

    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "reset", str(dataset_path), "--sims", "2"])
    assert cli.main() == 0
    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_array_equal(f["status/state"][:], np.array([1, 0, 1], dtype=np.uint8))


def test_cli_check_fails_with_pending(tmp_path: Path, monkeypatch):
    dataset_path, config_yaml = _prepare_cli_paths(tmp_path)
    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)

    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "check", str(dataset_path)])
    assert cli.main() == 1


def test_cli_check_passes_when_complete(tmp_path: Path, monkeypatch):
    dataset_path, config_yaml = _prepare_cli_paths(tmp_path)
    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)

    sim = TiptopSimulation()
    monkeypatch.setattr(sim_api, "create_simulation_from_payload", lambda _payload: sim)
    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "run", str(dataset_path)])
    assert cli.main() == 0

    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "check", str(dataset_path)])
    assert cli.main() == 0


def test_cli_simulate_requires_subcommand(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert int(exc.value.code) == 2


def test_cli_load_config_normalizes_key_case(tmp_path: Path):
    cfg_path = tmp_path / "config_case.yaml"
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(TIPTOP_INI_TEXT, encoding="utf-8")
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "Simulation": {"Name": "ao_predict.simulation.tiptop:TiptopSimulation", "Config_Path": str(ini_path)},
                "Setup": {"EE_APERTURES_MAS": [50.0, 100.0], "NGS_MAG_ZEROPOINT": 3.0e10},
                "Options": {
                    "Table": {
                        "Columns": ["WAVELENGTH_UM", "ZENITH_ANGLE_DEG"],
                        "Rows": [[1.65, 20.0]],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    simulation_cfg, setup_cfg, options_cfg = cli._load_config(str(cfg_path))
    assert "name" in simulation_cfg and "config_path" in simulation_cfg
    assert "ee_apertures_mas" in setup_cfg and "ngs_mag_zeropoint" in setup_cfg
    assert options_cfg["columns"] == ["wavelength_um", "zenith_angle_deg"]
