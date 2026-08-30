from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from astropy import units as u

import ao_predict.cli as cli
import ao_predict.simulation.api as sim_api
from ao_predict import __version__
from ao_predict.simulation.helpers import normalize_psf_pixel_sum
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
                    "wavelength",
                    "atm_profile_id",
                    "zenith_angle",
                    "r0",
                    "ngs1_r",
                    "ngs1_theta",
                    "ngs1_magnitude",
                    "ngs2_r",
                    "ngs2_theta",
                    "ngs2_magnitude",
                    "ngs3_r",
                    "ngs3_theta",
                    "ngs3_magnitude",
                ],
                "units": {
                    "wavelength": "um",
                    "zenith_angle": "deg",
                    "r0": "m",
                    "ngs1_r": "arcsec",
                    "ngs1_theta": "deg",
                    "ngs1_magnitude": "mag",
                    "ngs2_r": "arcsec",
                    "ngs2_theta": "deg",
                    "ngs2_magnitude": "mag",
                    "ngs3_r": "arcsec",
                    "ngs3_theta": "deg",
                    "ngs3_magnitude": "mag",
                },
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
            "ee_apertures": {"value": [50.0, 100.0], "unit": "mas"},
            "sr_method": schema.DEFAULT_SETUP_SR_METHOD,
            "fwhm_summary": schema.DEFAULT_SETUP_FWHM_SUMMARY,
            "ngs_magnitude_zeropoint": {"value": 1.1e13 / 368.0, "unit": "photon / s"},
            "sci_r": {"value": [0.0, 10.0, 20.0], "unit": "arcsec"},
            "sci_theta": {"value": [0.0, 90.0, 180.0], "unit": "deg"},
            "lgs_r": {"value": [30.0, 30.0, 30.0, 30.0], "unit": "arcsec"},
            "lgs_theta": {"value": [45.0, 135.0, 225.0, 315.0], "unit": "deg"},
            "atm_profiles": {
                "0": {
                    "name": "default",
                    "r0": {"value": 0.16, "unit": "m"},
                    "L0": {"value": 25.0, "unit": "m"},
                    "cn2_heights": {"value": [0.0, 5000.0], "unit": "m"},
                    "cn2_weights": {"value": [0.6, 0.4], "unit": "1"},
                    "wind_speed": {"value": [5.0, 10.0], "unit": "m / s"},
                    "wind_direction": {"value": [0.0, 90.0], "unit": "deg"},
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
            "pixel_scale": 4.0 * u.mas,
            "tel_diameter": 8.0 * u.m,
            "tel_pupil": np.ones((6, 6), dtype=np.float32) * u.one,
        },
        psfs=np.zeros((m, 4, 4), dtype=np.float32) if with_psfs else None,
    )
    if with_stats:
        result.stats = {
            "sr": np.linspace(0.1, 0.3, m, dtype=np.float32) * u.one,
            "ee": np.full((m, 2), 0.5, dtype=np.float32) * u.one,
            "fwhm": np.full((m,), 60.0, dtype=np.float32) * u.mas,
        }
    return result


class TiptopSimulation(Simulation):
    _NAME = "ao_predict.simulation.tiptop:TiptopSimulation"
    _VERSION = "0.0.1"
    ngs_mag_standard = "R"

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
        merged["atm_wavelength"] = merged.get("atm_wavelength", 0.5 * u.um)
        return merged

    def prepare_options_payload(self, num_sims, setup_payload, base_options_payload):
        _ = setup_payload
        out = {str(k): v.copy() if hasattr(v, "copy") else v for k, v in base_options_payload.items()}
        n = int(num_sims)
        out.setdefault("wavelength", np.full((n,), 1.65, dtype=float) * u.um)
        out.setdefault("zenith_angle", np.full((n,), 20.0, dtype=float) * u.deg)
        out.setdefault("atm_profile_id", np.zeros((n,), dtype=np.int32))
        out.setdefault("r0", np.full((n,), 0.16, dtype=float) * u.m)
        if not any(key in out for key in ("ngs_r", "ngs_theta", "ngs_magnitude")):
            out["ngs_r"] = np.full((n, 1), 1.0, dtype=float) * u.arcsec
            out["ngs_theta"] = np.full((n, 1), 0.0, dtype=float) * u.deg
            out["ngs_magnitude"] = np.full((n, 1), 15.0, dtype=float) * u.mag
        out["atm_profile_id"] = np.asarray(out["atm_profile_id"], dtype=np.int32).reshape(-1)
        return out

    def validate_options_payload(self, num_sims, options_payload):
        _ = num_sims
        _ = options_payload

    def load_setup_payload(self, setup_payload):
        self._setup = SimulationSetup(
            ee_apertures=setup_payload["ee_apertures"],
            sr_method=str(setup_payload["sr_method"]),
            fwhm_summary=str(setup_payload["fwhm_summary"]),
            ee_geometry=str(setup_payload["ee_geometry"]),
            atm_wavelength=setup_payload["atm_wavelength"],
            atm_profiles=dict(setup_payload["atm_profiles"]),
            lgs_r=setup_payload["lgs_r"],
            lgs_theta=setup_payload["lgs_theta"],
            sci_r=setup_payload["sci_r"],
            sci_theta=setup_payload["sci_theta"],
        )

    def validate_setup_payload(self, setup_payload):
        _ = SimulationSetup(
            ee_apertures=setup_payload["ee_apertures"],
            sr_method=str(setup_payload["sr_method"]),
            fwhm_summary=str(setup_payload["fwhm_summary"]),
            ee_geometry=str(setup_payload["ee_geometry"]),
            atm_wavelength=setup_payload["atm_wavelength"],
            atm_profiles=dict(setup_payload["atm_profiles"]),
            lgs_r=setup_payload["lgs_r"],
            lgs_theta=setup_payload["lgs_theta"],
            sci_r=setup_payload["sci_r"],
            sci_theta=setup_payload["sci_theta"],
        )

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


def test_cli_simulate_init_and_run(tmp_path: Path, monkeypatch):
    dataset_path, config_yaml = _prepare_cli_paths(tmp_path)
    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)

    sim = TiptopSimulation()
    monkeypatch.setattr(sim_api, "create_simulation_from_payload", lambda _payload: sim)

    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "run", str(dataset_path)])
    assert cli.main() == 0

    with h5py.File(dataset_path, "r") as f:
        assert float(f["setup/ngs_magnitude_zeropoint"][()]) > 0.0
        assert f["setup/sr_method"][()].decode("utf-8") == schema.DEFAULT_SETUP_SR_METHOD
        assert f["setup/fwhm_summary"][()].decode("utf-8") == schema.DEFAULT_SETUP_FWHM_SUMMARY
        assert f["setup/ee_geometry"][()].decode("utf-8") == schema.DEFAULT_SETUP_EE_GEOMETRY
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
                "zenith_angle": {"value": 20.0, "unit": "deg"},
            },
            "table": {
                "columns": [
                    "wavelength",
                    "atm_profile_id",
                    "r0",
                    "ngs1_r",
                    "ngs1_theta",
                    "ngs1_magnitude",
                ],
                "units": {
                    "wavelength": "um",
                    "r0": "m",
                    "ngs1_r": "arcsec",
                    "ngs1_theta": "deg",
                    "ngs1_magnitude": "mag",
                },
                "rows": [
                    [1.654, 0, 0.16, 1.0, 0.0, 15.0],
                    [2.179, 0, 0.16, 1.0, 0.0, 15.0],
                ],
            },
        },
    )

    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)

    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_allclose(f["options/wavelength"][:], np.array([1.654, 2.179], dtype=float))
        np.testing.assert_allclose(f["options/zenith_angle"][:], np.array([20.0, 20.0], dtype=float))


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


def test_cli_simulate_run_passes_parallel_controls(monkeypatch, tmp_path: Path):
    observed: dict[str, object] = {}
    dataset_path = tmp_path / "sim_data.h5"

    def _run(dataset, **kwargs):
        observed["dataset"] = dataset
        observed.update(kwargs)
        return sim_api.RunSummary(attempted=0, succeeded=0, failed=0)

    monkeypatch.setattr(cli, "run_simulations_by_state", _run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ao-predict",
            "simulate",
            "run",
            str(dataset_path),
            "--threads",
            "3",
            "--chunks",
            "2",
        ],
    )

    assert cli.main() == 0
    assert observed["dataset"] == str(dataset_path)
    assert observed["state"] == SimulationState.PENDING
    assert observed["num_workers"] == 3
    assert observed["chunk_multiple"] == 2


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


def test_cli_simulate_retry_passes_parallel_controls(monkeypatch, tmp_path: Path):
    observed: dict[str, object] = {}
    dataset_path = tmp_path / "sim_data.h5"

    def _run(dataset, **kwargs):
        observed["dataset"] = dataset
        observed.update(kwargs)
        return sim_api.RunSummary(attempted=0, succeeded=0, failed=0)

    monkeypatch.setattr(cli, "run_simulations_by_state", _run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ao-predict",
            "simulate",
            "retry",
            str(dataset_path),
            "--threads",
            "4",
            "--chunks",
            "3",
        ],
    )

    assert cli.main() == 0
    assert observed["dataset"] == str(dataset_path)
    assert observed["state"] == SimulationState.FAILED
    assert observed["num_workers"] == 4
    assert observed["chunk_multiple"] == 3


def test_cli_simulate_resume_retries_only_preexisting_failures(tmp_path: Path, monkeypatch):
    dataset_path, config_yaml = _prepare_cli_paths(tmp_path)
    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)
    store = sim_api.SimulationStore(dataset_path)
    store.write_simulation_failure(2)

    sim = TiptopSimulation(fail_idx=1)
    monkeypatch.setattr(sim_api, "create_simulation_from_payload", lambda _payload: sim)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ao-predict",
            "simulate",
            "resume",
            str(dataset_path),
            "--config",
            str(config_yaml),
        ],
    )

    assert cli.main() == 0
    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_array_equal(f["status/state"][:], np.array([1, 2, 1], dtype=np.uint8))


def test_cli_simulate_resume_passes_parallel_controls(monkeypatch, tmp_path: Path):
    observed: dict[str, object] = {}
    dataset_path = tmp_path / "sim_data.h5"

    def _resume(dataset, **kwargs):
        observed["dataset"] = dataset
        observed.update(kwargs)
        return sim_api.RunSummary(attempted=0, succeeded=0, failed=0)

    monkeypatch.setattr(cli, "resume_simulations", _resume)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ao-predict",
            "simulate",
            "resume",
            str(dataset_path),
            "--threads",
            "5",
            "--chunks",
            "2",
        ],
    )

    assert cli.main() == 0
    assert observed["dataset"] == str(dataset_path)
    assert observed["num_workers"] == 5
    assert observed["chunk_multiple"] == 2


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


def test_cli_check_with_config_passes_when_complete_and_matching(tmp_path: Path, monkeypatch):
    dataset_path, config_yaml = _prepare_cli_paths(tmp_path)
    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)

    sim = TiptopSimulation()
    monkeypatch.setattr(sim_api, "create_simulation_from_payload", lambda _payload: sim)
    monkeypatch.setattr(sys, "argv", ["ao-predict", "simulate", "run", str(dataset_path)])
    assert cli.main() == 0

    monkeypatch.setattr(
        sys,
        "argv",
        ["ao-predict", "simulate", "check", str(dataset_path), "--config", str(config_yaml)],
    )
    assert cli.main() == 0


def test_cli_check_with_config_reports_mismatch(tmp_path: Path, monkeypatch, capsys):
    dataset_path, config_yaml = _prepare_cli_paths(tmp_path)
    _cli_init_dataset(monkeypatch, config_yaml, dataset_path)

    with h5py.File(dataset_path, "r+") as f:
        f["setup/ee_apertures"][0] = 75.0

    monkeypatch.setattr(
        sys,
        "argv",
        ["ao-predict", "simulate", "check", str(dataset_path), "--config", str(config_yaml)],
    )
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert "Config mismatch: /setup/ee_apertures" in captured.out


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
                "Setup": {
                    "EE_APERTURES": {"Value": [50.0, 100.0], "Unit": "mas"},
                    "NGS_MAGNITUDE_ZEROPOINT": {"Value": 3.0e10, "Unit": "photon / s"},
                },
                "Options": {
                    "Table": {
                        "Columns": ["WAVELENGTH", "ZENITH_ANGLE"],
                        "Units": {"WAVELENGTH": "um", "ZENITH_ANGLE": "deg"},
                        "Rows": [[1.65, 20.0]],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    simulation_cfg, setup_cfg, options_cfg = cli._load_config(str(cfg_path))
    assert "name" in simulation_cfg and "config_path" in simulation_cfg
    assert "ee_apertures" in setup_cfg and "ngs_magnitude_zeropoint" in setup_cfg
    assert options_cfg["columns"] == ["wavelength", "zenith_angle"]
    assert options_cfg["units"] == {"wavelength": "um", "zenith_angle": "deg"}
