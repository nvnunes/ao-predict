from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy import units as u

import ao_predict.simulation.api as sim_api
from ao_predict.analysis import load_analysis_dataset
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
            ee_apertures=[50.0, 100.0] * u.mas,
            sr_method=schema.DEFAULT_SETUP_SR_METHOD,
            fwhm_summary=schema.DEFAULT_SETUP_FWHM_SUMMARY,
            specific_fields={
                "ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s
            },
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength": np.array([1.65, 1.65, 1.65], dtype=float) * u.um,
                "atm_profile_id": np.array([0, 0, 0], dtype=np.int32),
                "zenith_angle": np.array([20.0, 25.0, 30.0], dtype=float) * u.deg,
                "r0": np.array([0.16, 0.15, 0.14], dtype=float) * u.m,
                "ngs_r": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float) * u.arcsec,
                "ngs_theta": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float) * u.deg,
                "ngs_magnitude": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float) * u.mag,
            }
        ),
    )


def _success_result(m: int = 3, *, with_stats: bool = True, with_psfs: bool = True) -> SimulationResult:
    result = SimulationResult(
        state=SimulationState.SUCCEEDED,
        meta={
            "pixel_scale": 4.0 * u.mas,
            "tel_diameter": 8.0 * u.m,
            "tel_pupil": np.ones((6, 6), dtype=np.float32) * u.dimensionless_unscaled,
        },
        psfs=np.zeros((m, 4, 4), dtype=np.float32) if with_psfs else None,
    )
    if with_stats:
        result.stats = {
            "sr": np.linspace(0.1, 0.3, m, dtype=np.float32) * u.dimensionless_unscaled,
            "ee": np.full((m, 2), 0.5, dtype=np.float32) * u.dimensionless_unscaled,
            "fwhm": np.full((m,), 60.0, dtype=np.float32) * u.mas,
        }
    return result


class FakeSimulation(Simulation):
    _NAME = "ao_predict.simulation.tiptop:TiptopSimulation"
    _VERSION = "0.0.1"
    ngs_mag_standard = "R"

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
        merged["atm_wavelength"] = merged.get("atm_wavelength", 0.5 * u.um)
        return merged

    def prepare_options_payload(self, num_sims, setup_payload, base_options_payload):
        _ = setup_payload
        out = {str(k): v.copy() for k, v in base_options_payload.items()}
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
        _ = setup_payload["ee_apertures"]

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


def test_api_science_offsets_persist_sparsely_as_float32(tmp_path: Path):
    request = _base_request(tmp_path)
    assert isinstance(request.options, OptionsConfig)
    option_arrays = {
        **request.options.option_arrays,
        schema.KEY_OPTION_SCI_DX: np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            dtype=np.float64,
        ) * u.arcsec,
        schema.KEY_OPTION_SCI_DY: np.zeros((3, 3), dtype=np.float64) * u.arcsec,
    }
    request = replace(request, options=OptionsConfig(option_arrays=option_arrays))

    sim_api.init_dataset(request)

    store = sim_api.SimulationStore(request.dataset_path)
    options = store.read_options()
    assert options[schema.KEY_OPTION_SCI_DX].dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        options[schema.KEY_OPTION_SCI_DX],
        option_arrays[schema.KEY_OPTION_SCI_DX],
    )
    assert schema.KEY_OPTION_SCI_DY not in options
    row = store.read_sim_options(1)
    np.testing.assert_allclose(
        row[schema.KEY_OPTION_SCI_DX], np.array([3.0, 4.0, 5.0]) * u.arcsec
    )
    assert schema.KEY_OPTION_SCI_DY not in row
    sim_api.validate_dataset_matches_request(request.dataset_path, request)


def test_api_science_offsets_require_matching_science_dimension(tmp_path: Path):
    request = _base_request(tmp_path)
    assert isinstance(request.options, OptionsConfig)
    option_arrays = {
        **request.options.option_arrays,
        schema.KEY_OPTION_SCI_DX: np.ones((3, 2), dtype=np.float32) * u.arcsec,
    }
    request = replace(request, options=OptionsConfig(option_arrays=option_arrays))

    with pytest.raises(ValueError, match=r"shape \[N, M\]=\(3, 3\)"):
        sim_api.init_dataset(request)


def test_api_science_offsets_must_be_finite(tmp_path: Path):
    request = _base_request(tmp_path)
    assert isinstance(request.options, OptionsConfig)
    offsets = np.ones((3, 3), dtype=np.float32)
    offsets[1, 2] = np.nan
    option_arrays = {
        **request.options.option_arrays,
        schema.KEY_OPTION_SCI_DY: offsets * u.arcsec,
    }
    request = replace(request, options=OptionsConfig(option_arrays=option_arrays))

    with pytest.raises(ValueError, match="must be finite"):
        sim_api.init_dataset(request)


def test_api_science_offsets_require_quantities(tmp_path: Path):
    request = _base_request(tmp_path)
    assert isinstance(request.options, OptionsConfig)
    option_arrays = {
        **request.options.option_arrays,
        schema.KEY_OPTION_SCI_DX: np.full((3, 3), "invalid"),
    }
    request = replace(request, options=OptionsConfig(option_arrays=option_arrays))

    with pytest.raises(TypeError, match="must be an Astropy Quantity"):
        sim_api.init_dataset(request)


def test_api_normalizes_equivalent_quantities_to_canonical_persisted_units(
    tmp_path: Path,
) -> None:
    request = _base_request(tmp_path)
    assert isinstance(request.setup, SetupConfig)
    assert isinstance(request.options, OptionsConfig)
    option_arrays = dict(request.options.option_arrays)
    option_arrays["wavelength"] = option_arrays["wavelength"].to(u.nm)
    option_arrays["zenith_angle"] = option_arrays["zenith_angle"].to(u.rad)
    option_arrays["r0"] = option_arrays["r0"].to(u.cm)
    option_arrays["ngs_r"] = option_arrays["ngs_r"].to(u.mas)
    option_arrays["ngs_theta"] = option_arrays["ngs_theta"].to(u.rad)
    request = replace(
        request,
        setup=replace(request.setup, ee_apertures=request.setup.ee_apertures.to(u.arcsec)),
        options=OptionsConfig(option_arrays=option_arrays),
    )

    sim_api.init_dataset(request)

    with h5py.File(request.dataset_path, "r") as f:
        np.testing.assert_allclose(f["setup/ee_apertures"][...], [50.0, 100.0])
        np.testing.assert_allclose(f["options/wavelength"][...], [1.65, 1.65, 1.65])
        np.testing.assert_allclose(f["options/r0"][...], [0.16, 0.15, 0.14])
        assert f["setup/ee_apertures"].attrs["units"] == "mas"
        assert f["options/wavelength"].attrs["units"] == "um"
        assert f["options/r0"].attrs["units"] == "m"


def test_api_validate_dataset_matches_request_accepts_matching_payloads(tmp_path: Path):
    request = _base_request(tmp_path)
    sim_api.init_dataset(request)

    sim_api.validate_dataset_matches_request(request.dataset_path, request)


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        ("simulation/name", "different:Simulation", "/simulation/name"),
        ("setup/ee_apertures", np.array([75.0, 100.0]), "/setup/ee_apertures"),
        ("setup/ee_geometry", schema.STATS_EE_GEOMETRY_ENCIRCLED, "/setup/ee_geometry"),
        ("options/zenith_angle", np.array([21.0, 25.0, 30.0]), "/options/zenith_angle"),
    ],
)
def test_api_validate_dataset_matches_request_rejects_mismatches(
    tmp_path: Path,
    path: str,
    value,
    expected: str,
):
    request = _base_request(tmp_path)
    sim_api.init_dataset(request)

    with h5py.File(request.dataset_path, "r+") as f:
        del f[path]
        dataset = f.create_dataset(path, data=value)
        section, field = path.split("/", maxsplit=1)
        unit = {
            "setup": schema.SETUP_FIELD_UNITS,
            "options": schema.OPTION_FIELD_UNITS,
        }.get(section, {}).get(field)
        if unit is not None:
            dataset.attrs["units"] = unit.to_string()

    with pytest.raises(sim_api.DatasetConfigMismatchError, match=expected):
        sim_api.validate_dataset_matches_request(request.dataset_path, request)


def test_api_full_pipeline_with_test_simulation(tmp_path: Path):
    dataset_path = tmp_path / "test_simulation.h5"
    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="mock_simulation:MockSimulation"),
        setup=SetupConfig(ee_apertures=[50.0, 100.0] * u.mas),
        options=OptionsConfig(
            option_arrays={
                "zenith_angle": np.array([15.0, 25.0, 35.0], dtype=float) * u.deg,
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
        assert (
            f[f"{schema.KEY_SETUP_SECTION}/{schema.KEY_SETUP_EE_GEOMETRY}"][()].decode("utf-8")
            == schema.DEFAULT_SETUP_EE_GEOMETRY
        )


def test_api_init_persists_explicit_setup_stats_selectors(tmp_path: Path):
    request = _base_request(tmp_path)
    request = InitDatasetRequest(
        dataset_path=request.dataset_path,
        simulation=request.simulation,
        setup=SetupConfig(
            ee_apertures=[50.0, 100.0] * u.mas,
            sr_method=schema.STATS_SR_METHOD_PIXEL_MAX,
            fwhm_summary=schema.STATS_FWHM_SUMMARY_MAX,
            ee_geometry=schema.STATS_EE_GEOMETRY_ENCIRCLED,
            specific_fields={"ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s},
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
        assert (
            f[f"{schema.KEY_SETUP_SECTION}/{schema.KEY_SETUP_EE_GEOMETRY}"][()].decode("utf-8")
            == schema.STATS_EE_GEOMETRY_ENCIRCLED
        )


def test_api_init_persists_ngs_mag_standard_and_exposes_it_to_analysis(tmp_path: Path):
    request = _base_request(tmp_path)

    sim_api.init_dataset(request)

    store = sim_api.SimulationStore(request.dataset_path)
    assert store.read_simulation()[schema.KEY_SIMULATION_NGS_MAG_STANDARD] == "R"
    assert (
        load_analysis_dataset(request.dataset_path).simulation_payload[
            schema.KEY_SIMULATION_NGS_MAG_STANDARD
        ]
        == "R"
    )


def test_api_init_rejects_invalid_setup_stats_selector(tmp_path: Path):
    request = _base_request(tmp_path)
    request = InitDatasetRequest(
        dataset_path=request.dataset_path,
        simulation=request.simulation,
        setup=SetupConfig(
            ee_apertures=[50.0, 100.0] * u.mas,
            sr_method="bad_selector",
            specific_fields={"ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s},
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


def test_api_resume_retries_only_preexisting_failures(tmp_path: Path, monkeypatch):
    request = _base_request(tmp_path)
    sim_api.init_dataset(request)
    dataset_path = Path(request.dataset_path)
    store = sim_api.SimulationStore(dataset_path)
    store.write_simulation_failure(2)

    sim = FakeSimulation(fail_idx=1)
    monkeypatch.setattr(sim_api, "create_simulation_from_payload", lambda _payload: sim)

    summary = sim_api.resume_simulations(dataset_path, expected_request=request)

    assert summary.attempted == 3
    assert summary.succeeded == 2
    assert summary.failed == 1
    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_array_equal(f["status/state"][:], np.array([1, 2, 1], dtype=np.uint8))


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


def _r0_from_seeing_arcsec(seeing: np.ndarray, atm_wavelength: float = 0.5) -> np.ndarray:
    seeing_rad = np.asarray(seeing, dtype=float) * (math.pi / 648000.0)
    return 0.98 * (float(atm_wavelength) * 1e-6) / seeing_rad


def test_api_init_accepts_seeing_alias_columns(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "seeing_columns.h5"
    seeing = np.array([0.70, 0.80, 0.90], dtype=float)

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="Tiptop", base_path=str(Path(ini_path).parent), specific_fields={"config_path": str(ini_path)}),
        setup=SetupConfig(
            ee_apertures=[50.0, 100.0] * u.mas,
            specific_fields={"ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength": np.array([1.65, 1.65, 1.65], dtype=float) * u.um,
                "atm_profile_id": np.array([0, 0, 0], dtype=np.int32),
                "zenith_angle": np.array([20.0, 25.0, 30.0], dtype=float) * u.deg,
                    "seeing": seeing * u.arcsec,
                    "ngs_r": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float) * u.arcsec,
                    "ngs_theta": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float) * u.deg,
                    "ngs_magnitude": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float) * u.mag,
            }
        ),
    )
    sim_api.init_dataset(request)

    with h5py.File(dataset_path, "r") as f:
        assert "seeing" not in f["options"]
        np.testing.assert_allclose(
            np.asarray(f["options/r0"][:], dtype=float),
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
            ee_apertures=[50.0, 100.0] * u.mas,
            specific_fields={"ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s},
        ),
        options=TableOptionsConfig(
            broadcast={},
            columns=[
                "wavelength",
                "atm_profile_id",
                "zenith_angle",
                "seeing",
                "ngs1_r",
                "ngs1_theta",
                "ngs1_magnitude",
            ],
            units={
                "wavelength": u.um,
                "zenith_angle": u.deg,
                "seeing": u.arcsec,
                "ngs1_r": u.arcsec,
                "ngs1_theta": u.deg,
                "ngs1_magnitude": u.mag,
            },
            rows=[
                [1.65, 0, 20.0, float(seeing[0]), 1.0, 0.0, 14.0],
                [1.65, 0, 25.0, float(seeing[1]), 1.0, 0.0, 14.0],
                [1.65, 0, 30.0, float(seeing[2]), 1.0, 0.0, 14.0],
            ],
        ),
    )
    sim_api.init_dataset(request)

    with h5py.File(dataset_path, "r") as f:
        assert "seeing" not in f["options"]
        np.testing.assert_allclose(
            np.asarray(f["options/r0"][:], dtype=float),
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
            ee_apertures=[50.0, 100.0] * u.mas,
            specific_fields={"ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s},
        ),
        options=TableOptionsConfig(
            broadcast={},
            columns=[
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
            ],
            units={
                "wavelength": u.um,
                "zenith_angle": u.deg,
                "r0": u.m,
                "ngs1_r": u.arcsec,
                "ngs1_theta": u.deg,
                "ngs1_magnitude": u.mag,
                "ngs2_r": u.arcsec,
                "ngs2_theta": u.deg,
                "ngs2_magnitude": u.mag,
            },
            rows=[
                [1.65, 0, 20.0, 0.16, 1.0, 0.0, 14.0, 2.0, 180.0, 15.0],
                [1.65, 0, 25.0, 0.15, 1.5, 30.0, 14.5, None, None, None],
            ],
        ),
    )

    sim_api.init_dataset(request)

    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_allclose(
            np.asarray(f["options/ngs_r"][:], dtype=float),
            np.array([[1.0, 2.0], [1.5, np.nan]], dtype=float),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(f["options/ngs_theta"][:], dtype=float),
            np.array([[0.0, 180.0], [30.0, np.nan]], dtype=float),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(f["options/ngs_magnitude"][:], dtype=float),
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
        setup={"ee_apertures": [50.0, 100.0] * u.mas, "ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s},
        options={
            "broadcast": {},
            "columns": [
                "wavelength",
                "atm_profile_id",
                "zenith_angle",
                "seeing",
                "ngs1_r",
                "ngs1_theta",
                "ngs1_magnitude",
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
        setup={"ee_apertures": [50.0, 100.0] * u.mas, "ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s},
        options={
            "broadcast": {},
            "columns": [
                "wavelength",
                "atm_profile_id",
                "zenith_angle",
                "r0",
                "ngs1_r",
                "ngs1_theta",
                "ngs1_magnitude",
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
            ee_apertures=[50.0, 100.0] * u.mas,
            specific_fields={"ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength": np.array([1.65, 1.65, 1.65], dtype=float) * u.um,
                "atm_profile_id": np.array([0, 0, 0], dtype=np.int32),
                "zenith_angle": np.array([20.0, 25.0, 30.0], dtype=float) * u.deg,
                "seeing": seeing * u.arcsec,
                "r0": np.array([0.16, 0.15, 0.14], dtype=float) * u.m,
                "ngs_r": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float) * u.arcsec,
                "ngs_theta": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float) * u.deg,
                "ngs_magnitude": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float) * u.mag,
            }
        ),
    )

    with pytest.raises(ValueError, match="r0 and seeing"):
        sim_api.init_dataset(request)


def test_api_init_rejects_non_positive_seeing_values(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "seeing_non_positive.h5"

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="Tiptop", base_path=str(Path(ini_path).parent), specific_fields={"config_path": str(ini_path)}),
        setup=SetupConfig(
            ee_apertures=[50.0, 100.0] * u.mas,
            specific_fields={"ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength": np.array([1.65, 1.65], dtype=float) * u.um,
                "atm_profile_id": np.array([0, 0], dtype=np.int32),
                "zenith_angle": np.array([20.0, 25.0], dtype=float) * u.deg,
                "seeing": np.array([0.0, -0.1], dtype=float) * u.arcsec,
                "ngs_r": np.array([[1.0, 2.0], [1.0, 2.0]], dtype=float) * u.arcsec,
                "ngs_theta": np.array([[0.0, 180.0], [0.0, 180.0]], dtype=float) * u.deg,
                "ngs_magnitude": np.array([[14.0, 15.0], [14.0, 15.0]], dtype=float) * u.mag,
            }
        ),
    )

    with pytest.raises(ValueError, match="seeing values must be > 0"):
        sim_api.init_dataset(request)


def test_api_init_rejects_seeing_length_mismatch(tmp_path: Path):
    ini_path = _write_ini(tmp_path)
    dataset_path = tmp_path / "seeing_length_mismatch.h5"

    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="Tiptop", base_path=str(Path(ini_path).parent), specific_fields={"config_path": str(ini_path)}),
        setup=SetupConfig(
            ee_apertures=[50.0, 100.0] * u.mas,
            specific_fields={"ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength": np.array([1.65, 1.65, 1.65], dtype=float) * u.um,
                "atm_profile_id": np.array([0, 0, 0], dtype=np.int32),
                "zenith_angle": np.array([20.0, 25.0, 30.0], dtype=float) * u.deg,
                "seeing": np.array([[0.7, 0.7], [0.8, 0.8], [0.9, 0.9]], dtype=float) * u.arcsec,
                "ngs_r": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float) * u.arcsec,
                "ngs_theta": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float) * u.deg,
                "ngs_magnitude": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float) * u.mag,
            }
        ),
    )

    with pytest.raises(ValueError, match="seeing"):
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
            ee_apertures=[50.0, 100.0] * u.mas,
            specific_fields={"ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength": np.array([1.65, 1.65, 1.65], dtype=float) * u.um,
                "atm_profile_id": np.array([0, 0, 0], dtype=np.int32),
                "zenith_angle": np.array([20.0, 25.0, 30.0], dtype=float) * u.deg,
                "seeing": seeing * u.arcsec,
                    "r0": r0_partial * u.m,
                "ngs_r": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float) * u.arcsec,
                "ngs_theta": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float) * u.deg,
                "ngs_magnitude": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float) * u.mag,
            }
        ),
    )

    sim_api.init_dataset(request)
    with h5py.File(dataset_path, "r") as f:
        np.testing.assert_allclose(
            np.asarray(f["options/r0"][:], dtype=float),
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
            ee_apertures=[50.0, 100.0] * u.mas,
            specific_fields={"ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s},
        ),
        options=OptionsConfig(
            option_arrays={
                    "wavelength": 1.65 * u.um,
                "atm_profile_id": np.array([0, 0, 0], dtype=np.int32),
                "zenith_angle": np.array([20.0, 25.0, 30.0], dtype=float) * u.deg,
                "r0": np.array([0.16, 0.15, 0.14], dtype=float) * u.m,
                "ngs_r": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float) * u.arcsec,
                "ngs_theta": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float) * u.deg,
                "ngs_magnitude": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float) * u.mag,
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
            ee_apertures=[50.0, 100.0] * u.mas,
            specific_fields={"ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s},
        ),
        options=OptionsConfig(
            option_arrays={
                "wavelength": np.array([1.65, 1.65, 1.65], dtype=float) * u.um,
                "atm_profile_id": np.array([0, 0], dtype=np.int32),
                "zenith_angle": np.array([20.0, 25.0, 30.0], dtype=float) * u.deg,
                "r0": np.array([0.16, 0.15, 0.14], dtype=float) * u.m,
                "ngs_r": np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=float) * u.arcsec,
                "ngs_theta": np.array([[0.0, 180.0], [0.0, 180.0], [0.0, 180.0]], dtype=float) * u.deg,
                "ngs_magnitude": np.array([[14.0, 15.0], [14.0, 15.0], [14.0, 15.0]], dtype=float) * u.mag,
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
            "ee_apertures": [50.0, 100.0] * u.mas,
            "ngs_magnitude_zeropoint": (1.1e13 / 368.0) * u.photon / u.s,
        },
        options={
            "wavelength": np.array([1.65], dtype=float) * u.um,
            "atm_profile_id": np.array([0], dtype=np.int32),
            "zenith_angle": np.array([20.0], dtype=float) * u.deg,
            "r0": np.array([0.16], dtype=float) * u.m,
            "ngs_r": np.array([[1.0]], dtype=float) * u.arcsec,
            "ngs_theta": np.array([[0.0]], dtype=float) * u.deg,
            "ngs_magnitude": np.array([[15.0]], dtype=float) * u.mag,
        },
    )

    with pytest.raises(ValueError, match="must be lowercase"):
        sim_api.init_dataset(request)
