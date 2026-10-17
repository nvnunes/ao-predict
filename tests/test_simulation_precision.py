"""Dataset-bound precision, ordinary resume, and native-free analysis loading."""

from dataclasses import replace

import h5py
import numpy as np
import pytest
from astropy import units as u

from ao_predict import (
    DatasetConfigMismatchError,
    InitDatasetRequest,
    OptionsConfig,
    SetupConfig,
    SimulationConfig,
    init_dataset,
    run_simulations_by_state,
    validate_dataset_matches_request,
)
from ao_predict.analysis import load_analysis_dataset
from ao_predict.persistence import SimulationStore
from ao_predict.simulation.runner import create_simulation_from_config
from ao_predict.simulation.config import normalize_simulation_config
from test_hybrid_simulation import _write_hybrid_inputs, _setup_payload, _options


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hybrid_precision_execute_resume_and_analysis(tmp_path, monkeypatch, dtype):
    ini, science, ngs = _write_hybrid_inputs(tmp_path)
    ini.write_text(ini.read_text() + "\n[hybrid]\nprecision=float64\n")
    request = InitDatasetRequest(
        dataset_path=tmp_path / "dataset.h5",
        simulation=SimulationConfig(
            name="Hybrid",
            dtype=dtype,
            base_path=str(tmp_path),
            specific_fields={
                "config_path": ini.name,
                "science_ho_psf_interpolator_path": science.name,
                "ngs_ho_metric_interpolator_path": ngs.name,
                "diagnostics_level": "debug",
            },
        ),
        setup=SetupConfig(
            ee_apertures=[50.0] * u.mas, specific_fields=_setup_payload()
        ),
        options=OptionsConfig(
            {
                name: np.stack([value, value])
                for name, value in _options().items()
                if name != "ngs_used"
            }
        ),
        save_psfs=True,
    )

    class Mavis:
        error = False
        mas2nm = 2.0

        def __init__(self, *args, **kwargs):
            pass

        def computeTotalResidualMatrix(self, *args, **kwargs):
            return np.stack([np.eye(2), 4 * np.eye(2)])

    monkeypatch.setattr("hybrid_ao_psf._mastsel._load_mavis_lo", lambda: Mavis)
    init_dataset(request)
    store = SimulationStore(request.dataset_path)
    assert store.read_simulation()["precision"] == np.dtype(dtype).name
    assert (
        run_simulations_by_state(
            request.dataset_path, indexes=[0], num_workers=1
        ).succeeded
        == 1
    )
    assert (
        run_simulations_by_state(
            request.dataset_path, indexes=[1], num_workers=1
        ).succeeded
        == 1
    )
    store.validate_schema()
    for row in (0, 1):
        assert store.read_simulation_psfs(row).dtype == dtype
        for value in store.read_simulation_stats(row).values():
            assert value.dtype == dtype
        for value in store.read_simulation_meta(row).values():
            assert value.dtype == dtype
    with h5py.File(request.dataset_path) as root:
        for path in (
            "stats/sr",
            "stats/ee",
            "stats/fwhm",
            "stats/jitter",
            "meta/tel_pupil",
            "meta/pixel_scale",
            "meta/tel_diameter",
            "diagnostics/hybrid/ctot_angle",
            "diagnostics/hybrid/ngs/fwhm",
        ):
            assert root[path].dtype == dtype
        assert root["diagnostics/hybrid/psd_clipped"].dtype == bool
        assert root["diagnostics/hybrid/psd_valid_count"].dtype == np.int32
    other = np.float64 if dtype == np.float32 else np.float32
    with pytest.raises(DatasetConfigMismatchError, match="precision"):
        validate_dataset_matches_request(
            request.dataset_path,
            replace(request, simulation=replace(request.simulation, dtype=other)),
        )
    science.rename(science.with_suffix(".hidden"))
    ngs.rename(ngs.with_suffix(".hidden"))
    loaded = load_analysis_dataset(request.dataset_path)
    assert loaded is not None
    assert store.read_simulation_psfs(0).dtype == dtype


def test_ini_precision_and_typed_override(tmp_path):
    ini, science, ngs = _write_hybrid_inputs(tmp_path)
    ini.write_text(ini.read_text() + "\n[hybrid]\nprecision=float64\n")
    config = {
        "name": "Hybrid",
        "base_path": str(tmp_path),
        "config_path": ini.name,
        "science_ho_psf_interpolator_path": science.name,
        "ngs_ho_metric_interpolator_path": ngs.name,
    }
    simulation, payload = create_simulation_from_config(config)
    assert simulation.dtype == np.float64 and payload["precision"] == "float64"
    simulation, payload = create_simulation_from_config(
        {**config, "precision": "float32"}
    )
    assert simulation.dtype == np.float32 and payload["precision"] == "float32"
    assert "precision" not in normalize_simulation_config(
        SimulationConfig(name="Hybrid")
    )
    assert (
        normalize_simulation_config(SimulationConfig(name="Hybrid", dtype=np.float64))[
            "precision"
        ]
        == "float64"
    )
    for bad in (None, "float16", np.int32):
        with pytest.raises(ValueError, match="precision"):
            normalize_simulation_config({**config, "precision": bad})


def test_invalid_precision_load_keeps_bound_state(tmp_path):
    from ao_predict.simulation.hybrid import HybridSimulation
    from test_hybrid_simulation import _simulation_payload

    sim = HybridSimulation()
    payload = _simulation_payload(tmp_path)
    sim.load_simulation_payload(payload)
    with pytest.raises(ValueError):
        sim.load_simulation_payload(
            {
                **payload,
                "precision": "float64",
                "diagnostics_level": "debug",
                "diagnostic_fields": {},
            }
        )
    assert sim.dtype == np.float32 and sim.diagnostics_level == "none"


@pytest.mark.parametrize("dtype,diameters",[(np.float32,(8.1*u.m,810*u.cm)),(np.float64,(8.7*u.m,8700*u.mm))])
def test_telescope_invariants_allow_unit_conversion_roundoff(tmp_path,dtype,diameters):
    from ao_predict.persistence.store import _write_dataset_level_telescope_meta
    from ao_predict.simulation.interfaces import SimulationResult

    path = tmp_path / "units.h5"
    with h5py.File(path, "w") as root:
        root.create_dataset("stats/sr", data=np.zeros((2, 1), dtype=dtype))
        root.create_dataset("simulation/precision", data=np.dtype(dtype).name)
        root.create_dataset("meta/tel_diameter", data=dtype(np.nan))
        root.create_dataset(
            "meta/tel_pupil", data=np.full((2, 2), np.nan, dtype=dtype)
        )
        for diameter in diameters:
            result = SimulationResult(state=2)
            result.meta = {
                "tel_diameter": diameter,
                "tel_pupil": np.ones((2, 2)) * u.one,
            }
            _write_dataset_level_telescope_meta(root, result)
        result.meta["tel_diameter"] = 8.2 * u.m
        with pytest.raises(ValueError, match="does not match"):
            _write_dataset_level_telescope_meta(root, result)


def test_candidate_legacy_precision_is_independent_of_bound_state(tmp_path):
    from ao_predict.simulation.hybrid import HybridSimulation
    from test_hybrid_simulation import _simulation_payload
    from ao_predict._precision import diagnostic_precision
    sim=HybridSimulation()
    original=_simulation_payload(tmp_path,diagnostics_level='debug')
    double={**original,'precision':'float64','diagnostic_fields':diagnostic_precision(original['diagnostic_fields'],np.dtype('float64'))}
    sim.load_simulation_payload(double)
    legacy={key:value for key,value in original.items() if key!='precision'}
    sim.load_simulation_payload(legacy)
    assert sim.dtype==np.float32


def test_unlabelled_generic_diagnostics_keep_declared_precision():
    from ao_predict.simulation import TiptopSimulation

    class DoubleDiagnostics(TiptopSimulation):
        @property
        def supported_extra_diagnostics_levels(self):
            return ("debug",)

        def _diagnostic_field_specs(self, diagnostics_level):
            return {"native/value": {"dtype": "float64", "shape": (), "unit": u.one}}

    simulation = DoubleDiagnostics()
    payload = {
        "diagnostics_level": "debug",
        "diagnostic_fields": simulation._diagnostic_field_specs("debug"),
    }
    assert simulation._diagnostics_level_from_payload(payload) == "debug"
