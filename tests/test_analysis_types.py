from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

import ao_predict.simulation.api as sim_api
from ao_predict.analysis import (
    AnalysisDataset,
    AnalysisDatasetLoadPayload,
    AnalysisLoadContext,
    AnalysisLoadContribution,
    AnalysisSimulation,
    AnalysisSimulationLoadPayload,
    load_analysis_dataset,
)
from ao_predict.analysis._immutability import freeze_array, freeze_mapping
from ao_predict.persistence import SimulationStore
from ao_predict.simulation import SimulationState, schema
from ao_predict.simulation.api import InitDatasetRequest, OptionsConfig, SetupConfig, SimulationConfig


def test_freeze_array_returns_non_writeable_copy() -> None:
    source = np.array([1.0, 2.0], dtype=np.float32)

    frozen = freeze_array(source)
    source[0] = 9.0

    assert frozen.flags.writeable is False
    assert np.array_equal(frozen, np.array([1.0, 2.0], dtype=np.float32))


def test_freeze_mapping_recursively_freezes_nested_values() -> None:
    frozen = freeze_mapping(
        {
            "meta": {
                "pupil": np.ones((2, 2), dtype=np.float32),
            },
            "labels": ["a", "b"],
        }
    )

    assert isinstance(frozen, MappingProxyType)
    assert isinstance(frozen["meta"], MappingProxyType)
    assert frozen["meta"]["pupil"].flags.writeable is False
    assert frozen["labels"] == ("a", "b")

    with pytest.raises(TypeError):
        frozen["extra"] = 1


def test_analysis_package_exports() -> None:
    assert AnalysisDataset.__name__ == "AnalysisDataset"
    assert AnalysisSimulation.__name__ == "AnalysisSimulation"
    assert load_analysis_dataset.__name__ == "load_analysis_dataset"


def test_analysis_simulation_psfs_lazy_loads_once() -> None:
    calls: list[str] = []

    def _load_psfs() -> np.ndarray:
        calls.append("load")
        return np.ones((2, 3, 3), dtype=np.float32)

    simulation = AnalysisSimulation(
        _config=freeze_mapping({"setup": {}, "options": {}}),
        _meta=freeze_mapping({"pixel_scale_mas": 4.0}),
        _stats=freeze_mapping({"sr": np.array([0.1], dtype=np.float32)}),
        _extra_lazy_fields={"psfs": _load_psfs},
    )

    first = simulation.psfs
    second = simulation.psfs

    assert calls == ["load"]
    assert first is second
    assert first.flags.writeable is False


def test_analysis_simulation_requires_exact_config_shape() -> None:
    with pytest.raises(ValueError, match="exactly 'setup' and 'options'"):
        AnalysisSimulation(
            _config=freeze_mapping({"setup": {}, "options": {}, "simulation": {}}),
            _meta=freeze_mapping({"pixel_scale_mas": 4.0}),
            _stats=freeze_mapping({"sr": np.array([0.1], dtype=np.float32)}),
        )


def test_analysis_simulation_require_config_field_reads_setup_value() -> None:
    simulation = AnalysisSimulation(
        _config=freeze_mapping({"setup": {"mode": "LTAO"}, "options": {}}),
        _meta=freeze_mapping({}),
        _stats=freeze_mapping({}),
    )

    assert simulation._require_config_field("mode") == "LTAO"


def test_analysis_simulation_require_meta_field_reads_meta_value() -> None:
    simulation = AnalysisSimulation(
        _config=freeze_mapping({"setup": {}, "options": {}}),
        _meta=freeze_mapping({"mode": "SCAO"}),
        _stats=freeze_mapping({}),
    )

    assert simulation._require_meta_field("mode") == "SCAO"


def test_analysis_simulation_require_persisted_field_prefers_setup_over_meta() -> None:
    simulation = AnalysisSimulation(
        _config=freeze_mapping({"setup": {"mode": "LTAO"}, "options": {}}),
        _meta=freeze_mapping({"mode": "SCAO"}),
        _stats=freeze_mapping({}),
    )

    assert simulation._require_persisted_field("mode") == "LTAO"


def test_analysis_simulation_require_persisted_field_can_fall_back_to_meta() -> None:
    simulation = AnalysisSimulation(
        _config=freeze_mapping({"setup": {}, "options": {}}),
        _meta=freeze_mapping({"mode": "SCAO"}),
        _stats=freeze_mapping({}),
    )

    assert simulation._require_persisted_field("mode") == "SCAO"


def test_analysis_simulation_require_persisted_string_field_normalizes() -> None:
    simulation = AnalysisSimulation(
        _config=freeze_mapping({"setup": {"mode": "  LTAO  "}, "options": {}}),
        _meta=freeze_mapping({}),
        _stats=freeze_mapping({}),
    )

    assert simulation._require_persisted_string_field("mode", normalize=True) == "ltao"


def test_analysis_simulation_require_persisted_field_raises_on_missing_field() -> None:
    simulation = AnalysisSimulation(
        _config=freeze_mapping({"setup": {}, "options": {}}),
        _meta=freeze_mapping({}),
        _stats=freeze_mapping({}),
    )

    with pytest.raises(ValueError, match="Missing required persisted field 'mode'"):
        simulation._require_persisted_field("mode")


def test_analysis_simulation_require_persisted_string_field_raises_on_wrong_type() -> None:
    simulation = AnalysisSimulation(
        _config=freeze_mapping({"setup": {"mode": 3}, "options": {}}),
        _meta=freeze_mapping({}),
        _stats=freeze_mapping({}),
    )

    with pytest.raises(TypeError, match="Persisted field 'mode' must be a string"):
        simulation._require_persisted_string_field("mode")


def test_analysis_simulation_require_persisted_string_field_raises_on_empty_normalized_value() -> None:
    simulation = AnalysisSimulation(
        _config=freeze_mapping({"setup": {"mode": "   "}, "options": {}}),
        _meta=freeze_mapping({}),
        _stats=freeze_mapping({}),
    )

    with pytest.raises(ValueError, match="Persisted field 'mode' must be a non-empty string"):
        simulation._require_persisted_string_field("mode", normalize=True)


def test_analysis_dataset_sim_reuses_frozen_payloads() -> None:
    setup = freeze_mapping({"ee_apertures_mas": np.array([50.0], dtype=np.float32)})
    options = freeze_mapping({"wavelength_um": np.array([1.65], dtype=np.float64)})
    meta = freeze_mapping(
        {
            "pixel_scale_mas": np.array([4.0], dtype=np.float32),
            "tel_diameter_m": np.float32(8.0),
            "tel_pupil": np.ones((2, 2), dtype=np.float32),
        }
    )
    stats = freeze_mapping({"sr": np.array([[0.1]], dtype=np.float32)})
    dataset = AnalysisDataset(
        path=Path("/tmp/example.h5"),
        simulation_payload=freeze_mapping({"name": "demo"}),
        setup=setup,
        options=options,
        meta=meta,
        stats=stats,
        extra_stat_names=(),
    )

    sim = dataset.sim(0)

    assert len(dataset) == 1
    assert np.array_equal(sim.config["setup"]["ee_apertures_mas"], setup["ee_apertures_mas"])
    assert dataset.options["wavelength_um"].flags.writeable is False
    assert dataset.meta["tel_pupil"].flags.writeable is False
    assert dataset.stats["sr"].flags.writeable is False
    assert sim.config["options"]["wavelength_um"] == options["wavelength_um"][0]
    assert sim.meta["pixel_scale_mas"] == meta["pixel_scale_mas"][0]
    assert sim.meta["tel_diameter_m"] == meta["tel_diameter_m"]
    assert np.array_equal(sim.stats["sr"], stats["sr"][0])


def test_analysis_dataset_exposes_columnar_dataset_level_fields() -> None:
    dataset = AnalysisDataset(
        path=Path("/tmp/example.h5"),
        simulation_payload=freeze_mapping({"name": "demo"}),
        setup=freeze_mapping({"mode": "LTAO"}),
        options=freeze_mapping({"wavelength_um": np.array([1.25, 1.65], dtype=np.float64)}),
        meta=freeze_mapping(
            {
                "pixel_scale_mas": np.array([4.0, 5.0], dtype=np.float32),
                "tel_diameter_m": np.float32(8.0),
                "tel_pupil": np.ones((2, 2), dtype=np.float32),
            }
        ),
        stats=freeze_mapping({"sr": np.array([[0.1], [0.2]], dtype=np.float32)}),
        extra_stat_names=(),
    )

    assert dataset.setup["mode"] == "LTAO"
    np.testing.assert_allclose(dataset.options["wavelength_um"], np.array([1.25, 1.65], dtype=np.float64))
    np.testing.assert_allclose(dataset.meta["pixel_scale_mas"], np.array([4.0, 5.0], dtype=np.float32))
    assert dataset.meta["tel_diameter_m"] == np.float32(8.0)
    np.testing.assert_allclose(dataset.stats["sr"], np.array([[0.1], [0.2]], dtype=np.float32))


def test_analysis_dataset_rejects_mismatched_column_lengths() -> None:
    with pytest.raises(ValueError, match="share dataset size 2"):
        AnalysisDataset(
            path=Path("/tmp/example.h5"),
            simulation_payload=freeze_mapping({"name": "demo"}),
            setup=freeze_mapping({}),
            options=freeze_mapping({"wavelength_um": np.array([1.25, 1.65], dtype=np.float64)}),
            meta=freeze_mapping(
                {
                    "pixel_scale_mas": np.array([4.0], dtype=np.float32),
                    "tel_diameter_m": np.float32(8.0),
                    "tel_pupil": np.ones((2, 2), dtype=np.float32),
                }
            ),
            stats=freeze_mapping({"sr": np.array([[0.1], [0.2]], dtype=np.float32)}),
            extra_stat_names=(),
        )


def test_analysis_dataset_require_setup_field_reads_present_value() -> None:
    dataset = AnalysisDataset(
        path=Path("/tmp/example.h5"),
        simulation_payload=freeze_mapping({"name": "demo"}),
        setup=freeze_mapping({"mode": "LTAO"}),
        options=freeze_mapping({"wavelength_um": np.array([1.65], dtype=np.float64)}),
        meta=freeze_mapping(
            {
                "pixel_scale_mas": np.array([4.0], dtype=np.float32),
                "tel_diameter_m": np.float32(8.0),
                "tel_pupil": np.ones((2, 2), dtype=np.float32),
            }
        ),
        stats=freeze_mapping({"sr": np.array([[0.1]], dtype=np.float32)}),
        extra_stat_names=(),
    )

    assert dataset._require_setup_field("mode") == "LTAO"


def test_analysis_dataset_require_setup_string_field_normalizes() -> None:
    dataset = AnalysisDataset(
        path=Path("/tmp/example.h5"),
        simulation_payload=freeze_mapping({"name": "demo"}),
        setup=freeze_mapping({"mode": "  LTAO  "}),
        options=freeze_mapping({"wavelength_um": np.array([1.65], dtype=np.float64)}),
        meta=freeze_mapping(
            {
                "pixel_scale_mas": np.array([4.0], dtype=np.float32),
                "tel_diameter_m": np.float32(8.0),
                "tel_pupil": np.ones((2, 2), dtype=np.float32),
            }
        ),
        stats=freeze_mapping({"sr": np.array([[0.1]], dtype=np.float32)}),
        extra_stat_names=(),
    )

    assert dataset._require_setup_string_field("mode", normalize=True) == "ltao"


def test_analysis_dataset_require_setup_field_raises_on_missing_field() -> None:
    dataset = AnalysisDataset(
        path=Path("/tmp/example.h5"),
        simulation_payload=freeze_mapping({"name": "demo"}),
        setup=freeze_mapping({}),
        options=freeze_mapping({"wavelength_um": np.array([1.65], dtype=np.float64)}),
        meta=freeze_mapping(
            {
                "pixel_scale_mas": np.array([4.0], dtype=np.float32),
                "tel_diameter_m": np.float32(8.0),
                "tel_pupil": np.ones((2, 2), dtype=np.float32),
            }
        ),
        stats=freeze_mapping({"sr": np.array([[0.1]], dtype=np.float32)}),
        extra_stat_names=(),
    )

    with pytest.raises(ValueError, match="Missing required setup field 'mode'"):
        dataset._require_setup_field("mode")


def test_analysis_dataset_require_setup_string_field_raises_on_wrong_type() -> None:
    dataset = AnalysisDataset(
        path=Path("/tmp/example.h5"),
        simulation_payload=freeze_mapping({"name": "demo"}),
        setup=freeze_mapping({"mode": 3}),
        options=freeze_mapping({"wavelength_um": np.array([1.65], dtype=np.float64)}),
        meta=freeze_mapping(
            {
                "pixel_scale_mas": np.array([4.0], dtype=np.float32),
                "tel_diameter_m": np.float32(8.0),
                "tel_pupil": np.ones((2, 2), dtype=np.float32),
            }
        ),
        stats=freeze_mapping({"sr": np.array([[0.1]], dtype=np.float32)}),
        extra_stat_names=(),
    )

    with pytest.raises(TypeError, match="Setup field 'mode' must be a string"):
        dataset._require_setup_string_field("mode")


def test_analysis_dataset_simulation_payload_helpers_read_present_value() -> None:
    dataset = AnalysisDataset(
        path=Path("/tmp/example.h5"),
        simulation_payload=freeze_mapping({"mode": "LTAO", "name": "demo"}),
        setup=freeze_mapping({}),
        options=freeze_mapping({"wavelength_um": np.array([1.65], dtype=np.float64)}),
        meta=freeze_mapping(
            {
                "pixel_scale_mas": np.array([4.0], dtype=np.float32),
                "tel_diameter_m": np.float32(8.0),
                "tel_pupil": np.ones((2, 2), dtype=np.float32),
            }
        ),
        stats=freeze_mapping({"sr": np.array([[0.1]], dtype=np.float32)}),
        extra_stat_names=(),
    )

    assert dataset._require_simulation_payload_field("mode") == "LTAO"


def test_analysis_dataset_rejects_out_of_range_indexes() -> None:
    dataset = AnalysisDataset(
        path=Path("/tmp/example.h5"),
        simulation_payload=freeze_mapping({"name": "demo"}),
        setup=freeze_mapping({}),
        options=freeze_mapping({"wavelength_um": np.array([1.65], dtype=np.float64)}),
        meta=freeze_mapping(
            {
                "pixel_scale_mas": np.array([4.0], dtype=np.float32),
                "tel_diameter_m": np.float32(8.0),
                "tel_pupil": np.ones((2, 2), dtype=np.float32),
            }
        ),
        stats=freeze_mapping({"sr": np.array([[0.1]], dtype=np.float32)}),
        extra_stat_names=(),
    )

    with pytest.raises(IndexError, match=">= 0"):
        dataset.sim(-1)

    with pytest.raises(IndexError, match="out of range"):
        dataset.sim(1)


def test_analysis_dataset_subclasses_can_customize_sim_payload_without_reimplementing_sim() -> None:
    class CustomAnalysisSimulation(AnalysisSimulation):
        @property
        def mode(self) -> str:
            return self._require_extra_field("mode")

    class CustomAnalysisDataset(AnalysisDataset):
        def _build_simulation_load_payload(self, sim_idx: int) -> AnalysisSimulationLoadPayload:
            payload = super()._build_simulation_load_payload(sim_idx)
            return AnalysisSimulationLoadPayload(
                config=payload.config,
                meta=payload.meta,
                stats=payload.stats,
                extra_fields=freeze_mapping({"mode": f"mode-{sim_idx}", **dict(payload.extra_fields)}),
                extra_lazy_fields=payload.extra_lazy_fields,
            )

    dataset = CustomAnalysisDataset(
        path=Path("/tmp/example.h5"),
        simulation_payload=freeze_mapping({"name": "demo"}),
        setup=freeze_mapping({}),
        options=freeze_mapping({"wavelength_um": np.array([1.65], dtype=np.float64)}),
        meta=freeze_mapping(
            {
                "pixel_scale_mas": np.array([4.0], dtype=np.float32),
                "tel_diameter_m": np.float32(8.0),
                "tel_pupil": np.ones((2, 2), dtype=np.float32),
            }
        ),
        stats=freeze_mapping({"sr": np.array([[0.1]], dtype=np.float32)}),
        extra_stat_names=(),
        _simulation_cls=CustomAnalysisSimulation,
    )

    sim = dataset.sim(0)

    assert isinstance(sim, CustomAnalysisSimulation)
    assert sim.mode == "mode-0"


def test_load_analysis_dataset_loads_eager_non_psf_payloads(tmp_path: Path) -> None:
    data_path = tmp_path / "analysis_dataset.h5"
    store = SimulationStore(data_path)
    store.create(
        {
            "name": "ao_predict.simulation.tiptop:TiptopSimulation",
            "version": "x.y",
            "extra_stat_names": np.array(["halo_mas"], dtype=str),
            "base_config": "[section]\nvalue=1\n",
        },
        {
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
        },
        {
            "wavelength_um": np.full((2,), 1.65, dtype=float),
            "atm_profile_id": np.zeros((2,), dtype=np.int32),
            "zenith_angle_deg": np.full((2,), 20.0, dtype=float),
            "r0_m": np.full((2,), 0.16, dtype=float),
            "ngs_r_arcsec": np.ones((2, 3), dtype=float),
            "ngs_theta_deg": np.zeros((2, 3), dtype=float),
            "ngs_mag": np.full((2, 3), 15.0, dtype=float),
        },
        save_psfs=True,
    )

    result0_stats = {
        schema.KEY_STATS_SR: np.array([0.1, 0.2, 0.3], dtype=np.float32),
        schema.KEY_STATS_EE: np.full((3, 2), 0.5, dtype=np.float32),
        schema.KEY_STATS_FWHM_MAS: np.full((3,), 60.0, dtype=np.float32),
        "halo_mas": np.full((3,), 7.0, dtype=np.float32),
    }
    result1_stats = {
        schema.KEY_STATS_SR: np.array([0.4, 0.5, 0.6], dtype=np.float32),
        schema.KEY_STATS_EE: np.full((3, 2), 0.7, dtype=np.float32),
        schema.KEY_STATS_FWHM_MAS: np.full((3,), 80.0, dtype=np.float32),
        "halo_mas": np.full((3,), 9.0, dtype=np.float32),
    }
    result_meta = {
        schema.KEY_META_PIXEL_SCALE_MAS: 4.0,
        schema.KEY_META_TEL_DIAMETER_M: 8.0,
        schema.KEY_META_TEL_PUPIL: np.ones((6, 6), dtype=np.float32),
    }
    store.write_simulation_success(
        0,
        _success_result(stats=result0_stats, meta=result_meta, psfs=np.full((3, 4, 4), 0.1, dtype=np.float32)),
    )
    store.write_simulation_success(
        1,
        _success_result(stats=result1_stats, meta=result_meta, psfs=np.full((3, 4, 4), 0.2, dtype=np.float32)),
    )

    dataset = load_analysis_dataset(data_path)
    sim0 = dataset.sim(0)
    sim1 = dataset.sim(1)

    assert len(dataset) == 2
    assert dataset.path == data_path
    assert dataset.extra_stat_names == ("halo_mas",)
    assert dataset.simulation_payload["name"] == "ao_predict.simulation.tiptop:TiptopSimulation"
    np.testing.assert_allclose(dataset.options["wavelength_um"], np.full((2,), 1.65, dtype=float))
    np.testing.assert_allclose(
        dataset.meta[schema.KEY_META_PIXEL_SCALE_MAS],
        np.full((2,), 4.0, dtype=np.float32),
    )
    assert dataset.meta[schema.KEY_META_TEL_DIAMETER_M] == np.float32(8.0)
    np.testing.assert_allclose(dataset.stats[schema.KEY_STATS_SR][0], result0_stats[schema.KEY_STATS_SR])
    np.testing.assert_allclose(dataset.stats["halo_mas"][1], result1_stats["halo_mas"])
    assert tuple(sim0.config.keys()) == ("setup", "options")
    assert sim0.config["options"]["wavelength_um"] == np.float64(1.65)
    np.testing.assert_allclose(sim0.config["setup"]["ee_apertures_mas"], np.array([50.0, 100.0]))
    assert sim0.meta[schema.KEY_META_PIXEL_SCALE_MAS] == np.float32(4.0)
    assert sim0.meta[schema.KEY_META_TEL_DIAMETER_M] == np.float32(8.0)
    np.testing.assert_allclose(sim0.meta[schema.KEY_META_TEL_PUPIL], np.ones((6, 6), dtype=np.float32))
    np.testing.assert_allclose(sim0.stats[schema.KEY_STATS_SR], result0_stats[schema.KEY_STATS_SR])
    np.testing.assert_allclose(sim1.stats["halo_mas"], result1_stats["halo_mas"])
    assert "simulation" not in sim0.config
    assert dataset.options["wavelength_um"].flags.writeable is False
    assert dataset.meta[schema.KEY_META_PIXEL_SCALE_MAS].flags.writeable is False
    assert dataset.stats[schema.KEY_STATS_SR].flags.writeable is False
    assert sim0.config["setup"]["ee_apertures_mas"].flags.writeable is False
    assert sim0.meta[schema.KEY_META_TEL_PUPIL].flags.writeable is False
    assert sim0.stats[schema.KEY_STATS_SR].flags.writeable is False


def test_load_analysis_dataset_preserves_analysis_visible_store_slice(tmp_path: Path) -> None:
    data_path = tmp_path / "analysis_dataset_overlap.h5"
    store = SimulationStore(data_path)
    store.create(
        {
            "name": "ao_predict.simulation.tiptop:TiptopSimulation",
            "version": "x.y",
            "extra_stat_names": np.array(["halo_mas"], dtype=str),
            "base_config": "[section]\nvalue=1\n",
        },
        {
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
        },
        {
            "wavelength_um": np.full((1,), 1.65, dtype=float),
            "atm_profile_id": np.zeros((1,), dtype=np.int32),
            "zenith_angle_deg": np.full((1,), 20.0, dtype=float),
            "r0_m": np.full((1,), 0.16, dtype=float),
            "ngs_r_arcsec": np.ones((1, 3), dtype=float),
            "ngs_theta_deg": np.zeros((1, 3), dtype=float),
            "ngs_mag": np.full((1, 3), 15.0, dtype=float),
        },
        save_psfs=True,
    )
    expected_stats = {
        schema.KEY_STATS_SR: np.array([0.1, 0.2, 0.3], dtype=np.float32),
        schema.KEY_STATS_EE: np.full((3, 2), 0.5, dtype=np.float32),
        schema.KEY_STATS_FWHM_MAS: np.full((3,), 60.0, dtype=np.float32),
        "halo_mas": np.full((3,), 7.0, dtype=np.float32),
    }
    expected_meta = {
        schema.KEY_META_PIXEL_SCALE_MAS: 4.0,
        schema.KEY_META_TEL_DIAMETER_M: 8.0,
        schema.KEY_META_TEL_PUPIL: np.ones((6, 6), dtype=np.float32),
    }
    expected_psfs = np.full((3, 4, 4), 0.1, dtype=np.float32)
    store.write_simulation_success(
        0,
        _success_result(
            psfs=expected_psfs,
            stats=expected_stats,
            meta=expected_meta,
        ),
    )

    dataset = load_analysis_dataset(data_path)
    sim = dataset.sim(0)

    expected_options = store.read_sim_options(0)
    expected_meta_row = store.read_simulation_meta(0)
    expected_stats_row = store.read_simulation_stats(0)

    assert len(dataset) == 1
    assert dataset.path == data_path
    assert sim.config["options"].keys() == expected_options.keys()
    for key, value in expected_options.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(sim.config["options"][key], value)
        else:
            assert sim.config["options"][key] == value
    assert sim.meta.keys() == expected_meta_row.keys()
    assert sim.stats.keys() == expected_stats_row.keys()
    assert sim.meta[schema.KEY_META_PIXEL_SCALE_MAS] == expected_meta_row[schema.KEY_META_PIXEL_SCALE_MAS]
    assert sim.meta[schema.KEY_META_TEL_DIAMETER_M] == expected_meta_row[schema.KEY_META_TEL_DIAMETER_M]
    np.testing.assert_allclose(
        sim.meta[schema.KEY_META_TEL_PUPIL],
        expected_meta_row[schema.KEY_META_TEL_PUPIL],
    )
    np.testing.assert_allclose(sim.stats[schema.KEY_STATS_SR], expected_stats_row[schema.KEY_STATS_SR])
    np.testing.assert_allclose(sim.stats[schema.KEY_STATS_EE], expected_stats_row[schema.KEY_STATS_EE])
    np.testing.assert_allclose(
        sim.stats[schema.KEY_STATS_FWHM_MAS],
        expected_stats_row[schema.KEY_STATS_FWHM_MAS],
    )
    np.testing.assert_allclose(sim.stats["halo_mas"], expected_stats_row["halo_mas"])
    assert sim.psfs.shape == expected_psfs.shape
    np.testing.assert_allclose(sim.psfs, expected_psfs)


def test_load_analysis_dataset_supports_custom_dataset_cls(tmp_path: Path) -> None:
    data_path = tmp_path / "analysis_dataset_custom_dataset.h5"
    store = SimulationStore(data_path)
    _write_single_analysis_result(store, save_psfs=False)

    class CustomAnalysisDataset(AnalysisDataset):
        @classmethod
        def from_load_payload(
            cls,
            payload: AnalysisDatasetLoadPayload,
            simulation_cls: type[AnalysisSimulation] = AnalysisSimulation,
        ) -> "CustomAnalysisDataset":
            dataset = super().from_load_payload(payload, simulation_cls=simulation_cls)
            object.__setattr__(dataset, "_label", "custom-dataset")
            return dataset

        @property
        def label(self) -> str:
            return self._label

    dataset = load_analysis_dataset(data_path, dataset_cls=CustomAnalysisDataset)

    assert isinstance(dataset, CustomAnalysisDataset)
    assert dataset.label == "custom-dataset"
    assert len(dataset) == 1
    assert dataset.path == data_path
    assert dataset.simulation_payload["name"] == "ao_predict.simulation.tiptop:TiptopSimulation"


def test_load_analysis_dataset_supports_custom_simulation_cls(tmp_path: Path) -> None:
    data_path = tmp_path / "analysis_dataset_custom_simulation.h5"
    store = SimulationStore(data_path)
    _write_single_analysis_result(store, save_psfs=True)

    class CustomAnalysisSimulation(AnalysisSimulation):
        @classmethod
        def from_load_payload(
            cls,
            payload: AnalysisSimulationLoadPayload,
        ) -> "CustomAnalysisSimulation":
            return super().from_load_payload(payload)

        @property
        def pixel_scale(self) -> np.float32:
            return self.meta[schema.KEY_META_PIXEL_SCALE_MAS]

    dataset = load_analysis_dataset(data_path, simulation_cls=CustomAnalysisSimulation)
    sim = dataset.sim(0)

    assert isinstance(sim, CustomAnalysisSimulation)
    assert sim.config["options"]["wavelength_um"] == np.float64(1.65)
    assert sim.pixel_scale == np.float32(4.0)
    np.testing.assert_allclose(sim.stats[schema.KEY_STATS_SR], np.array([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(sim.psfs, np.full((3, 4, 4), 0.1, dtype=np.float32))


def test_load_analysis_dataset_supports_combined_dataset_and_simulation_classes(tmp_path: Path) -> None:
    data_path = tmp_path / "analysis_dataset_combined_classes.h5"
    store = SimulationStore(data_path)
    _write_single_analysis_result(store, save_psfs=False)

    calls: list[str] = []

    class CustomAnalysisDataset(AnalysisDataset):
        @classmethod
        def from_load_payload(
            cls,
            payload: AnalysisDatasetLoadPayload,
            simulation_cls: type[AnalysisSimulation] = AnalysisSimulation,
        ) -> "CustomAnalysisDataset":
            calls.append("dataset")
            return super().from_load_payload(payload, simulation_cls=simulation_cls)

    class CustomAnalysisSimulation(AnalysisSimulation):
        @classmethod
        def from_load_payload(
            cls,
            payload: AnalysisSimulationLoadPayload,
        ) -> "CustomAnalysisSimulation":
            calls.append("simulation")
            return super().from_load_payload(payload)

    dataset = load_analysis_dataset(
        data_path,
        dataset_cls=CustomAnalysisDataset,
        simulation_cls=CustomAnalysisSimulation,
    )
    sim = dataset.sim(0)

    assert isinstance(dataset, CustomAnalysisDataset)
    assert isinstance(sim, CustomAnalysisSimulation)
    assert calls == ["dataset", "simulation"]


def test_load_analysis_dataset_subclasses_can_use_load_payload_without_restating_dataset_fields(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "analysis_dataset_subclass_payload_helper.h5"
    store = SimulationStore(data_path)
    _write_single_analysis_result(store, save_psfs=False)

    class CustomAnalysisDataset(AnalysisDataset):
        @classmethod
        def from_load_payload(
            cls,
            payload: AnalysisDatasetLoadPayload,
            simulation_cls: type[AnalysisSimulation] = AnalysisSimulation,
        ) -> "CustomAnalysisDataset":
            dataset = super().from_load_payload(payload, simulation_cls=simulation_cls)
            object.__setattr__(dataset, "_payload_path", payload.path)
            return dataset

    dataset = load_analysis_dataset(data_path, dataset_cls=CustomAnalysisDataset)

    assert isinstance(dataset, CustomAnalysisDataset)
    assert dataset._payload_path == data_path
    assert len(dataset) == 1


def test_load_analysis_dataset_custom_simulation_cls_receives_generic_lazy_extra_loader(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "analysis_dataset_extra_loader.h5"
    store = SimulationStore(data_path)
    _write_single_analysis_result(store, save_psfs=False)

    extra_calls: list[int] = []

    class CustomAnalysisSimulation(AnalysisSimulation):
        @property
        def extra_cube(self) -> np.ndarray:
            return self._require_extra_field("extra_cube")

        @classmethod
        def from_load_payload(
            cls,
            payload: AnalysisSimulationLoadPayload,
        ) -> "CustomAnalysisSimulation":
            return super().from_load_payload(payload)

    def _extra_cube_extractor(ctx: AnalysisLoadContext) -> AnalysisLoadContribution:
        return AnalysisLoadContribution(
            simulation_lazy_fields=tuple(
                {"extra_cube": lambda sim_index=sim_idx: _record_extra_loader(extra_calls, sim_index)}
                for sim_idx in range(ctx.num_sims)
            )
        )

    dataset = load_analysis_dataset(
        data_path,
        simulation_cls=CustomAnalysisSimulation,
        extra_field_extractors=[_extra_cube_extractor],
    )
    sim = dataset.sim(0)

    assert extra_calls == []
    assert isinstance(sim, CustomAnalysisSimulation)
    np.testing.assert_allclose(sim.extra_cube, np.array([0.0], dtype=np.float32))
    assert extra_calls == [0]


def test_load_analysis_dataset_extra_field_extractors_support_eager_per_sim_fields(tmp_path: Path) -> None:
    data_path = tmp_path / "analysis_dataset_eager_extra.h5"
    store = SimulationStore(data_path)
    _write_single_analysis_result(store, save_psfs=False)

    class CustomAnalysisSimulation(AnalysisSimulation):
        @property
        def mode(self) -> str:
            return self._require_extra_field("mode")

    def _mode_extractor(ctx: AnalysisLoadContext) -> AnalysisLoadContribution:
        return AnalysisLoadContribution(
            simulation_fields=tuple({"mode": f"mode-{sim_idx}"} for sim_idx in range(ctx.num_sims))
        )

    dataset = load_analysis_dataset(
        data_path,
        simulation_cls=CustomAnalysisSimulation,
        extra_field_extractors=[_mode_extractor],
    )

    assert dataset.sim(0).mode == "mode-0"


def test_load_analysis_dataset_extra_field_extractors_support_dataset_level_fields(tmp_path: Path) -> None:
    data_path = tmp_path / "analysis_dataset_dataset_extra.h5"
    store = SimulationStore(data_path)
    _write_single_analysis_result(store, save_psfs=False)

    class CustomAnalysisDataset(AnalysisDataset):
        @property
        def mode_summary(self) -> str:
            return self._require_extra_field("mode_summary")

    def _dataset_extractor(ctx: AnalysisLoadContext) -> AnalysisLoadContribution:
        return AnalysisLoadContribution(dataset_fields={"mode_summary": f"count={ctx.num_sims}"})

    dataset = load_analysis_dataset(
        data_path,
        dataset_cls=CustomAnalysisDataset,
        extra_field_extractors=[_dataset_extractor],
    )

    assert dataset.mode_summary == "count=1"


def test_analysis_simulation_require_extra_field_raises_cleanly_when_missing() -> None:
    class CustomAnalysisSimulation(AnalysisSimulation):
        @property
        def grouped_psfs(self) -> np.ndarray:
            return self._require_extra_field("grouped_psfs")

    simulation = CustomAnalysisSimulation(
        _config=freeze_mapping({"setup": {}, "options": {}}),
        _meta=freeze_mapping({"pixel_scale_mas": 4.0}),
        _stats=freeze_mapping({"sr": np.array([0.1], dtype=np.float32)}),
    )

    with pytest.raises(ValueError, match="grouped_psfs"):
        _ = simulation.grouped_psfs


def test_load_analysis_dataset_fails_early_on_invalid_schema(tmp_path: Path) -> None:
    data_path = tmp_path / "invalid_analysis_dataset.h5"
    data_path.write_text("not an hdf5 dataset", encoding="utf-8")

    with pytest.raises(ValueError, match="Schema validation failed"):
        load_analysis_dataset(data_path)


def test_load_analysis_dataset_does_not_build_analysis_objects_before_schema_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "invalid_analysis_dataset_prevalidation.h5"
    data_path.write_text("not an hdf5 dataset", encoding="utf-8")
    calls: list[SimulationStore] = []

    def _unexpected_builder(store: SimulationStore, **_: Any) -> AnalysisDataset:
        calls.append(store)
        raise AssertionError("analysis composition should not run before schema validation")

    monkeypatch.setattr("ao_predict.analysis._compose._load_analysis_dataset_from_store", _unexpected_builder)

    with pytest.raises(ValueError, match="Schema validation failed"):
        load_analysis_dataset(data_path)

    assert calls == []


def test_load_analysis_dataset_reads_dataset_built_via_init_and_run_pipeline(tmp_path: Path) -> None:
    dataset_path = tmp_path / "analysis_pipeline_dataset.h5"
    request = InitDatasetRequest(
        dataset_path=dataset_path,
        simulation=SimulationConfig(name="mock_simulation:MockSimulation"),
        setup=SetupConfig(ee_apertures_mas=[50.0, 100.0]),
        options=OptionsConfig(
            option_arrays={
                "zenith_angle_deg": np.array([15.0, 25.0, 35.0], dtype=float),
            }
        ),
        save_psfs=True,
    )

    num_sims = sim_api.init_dataset(request)
    assert num_sims == 3

    summary = sim_api.run_simulations_by_state(dataset_path, state=SimulationState.PENDING)
    assert summary.attempted == 3
    assert summary.succeeded == 3
    assert summary.failed == 0

    dataset = load_analysis_dataset(dataset_path)
    sim0 = dataset.sim(0)
    sim1 = dataset.sim(1)

    assert len(dataset) == 3
    assert dataset.path == dataset_path
    assert sim0.config["options"]["zenith_angle_deg"] == np.float64(15.0)
    assert sim1.config["options"]["zenith_angle_deg"] == np.float64(25.0)
    assert sim0.meta[schema.KEY_META_PIXEL_SCALE_MAS] == np.float32(5.0)
    assert sim0.stats[schema.KEY_STATS_SR].shape == (1,)
    assert np.all(np.isfinite(sim0.stats[schema.KEY_STATS_SR]))
    np.testing.assert_allclose(sim0.psfs, np.full((1, 4, 4), 0.1, dtype=np.float32))
    np.testing.assert_allclose(sim1.psfs, np.full((1, 4, 4), 0.2, dtype=np.float32))
    assert sim0.psfs.flags.writeable is False


def test_analysis_simulation_views_are_read_only_to_callers(tmp_path: Path) -> None:
    data_path = tmp_path / "analysis_dataset_read_only.h5"
    store = SimulationStore(data_path)
    store.create(
        {
            "name": "ao_predict.simulation.tiptop:TiptopSimulation",
            "version": "x.y",
            "extra_stat_names": np.array(["halo_mas"], dtype=str),
            "base_config": "[section]\nvalue=1\n",
        },
        {
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
        },
        {
            "wavelength_um": np.full((1,), 1.65, dtype=float),
            "atm_profile_id": np.zeros((1,), dtype=np.int32),
            "zenith_angle_deg": np.full((1,), 20.0, dtype=float),
            "r0_m": np.full((1,), 0.16, dtype=float),
            "ngs_r_arcsec": np.ones((1, 3), dtype=float),
            "ngs_theta_deg": np.zeros((1, 3), dtype=float),
            "ngs_mag": np.full((1, 3), 15.0, dtype=float),
        },
        save_psfs=True,
    )
    store.write_simulation_success(
        0,
        _success_result(
            psfs=np.full((3, 4, 4), 0.1, dtype=np.float32),
            stats={
                schema.KEY_STATS_SR: np.array([0.1, 0.2, 0.3], dtype=np.float32),
                schema.KEY_STATS_EE: np.full((3, 2), 0.5, dtype=np.float32),
                schema.KEY_STATS_FWHM_MAS: np.full((3,), 60.0, dtype=np.float32),
                "halo_mas": np.full((3,), 7.0, dtype=np.float32),
            },
            meta={
                schema.KEY_META_PIXEL_SCALE_MAS: 4.0,
                schema.KEY_META_TEL_DIAMETER_M: 8.0,
                schema.KEY_META_TEL_PUPIL: np.ones((6, 6), dtype=np.float32),
            },
        ),
    )

    sim = load_analysis_dataset(data_path).sim(0)

    with pytest.raises(TypeError):
        sim.config["new"] = {}
    with pytest.raises(TypeError):
        sim.meta["new"] = 1
    with pytest.raises(TypeError):
        sim.stats["new"] = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="read-only"):
        sim.config["setup"]["ee_apertures_mas"][0] = 99.0
    with pytest.raises(ValueError, match="read-only"):
        sim.meta[schema.KEY_META_TEL_PUPIL][0, 0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        sim.stats["halo_mas"][0] = -1.0
    with pytest.raises(ValueError, match="read-only"):
        sim.psfs[0, 0, 0] = -1.0


def test_analysis_dataset_psfs_remain_lazy_until_simulation_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_path = tmp_path / "analysis_dataset_psfs.h5"
    store = SimulationStore(data_path)
    store.create(
        {
            "name": "ao_predict.simulation.tiptop:TiptopSimulation",
            "version": "x.y",
            "extra_stat_names": np.array([], dtype=str),
            "base_config": "[section]\nvalue=1\n",
        },
        {
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
        },
        {
            "wavelength_um": np.full((1,), 1.65, dtype=float),
            "atm_profile_id": np.zeros((1,), dtype=np.int32),
            "zenith_angle_deg": np.full((1,), 20.0, dtype=float),
            "r0_m": np.full((1,), 0.16, dtype=float),
            "ngs_r_arcsec": np.ones((1, 3), dtype=float),
            "ngs_theta_deg": np.zeros((1, 3), dtype=float),
            "ngs_mag": np.full((1, 3), 15.0, dtype=float),
        },
        save_psfs=True,
    )
    store.write_simulation_success(
        0,
        _success_result(
            psfs=np.full((3, 4, 4), 0.1, dtype=np.float32),
            stats={
                schema.KEY_STATS_SR: np.array([0.1, 0.2, 0.3], dtype=np.float32),
                schema.KEY_STATS_EE: np.full((3, 2), 0.5, dtype=np.float32),
                schema.KEY_STATS_FWHM_MAS: np.full((3,), 60.0, dtype=np.float32),
            },
            meta={
                schema.KEY_META_PIXEL_SCALE_MAS: 4.0,
                schema.KEY_META_TEL_DIAMETER_M: 8.0,
                schema.KEY_META_TEL_PUPIL: np.ones((6, 6), dtype=np.float32),
            },
        ),
    )

    calls: list[int] = []
    original_reader = AnalysisLoadContext.read_sim_array

    def _recording_reader(self: AnalysisLoadContext, path: str, sim_idx: int) -> np.ndarray:
        if path == "/psfs/data":
            calls.append(sim_idx)
        return original_reader(self, path, sim_idx)

    monkeypatch.setattr(AnalysisLoadContext, "read_sim_array", _recording_reader)

    dataset = load_analysis_dataset(data_path)
    sim = dataset.sim(0)

    assert calls == []
    _ = sim.stats
    assert calls == []

    first = sim.psfs
    second = sim.psfs

    assert calls == [0]
    assert first is second
    assert first.flags.writeable is False


def test_analysis_simulation_psfs_raises_clear_error_when_dataset_has_no_psfs(tmp_path: Path) -> None:
    data_path = tmp_path / "analysis_dataset_missing_psfs.h5"
    store = SimulationStore(data_path)
    store.create(
        {
            "name": "ao_predict.simulation.tiptop:TiptopSimulation",
            "version": "x.y",
            "extra_stat_names": np.array([], dtype=str),
            "base_config": "[section]\nvalue=1\n",
        },
        {
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
        },
        {
            "wavelength_um": np.full((1,), 1.65, dtype=float),
            "atm_profile_id": np.zeros((1,), dtype=np.int32),
            "zenith_angle_deg": np.full((1,), 20.0, dtype=float),
            "r0_m": np.full((1,), 0.16, dtype=float),
            "ngs_r_arcsec": np.ones((1, 3), dtype=float),
            "ngs_theta_deg": np.zeros((1, 3), dtype=float),
            "ngs_mag": np.full((1, 3), 15.0, dtype=float),
        },
        save_psfs=False,
    )
    store.write_simulation_success(
        0,
        _success_result(
            psfs=np.full((3, 4, 4), 0.1, dtype=np.float32),
            stats={
                schema.KEY_STATS_SR: np.array([0.1, 0.2, 0.3], dtype=np.float32),
                schema.KEY_STATS_EE: np.full((3, 2), 0.5, dtype=np.float32),
                schema.KEY_STATS_FWHM_MAS: np.full((3,), 60.0, dtype=np.float32),
            },
            meta={
                schema.KEY_META_PIXEL_SCALE_MAS: 4.0,
                schema.KEY_META_TEL_DIAMETER_M: 8.0,
                schema.KEY_META_TEL_PUPIL: np.ones((6, 6), dtype=np.float32),
            },
        ),
    )

    dataset = load_analysis_dataset(data_path)

    with pytest.raises(ValueError, match="PSFs are not available in this dataset"):
        _ = dataset.sim(0).psfs


def test_analysis_dataset_loads_via_supported_loader_entrypoint(tmp_path: Path) -> None:
    data_path = tmp_path / "analysis_dataset_loader_entrypoint.h5"
    store = SimulationStore(data_path)
    store.create(
        {
            "name": "ao_predict.simulation.tiptop:TiptopSimulation",
            "version": "x.y",
            "extra_stat_names": np.array([], dtype=str),
            "base_config": "[section]\nvalue=1\n",
        },
        {
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
        },
        {
            "wavelength_um": np.full((1,), 1.65, dtype=float),
            "atm_profile_id": np.zeros((1,), dtype=np.int32),
            "zenith_angle_deg": np.full((1,), 20.0, dtype=float),
            "r0_m": np.full((1,), 0.16, dtype=float),
            "ngs_r_arcsec": np.ones((1, 3), dtype=float),
            "ngs_theta_deg": np.zeros((1, 3), dtype=float),
            "ngs_mag": np.full((1, 3), 15.0, dtype=float),
        },
        save_psfs=False,
    )
    store.write_simulation_success(
        0,
        _success_result(
            psfs=np.full((3, 4, 4), 0.1, dtype=np.float32),
            stats={
                schema.KEY_STATS_SR: np.array([0.1, 0.2, 0.3], dtype=np.float32),
                schema.KEY_STATS_EE: np.full((3, 2), 0.5, dtype=np.float32),
                schema.KEY_STATS_FWHM_MAS: np.full((3,), 60.0, dtype=np.float32),
            },
            meta={
                schema.KEY_META_PIXEL_SCALE_MAS: 4.0,
                schema.KEY_META_TEL_DIAMETER_M: 8.0,
                schema.KEY_META_TEL_PUPIL: np.ones((6, 6), dtype=np.float32),
            },
        ),
    )

    dataset = load_analysis_dataset(data_path)

    assert len(dataset) == 1
    assert dataset.path == data_path


def test_analysis_dataset_exposes_supported_scientific_view_contract(tmp_path: Path) -> None:
    data_path = tmp_path / "analysis_dataset_contract.h5"
    store = SimulationStore(data_path)
    store.create(
        {
            "name": "ao_predict.simulation.tiptop:TiptopSimulation",
            "version": "x.y",
            "extra_stat_names": np.array(["halo_mas"], dtype=str),
            "base_config": "[section]\nvalue=1\n",
        },
        {
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
        },
        {
            "wavelength_um": np.array([1.25, 1.65], dtype=float),
            "atm_profile_id": np.zeros((2,), dtype=np.int32),
            "zenith_angle_deg": np.array([10.0, 25.0], dtype=float),
            "r0_m": np.array([0.15, 0.16], dtype=float),
            "ngs_r_arcsec": np.array([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]], dtype=float),
            "ngs_theta_deg": np.array([[0.0, 45.0, 90.0], [135.0, 180.0, 225.0]], dtype=float),
            "ngs_mag": np.array([[14.0, 15.0, 16.0], [17.0, 18.0, 19.0]], dtype=float),
        },
        save_psfs=False,
    )
    expected_rows: list[dict[str, Any]] = [
        {
            "stats": {
                schema.KEY_STATS_SR: np.array([0.1, 0.2, 0.3], dtype=np.float32),
                schema.KEY_STATS_EE: np.full((3, 2), 0.5, dtype=np.float32),
                schema.KEY_STATS_FWHM_MAS: np.full((3,), 60.0, dtype=np.float32),
                "halo_mas": np.full((3,), 7.0, dtype=np.float32),
            },
            "meta": {
                schema.KEY_META_PIXEL_SCALE_MAS: 4.0,
                schema.KEY_META_TEL_DIAMETER_M: 8.0,
                schema.KEY_META_TEL_PUPIL: np.ones((6, 6), dtype=np.float32),
            },
        },
        {
            "stats": {
                schema.KEY_STATS_SR: np.array([0.4, 0.5, 0.6], dtype=np.float32),
                schema.KEY_STATS_EE: np.full((3, 2), 0.7, dtype=np.float32),
                schema.KEY_STATS_FWHM_MAS: np.full((3,), 80.0, dtype=np.float32),
                "halo_mas": np.full((3,), 9.0, dtype=np.float32),
            },
            "meta": {
                schema.KEY_META_PIXEL_SCALE_MAS: 5.0,
                schema.KEY_META_TEL_DIAMETER_M: 8.0,
                schema.KEY_META_TEL_PUPIL: np.ones((6, 6), dtype=np.float32),
            },
        },
    ]
    for sim_idx, row in enumerate(expected_rows):
        store.write_simulation_success(
            sim_idx,
            _success_result(
                psfs=np.full((3, 4, 4), 0.1 + 0.1 * sim_idx, dtype=np.float32),
                stats=row["stats"],
                meta=row["meta"],
            ),
        )

    dataset = load_analysis_dataset(data_path)
    sim0 = dataset.sim(0)
    sim1 = dataset.sim(1)

    assert len(dataset) == 2
    assert tuple(sim0.config.keys()) == ("setup", "options")
    np.testing.assert_allclose(
        sim0.config["setup"]["ee_apertures_mas"],
        np.array([50.0, 100.0], dtype=float),
    )
    assert sim0.config["setup"]["sr_method"] == schema.DEFAULT_SETUP_SR_METHOD
    assert sim0.config["options"]["wavelength_um"] == np.float64(1.25)
    assert sim1.config["options"]["zenith_angle_deg"] == np.float64(25.0)
    assert sim0.meta[schema.KEY_META_PIXEL_SCALE_MAS] == np.float32(4.0)
    assert sim1.meta[schema.KEY_META_PIXEL_SCALE_MAS] == np.float32(5.0)
    assert sim0.meta[schema.KEY_META_TEL_DIAMETER_M] == np.float32(8.0)
    np.testing.assert_allclose(sim1.meta[schema.KEY_META_TEL_PUPIL], np.ones((6, 6), dtype=np.float32))
    np.testing.assert_allclose(sim0.stats[schema.KEY_STATS_SR], expected_rows[0]["stats"][schema.KEY_STATS_SR])
    np.testing.assert_allclose(sim0.stats[schema.KEY_STATS_EE], expected_rows[0]["stats"][schema.KEY_STATS_EE])
    np.testing.assert_allclose(
        sim1.stats[schema.KEY_STATS_FWHM_MAS],
        expected_rows[1]["stats"][schema.KEY_STATS_FWHM_MAS],
    )
    np.testing.assert_allclose(sim1.stats["halo_mas"], expected_rows[1]["stats"]["halo_mas"])


def _success_result(
    *,
    stats: dict[str, np.ndarray],
    meta: dict[str, object],
    psfs: np.ndarray,
) -> object:
    from ao_predict.simulation import SimulationResult, SimulationState

    return SimulationResult(
        state=SimulationState.SUCCEEDED,
        stats=stats,
        meta=dict(meta),
        psfs=psfs,
    )


def _record_extra_loader(calls: list[int], sim_idx: int) -> np.ndarray:
    calls.append(sim_idx)
    return np.array([float(sim_idx)], dtype=np.float32)


def _write_single_analysis_result(store: SimulationStore, *, save_psfs: bool) -> None:
    store.create(
        {
            "name": "ao_predict.simulation.tiptop:TiptopSimulation",
            "version": "x.y",
            "extra_stat_names": np.array([], dtype=str),
            "base_config": "[section]\nvalue=1\n",
        },
        {
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
        },
        {
            "wavelength_um": np.full((1,), 1.65, dtype=float),
            "atm_profile_id": np.zeros((1,), dtype=np.int32),
            "zenith_angle_deg": np.full((1,), 20.0, dtype=float),
            "r0_m": np.full((1,), 0.16, dtype=float),
            "ngs_r_arcsec": np.ones((1, 3), dtype=float),
            "ngs_theta_deg": np.zeros((1, 3), dtype=float),
            "ngs_mag": np.full((1, 3), 15.0, dtype=float),
        },
        save_psfs=save_psfs,
    )
    store.write_simulation_success(
        0,
        _success_result(
            psfs=np.full((3, 4, 4), 0.1, dtype=np.float32),
            stats={
                schema.KEY_STATS_SR: np.array([0.1, 0.2, 0.3], dtype=np.float32),
                schema.KEY_STATS_EE: np.full((3, 2), 0.5, dtype=np.float32),
                schema.KEY_STATS_FWHM_MAS: np.full((3,), 60.0, dtype=np.float32),
            },
            meta={
                schema.KEY_META_PIXEL_SCALE_MAS: 4.0,
                schema.KEY_META_TEL_DIAMETER_M: 8.0,
                schema.KEY_META_TEL_PUPIL: np.ones((6, 6), dtype=np.float32),
            },
        ),
    )
