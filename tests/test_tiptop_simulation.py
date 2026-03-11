from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest

from ao_predict.simulation import SimulationResult, TiptopSimulation


def _ini_text() -> str:
    return (
        "[main]\nvalue = 42\n"
        "[science]\nwavelength_um = 1.65\n"
        "[telescope]\nZenithAngle=20\nTelescopeDiameter=8.0\n"
        "[atmosphere]\nWavelength=500e-9\nr0_Value=0.16\nL0=25.0\nCn2Heights=[0,5000]\nCn2Weights=[0.6,0.4]\nWindSpeed=[5,10]\nWindDirection=[0,90]\n"
        "[RTC]\nSensorFrameRate_LO=500.0\n"
        "[sensor_science]\nTelescopeDiameterForPixelScale=9.6556\nPixelScale=8.8\n"
        "[sensor_LO]\nNumberLenslets=[16]\nNumberPhotons=[100]\n"
        "[sources_LO]\nWavelength=[710e-9]\nZenith=[1.0]\nAzimuth=[0.0]\n"
        "[sources_science]\nWavelength=[1.65e-06]\nZenith=[0.0,1.0]\nAzimuth=[0.0,90.0]\n"
    )


def test_tiptop_config_roundtrip(tmp_path: Path):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(_ini_text(), encoding="utf-8")

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    assert "base_config" in simulation_payload
    assert isinstance(simulation_payload["base_config"], str)

    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})
    assert sim._base_config is not None
    assert sim._base_config.parser["main"]["value"] == "42"
    assert sim._base_config.parser["science"]["wavelength_um"] == "1.65"


def test_tiptop_create_context(tmp_path: Path):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(_ini_text(), encoding="utf-8")

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    setup = {
        "ee_apertures_mas": np.array([50.0, 100.0]),
        "ngs_mag_zeropoint": 1.1e13 / 368.0,
        "sci_r_arcsec": np.array([0.0, 1.0]),
        "atm_profiles": {
            "0": {
                "name": "default",
                "r0_m": 0.16,
                "L0_m": 25.0,
                "cn2_heights_m": np.array([0.0, 5000.0]),
                "cn2_weights": np.array([0.6, 0.4]),
                "wind_speed_mps": np.array([5.0, 10.0]),
                "wind_direction_deg": np.array([0.0, 90.0]),
            },
            "1": {
                "name": "alt",
                "r0_m": 0.20,
                "L0_m": 30.0,
                "cn2_heights_m": np.array([0.0, 4000.0, 8000.0]),
                "cn2_weights": np.array([0.5, 0.3, 0.2]),
                "wind_speed_mps": np.array([4.0, 8.0, 12.0]),
                "wind_direction_deg": np.array([10.0, 50.0, 90.0]),
            },
        },
    }
    options = {
        "wavelength_um": 1.65,
        "zenith_angle_deg": 25.0,
        "r0_m": 0.12,
        "ngs_r_arcsec": np.array([10.0, np.nan, 20.0]),
        "ngs_theta_deg": np.array([0.0, np.nan, 180.0]),
        "ngs_mag": np.array([14.0, np.nan, 15.0]),
        "ngs_used": np.array([True, False, True]),
        "atm_profile_id": 1,
    }
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})
    setup_payload = sim.prepare_setup_payload({"ee_apertures_mas": setup["ee_apertures_mas"]}, setup)
    sim.load_setup_payload(setup_payload)
    ctx = sim.create(index=3, options=options)

    assert ctx.index == 3
    assert ctx.setup is not None
    assert ctx.runtime["effective_parser"] is not None
    assert sim._base_config is not None
    assert ctx.runtime["effective_parser"] is not sim._base_config.parser
    assert float(ctx.setup.ngs_mag_zeropoint) > 0.0
    assert float(ctx.setup.atm_wavelength_um) == pytest.approx(0.5)
    assert ctx.runtime["effective_parser"]["sources_science"]["Wavelength"] == "[1.650000e-06]"
    # Pixel scale remains whatever is defined in the base INI for generic TIPTOP behavior.
    assert float(ctx.runtime["effective_parser"]["sensor_science"]["PixelScale"]) == pytest.approx(8.8, abs=1e-4)
    # NGS overrides keep only active stars and compute LO photons from ngs_mag.
    assert ctx.runtime["effective_parser"]["sources_LO"]["Zenith"] == "[10,20]"
    assert ctx.runtime["effective_parser"]["sources_LO"]["Azimuth"] == "[0,180]"
    np.testing.assert_array_equal(ctx.options["ngs_used"], np.array([True, False, True]))
    photons = np.fromstring(ctx.runtime["effective_parser"]["sensor_LO"]["NumberPhotons"].strip("[]"), sep=",")
    assert photons.size == 2
    assert np.all(photons > 0.0)
    assert photons[0] > photons[1]
    # Selected atmospheric profile is applied; effective config carries Seeing (not r0_Value).
    assert "r0_Value" not in ctx.runtime["effective_parser"]["atmosphere"]
    assert float(ctx.runtime["effective_parser"]["atmosphere"]["Seeing"]) > 0.0
    assert float(ctx.runtime["effective_parser"]["atmosphere"]["L0"]) == pytest.approx(30.0)
    np.testing.assert_allclose(
        np.fromstring(ctx.runtime["effective_parser"]["atmosphere"]["Cn2Heights"].strip("[]"), sep=","),
        np.array([0.0, 4000.0, 8000.0]),
    )
    np.testing.assert_allclose(
        np.fromstring(ctx.runtime["effective_parser"]["atmosphere"]["Cn2Weights"].strip("[]"), sep=","),
        np.array([0.5, 0.3, 0.2]),
    )
    np.testing.assert_allclose(
        np.fromstring(ctx.runtime["effective_parser"]["atmosphere"]["WindSpeed"].strip("[]"), sep=","),
        np.array([4.0, 8.0, 12.0]),
    )
    np.testing.assert_allclose(
        np.fromstring(ctx.runtime["effective_parser"]["atmosphere"]["WindDirection"].strip("[]"), sep=","),
        np.array([10.0, 50.0, 90.0]),
    )


def test_tiptop_validate_setup_payload_does_not_rebind_loaded_setup(tmp_path: Path):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(_ini_text(), encoding="utf-8")

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})

    original_setup_payload = sim.prepare_setup_payload(
        {"ee_apertures_mas": np.array([50.0, 100.0])},
        {"ngs_mag_zeropoint": 3.0e10},
    )
    sim.load_setup_payload(original_setup_payload)
    original_setup = sim.setup

    candidate_setup_payload = sim.prepare_setup_payload(
        {"ee_apertures_mas": np.array([25.0])},
        {"ngs_mag_zeropoint": 4.0e10},
    )
    sim.validate_setup_payload(candidate_setup_payload)

    assert sim.setup is original_setup
    np.testing.assert_allclose(sim.setup.ee_apertures_mas, np.array([50.0, 100.0]))
    assert float(sim.setup.ngs_mag_zeropoint) == pytest.approx(3.0e10)


def test_tiptop_create_context_leaves_ini_ngs_when_options_omit_ngs(tmp_path: Path):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(_ini_text(), encoding="utf-8")

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})

    setup_payload = sim.prepare_setup_payload(
        {"ee_apertures_mas": np.array([50.0, 100.0])},
        {"ngs_mag_zeropoint": 3.0e10},
    )
    sim.load_setup_payload(setup_payload)

    ctx = sim.create(
        index=0,
        options={
            "wavelength_um": 1.65,
            "zenith_angle_deg": 25.0,
            "r0_m": 0.12,
            "atm_profile_id": 0,
        },
    )

    assert ctx.runtime["effective_parser"]["sources_LO"]["Zenith"] == "[1.0]"
    assert ctx.runtime["effective_parser"]["sources_LO"]["Azimuth"] == "[0.0]"
    assert ctx.runtime["effective_parser"]["sensor_LO"]["NumberPhotons"] == "[100]"


def test_tiptop_prepare_options_payload_loads_ngs_defaults_from_ini(tmp_path: Path):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(_ini_text(), encoding="utf-8")

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})

    setup_payload = sim.prepare_setup_payload(
        {"ee_apertures_mas": np.array([50.0, 100.0])},
        {"ngs_mag_zeropoint": 3.0e10},
    )

    options_payload = sim.prepare_options_payload(
        2,
        setup_payload,
        {
            "wavelength_um": np.array([1.65, 1.65], dtype=float),
            "zenith_angle_deg": np.array([20.0, 25.0], dtype=float),
            "atm_profile_id": np.array([0, 0], dtype=np.int32),
            "r0_m": np.array([0.16, 0.14], dtype=float),
        },
    )

    np.testing.assert_allclose(options_payload["ngs_r_arcsec"], np.array([[1.0], [1.0]]))
    np.testing.assert_allclose(options_payload["ngs_theta_deg"], np.array([[0.0], [0.0]]))
    assert options_payload["ngs_mag"].shape == (2, 1)
    assert np.all(np.isfinite(options_payload["ngs_mag"]))


def test_tiptop_run_finalize_with_stubbed_tiptop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(
        (
            "[telescope]\nZenithAngle=20\nTelescopeDiameter=8.0\n"
            "[atmosphere]\nWavelength=500e-9\nr0_Value=0.16\nL0=25\nCn2Heights=[0,5000]\nCn2Weights=[0.6,0.4]\nWindSpeed=[5,10]\nWindDirection=[0,90]\n"
            "[RTC]\nSensorFrameRate_LO=500.0\n"
            "[sensor_LO]\nNumberLenslets=[16]\nNumberPhotons=[100]\n"
            "[sources_LO]\nWavelength=[710e-9]\nZenith=[10.0,20.0,30.0]\nAzimuth=[0.0,120.0,240.0]\n"
            "[sources_science]\nWavelength=[1.65e-06]\nZenith=[0.0,1.0,2.0]\nAzimuth=[0.0,90.0,180.0]\n"
        ),
        encoding="utf-8",
    )

    class _Img:
        def __init__(self, value: float):
            self.sampling = np.full((4, 4), value, dtype=np.float32)

    class FakeBaseSimulation:
        def __init__(self, *_args, **kwargs):
            self.kwargs = kwargs
            self.sr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            self.ee = np.array([0.4, 0.5, 0.6], dtype=np.float32)
            self.fwhm = np.array([50.0, 60.0, 70.0], dtype=np.float32)
            self.results = [_Img(0.1), _Img(0.2), _Img(0.3)]
            self.tel_radius = 4.0
            self.psInMas = 2.0
            self.fao = types.SimpleNamespace(
                ao=types.SimpleNamespace(
                    tel=types.SimpleNamespace(
                        pupil=np.ones((6, 6), dtype=np.float32),
                    )
                )
            )

        def doOverallSimulation(self, **_kwargs):
            return None

    tiptop_mod = types.ModuleType("tiptop.tiptop")
    tiptop_mod.baseSimulation = FakeBaseSimulation
    tiptop_pkg = types.ModuleType("tiptop")
    tiptop_pkg.tiptop = tiptop_mod
    monkeypatch.setitem(sys.modules, "tiptop", tiptop_pkg)
    monkeypatch.setitem(sys.modules, "tiptop.tiptop", tiptop_mod)

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    setup = {
        "ee_apertures_mas": np.array([50.0, 100.0]),
        "ngs_mag_zeropoint": 1.1e13 / 368.0,
        "sci_r_arcsec": np.array([0.0, 1.0, 2.0]),
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
    }
    options = {
        "wavelength_um": 1.25,
        "zenith_angle_deg": 35.0,
        "ngs_r_arcsec": np.array([10.0, 20.0, 30.0]),
        "ngs_theta_deg": np.array([0.0, 120.0, 240.0]),
        "ngs_mag": np.array([14.0, 15.0, 16.0]),
        "ngs_used": np.array([True, True, True]),
        "atm_profile_id": 0,
    }
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})
    setup_payload = sim.prepare_setup_payload({"ee_apertures_mas": setup["ee_apertures_mas"]}, setup)
    sim.load_setup_payload(setup_payload)
    ctx = sim.create(index=0, options=options)
    sim.run(ctx)
    sim.finalize(ctx)

    assert isinstance(ctx.result, SimulationResult)
    assert ctx.result.state == 1
    assert ctx.result.psfs is not None
    assert ctx.result.psfs.shape == (3, 4, 4)
    assert ctx.result.meta["pixel_scale_mas"] == np.float32(2.0)
    assert ctx.result.meta["tel_diameter_m"] == np.float32(8.0)
    assert ctx.result.meta["tel_pupil"].shape == (6, 6)
    assert ctx.result.stats == {}

    # Verify first EE aperture width (50 mas) was converted to radius in TIPTOP call.
    runtime_sim = ctx.runtime["tiptop_simulation"]
    assert runtime_sim.kwargs["eeRadiusInMas"] == 25.0


def test_tiptop_derives_r0_from_seeing_when_missing(tmp_path: Path):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop_seeing.ini"
    ini_path.write_text(
        (
            "[telescope]\nZenithAngle=20\nTelescopeDiameter=8.0\n"
            "[atmosphere]\nWavelength=500e-9\nSeeing=0.83\nL0=25\nCn2Heights=[0,5000]\nCn2Weights=[0.6,0.4]\nWindSpeed=[5,10]\nWindDirection=[0,90]\n"
            "[RTC]\nSensorFrameRate_LO=500.0\n"
            "[sensor_LO]\nNumberLenslets=[16]\nNumberPhotons=[100]\n"
            "[sources_LO]\nWavelength=[710e-9]\nZenith=[10.0]\nAzimuth=[0.0]\n"
            "[sources_science]\nWavelength=[1.65e-06]\nZenith=[0.0]\nAzimuth=[0.0]\n"
        ),
        encoding="utf-8",
    )

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})
    setup_payload = sim.prepare_setup_payload(
        {"ee_apertures_mas": np.array([50.0])},
        {"ngs_mag_zeropoint": 1.1e13 / 368.0},
    )

    assert 0 in setup_payload["atm_profiles"]
    assert "r0_m" in setup_payload["atm_profiles"][0]
    assert float(setup_payload["atm_profiles"][0]["r0_m"]) > 0.0


def test_tiptop_uses_only_seeing_in_effective_config_when_r0_present(tmp_path: Path):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop_seeing.ini"
    ini_path.write_text(
        (
            "[telescope]\nZenithAngle=20\nTelescopeDiameter=8.0\n"
            "[atmosphere]\nWavelength=500e-9\nSeeing=0.83\nL0=25\nCn2Heights=[0,5000]\nCn2Weights=[0.6,0.4]\nWindSpeed=[5,10]\nWindDirection=[0,90]\n"
            "[RTC]\nSensorFrameRate_LO=500.0\n"
            "[sensor_LO]\nNumberLenslets=[16]\nNumberPhotons=[100]\n"
            "[sources_LO]\nWavelength=[710e-9]\nZenith=[10.0]\nAzimuth=[0.0]\n"
            "[sources_science]\nWavelength=[1.65e-06]\nZenith=[0.0]\nAzimuth=[0.0]\n"
        ),
        encoding="utf-8",
    )

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})
    setup_payload = sim.prepare_setup_payload(
        {"ee_apertures_mas": np.array([50.0])},
        {"ngs_mag_zeropoint": 3.0e10},
    )
    sim.load_setup_payload(setup_payload)

    ctx = sim.create(
        index=0,
        options={
            "wavelength_um": 1.65,
            "zenith_angle_deg": 20.0,
            "atm_profile_id": 0,
            "r0_m": 0.16,
            "ngs_r_arcsec": np.array([10.0]),
            "ngs_theta_deg": np.array([0.0]),
            "ngs_mag": np.array([14.0]),
            "ngs_used": np.array([True]),
        },
    )
    atm = ctx.runtime["effective_parser"]["atmosphere"]
    assert "Seeing" in atm
    assert float(atm["Seeing"]) > 0.0
    assert "r0_Value" not in atm


def test_tiptop_accepts_seeing_arcsec_in_atm_profiles(tmp_path: Path):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(_ini_text(), encoding="utf-8")

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})

    setup_payload = sim.prepare_setup_payload(
        {"ee_apertures_mas": np.array([50.0, 100.0])},
        {
            "ngs_mag_zeropoint": 1.1e13 / 368.0,
            "atm_profiles": {
                "0": {
                    "name": "seeing_profile",
                    "seeing_arcsec": 0.8,
                    "L0_m": 25.0,
                    "cn2_heights_m": np.array([0.0, 5000.0]),
                    "cn2_weights": np.array([0.6, 0.4]),
                    "wind_speed_mps": np.array([5.0, 10.0]),
                    "wind_direction_deg": np.array([0.0, 90.0]),
                }
            }
        },
    )

    profile = setup_payload["atm_profiles"][0]
    assert "seeing_arcsec" not in profile
    assert "r0_m" in profile
    assert float(profile["r0_m"]) > 0.0


def test_tiptop_requires_runtime_ngs_used_when_ngs_options_present(tmp_path: Path):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(_ini_text(), encoding="utf-8")

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})
    setup_payload = sim.prepare_setup_payload(
        {"ee_apertures_mas": np.array([50.0])},
        {"ngs_mag_zeropoint": 3.0e10},
    )
    sim.load_setup_payload(setup_payload)

    with pytest.raises(ValueError, match="ngs_used"):
        sim.create(
            index=0,
            options={
                "wavelength_um": 1.65,
                "zenith_angle_deg": 20.0,
                "atm_profile_id": 0,
                "r0_m": 0.16,
                "ngs_r_arcsec": np.array([10.0]),
                "ngs_theta_deg": np.array([0.0]),
                "ngs_mag": np.array([14.0]),
            },
        )


@pytest.mark.parametrize(("field", "value"), [("r0_m", 0.0), ("r0_m", -0.1), ("L0_m", 0.0), ("L0_m", -1.0)])
def test_tiptop_rejects_non_positive_atm_profile_scalars(tmp_path: Path, field: str, value: float):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(_ini_text(), encoding="utf-8")

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})

    profile = {
        "name": "invalid_profile",
        "r0_m": 0.16,
        "L0_m": 25.0,
        "cn2_heights_m": np.array([0.0, 5000.0]),
        "cn2_weights": np.array([0.6, 0.4]),
        "wind_speed_mps": np.array([5.0, 10.0]),
        "wind_direction_deg": np.array([0.0, 90.0]),
    }
    profile[field] = value

    with pytest.raises(ValueError, match=field):
        sim.prepare_setup_payload(
            {"ee_apertures_mas": np.array([50.0, 100.0])},
            {
                "ngs_mag_zeropoint": 1.1e13 / 368.0,
                "atm_profiles": {"0": profile},
            },
        )


def test_tiptop_rejects_inconsistent_atm_profile_r0_and_seeing(tmp_path: Path):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop.ini"
    ini_path.write_text(_ini_text(), encoding="utf-8")

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})

    with pytest.raises(ValueError, match="Inconsistent atmospheric profile"):
        sim.prepare_setup_payload(
            {"ee_apertures_mas": np.array([50.0, 100.0])},
            {
                "ngs_mag_zeropoint": 1.1e13 / 368.0,
                "atm_profiles": {
                    "1": {
                        "name": "conflict_profile",
                        "seeing_arcsec": 0.8,
                        "r0_m": 0.16,
                        "L0_m": 25.0,
                        "cn2_heights_m": np.array([0.0, 5000.0]),
                        "cn2_weights": np.array([0.6, 0.4]),
                        "wind_speed_mps": np.array([5.0, 10.0]),
                        "wind_direction_deg": np.array([0.0, 90.0]),
                    }
                }
            },
        )


def test_tiptop_requires_ngs_mag_zeropoint(tmp_path: Path):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop_missing_zp.ini"
    ini_path.write_text(
        (
            "[telescope]\nZenithAngle=20\nTelescopeDiameter=8.0\n"
            "[atmosphere]\nWavelength=500e-9\nr0_Value=0.16\nL0=25\nCn2Heights=[0,5000]\nCn2Weights=[0.6,0.4]\nWindSpeed=[5,10]\nWindDirection=[0,90]\n"
            "[sources_LO]\nWavelength=[850e-9]\nZenith=[10.0]\nAzimuth=[0.0]\n"
            "[sources_science]\nWavelength=[1.65e-06]\nZenith=[0.0]\nAzimuth=[0.0]\n"
            "[RTC]\nSensorFrameRate_LO=500.0\n"
            "[sensor_LO]\nNumberLenslets=[16]\nNumberPhotons=[100]\n"
        ),
        encoding="utf-8",
    )

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})
    with pytest.raises(ValueError, match="ngs_mag_zeropoint"):
        sim.prepare_setup_payload({"ee_apertures_mas": np.array([50.0])}, {})


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ("SensorFrameRate_LO=fast", "RTC.SensorFrameRate_LO must be numeric"),
        ("SensorFrameRate_LO=0.0", "RTC.SensorFrameRate_LO must be > 0"),
    ],
)
def test_tiptop_rejects_invalid_lo_frame_rate_for_ngs_mag(tmp_path: Path, line: str, match: str):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop_bad_frame_rate.ini"
    ini_path.write_text(_ini_text().replace("SensorFrameRate_LO=500.0", line), encoding="utf-8")

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})
    setup_payload = sim.prepare_setup_payload(
        {"ee_apertures_mas": np.array([50.0, 100.0])},
        {"ngs_mag_zeropoint": 1.1e13 / 368.0},
    )
    sim.load_setup_payload(setup_payload)

    with pytest.raises(ValueError, match=match):
        sim.create(
            index=0,
            options={
                "wavelength_um": 1.65,
                "zenith_angle_deg": 20.0,
                "atm_profile_id": 0,
                "r0_m": 0.16,
                "ngs_r_arcsec": np.array([10.0]),
                "ngs_theta_deg": np.array([0.0]),
                "ngs_mag": np.array([14.0]),
                "ngs_used": np.array([True]),
            },
        )


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ("TelescopeDiameter=wide", "telescope.TelescopeDiameter must be numeric"),
        ("TelescopeDiameter=0.0", "telescope.TelescopeDiameter must be > 0"),
    ],
)
def test_tiptop_rejects_invalid_telescope_diameter_for_ngs_mag(tmp_path: Path, line: str, match: str):
    sim = TiptopSimulation()
    ini_path = tmp_path / "tiptop_bad_telescope.ini"
    ini_path.write_text(_ini_text().replace("TelescopeDiameter=8.0", line), encoding="utf-8")

    simulation_payload = sim.prepare_simulation_payload({"config_path": str(ini_path)})
    sim.load_simulation_payload({"name": sim.name, "version": sim.version, **simulation_payload})
    setup_payload = sim.prepare_setup_payload(
        {"ee_apertures_mas": np.array([50.0, 100.0])},
        {"ngs_mag_zeropoint": 1.1e13 / 368.0},
    )
    sim.load_setup_payload(setup_payload)

    with pytest.raises(ValueError, match=match):
        sim.create(
            index=0,
            options={
                "wavelength_um": 1.65,
                "zenith_angle_deg": 20.0,
                "atm_profile_id": 0,
                "r0_m": 0.16,
                "ngs_r_arcsec": np.array([10.0]),
                "ngs_theta_deg": np.array([0.0]),
                "ngs_mag": np.array([14.0]),
                "ngs_used": np.array([True]),
            },
        )
