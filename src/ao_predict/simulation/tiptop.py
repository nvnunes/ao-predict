"""TIPTOP simulation implementation."""

from __future__ import annotations

from configparser import ConfigParser
from copy import deepcopy
from dataclasses import dataclass
import io
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from . import schema
from . import atm
from .base import BaseSimulation, BaseSimulationSetup, PsfParameters
from .helpers import r0_to_seeing_arcsec, seeing_arcsec_to_r0_m
from .interfaces import SimulationContext, SimulationSetup
from .photometry import (
    WFSPhotometryConfig,
    magnitudes_to_photons_per_frame,
    photons_per_frame_to_magnitudes,
)
from ..utils import as_float_vector


@dataclass
class TiptopBaseConfig:
    """Parsed TIPTOP base configuration and source text."""

    ini_text: str
    parser: ConfigParser


@dataclass(frozen=True)
class TiptopSetup(BaseSimulationSetup):
    """Typed setup payload for ``TiptopSimulation``.

    This currently adds no fields beyond ``BaseSimulationSetup`` but keeps a
    dedicated type for TIPTOP-specific extensions.
    """


# INI format helpers

def _serialize_parser(parser: ConfigParser) -> str:
    """Serialize ``ConfigParser`` back to INI text."""
    buf = io.StringIO()
    parser.write(buf)
    return buf.getvalue()


def _parse_ini_text(ini_text: str) -> ConfigParser:
    """Parse INI text into a case-preserving ``ConfigParser``."""
    parser = ConfigParser(interpolation=None)
    parser.optionxform = str  # preserve original option case
    parser.read_string(ini_text)
    return parser


def _parse_ini_array(raw: str) -> np.ndarray:
    """Parse TIPTOP bracket-array text (``[a,b,c]``) into a float array."""
    return np.fromstring(raw.strip("[]"), dtype=float, sep=",")


def _format_ini_array(values: np.ndarray) -> str:
    """Format a numeric vector into TIPTOP bracket-array text."""
    arr = np.asarray(values).reshape(-1)
    return "[" + ",".join(f"{float(v):.6g}" for v in arr) + "]"


def _get_ini_array(parser: ConfigParser, section: str, key: str) -> np.ndarray | None:
    """Read a vector-valued INI field as float array."""
    if not parser.has_section(section) or key not in parser[section]:
        return None
    values = _parse_ini_array(parser[section][key])
    if values.size == 0:
        return None
    return np.asarray(values, dtype=float).reshape(-1)


def _get_ini_float(parser: ConfigParser, section: str, key: str) -> float | None:
    """Read a scalar-valued INI field as float."""
    if not parser.has_section(section) or key not in parser[section]:
        return None
    try:
        return float(parser[section][key])
    except ValueError:
        return None


# Tiptop Simulation

class TiptopSimulation(BaseSimulation):
    """TIPTOP-backed `Simulation` implementation."""

    _VERSION = "0.0.1"

    KEY_SETUP_CONFIG_PATH = "config_path"
    KEY_SETUP_BASE_CONFIG = "base_config"
    KEY_RUNTIME_EFFECTIVE_PARSER = "effective_parser"
    KEY_RUNTIME_SIMULATION = "tiptop_simulation"
    ATM_PROFILE_KEYS_TO_INI_FIELDS = {
        atm.KEY_SETUP_ATM_PROFILE_R0_M: "r0_Value",
        atm.KEY_SETUP_ATM_PROFILE_L0_M: "L0",
        atm.KEY_SETUP_ATM_PROFILE_CN2_HEIGHTS_M: "Cn2Heights",
        atm.KEY_SETUP_ATM_PROFILE_CN2_WEIGHTS: "Cn2Weights",
        atm.KEY_SETUP_ATM_PROFILE_WIND_SPEED_MPS: "WindSpeed",
        atm.KEY_SETUP_ATM_PROFILE_WIND_DIRECTION_DEG: "WindDirection",
    }

    # Properties

    @property
    def base_config(self) -> TiptopBaseConfig:
        """Return loaded TIPTOP base config or raise if not configured."""
        base_config = self._base_config
        if not isinstance(base_config, TiptopBaseConfig):
            raise TypeError("TiptopSimulation base config is not configured. Call load_simulation_payload(...) first.")
        return base_config

    # Construction and properties

    def __init__(self) -> None:
        """Initialize unbound TIPTOP simulation state."""
        super().__init__()
        self._base_config: TiptopBaseConfig | None = None

    # Simulation payload lifecycle

    def prepare_simulation_payload(self, simulation_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
        """Build persisted ``/simulation`` payload from config input.

        Args:
            simulation_cfg: Normalized simulation config mapping.

        Returns:
            Persisted simulation payload containing serialized TIPTOP INI text.

        Raises:
            ValueError: If required fields are missing.
            TypeError: If input field types are invalid.
            FileNotFoundError: If referenced INI file is missing.
        """
        source_path = simulation_cfg.get(self.KEY_SETUP_CONFIG_PATH)
        if source_path is None:
            raise ValueError(
                f"TiptopSimulation.prepare_simulation_payload requires simulation['{self.KEY_SETUP_CONFIG_PATH}'] in YAML input."
            )
        if not isinstance(source_path, str):
            raise TypeError(f"simulation['{self.KEY_SETUP_CONFIG_PATH}'] must be a string.")

        ini_path = Path(source_path)
        if not ini_path.is_absolute():
            base_path = simulation_cfg.get(schema.KEY_CFG_SIMULATION_BASE_PATH)
            if base_path is not None:
                if not isinstance(base_path, str):
                    raise TypeError(f"simulation['{schema.KEY_CFG_SIMULATION_BASE_PATH}'] must be a string when provided.")
                ini_path = Path(base_path) / ini_path
        if not ini_path.is_file():
            raise FileNotFoundError(f"TIPTOP INI file not found: {ini_path}")

        ini_text = ini_path.read_text(encoding="utf-8")

        simulation_payload = self._build_simulation_payload(
            simulation_cfg,
            exclude_keys={
                self.KEY_SETUP_CONFIG_PATH,
            },
        )
        simulation_payload[self.KEY_SETUP_BASE_CONFIG] = ini_text
        return simulation_payload

    def validate_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Validate persisted ``/simulation`` payload expected by TIPTOP backend.

        Args:
            simulation_payload: Candidate persisted simulation payload.

        Raises:
            ValueError: If required fields are missing or malformed.
            TypeError: If required fields have invalid types.
        """
        super().validate_simulation_payload(simulation_payload)
        base_config_str = self._get_required_base_config_text(simulation_payload)
        # Parse the INI text and raise if it is malformed.
        _ = _parse_ini_text(base_config_str)

    def load_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Load and parse persisted ``/simulation/base_config`` INI text.

        Args:
            simulation_payload: Persisted simulation payload.

        Raises:
            ValueError: If required fields are missing.
            TypeError: If required fields have invalid types.
        """
        base_config_str = self._get_required_base_config_text(simulation_payload)
        self._base_config = TiptopBaseConfig(
            ini_text=base_config_str,
            parser=_parse_ini_text(base_config_str),
        )

    # Setup payload lifecycle

    def prepare_setup_payload(
        self,
        base_setup_payload: Mapping[str, Any],
        setup_cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Resolve and return persisted ``/setup`` payload for TIPTOP.

        Args:
            base_setup_payload: Core-prepared setup payload.
            setup_cfg: Full setup config mapping.

        Returns:
            Persisted setup payload with TIPTOP-resolved defaults.
        """
        parser = self.base_config.parser

        atm_wavelength_um = self._get_required_atm_wavelength_m(parser, "prepare setup payload") * 1e6
        default_atm_profile = self._get_default_atm_profile_from_ini(parser)

        lgs_r_arcsec = _get_ini_array(parser, "sources_HO", "Zenith")
        lgs_theta_deg = _get_ini_array(parser, "sources_HO", "Azimuth")
        if lgs_r_arcsec is None:
            lgs_r_arcsec = np.asarray([], dtype=float)
        if lgs_theta_deg is None:
            lgs_theta_deg = np.asarray([], dtype=float)

        sci_r_arcsec = _get_ini_array(parser, "sources_science", "Zenith")
        sci_theta_deg = _get_ini_array(parser, "sources_science", "Azimuth")

        setup_payload = self._build_setup_payload(
            base_setup_payload,
            setup_cfg,
            default_atm_wavelength_um=atm_wavelength_um,
            default_atm_profile=default_atm_profile,
            default_lgs_r_arcsec=lgs_r_arcsec,
            default_lgs_theta_deg=lgs_theta_deg,
            default_sci_r_arcsec=sci_r_arcsec,
            default_sci_theta_deg=sci_theta_deg,
        )
        return setup_payload

    def _parse_setup_payload(self, setup_payload: Mapping[str, Any]) -> TiptopSetup:
        """Parse and validate TIPTOP persisted ``/setup`` without binding it.

        Args:
            setup_payload: Candidate persisted setup payload.

        Returns:
            Parsed ``TiptopSetup`` instance ready to be bound by
            ``load_setup_payload()``.

        Raises:
            TypeError: If persisted setup field types are invalid.
            ValueError: If required setup fields are missing or invalid.
        """
        return self._parse_base_setup_payload(setup_payload, TiptopSetup)

    def load_setup_payload(self, setup_payload: Mapping[str, Any]) -> None:
        """Bind typed TIPTOP setup from persisted ``/setup`` payload.

        Args:
            setup_payload: Persisted setup payload read from dataset storage.

        Raises:
            TypeError: If persisted setup field types are invalid.
            ValueError: If required setup fields are missing or invalid.
        """
        self._load_base_setup_payload(setup_payload, TiptopSetup)

    # Options payload lifecycle

    def prepare_options_payload(
        self,
        num_sims: int,
        setup_payload: Mapping[str, Any],
        base_options_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Complete missing per-simulation options using TIPTOP INI defaults.

        Args:
            num_sims: Number of simulations ``N``.
            setup_payload: Persisted setup payload for setup-dependent defaults.
            base_options_payload: Core-prepared options payload.

        Returns:
            Completed per-simulation options payload.
        """
        num_sims = int(num_sims)
        if num_sims <= 0:
            raise ValueError("num_sims must be > 0.")

        parser = self.base_config.parser
        options_payload = {str(key): np.asarray(value).copy() for key, value in base_options_payload.items()}

        default_options: dict[str, Any] = {}
        if schema.KEY_OPTION_WAVELENGTH_UM not in options_payload:
            default_options[schema.KEY_OPTION_WAVELENGTH_UM] = float(self._get_default_wavelength_m_from_ini(parser) * 1e6)
        if schema.KEY_OPTION_ZENITH_ANGLE_DEG not in options_payload:
            default_options[schema.KEY_OPTION_ZENITH_ANGLE_DEG] = float(self._get_default_zenith_angle_deg_from_ini(parser))
        if schema.KEY_OPTION_ATM_PROFILE_ID not in options_payload:
            default_options[schema.KEY_OPTION_ATM_PROFILE_ID] = np.int32(0)
        if schema.KEY_OPTION_R0_M not in options_payload:
            default_options[schema.KEY_OPTION_R0_M] = float(self._get_default_r0_m_from_ini(parser))

        if not any(key in options_payload for key in schema.OPTION_KEYS_NGS):
            default_ngs_options = self._get_default_ngs_options_from_ini(
                parser,
                self._get_ngs_photometry_config(
                    parser,
                    float(setup_payload[self.KEY_SETUP_NGS_MAG_ZEROPOINT]),
                ),
            )
            if default_ngs_options is not None:
                for key, values in default_ngs_options.items():
                    values = np.asarray(values, dtype=float).reshape(1, -1)
                    options_payload[key] = np.broadcast_to(values, (num_sims, values.shape[1])).copy()

        return self._build_options_payload(
            num_sims,
            options_payload,
            default_options=default_options,
        )

    # INI field helpers

    def _get_required_base_config_text(self, simulation_payload: Mapping[str, Any]) -> str:
        """Read required serialized TIPTOP base config text from ``/simulation``."""
        if self.KEY_SETUP_BASE_CONFIG not in simulation_payload:
            raise ValueError(f"TiptopSimulation requires simulation['{self.KEY_SETUP_BASE_CONFIG}'].")
        base_config_text = simulation_payload[self.KEY_SETUP_BASE_CONFIG]
        if not isinstance(base_config_text, str):
            raise TypeError(f"simulation['{self.KEY_SETUP_BASE_CONFIG}'] must be a string for TiptopSimulation.")
        return base_config_text

    @staticmethod
    def _get_required_atm_wavelength_m(parser: ConfigParser, purpose: str) -> float:
        """Read required atmosphere wavelength from INI in meters."""
        if not parser.has_section("atmosphere") or "Wavelength" not in parser["atmosphere"]:
            raise ValueError(f"TIPTOP atmosphere.Wavelength must be present to {purpose}.")
        try:
            atm_wavelength_m = float(parser["atmosphere"]["Wavelength"])
        except ValueError as exc:
            raise ValueError(f"TIPTOP atmosphere.Wavelength must be numeric to {purpose}.") from exc
        if atm_wavelength_m <= 0.0:
            raise ValueError(f"TIPTOP atmosphere.Wavelength must be > 0 to {purpose}.")
        return atm_wavelength_m

    @staticmethod
    def _get_frame_rate_lo(parser: ConfigParser) -> float:
        """Read LO WFS frame rate needed for magnitude/photon conversions."""
        if not parser.has_section("RTC") or "SensorFrameRate_LO" not in parser["RTC"]:
            raise ValueError("TIPTOP config missing RTC.SensorFrameRate_LO required for ngs_mag conversion.")
        try:
            frame_rate_hz = float(parser["RTC"]["SensorFrameRate_LO"])
        except ValueError as exc:
            raise ValueError("TIPTOP RTC.SensorFrameRate_LO must be numeric for ngs_mag conversion.") from exc
        if frame_rate_hz <= 0.0:
            raise ValueError("TIPTOP RTC.SensorFrameRate_LO must be > 0 for ngs_mag conversion.")
        return frame_rate_hz

    @staticmethod
    def _get_n_lenslets_lo(parser: ConfigParser) -> float:
        """Read LO sensor lenslet count for magnitude/photon conversions."""
        if not parser.has_section("sensor_LO") or "NumberLenslets" not in parser["sensor_LO"]:
            raise ValueError("TIPTOP config missing sensor_LO.NumberLenslets required for ngs_mag conversion.")
        n_lenslets = _parse_ini_array(parser["sensor_LO"]["NumberLenslets"])
        if n_lenslets.size == 0:
            raise ValueError("sensor_LO.NumberLenslets is empty.")
        return float(n_lenslets[0])

    @staticmethod
    def _get_telescope_diameter_m(parser: ConfigParser) -> float:
        """Read telescope diameter used in photon normalization."""
        if not parser.has_section("telescope") or "TelescopeDiameter" not in parser["telescope"]:
            raise ValueError("TIPTOP config missing telescope.TelescopeDiameter required for ngs_mag conversion.")
        try:
            telescope_diameter_m = float(parser["telescope"]["TelescopeDiameter"])
        except ValueError as exc:
            raise ValueError("TIPTOP telescope.TelescopeDiameter must be numeric for ngs_mag conversion.") from exc
        if telescope_diameter_m <= 0.0:
            raise ValueError("TIPTOP telescope.TelescopeDiameter must be > 0 for ngs_mag conversion.")
        return telescope_diameter_m

    @classmethod
    def _get_ngs_photometry_config(
        cls,
        parser: ConfigParser,
        ngs_mag_zeropoint: float,
    ) -> WFSPhotometryConfig:
        """Read parser-backed inputs needed for NGS magnitude/photon conversions."""
        return WFSPhotometryConfig(
            telescope_diameter_m=cls._get_telescope_diameter_m(parser),
            n_channels=cls._get_n_lenslets_lo(parser),
            frame_rate_hz=cls._get_frame_rate_lo(parser),
            zeropoint=float(ngs_mag_zeropoint),
        )

    def _get_default_r0_m_from_ini(self, parser: ConfigParser) -> float:
        """Read default r0 option from INI (or derive from Seeing)."""
        r0_m = _get_ini_float(parser, "atmosphere", "r0_Value")
        if r0_m is not None:
            return float(r0_m)

        seeing_arcsec = _get_ini_float(parser, "atmosphere", "Seeing")
        wavelength_m = _get_ini_float(parser, "atmosphere", "Wavelength")
        if seeing_arcsec is None or wavelength_m is None:
            raise ValueError(
                "TIPTOP config must provide atmosphere.r0_Value, or both atmosphere.Seeing and atmosphere.Wavelength "
                "for default r0_m option."
            )
        if seeing_arcsec <= 0.0:
            raise ValueError("TIPTOP atmosphere.Seeing must be > 0 when deriving default r0_m.")
        if wavelength_m <= 0.0:
            raise ValueError("TIPTOP atmosphere.Wavelength must be > 0 when deriving default r0_m.")

        return seeing_arcsec_to_r0_m(float(seeing_arcsec), float(wavelength_m))

    # TIPTOP default helpers

    def _get_default_atm_profile_from_ini(self, parser: ConfigParser) -> dict[str, Any]:
        """Construct default atmospheric profile (id=0) from base INI."""
        profile: dict[str, Any] = {atm.KEY_SETUP_ATM_PROFILE_NAME: "ini_default"}
        scalar_map = {
            atm.KEY_SETUP_ATM_PROFILE_L0_M: ("atmosphere", "L0"),
        }
        array_map = {
            atm.KEY_SETUP_ATM_PROFILE_CN2_HEIGHTS_M: ("atmosphere", "Cn2Heights"),
            atm.KEY_SETUP_ATM_PROFILE_CN2_WEIGHTS: ("atmosphere", "Cn2Weights"),
            atm.KEY_SETUP_ATM_PROFILE_WIND_SPEED_MPS: ("atmosphere", "WindSpeed"),
            atm.KEY_SETUP_ATM_PROFILE_WIND_DIRECTION_DEG: ("atmosphere", "WindDirection"),
        }

        for dst_key, (section, key) in scalar_map.items():
            value = _get_ini_float(parser, section, key)
            if value is not None:
                profile[dst_key] = float(value)
        profile[atm.KEY_SETUP_ATM_PROFILE_R0_M] = float(self._get_default_r0_m_from_ini(parser))
        for dst_key, (section, key) in array_map.items():
            value = _get_ini_array(parser, section, key)
            if value is not None:
                profile[dst_key] = value
        return profile

    def _get_default_wavelength_m_from_ini(self, parser: ConfigParser) -> float:
        """Read default science wavelength option from INI in meters."""
        wavelength_m = _get_ini_array(parser, "sources_science", "Wavelength")
        if wavelength_m is None or wavelength_m.size == 0:
            raise ValueError("TIPTOP config missing sources_science.Wavelength for default wavelength_um option.")
        return float(wavelength_m[0])

    def _get_default_zenith_angle_deg_from_ini(self, parser: ConfigParser) -> float:
        """Read default zenith angle option from INI."""
        zenith_angle_deg = _get_ini_float(parser, "telescope", "ZenithAngle")
        if zenith_angle_deg is None:
            raise ValueError("TIPTOP config missing telescope.ZenithAngle for default zenith_angle_deg option.")
        return float(zenith_angle_deg)

    def _get_default_ngs_options_from_ini(
        self,
        parser: ConfigParser,
        photometry: WFSPhotometryConfig,
    ) -> dict[str, np.ndarray] | None:
        """Read default NGS geometry/magnitude options from INI."""
        ngs_r = _get_ini_array(parser, "sources_LO", "Zenith")
        ngs_theta = _get_ini_array(parser, "sources_LO", "Azimuth")
        if ngs_r is None or ngs_theta is None:
            return None
        if ngs_r.shape != ngs_theta.shape:
            raise ValueError("TIPTOP sources_LO Zenith/Azimuth arrays must have identical shape.")

        if not parser.has_section("sensor_LO") or "NumberPhotons" not in parser["sensor_LO"]:
            return None
        photons = _parse_ini_array(parser["sensor_LO"]["NumberPhotons"]).reshape(-1)
        if photons.size == 0:
            raise ValueError("TIPTOP sensor_LO.NumberPhotons cannot be empty.")
        if photons.size == 1 and ngs_r.size > 1:
            photons = np.full((ngs_r.size,), float(photons[0]), dtype=float)
        if photons.size != ngs_r.size:
            raise ValueError(
                "TIPTOP sensor_LO.NumberPhotons length must match NGS count from sources_LO Zenith/Azimuth."
            )

        ngs_mag = photons_per_frame_to_magnitudes(photons, photometry)
        return {
            schema.KEY_OPTION_NGS_R_ARCSEC: np.asarray(ngs_r, dtype=float).reshape(-1),
            schema.KEY_OPTION_NGS_THETA_DEG: np.asarray(ngs_theta, dtype=float).reshape(-1),
            schema.KEY_OPTION_NGS_MAG: np.asarray(ngs_mag, dtype=float).reshape(-1),
        }

    # Runtime INI update helpers

    def _create_runtime_context(self, index: int, options: dict[str, Any], setup: SimulationSetup) -> dict[str, Any]:
        """Create runtime scratch state for one TIPTOP simulation.

        This derives the per-simulation effective INI parser by copying the
        loaded base config and applying setup- and option-dependent runtime
        overrides.

        Args:
            index: Zero-based simulation index.
            options: Copied per-simulation options mapping.
            setup: Bound typed setup object for this simulation instance.

        Returns:
            Runtime scratch mapping containing the effective TIPTOP parser.

        Raises:
            TypeError: If ``setup`` is not ``TiptopSetup``.
            ValueError: If runtime INI updates require missing or invalid
                inputs.
        """
        del index
        if not isinstance(setup, TiptopSetup):
            raise TypeError("TiptopSimulation setup must be TiptopSetup.")
        # Per-simulation INI edits must not mutate the shared base parser.
        parser = deepcopy(self.base_config.parser)
        self._update_atmosphere_in_ini(parser, options, setup)
        self._update_science_in_ini(parser, options, setup)
        self._update_lgs_in_ini(parser, setup)
        self._update_ngs_in_ini(parser, options, setup)
        return {self.KEY_RUNTIME_EFFECTIVE_PARSER: parser}

    def _update_atmosphere_in_ini(
        self,
        parser: ConfigParser,
        options: Mapping[str, Any],
        setup: TiptopSetup,
    ) -> None:
        """Apply atmosphere/profile/r0 runtime updates."""
        if schema.KEY_OPTION_ZENITH_ANGLE_DEG in options and parser.has_section("telescope"):
            parser["telescope"]["ZenithAngle"] = f"{float(options[schema.KEY_OPTION_ZENITH_ANGLE_DEG]):.6g}"

        if not parser.has_section("atmosphere"):
            return

        atmosphere_section = parser["atmosphere"]

        if schema.KEY_OPTION_ATM_PROFILE_ID in options:
            atm_profile_id = int(np.asarray(options.get(schema.KEY_OPTION_ATM_PROFILE_ID, 0)).item())
            atm_profile = atm.select_atm_profile(setup.atm_profiles, atm_profile_id)
            for src_key, dst_key in self.ATM_PROFILE_KEYS_TO_INI_FIELDS.items():
                if src_key not in atm_profile:
                    continue
                value = atm_profile[src_key]
                if isinstance(value, np.ndarray):
                    atmosphere_section[dst_key] = _format_ini_array(value)
                else:
                    atmosphere_section[dst_key] = f"{float(value):.6g}"

        if schema.KEY_OPTION_R0_M in options:
            r0_m = float(options[schema.KEY_OPTION_R0_M])
            atm_wavelength_m = self._get_required_atm_wavelength_m(parser, "convert r0_m to Seeing")
            atmosphere_section["Seeing"] = f"{r0_to_seeing_arcsec(r0_m, atm_wavelength_m):.6g}"
            # TIPTOP runtime input is written as Seeing rather than r0_Value.
            if "r0_Value" in atmosphere_section:
                del atmosphere_section["r0_Value"]

    def _update_science_in_ini(
        self,
        parser: ConfigParser,
        options: Mapping[str, Any],
        setup: TiptopSetup,
    ) -> None:
        """Apply science geometry and science-wavelength updates."""
        if parser.has_section("sources_science"):
            parser["sources_science"]["Zenith"] = _format_ini_array(setup.sci_r_arcsec)
            parser["sources_science"]["Azimuth"] = _format_ini_array(setup.sci_theta_deg)
            if schema.KEY_OPTION_WAVELENGTH_UM in options:
                parser["sources_science"]["Wavelength"] = f"[{float(options[schema.KEY_OPTION_WAVELENGTH_UM]) * 1e-6:.6e}]"

    def _update_lgs_in_ini(self, parser: ConfigParser, setup: TiptopSetup) -> None:
        """Apply invariant LGS geometry from setup."""
        del self
        if setup.lgs_r_arcsec.size > 0 and parser.has_section("sources_HO"):
            parser["sources_HO"]["Zenith"] = _format_ini_array(setup.lgs_r_arcsec)
        if setup.lgs_theta_deg.size > 0 and parser.has_section("sources_HO"):
            parser["sources_HO"]["Azimuth"] = _format_ini_array(setup.lgs_theta_deg)

    def _update_ngs_in_ini(
        self,
        parser: ConfigParser,
        options: Mapping[str, Any],
        setup: TiptopSetup,
    ) -> None:
        """Apply active-NGS geometry and photon updates."""
        required_ngs_keys = (
            schema.KEY_OPTION_NGS_R_ARCSEC,
            schema.KEY_OPTION_NGS_THETA_DEG,
            schema.KEY_OPTION_NGS_MAG,
        )
        if not all(key in options for key in required_ngs_keys):
            return
        if schema.KEY_OPTION_NGS_USED not in options:
            raise ValueError(
                "Missing required runtime option 'ngs_used' while applying NGS per-simulation overrides. "
                "Call runner.prepare_options_payload(...) (or api.init_dataset(...)) so core derives runtime fields first."
            )

        ngs_mag = np.asarray(options[schema.KEY_OPTION_NGS_MAG], dtype=float).reshape(-1)
        ngs_used = np.asarray(options[schema.KEY_OPTION_NGS_USED], dtype=bool).reshape(-1)
        if not np.any(ngs_used):
            return
        photometry = self._get_ngs_photometry_config(parser, setup.ngs_mag_zeropoint)

        if parser.has_section("sources_LO"):
            parser["sources_LO"]["Zenith"] = _format_ini_array(
                as_float_vector(options[schema.KEY_OPTION_NGS_R_ARCSEC], label=schema.KEY_OPTION_NGS_R_ARCSEC)[ngs_used]
            )
            parser["sources_LO"]["Azimuth"] = _format_ini_array(
                as_float_vector(options[schema.KEY_OPTION_NGS_THETA_DEG], label=schema.KEY_OPTION_NGS_THETA_DEG)[ngs_used]
            )

        photons_per_frame = magnitudes_to_photons_per_frame(
            ngs_mag[ngs_used],
            photometry,
        )
        if parser.has_section("sensor_LO"):
            parser["sensor_LO"]["NumberPhotons"] = _format_ini_array(np.round(photons_per_frame, 0))
            if "NumberLenslets" in parser["sensor_LO"]:
                parser["sensor_LO"]["NumberLenslets"] = _format_ini_array(
                    np.full((int(np.sum(ngs_used)),), photometry.n_channels, dtype=float)
                )

    # Runtime lifecycle

    def run(self, context: SimulationContext) -> None:
        """Execute TIPTOP simulation and cache raw output object in runtime.

        Args:
            context: Simulation context produced by ``create()``.
        """
        parser = context.runtime.get(self.KEY_RUNTIME_EFFECTIVE_PARSER)
        if not isinstance(parser, ConfigParser):
            raise TypeError("context.runtime['effective_parser'] must be a ConfigParser. Did create() run?")

        setup = context.setup
        if not isinstance(setup, TiptopSetup):
            raise TypeError("context.setup must be TiptopSetup.")
        ee_apertures_mas = np.asarray(setup.ee_apertures_mas, dtype=float).reshape(-1)
        if ee_apertures_mas.size == 0:
            raise ValueError("ee_apertures_mas must contain at least one value.")
        ee_radius_mas = float(ee_apertures_mas[0]) * 0.5
        ini_text = _serialize_parser(parser)

        with tempfile.TemporaryDirectory(prefix="ao_predict_tiptop_") as tmpdir:
            ini_path = Path(tmpdir) / "sim.ini"
            ini_path.write_text(ini_text, encoding="utf-8")

            path2param = str(ini_path.parent)
            parameters_file = ini_path.stem
            output_dir = str(ini_path.parent)
            output_file = ini_path.stem

            def _cpu_array_safe(v: Any) -> Any:
                return v.get() if hasattr(v, "get") else v

            # Compatibility patch for upstream astro-tiptop CPU-only runs where
            # cpuArray may receive numpy values without a `.get()` method.
            try:  # pragma: no cover - depends on external package internals
                import tiptop.tiptopUtils as tiptop_utils  # pylint: disable=import-outside-toplevel

                tiptop_utils.cpuArray = _cpu_array_safe
            except Exception:
                pass
            try:  # pragma: no cover - depends on external package internals
                import tiptop.baseSimulation as tiptop_base_module  # pylint: disable=import-outside-toplevel

                tiptop_base_module.cpuArray = _cpu_array_safe
            except Exception:
                pass

            from tiptop.tiptop import baseSimulation  # pylint: disable=import-outside-toplevel

            simulation = baseSimulation(
                path2param,
                parameters_file,
                output_dir,
                output_file,
                doConvolve=True,
                getHoErrorBreakDown=True,
                ensquaredEnergy=True,
                eeRadiusInMas=ee_radius_mas,
                doPlot=False,
                verbose=False,
            )
            simulation.doOverallSimulation()
            context.runtime[self.KEY_RUNTIME_SIMULATION] = simulation

    def _extract_psfs(self, context: SimulationContext) -> np.ndarray | None:
        """Extract the PSF cube from the completed TIPTOP runtime object.

        Args:
            context: Completed simulation context.

        Returns:
            PSF cube with shape ``[M, Ny, Nx]`` or ``None`` when TIPTOP did
            not expose per-science sampling results.

        Raises:
            ValueError: If the TIPTOP runtime object is missing.
        """
        simulation = context.runtime.get(self.KEY_RUNTIME_SIMULATION)
        if simulation is None:
            raise ValueError("Missing TIPTOP simulation in context.runtime. Did you call run()?")

        psfs: np.ndarray | None = None
        if hasattr(simulation, "results"):
            results = getattr(simulation, "results")
            psfs = np.asarray(
                [
                    np.asarray(getattr(item, "sampling", item), dtype=np.float32)
                    for item in results
                ],
                dtype=np.float32,
            )

        return psfs

    def _extract_psf_parameters(self, context: SimulationContext) -> PsfParameters:
        """Extract persisted PSF metadata from completed TIPTOP runtime state.

        Args:
            context: Completed simulation context.

        Returns:
            Pixel scale, telescope diameter, and telescope pupil associated
            with the extracted PSFs.

        Raises:
            TypeError: If the effective parser is missing or invalid.
            ValueError: If required TIPTOP runtime outputs are unavailable.
        """
        simulation = context.runtime.get(self.KEY_RUNTIME_SIMULATION)
        if simulation is None:
            raise ValueError("Missing TIPTOP simulation in context.runtime. Did you call run()?")
        parser = context.runtime.get(self.KEY_RUNTIME_EFFECTIVE_PARSER)
        if not isinstance(parser, ConfigParser):
            raise TypeError("context.runtime['effective_parser'] must be a ConfigParser. Did create() run?")

        if hasattr(simulation, "psInMas"):
            pixel_scale_mas = float(getattr(simulation, "psInMas"))
        elif parser.has_section("sensor_science") and "PixelScale" in parser["sensor_science"]:
            pixel_scale_mas = float(parser["sensor_science"]["PixelScale"])
        else:
            raise ValueError("Unable to resolve pixel scale (mas) from TIPTOP outputs or INI.")

        if hasattr(simulation, "tel_radius"):
            tel_diameter_m = float(getattr(simulation, "tel_radius")) * 2.0
        elif parser.has_section("telescope") and "TelescopeDiameter" in parser["telescope"]:
            tel_diameter_m = float(parser["telescope"]["TelescopeDiameter"])
        else:
            raise ValueError("Unable to resolve telescope diameter (m) from TIPTOP outputs or INI.")

        try:
            tel_pupil = np.asarray(simulation.fao.ao.tel.pupil, dtype=np.float32)
        except Exception as exc:
            raise ValueError("Unable to resolve telescope pupil from TIPTOP output object.") from exc

        return PsfParameters(
            pixel_scale_mas=pixel_scale_mas,
            tel_diameter_m=tel_diameter_m,
            tel_pupil=tel_pupil,
        )
