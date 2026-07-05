"""TIPTOP/MASTSEL configuration-backed simulation base."""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
import io
from typing import Any, Mapping

import numpy as np

from . import atm
from . import schema
from .config_backed import ConfigBackedSimulation
from .helpers import seeing_arcsec_to_r0_m
from .photometry import WFSPhotometryConfig, photons_per_frame_to_magnitudes


@dataclass
class TiptopBaseConfig:
    """Parsed TIPTOP/MASTSEL-style base configuration.

    Attributes:
        ini_text: Exact persisted INI source text loaded from
            ``/simulation/base_config``.
        parser: Case-preserving parser for the same INI text. Runtime
            simulations must copy this parser before applying per-run edits.
    """

    ini_text: str
    parser: ConfigParser


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


class TiptopConfigBackedSimulation(ConfigBackedSimulation):
    """Base for simulations that use the shared TIPTOP/MASTSEL INI dialect.

    The class interprets the persisted base configuration as a case-preserving
    INI file and derives AO Predict setup/options defaults from fields that are
    common to the TIPTOP/MASTSEL configuration family. Runtime execution,
    runtime parser mutation, and output extraction remain subclass
    responsibilities.

    Attributes:
        base_config: Parsed ``TiptopBaseConfig`` bound from the persisted
            simulation payload.
    """

    ATM_PROFILE_KEYS_TO_INI_FIELDS = {
        atm.KEY_SETUP_ATM_PROFILE_L0_M: "L0",
        atm.KEY_SETUP_ATM_PROFILE_CN2_HEIGHTS_M: "Cn2Heights",
        atm.KEY_SETUP_ATM_PROFILE_CN2_WEIGHTS: "Cn2Weights",
        atm.KEY_SETUP_ATM_PROFILE_WIND_SPEED_MPS: "WindSpeed",
        atm.KEY_SETUP_ATM_PROFILE_WIND_DIRECTION_DEG: "WindDirection",
    }

    def __init__(self) -> None:
        """Initialize unbound TIPTOP-config-backed simulation state."""
        super().__init__()
        self._base_config: TiptopBaseConfig | None = None

    @property
    def base_config(self) -> TiptopBaseConfig:
        """Return the loaded TIPTOP/MASTSEL-style base configuration.

        Returns:
            Parsed base configuration text and parser.

        Raises:
            TypeError: If ``load_simulation_payload()`` has not been called
                successfully.
        """
        base_config = self._base_config
        if not isinstance(base_config, TiptopBaseConfig):
            raise TypeError(f"{type(self).__name__} base config is not configured. Call load_simulation_payload(...) first.")
        return base_config

    # Simulation payload lifecycle

    def _prepare_base_config_binding(self, base_config_text: str) -> TiptopBaseConfig:
        """Parse serialized INI text into bindable config state."""
        return TiptopBaseConfig(
            ini_text=base_config_text,
            parser=_parse_ini_text(base_config_text),
        )

    def _bind_base_config(self, base_config: Any) -> None:
        """Bind parsed INI config state for later lifecycle stages."""
        if not isinstance(base_config, TiptopBaseConfig):
            raise TypeError(f"{type(self).__name__} base config must be a TiptopBaseConfig.")
        self._base_config = base_config

    # Setup payload lifecycle

    def prepare_setup_payload(
        self,
        base_setup_payload: Mapping[str, Any],
        setup_cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Resolve a persisted ``/setup`` payload from user values and INI defaults.

        Args:
            base_setup_payload: Setup fields prepared by ``BaseSimulation``.
            setup_cfg: User setup mapping whose explicit values override
                defaults derived from the bound base INI.

        Returns:
            A setup payload with atmospheric, LGS, and science geometry
            defaults filled from the base INI where the user did not provide
            explicit values.

        Raises:
            TypeError: If the base configuration has not been loaded or setup
                field types are invalid.
            ValueError: If required INI fields needed for default derivation
                are missing or invalid.
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

        return self._build_setup_payload(
            base_setup_payload,
            setup_cfg,
            default_atm_wavelength_um=atm_wavelength_um,
            default_atm_profile=default_atm_profile,
            default_lgs_r_arcsec=lgs_r_arcsec,
            default_lgs_theta_deg=lgs_theta_deg,
            default_sci_r_arcsec=sci_r_arcsec,
            default_sci_theta_deg=sci_theta_deg,
        )

    # Options payload lifecycle

    def prepare_options_payload(
        self,
        num_sims: int,
        setup_payload: Mapping[str, Any],
        base_options_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Complete persisted per-simulation options using INI defaults.

        Args:
            num_sims: Number of option rows to prepare.
            setup_payload: Persisted setup payload used for NGS photometry
                default conversion.
            base_options_payload: User-provided option arrays and scalars.

        Returns:
            A normalized options payload with missing wavelength, zenith angle,
            atmospheric profile, ``r0_m``, and NGS defaults filled from the
            bound base INI when possible.

        Raises:
            TypeError: If the base configuration has not been loaded or option
                field types are invalid.
            ValueError: If ``num_sims`` is not positive or required INI fields
                needed for default derivation are missing or invalid.
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

    @classmethod
    def _write_atmosphere_profile_fields(
        cls,
        parser: ConfigParser,
        profile: Mapping[str, Any],
    ) -> None:
        """Write shared atmosphere profile fields into a TIPTOP/MASTSEL INI.

        This helper owns only the common INI dialect for profile fields such as
        ``L0`` and ``Cn2Weights``. Backend-specific strength policy, including
        whether runtime ``r0_m`` is represented as ``Seeing`` or ``r0_Value``,
        remains the subclass responsibility.
        """
        if not parser.has_section("atmosphere"):
            return
        atmosphere_section = parser["atmosphere"]
        for src_key, dst_key in cls.ATM_PROFILE_KEYS_TO_INI_FIELDS.items():
            if src_key not in profile:
                continue
            value = profile[src_key]
            if isinstance(value, np.ndarray):
                atmosphere_section[dst_key] = _format_ini_array(value)
            else:
                atmosphere_section[dst_key] = f"{float(value):.6g}"

    @staticmethod
    def _write_source_geometry_fields(
        parser: ConfigParser,
        section: str,
        r_arcsec: np.ndarray,
        theta_deg: np.ndarray,
    ) -> None:
        """Write polar source geometry fields into a TIPTOP/MASTSEL INI section."""
        if not parser.has_section(section):
            return
        parser[section]["Zenith"] = _format_ini_array(r_arcsec)
        parser[section]["Azimuth"] = _format_ini_array(theta_deg)

    @classmethod
    def _write_science_source_fields(
        cls,
        parser: ConfigParser,
        r_arcsec: np.ndarray,
        theta_deg: np.ndarray,
        wavelength_um: float | None,
    ) -> None:
        """Write science source geometry and optional wavelength into the INI."""
        cls._write_source_geometry_fields(parser, "sources_science", r_arcsec, theta_deg)
        if wavelength_um is not None and parser.has_section("sources_science"):
            parser["sources_science"]["Wavelength"] = f"[{float(wavelength_um) * 1e-6:.6e}]"

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
        """Read default r0 option from INI or derive it from Seeing."""
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

    # Default helpers

    def _get_default_atm_profile_from_ini(self, parser: ConfigParser) -> dict[str, Any]:
        """Construct default atmospheric profile from base INI."""
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
