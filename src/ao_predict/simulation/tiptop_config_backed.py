"""TIPTOP/MASTSEL configuration-backed simulation base."""

from __future__ import annotations

import io
from collections.abc import Mapping
from configparser import ConfigParser
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from ao_core.models.config import (
    AtmosphereConfig,
    DetectorConfig,
    LoopConfig,
    SourceConfig,
    TelescopeConfig,
    parse_ini,
    read_ini_scalar,
)
from ao_core.photometry import WFSPhotometryConfig, photons_per_frame_to_magnitudes
from astropy import units as u

from .._units import quantity_value
from . import atm, schema
from .config_backed import ConfigBackedSimulation
from .helpers import seeing_to_r0


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


def _format_ini_array(values: np.ndarray) -> str:
    """Format a numeric vector into TIPTOP bracket-array text."""
    arr = np.asarray(values).reshape(-1)
    return "[" + ",".join(f"{float(v):.6g}" for v in arr) + "]"


class TiptopConfigBackedSimulation(ConfigBackedSimulation):
    """Base for simulations that use the shared TIPTOP/MASTSEL INI dialect.

    AO Core decodes the shared fields of the case-preserving base INI.
    This class uses those fields to resolve AO Predict setup and option
    defaults. Runtime execution,
    runtime parser mutation, and output extraction remain subclass
    responsibilities.

    Attributes:
        base_config: Parsed ``TiptopBaseConfig`` bound from the persisted
            simulation payload.
    """

    ATM_PROFILE_KEYS_TO_INI_FIELDS: ClassVar[dict[str, str]] = {
        atm.KEY_SETUP_ATM_PROFILE_L0: "L0",
        atm.KEY_SETUP_ATM_PROFILE_CN2_HEIGHTS: "Cn2Heights",
        atm.KEY_SETUP_ATM_PROFILE_CN2_WEIGHTS: "Cn2Weights",
        atm.KEY_SETUP_ATM_PROFILE_WIND_SPEED: "WindSpeed",
        atm.KEY_SETUP_ATM_PROFILE_WIND_DIRECTION: "WindDirection",
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
            parser=parse_ini(base_config_text),
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

        atm_wavelength = self._get_required_atm_wavelength_m(parser, "prepare setup payload") * u.m
        default_atm_profile = self._get_default_atm_profile_from_ini(parser)

        high_order_sources = SourceConfig.from_ini(parser, "sources_HO")
        science_sources = SourceConfig.from_ini(parser, "sources_science")
        lgs_zenith_offsets = (
            None if high_order_sources is None else high_order_sources.zenith
        )
        lgs_azimuths = None if high_order_sources is None else high_order_sources.azimuth
        science_zenith_offsets = (
            None if science_sources is None else science_sources.zenith
        )
        science_azimuths = None if science_sources is None else science_sources.azimuth

        return self._build_setup_payload(
            base_setup_payload,
            setup_cfg,
            default_atm_wavelength=atm_wavelength,
            default_atm_profile=default_atm_profile,
            default_lgs_r=(
                np.empty(0) * u.arcsec
                if lgs_zenith_offsets is None
                else lgs_zenith_offsets.copy()
            ),
            default_lgs_theta=(
                np.empty(0) * u.deg if lgs_azimuths is None else lgs_azimuths.copy()
            ),
            default_sci_r=(
                None if science_zenith_offsets is None else science_zenith_offsets.copy()
            ),
            default_sci_theta=None if science_azimuths is None else science_azimuths.copy(),
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
            atmospheric profile, ``r0``, and NGS defaults filled from the
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
        options_payload = {str(key): value.copy() if hasattr(value, "copy") else value for key, value in base_options_payload.items()}

        default_options: dict[str, Any] = {}
        if schema.KEY_OPTION_WAVELENGTH not in options_payload:
            default_options[schema.KEY_OPTION_WAVELENGTH] = self._get_default_wavelength_m_from_ini(parser) * u.m
        if schema.KEY_OPTION_ZENITH_ANGLE not in options_payload:
            default_options[schema.KEY_OPTION_ZENITH_ANGLE] = self._get_default_zenith_angle_deg_from_ini(parser) * u.deg
        if schema.KEY_OPTION_ATM_PROFILE_ID not in options_payload:
            default_options[schema.KEY_OPTION_ATM_PROFILE_ID] = np.int32(0)
        if schema.KEY_OPTION_R0 not in options_payload:
            default_options[schema.KEY_OPTION_R0] = self._default_r0(AtmosphereConfig.from_ini(parser))

        if not any(key in options_payload for key in schema.OPTION_KEYS_NGS):
            default_ngs_options = self._get_default_ngs_options_from_ini(
                parser,
                setup_payload[self.KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT],
            )
            if default_ngs_options is not None:
                for key, values in default_ngs_options.items():
                    values = np.asarray(values, dtype=float).reshape(1, -1)
                    options_payload[key] = np.broadcast_to(values, (num_sims, values.shape[1])).copy() * schema.OPTION_FIELD_UNITS[key]

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
        whether runtime ``r0`` is represented as ``Seeing`` or ``r0_Value``,
        remains the subclass responsibility.
        """
        if not parser.has_section("atmosphere"):
            return
        atmosphere_section = parser["atmosphere"]
        for src_key, dst_key in cls.ATM_PROFILE_KEYS_TO_INI_FIELDS.items():
            if src_key not in profile:
                continue
            value = quantity_value(
                profile[src_key],
                atm.ATM_PROFILE_FIELD_UNITS[src_key],
                label=f"atmosphere profile {src_key}",
            )
            if value.ndim > 0:
                atmosphere_section[dst_key] = _format_ini_array(value)
            else:
                atmosphere_section[dst_key] = f"{float(value.item()):.6g}"

    @staticmethod
    def _write_source_geometry_fields(
        parser: ConfigParser,
        section: str,
        r: u.Quantity,
        theta: u.Quantity,
    ) -> None:
        """Write polar source geometry fields into a TIPTOP/MASTSEL INI section."""
        if not parser.has_section(section):
            return
        parser[section]["Zenith"] = _format_ini_array(quantity_value(r, u.arcsec, label=f"{section}.r"))
        parser[section]["Azimuth"] = _format_ini_array(quantity_value(theta, u.deg, label=f"{section}.theta"))

    @classmethod
    def _write_science_source_fields(
        cls,
        parser: ConfigParser,
        r: u.Quantity,
        theta: u.Quantity,
        wavelength: u.Quantity | None,
    ) -> None:
        """Write science source geometry and optional wavelength into the INI."""
        cls._write_source_geometry_fields(parser, "sources_science", r, theta)
        if wavelength is not None and parser.has_section("sources_science"):
            parser["sources_science"]["Wavelength"] = f"[{wavelength.to_value(u.m):.6e}]"

    @staticmethod
    def _get_required_atm_wavelength_m(parser: ConfigParser, purpose: str) -> float:
        """Read required atmosphere wavelength from INI in meters."""
        if not parser.has_section("atmosphere") or "Wavelength" not in parser["atmosphere"]:
            raise ValueError(f"TIPTOP atmosphere.Wavelength must be present to {purpose}.")
        try:
            atm_wavelength_m = read_ini_scalar(
                parser["atmosphere"], "Wavelength", label="atmosphere.Wavelength"
            )
        except ValueError as exc:
            raise ValueError(f"TIPTOP atmosphere.Wavelength must be numeric to {purpose}.") from exc
        if atm_wavelength_m <= 0.0:
            raise ValueError(f"TIPTOP atmosphere.Wavelength must be > 0 to {purpose}.")
        return atm_wavelength_m

    @staticmethod
    def _get_ngs_photometry_config(
        parser: ConfigParser,
        low_order_detector: DetectorConfig | None,
        ngs_magnitude_zeropoint: u.Quantity,
    ) -> WFSPhotometryConfig:
        """Build the NGS photometry inputs from shared telescope and LO fields."""
        telescope_config = TelescopeConfig.from_ini(parser)
        low_order_loop = LoopConfig.from_ini(parser, stage="LO")
        if telescope_config is None or telescope_config.diameter is None:
            raise ValueError("TIPTOP config missing telescope.TelescopeDiameter required for ngs_magnitude conversion.")
        if low_order_detector is None or low_order_detector.lenslet_counts is None:
            raise ValueError("TIPTOP config missing sensor_LO.NumberLenslets required for ngs_magnitude conversion.")
        if low_order_loop is None or low_order_loop.frame_rate is None:
            raise ValueError("TIPTOP config missing RTC.SensorFrameRate_LO required for ngs_magnitude conversion.")
        return WFSPhotometryConfig(
            telescope_diameter=telescope_config.diameter,
            n_channels=float(low_order_detector.lenslet_counts[0].value),
            frame_rate=low_order_loop.frame_rate,
            zeropoint=ngs_magnitude_zeropoint.to(u.photon / (u.m**2 * u.s)),
        )

    @staticmethod
    def _default_r0(atmosphere_config: AtmosphereConfig | None) -> u.Quantity:
        """Resolve the default r0, preferring an explicit Fried parameter."""
        if atmosphere_config is not None and atmosphere_config.fried_parameter is not None:
            return atmosphere_config.fried_parameter
        if (
            atmosphere_config is None
            or atmosphere_config.seeing is None
            or atmosphere_config.reference_wavelength is None
        ):
            raise ValueError(
                "TIPTOP config must provide atmosphere.r0_Value, or both atmosphere.Seeing and atmosphere.Wavelength "
                "for default r0 option."
            )
        return seeing_to_r0(
            atmosphere_config.seeing, atmosphere_config.reference_wavelength
        ).to(u.m)

    # Default helpers

    def _get_default_atm_profile_from_ini(self, parser: ConfigParser) -> dict[str, Any]:
        """Construct default atmospheric profile from base INI."""
        atmosphere_config = AtmosphereConfig.from_ini(parser)
        r0 = self._default_r0(atmosphere_config)
        assert atmosphere_config is not None
        profile = {
            atm.KEY_SETUP_ATM_PROFILE_NAME: "ini_default",
            atm.KEY_SETUP_ATM_PROFILE_L0: atmosphere_config.outer_scale,
            atm.KEY_SETUP_ATM_PROFILE_R0: r0,
            atm.KEY_SETUP_ATM_PROFILE_CN2_HEIGHTS: atmosphere_config.layer_heights,
            atm.KEY_SETUP_ATM_PROFILE_CN2_WEIGHTS: atmosphere_config.layer_weights,
            atm.KEY_SETUP_ATM_PROFILE_WIND_SPEED: atmosphere_config.wind_speeds,
            atm.KEY_SETUP_ATM_PROFILE_WIND_DIRECTION: atmosphere_config.wind_directions,
        }
        return {key: value for key, value in profile.items() if value is not None}

    @staticmethod
    def _get_default_wavelength_m_from_ini(parser: ConfigParser) -> float:
        """Read default science wavelength option from INI in meters."""
        science_sources = SourceConfig.from_ini(parser, "sources_science")
        if science_sources is None or science_sources.wavelengths is None:
            raise ValueError("TIPTOP config missing sources_science.Wavelength for default wavelength option.")
        return float(science_sources.wavelengths[0].to_value(u.m))

    def _get_default_zenith_angle_deg_from_ini(self, parser: ConfigParser) -> float:
        """Read default zenith angle option from INI."""
        telescope_config = TelescopeConfig.from_ini(parser)
        if telescope_config is None or telescope_config.zenith_angle is None:
            raise ValueError("TIPTOP config missing telescope.ZenithAngle for default zenith_angle option.")
        return float(telescope_config.zenith_angle.to_value(u.deg))

    def _get_default_ngs_options_from_ini(
        self,
        parser: ConfigParser,
        ngs_magnitude_zeropoint: u.Quantity,
    ) -> dict[str, np.ndarray] | None:
        """Resolve the default NGS geometry and convert photon counts to magnitudes."""
        low_order_detector = DetectorConfig.low_order_from_ini(parser)
        photometry = self._get_ngs_photometry_config(
            parser, low_order_detector, ngs_magnitude_zeropoint
        )
        low_order_sources = SourceConfig.from_ini(parser, "sources_LO")
        if (
            low_order_sources is None
            or low_order_sources.zenith is None
            or low_order_sources.azimuth is None
        ):
            return None
        if low_order_detector is None or low_order_detector.photons_per_frame is None:
            return None
        photons = low_order_detector.photons_per_frame
        if photons.size == 1 and low_order_sources.zenith.size > 1:
            photons = np.full(low_order_sources.zenith.size, photons[0].value) * u.photon
        if photons.size != low_order_sources.zenith.size:
            raise ValueError(
                "TIPTOP sensor_LO.NumberPhotons length must match NGS count from sources_LO Zenith/Azimuth."
            )

        ngs_magnitude = photons_per_frame_to_magnitudes(photons, photometry).to_value(u.mag)
        return {
            schema.KEY_OPTION_NGS_R: low_order_sources.zenith.to_value(u.arcsec),
            schema.KEY_OPTION_NGS_THETA: low_order_sources.azimuth.to_value(u.deg),
            schema.KEY_OPTION_NGS_MAGNITUDE: np.asarray(ngs_magnitude, dtype=float).reshape(-1),
        }
