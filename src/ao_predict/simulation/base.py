"""Shared simulation lifecycle scaffolding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Mapping, TypeVar

import numpy as np
from astropy import units as u

from . import schema
from . import atm
from .coordinates import resolve_science_coordinates
from .helpers import _MISSING, select_mapping_value
from .interfaces import Simulation, SimulationContext, SimulationResult, SimulationSetup, SimulationState
from .stats import clip_and_sum_normalize_psfs
from .._units import quantity_value, require_quantity, unit_string


@dataclass(frozen=True)
class BaseSimulationSetup(SimulationSetup):
    """Typed setup payload shared by BaseSimulation subclasses.

    This extends the core ``SimulationSetup`` contract with the common
    NGS magnitude zero-point used by simulations that derive WFS photon
    inputs from magnitudes.
    """

    ngs_magnitude_zeropoint: u.Quantity


@dataclass(frozen=True)
class PsfParameters:
    """PSF metadata extracted from one completed simulation.

    These values are an internal extraction helper for ``BaseSimulation``
    subclasses. ``finalize()`` flattens them into ``SimulationResult.meta``
    for persistence and core post-processing.
    """

    pixel_scale: u.Quantity
    tel_diameter: u.Quantity
    tel_pupil: u.Quantity


TBaseSetup = TypeVar("TBaseSetup", bound=BaseSimulationSetup)


def _normalize_diagnostic_field_specs(fields: Any, *, prefix: str = "") -> dict[str, dict[str, Any]]:
    """Return normalized slash-delimited diagnostic field specs.

    The simulation payload accepts flat slash-delimited specs or nested specs
    that mirror the eventual ``/diagnostics`` group structure. This helper
    normalizes both forms so payload validation can compare the complete
    persisted diagnostic contract, not just the declared field names.
    """
    if not isinstance(fields, Mapping):
        raise TypeError(f"simulation['{schema.KEY_SIMULATION_DIAGNOSTIC_FIELDS}'] must be a mapping.")
    specs: dict[str, dict[str, Any]] = {}
    for key, value in fields.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if not isinstance(value, Mapping):
            raise TypeError(f"diagnostic field spec for '{name}' must be a mapping.")
        if "dtype" in value or "shape" in value:
            field_name = name.strip("/")
            if not field_name:
                raise ValueError("diagnostic field names must be non-empty.")
            dtype = str(value.get("dtype", "float32")).strip()
            if not dtype:
                raise ValueError(f"diagnostic field spec for '{field_name}' must declare a dtype.")
            shape_values: list[str | int] = []
            for shape_value in np.asarray(value.get("shape", ()), dtype=object).reshape(-1).tolist():
                if isinstance(shape_value, str) and shape_value not in {"num_sci", "num_ngs"}:
                    try:
                        shape_value = int(shape_value)
                    except ValueError as exc:
                        raise ValueError(
                            f"diagnostic field spec for '{field_name}' has unsupported shape token {shape_value!r}."
                        ) from exc
                elif not isinstance(shape_value, str):
                    shape_value = int(shape_value)
                shape_values.append(shape_value)
            spec = {"dtype": dtype, "shape": shape_values}
            if "unit" in value:
                spec["unit"] = unit_string(value["unit"])
            specs[field_name] = spec
        else:
            specs.update(_normalize_diagnostic_field_specs(value, prefix=name))
    return specs


class BaseSimulation(Simulation, ABC):
    """Partial `Simulation` implementation with shared lifecycle scaffolding.

    ``BaseSimulation`` centralizes the common work needed by concrete
    simulators that follow ao-predict's persistence contract:
    - build and validate shared `/setup` fields
    - complete `/options` using shared per-simulation rules
    - create ``SimulationContext`` objects with bound setup state
    - finalize successful runs from extracted PSFs and PSF metadata

    Subclasses remain responsible for simulator-specific configuration,
    runtime execution, and extraction of backend outputs.
    """

    KEY_SETUP_LGS_R = schema.KEY_SETUP_LGS_R
    KEY_SETUP_LGS_THETA = schema.KEY_SETUP_LGS_THETA
    KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT = schema.KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT

    SETUP_KEYS_BASE = (
        schema.KEY_SETUP_ATM_WAVELENGTH,
        schema.KEY_SETUP_ATM_PROFILES,
        KEY_SETUP_LGS_R,
        KEY_SETUP_LGS_THETA,
        KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT,
        schema.KEY_SETUP_SCI_R,
        schema.KEY_SETUP_SCI_THETA,
    )
    # Construction and properties

    def __init__(self) -> None:
        """Initialize base simulation state."""
        self._setup: BaseSimulationSetup | None = None
        self._diagnostics_level: str = schema.DIAGNOSTICS_LEVEL_NONE

    @property
    def setup(self) -> BaseSimulationSetup:
        """Return bound setup or raise if setup has not been loaded."""
        if self._setup is None:
            raise TypeError(f"{type(self).__name__} setup is not configured. Call load_setup_payload(...) first.")
        return self._setup

    @property
    def diagnostics_level(self) -> str:
        """Return the bound generic diagnostics level for this simulation."""
        return self._diagnostics_level

    @property
    def ngs_mag_standard(self) -> str:
        """Return the default ``R`` standard for persisted NGS magnitudes.

        Subclasses whose ``options/ngs_magnitude`` values use another photometric
        standard must override this property.
        """
        return schema.DEFAULT_NGS_MAG_STANDARD

    @property
    def supported_extra_diagnostics_levels(self) -> tuple[str, ...]:
        """Return non-``none`` diagnostics levels implemented by this simulation.

        The base implementation opts out of diagnostics. Subclasses that
        produce per-run diagnostics should return a subset of the core
        diagnostics vocabulary excluding ``none``.
        """
        return ()

    # Simulation payload lifecycle

    def _build_simulation_payload(
        self,
        base_simulation_payload: Mapping[str, Any],
        simulation_cfg: Mapping[str, Any],
        *,
        exclude_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        """Build persisted ``/simulation`` payload from core fields plus copied config fields.

        This helper preserves the core-owned `/simulation` fields supplied by
        ao-predict and appends only simulation-specific fields derived from the
        normalized simulation config.
        """
        exclude_keys = {
            *schema.SIMULATION_KEYS_CORE,
            schema.KEY_CFG_SIMULATION_BASE_PATH,
            schema.KEY_SIMULATION_DIAGNOSTIC_FIELDS,
            schema.KEY_SIMULATION_META_FIELDS,
            *(str(k) for k in exclude_keys or ()),
        }
        diagnostics_level = self._normalize_diagnostics_level(
            simulation_cfg.get(schema.KEY_SIMULATION_DIAGNOSTICS_LEVEL, schema.DIAGNOSTICS_LEVEL_NONE)
        )
        payload = {str(k): v for k, v in dict(base_simulation_payload).items()}
        payload.update(
            {
                str(k): v
                for k, v in dict(simulation_cfg).items()
                if str(k) not in exclude_keys
            }
        )
        payload[schema.KEY_SIMULATION_DIAGNOSTICS_LEVEL] = diagnostics_level
        if diagnostics_level != schema.DIAGNOSTICS_LEVEL_NONE:
            diagnostic_fields = _normalize_diagnostic_field_specs(
                self._diagnostic_field_specs(diagnostics_level)
            )
            if not diagnostic_fields:
                raise ValueError(f"{type(self).__name__} diagnostics_level={diagnostics_level!r} declared no diagnostics fields.")
            payload[schema.KEY_SIMULATION_DIAGNOSTIC_FIELDS] = diagnostic_fields
        return payload

    def validate_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Validate simulation-specific persisted ``/simulation`` fields.

        The base implementation validates the generic diagnostics-level
        contract. Core identity/version checks are handled in core validation
        code before this hook is called.
        """
        self._diagnostics_level_from_payload(simulation_payload)

    def _load_base_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Bind generic simulation-level state from persisted ``/simulation``."""
        self._diagnostics_level = self._diagnostics_level_from_payload(simulation_payload)

    def _normalize_diagnostics_level(self, value: Any) -> str:
        """Normalize and validate a requested diagnostics level."""
        level = str(value).strip().lower()
        if level not in schema.DIAGNOSTICS_LEVELS:
            allowed = ", ".join(schema.DIAGNOSTICS_LEVELS)
            raise ValueError(f"diagnostics_level must be one of: {allowed}.")
        if level != schema.DIAGNOSTICS_LEVEL_NONE and level not in self.supported_extra_diagnostics_levels:
            supported = ", ".join(self.supported_extra_diagnostics_levels) or "none"
            raise ValueError(
                f"{type(self).__name__} does not support diagnostics_level={level!r}; "
                f"supported extra diagnostics levels: {supported}."
            )
        return level

    def _diagnostics_level_from_payload(self, simulation_payload: Mapping[str, Any]) -> str:
        """Read and validate persisted ``diagnostics_level`` without binding."""
        level = self._normalize_diagnostics_level(
            simulation_payload.get(schema.KEY_SIMULATION_DIAGNOSTICS_LEVEL, schema.DIAGNOSTICS_LEVEL_NONE)
        )
        if level == schema.DIAGNOSTICS_LEVEL_NONE:
            return level
        if schema.KEY_SIMULATION_DIAGNOSTIC_FIELDS not in simulation_payload:
            raise ValueError(
                f"{type(self).__name__} diagnostics_level={level!r} requires "
                f"simulation['{schema.KEY_SIMULATION_DIAGNOSTIC_FIELDS}']."
            )
        expected_fields = dict(self._diagnostic_field_specs(level))
        if not expected_fields:
            raise ValueError(f"{type(self).__name__} diagnostics_level={level!r} declared no diagnostics fields.")
        payload_fields = _normalize_diagnostic_field_specs(simulation_payload[schema.KEY_SIMULATION_DIAGNOSTIC_FIELDS])
        expected_fields = _normalize_diagnostic_field_specs(expected_fields)
        if payload_fields != expected_fields:
            missing = sorted(set(expected_fields) - set(payload_fields))
            unexpected = sorted(set(payload_fields) - set(expected_fields))
            changed = sorted(
                name
                for name in set(expected_fields) & set(payload_fields)
                if payload_fields[name] != expected_fields[name]
            )
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            if changed:
                details.append(f"changed specs: {', '.join(changed)}")
            raise ValueError(
                f"{type(self).__name__} diagnostics fields do not match diagnostics_level={level!r} "
                + "; ".join(details)
            )
        return level

    def _diagnostic_field_specs(self, diagnostics_level: str) -> Mapping[str, Mapping[str, Any]]:
        """Return declared ``/diagnostics`` field specs for one level.

        Subclasses that support non-``none`` diagnostics levels must override
        this hook and return specs keyed by slash-delimited diagnostic paths.
        Each spec declares a storage ``dtype`` and a per-simulation shape
        excluding the leading simulation dimension ``N``. The base
        implementation declares no diagnostics fields.

        Args:
            diagnostics_level: Normalized non-``none`` diagnostics level.

        Returns:
            Mapping from diagnostic path to dtype/shape spec.
        """
        del diagnostics_level
        return {}

    # Setup payload lifecycle

    def _build_atm_profiles(
        self,
        base_setup_payload: Mapping[str, Any],
        setup_cfg: Mapping[str, Any],
        atm_wavelength: u.Quantity | None,
        *,
        _default_atm_profile: Mapping[str, Any] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Resolve setup atmospheric profiles with optional simulation defaults.

        Values provided in setup payload/config take precedence. Default
        profiles are used only when setup/config provides no profiles.
        """
        raw_profiles = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            schema.KEY_SETUP_ATM_PROFILES,
            default={},
        )
        profiles = atm.parse_atm_profiles(raw_profiles)

        if _default_atm_profile and not profiles:
            parsed_defaults = atm.parse_atm_profiles({0: _default_atm_profile})
            for profile_id, profile in parsed_defaults.items():
                profiles[int(profile_id)] = dict(profile)

        profiles = atm.normalize_atm_profiles_with_seeing_alias(profiles, atm_wavelength)
        atm.validate_standard_atm_profiles(profiles)
        return profiles

    @classmethod
    def _validate_base_setup(cls, setup: BaseSimulationSetup) -> None:
        """Validate shared semantic constraints on a typed base setup object."""
        lgs_r = quantity_value(setup.lgs_r, u.arcsec, label=cls.KEY_SETUP_LGS_R, dtype=float).reshape(-1)
        lgs_theta = quantity_value(setup.lgs_theta, u.deg, label=cls.KEY_SETUP_LGS_THETA, dtype=float).reshape(-1)
        cls._validate_base_setup_values(
            ngs_magnitude_zeropoint=setup.ngs_magnitude_zeropoint,
            lgs_r=lgs_r,
            lgs_theta=lgs_theta,
            atm_profiles=setup.atm_profiles,
        )

    @classmethod
    def _validate_base_setup_values(
        cls,
        ngs_magnitude_zeropoint: u.Quantity,
        lgs_r: np.ndarray,
        lgs_theta: np.ndarray,
        atm_profiles: Mapping[int, Mapping[str, Any]],
    ) -> None:
        """Validate normalized shared setup values before persistence or binding."""
        ngs_magnitude_zeropoint = float(
            quantity_value(
                ngs_magnitude_zeropoint,
                u.photon / u.s,
                label=cls.KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT,
                dtype=float,
            ).item()
        )
        if not np.isfinite(ngs_magnitude_zeropoint) or ngs_magnitude_zeropoint <= 0.0:
            raise ValueError(f"setup['{cls.KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT}'] must be a positive finite scalar.")

        if lgs_r.shape != lgs_theta.shape:
            raise ValueError(
                f"setup['{cls.KEY_SETUP_LGS_R}'] and setup['{cls.KEY_SETUP_LGS_THETA}'] must have identical shape."
            )
        if lgs_r.size > 0 and (not np.all(np.isfinite(lgs_r)) or not np.all(np.isfinite(lgs_theta))):
            raise ValueError("setup LGS coordinates must be finite.")

        atm.validate_standard_atm_profiles(atm_profiles)

    def _build_setup_payload(
        self,
        base_setup_payload: Mapping[str, Any],
        setup_cfg: Mapping[str, Any],
        *,
        default_atm_wavelength: Any = _MISSING,
        default_atm_profile: Mapping[str, Any] | None = None,
        default_lgs_r: Any = _MISSING,
        default_lgs_theta: Any = _MISSING,
        default_sci_r: Any = _MISSING,
        default_sci_theta: Any = _MISSING,
        default_ngs_mag_zeropoint: Any = None,
    ) -> dict[str, Any]:
        """Build, validate, and serialize setup payload using shared base fields.

        This build path is persistence-oriented and intentionally does not
        require simulation-specific setup subclasses.
        """
        ee_apertures = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            schema.KEY_SETUP_EE_APERTURES,
        )

        atm_wavelength = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            schema.KEY_SETUP_ATM_WAVELENGTH,
            default=default_atm_wavelength,
        )
        atm_profiles = self._build_atm_profiles(
            base_setup_payload,
            setup_cfg,
            require_quantity(atm_wavelength, u.um, label=schema.KEY_SETUP_ATM_WAVELENGTH) if atm_wavelength is not None else None,
            _default_atm_profile=default_atm_profile,
        )

        lgs_r = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            self.KEY_SETUP_LGS_R,
            default=default_lgs_r,
        )
        lgs_theta = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            self.KEY_SETUP_LGS_THETA,
            default=default_lgs_theta,
        )

        ngs_magnitude_zeropoint = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            self.KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT,
            default=default_ngs_mag_zeropoint,
        )
        if ngs_magnitude_zeropoint is None:
            raise ValueError(f"{type(self).__name__} requires setup['{self.KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT}'].")

        sci_r = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            schema.KEY_SETUP_SCI_R,
            default=default_sci_r,
        )
        sci_theta = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            schema.KEY_SETUP_SCI_THETA,
            default=default_sci_theta,
        )

        ee_apertures = require_quantity(ee_apertures, u.mas, label=schema.KEY_SETUP_EE_APERTURES)
        atm_wavelength_quantity = require_quantity(atm_wavelength, u.um, label=schema.KEY_SETUP_ATM_WAVELENGTH)
        atm_profiles_map = {int(k): dict(v) for k, v in atm_profiles.items()}
        lgs_r = require_quantity(lgs_r, u.arcsec, label=self.KEY_SETUP_LGS_R)
        lgs_theta = require_quantity(lgs_theta, u.deg, label=self.KEY_SETUP_LGS_THETA)
        ngs_magnitude_zeropoint = require_quantity(
            ngs_magnitude_zeropoint,
            u.photon / u.s,
            label=self.KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT,
        )
        sci_r = require_quantity(sci_r, u.arcsec, label=schema.KEY_SETUP_SCI_R)
        sci_theta = require_quantity(sci_theta, u.deg, label=schema.KEY_SETUP_SCI_THETA)

        self._validate_base_setup_values(
            ngs_magnitude_zeropoint,
            lgs_r,
            lgs_theta,
            atm_profiles_map,
        )

        return {
            schema.KEY_SETUP_EE_APERTURES: ee_apertures,
            schema.KEY_SETUP_SR_METHOD: str(
                select_mapping_value(
                    base_setup_payload,
                    setup_cfg,
                    schema.KEY_SETUP_SR_METHOD,
                    default=schema.DEFAULT_SETUP_SR_METHOD,
                )
            ).strip(),
            schema.KEY_SETUP_FWHM_SUMMARY: str(
                select_mapping_value(
                    base_setup_payload,
                    setup_cfg,
                    schema.KEY_SETUP_FWHM_SUMMARY,
                    default=schema.DEFAULT_SETUP_FWHM_SUMMARY,
                )
            ).strip(),
            schema.KEY_SETUP_EE_GEOMETRY: str(
                select_mapping_value(
                    base_setup_payload,
                    setup_cfg,
                    schema.KEY_SETUP_EE_GEOMETRY,
                    default=schema.DEFAULT_SETUP_EE_GEOMETRY,
                )
            ).strip(),
            schema.KEY_SETUP_ATM_WAVELENGTH: atm_wavelength_quantity,
            schema.KEY_SETUP_ATM_PROFILES: atm_profiles_map,
            self.KEY_SETUP_LGS_R: lgs_r,
            self.KEY_SETUP_LGS_THETA: lgs_theta,
            self.KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT: ngs_magnitude_zeropoint,
            schema.KEY_SETUP_SCI_R: sci_r,
            schema.KEY_SETUP_SCI_THETA: sci_theta,
        }

    def _parse_base_setup_payload(
        self,
        setup_payload: Mapping[str, Any],
        setup_cls: type[TBaseSetup],
    ) -> TBaseSetup:
        """Deserialize and validate shared setup fields into ``setup_cls``."""
        lgs_r_raw = setup_payload.get(self.KEY_SETUP_LGS_R, [])
        lgs_theta_raw = setup_payload.get(self.KEY_SETUP_LGS_THETA, [])
        setup = setup_cls(
            ee_apertures=require_quantity(setup_payload[schema.KEY_SETUP_EE_APERTURES], u.mas, label=schema.KEY_SETUP_EE_APERTURES),
            sr_method=str(setup_payload[schema.KEY_SETUP_SR_METHOD]).strip(),
            fwhm_summary=str(setup_payload[schema.KEY_SETUP_FWHM_SUMMARY]).strip(),
            ee_geometry=str(setup_payload[schema.KEY_SETUP_EE_GEOMETRY]).strip(),
            atm_wavelength=require_quantity(setup_payload[schema.KEY_SETUP_ATM_WAVELENGTH], u.um, label=schema.KEY_SETUP_ATM_WAVELENGTH),
            atm_profiles=atm.parse_atm_profiles(setup_payload[schema.KEY_SETUP_ATM_PROFILES]),
            lgs_r=require_quantity(lgs_r_raw, u.arcsec, label=self.KEY_SETUP_LGS_R),
            lgs_theta=require_quantity(lgs_theta_raw, u.deg, label=self.KEY_SETUP_LGS_THETA),
            ngs_magnitude_zeropoint=require_quantity(setup_payload[self.KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT], u.photon / u.s, label=self.KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT),
            sci_r=require_quantity(setup_payload[schema.KEY_SETUP_SCI_R], u.arcsec, label=schema.KEY_SETUP_SCI_R),
            sci_theta=require_quantity(setup_payload[schema.KEY_SETUP_SCI_THETA], u.deg, label=schema.KEY_SETUP_SCI_THETA),
        )
        self._validate_base_setup(setup)
        return setup

    def _load_base_setup_payload(
        self,
        setup_payload: Mapping[str, Any],
        setup_cls: type[TBaseSetup],
    ) -> TBaseSetup:
        """Deserialize shared setup fields into ``setup_cls`` and bind the result."""
        setup = self._parse_base_setup_payload(setup_payload, setup_cls)
        self._setup = setup
        return setup

    def validate_setup_payload(self, setup_payload: Mapping[str, Any]) -> None:
        """Validate persisted ``/setup`` without mutating bound setup state.

        Args:
            setup_payload: Candidate persisted setup payload.

        Raises:
            TypeError: If ``setup_payload`` is not a mapping.
            ValueError: If setup loading/validation fails.
        """
        if not isinstance(setup_payload, Mapping):
            raise TypeError("setup_payload must be a mapping.")
        self._parse_setup_payload(setup_payload)

    @abstractmethod
    def _parse_setup_payload(self, setup_payload: Mapping[str, Any]) -> BaseSimulationSetup:
        """Parse and validate persisted ``/setup`` without binding it.

        This hook is the non-mutating counterpart to ``load_setup_payload()``.
        Implementations should deserialize the persisted mapping into the
        subclass's typed setup object, run any shared or subclass-specific
        semantic checks, and return the parsed setup instance. They must not
        assign ``self._setup`` or mutate other bound setup state.

        Args:
            setup_payload: Candidate persisted setup payload.

        Returns:
            Parsed typed setup object ready to be bound by
            ``load_setup_payload()``.

        Raises:
            TypeError: If required payload fields have invalid types.
            ValueError: If required payload fields are missing or invalid.
        """

    @abstractmethod
    def load_setup_payload(self, setup_payload: Mapping[str, Any]) -> None:
        """Load and bind persisted ``/setup`` into subclass-specific typed state.

        BaseSimulation subclasses should deserialize any simulator-specific
        setup fields here and leave ``self._setup`` bound to the final typed
        setup object used by ``create()``, ``run()``, and ``finalize()``.

        Args:
            setup_payload: Persisted setup payload.
        """

    # Options payload lifecycle

    def _build_options_payload(
        self,
        num_sims: int,
        base_options_payload: Mapping[str, Any],
        *,
        default_options: Mapping[str, Any] | None = None,
    ) -> dict[str, np.ndarray | u.Quantity]:
        """Build a complete persisted ``/options`` payload from partial inputs.

        This shared builder fills missing 1D option keys from scalar defaults,
        coerces any explicit NGS matrices to float arrays, and normalizes
        ``atm_profile_id`` to persisted ``int32`` storage. Core ``/options``
        validation runs later in the validation layer.

        Args:
            num_sims: Required number of simulations ``N``.
            base_options_payload: Partial options payload prepared by caller or
                subclass.
            default_options: Optional scalar defaults applied only to missing
                1D option keys.

        Returns:
            Completed persisted ``/options`` payload ready for core validation.

        Raises:
            ValueError: If ``num_sims`` is not positive.
        """
        num_sims = int(num_sims)
        if num_sims <= 0:
            raise ValueError("num_sims must be > 0.")
        default_options = dict(default_options or {})

        options_payload: dict[str, Any] = {}
        for key, value in base_options_payload.items():
            unit = schema.OPTION_FIELD_UNITS.get(str(key))
            if unit is None:
                options_payload[str(key)] = np.asarray(value).copy()
            else:
                options_payload[str(key)] = quantity_value(
                    value,
                    unit,
                    label=f"options.{key}",
                ).copy() * unit

        for key, value in default_options.items():
            if key not in options_payload:
                unit = schema.OPTION_FIELD_UNITS.get(key)
                if unit is None:
                    raw = np.asarray(value)
                    dtype = raw.dtype if raw.ndim == 0 else float
                    options_payload[key] = np.full((num_sims,), raw, dtype=dtype)
                else:
                    raw = quantity_value(value, unit, label=f"default options.{key}")
                    options_payload[key] = np.full((num_sims,), raw.item(), dtype=float) * unit

        for key in schema.OPTION_KEYS_NGS:
            if key in options_payload:
                unit = schema.OPTION_FIELD_UNITS[key]
                options_payload[key] = quantity_value(
                    options_payload[key], unit, label=f"options.{key}", dtype=float
                ) * unit

        if schema.KEY_OPTION_ATM_PROFILE_ID in options_payload:
            options_payload[schema.KEY_OPTION_ATM_PROFILE_ID] = np.asarray(
                options_payload[schema.KEY_OPTION_ATM_PROFILE_ID],
                dtype=np.int32,
            ).reshape(-1)

        return options_payload

    # Runtime lifecycle

    @abstractmethod
    def _create_runtime_context(self, index: int, options: dict[str, Any], setup: SimulationSetup) -> dict[str, Any]:
        """Create runtime scratch state for one simulation.

        This hook receives one copied options row and a typed runtime setup.
        The runtime setup is the bound setup when science offsets are absent.
        When offsets are present, it is a transient copy whose science
        coordinate fields contain the resolved execution positions. The hook
        should return transient runtime data needed by ``run()`` and
        ``finalize()`` without mutating persisted dataset content.

        Args:
            index: Zero-based simulation index.
            options: Copied per-simulation options mapping.
            setup: Typed runtime setup with resolved science coordinates.

        Returns:
            Runtime scratch mapping stored in ``SimulationContext.runtime``.

        Raises:
            TypeError: If subclass-specific setup or option types are invalid.
            ValueError: If required runtime inputs are missing or invalid.
        """

    def create(self, index: int, options: Mapping[str, Any]) -> SimulationContext:
        """Create one execution context from bound setup and one options row.

        This shared implementation copies the per-simulation options row,
        resolves effective science coordinates, creates runtime scratch state
        via ``_create_runtime_context()``, and returns the ``SimulationContext``
        consumed by ``run()`` and ``finalize()``. ``context.setup`` remains the
        bound invariant setup; effective coordinates are stored in the
        ``resolved_sci_*`` context fields.

        Args:
            index: Zero-based simulation index.
            options: Per-simulation options mapping.

        Returns:
            Bound simulation context for one simulation.
        """
        options_row = dict(options)
        setup = self.setup
        resolved_sci_r = setup.sci_r
        resolved_sci_theta = setup.sci_theta
        runtime_setup = setup
        if any(key in options_row for key in schema.OPTION_KEYS_SCI_OFFSETS):
            science = resolve_science_coordinates(setup, options_row)
            resolved_sci_r = science.r
            resolved_sci_theta = science.theta
            runtime_setup = replace(
                setup,
                sci_r=resolved_sci_r,
                sci_theta=resolved_sci_theta,
            )
        runtime = self._create_runtime_context(index=int(index), options=options_row, setup=runtime_setup)
        return SimulationContext(
            index=int(index),
            setup=setup,
            options=options_row,
            resolved_sci_r=resolved_sci_r,
            resolved_sci_theta=resolved_sci_theta,
            runtime=runtime,
        )

    @staticmethod
    def _resolved_science_setup(context: SimulationContext) -> SimulationSetup:
        """Return a transient setup carrying one context's resolved science field.

        ``SimulationContext.setup`` remains the bound persisted setup. Runtime
        implementations that need the effective science field can use this
        helper without changing the context's setup contract.
        """
        resolved_r = context.resolved_sci_r
        resolved_theta = context.resolved_sci_theta
        if resolved_r is None and resolved_theta is None:
            return context.setup
        if resolved_r is None or resolved_theta is None:
            raise ValueError("SimulationContext resolved science coordinates must be provided together.")
        return replace(
            context.setup,
            sci_r=resolved_r,
            sci_theta=resolved_theta,
        )

    @abstractmethod
    def _extract_psfs(self, context: SimulationContext) -> np.ndarray | None:
        """Extract the PSF cube for one completed simulation context.

        Return ``None`` when the backend did not expose PSFs; the shared
        finalize path converts that into a clear error.

        Args:
            context: Completed simulation context.

        Returns:
            PSF cube with shape ``[M, Ny, Nx]`` or ``None`` if unavailable.
        """

    @abstractmethod
    def _extract_psf_parameters(self, context: SimulationContext) -> PsfParameters:
        """Extract PSF metadata needed for persistence and core post-processing.

        Subclasses should return the pixel scale, telescope diameter, and
        telescope pupil associated with the PSFs extracted from the same
        completed runtime context.

        Args:
            context: Completed simulation context.

        Returns:
            Extracted PSF metadata for the completed simulation.
        """

    def _extract_diagnostics(self, context: SimulationContext) -> Mapping[str, Any]:
        """Extract optional diagnostics for one completed simulation context.

        Subclasses that declare diagnostics fields should override this hook
        and return values matching their persisted field specs. The base
        implementation returns no diagnostics.
        """
        del context
        return {}

    def finalize(self, context: SimulationContext) -> None:
        """Populate ``context.result`` from subclass PSF extraction hooks.

        This shared finalize path extracts the PSF cube and PSF metadata,
        flattens the metadata into ``result.meta``, and marks the result as a
        successful simulation output. Core PSF validation, extra-stats
        collection, and stats computation run later in the
        runner/result-validation layer.

        Args:
            context: Completed simulation context.

        Raises:
            ValueError: If the subclass does not expose a PSF cube.
        """
        psfs = self._extract_psfs(context)
        if psfs is None:
            raise ValueError(f"{type(self).__name__} did not expose a PSF cube for finalize().")

        psf_parameters = self._extract_psf_parameters(context)

        context.result = SimulationResult(
            state=SimulationState.SUCCEEDED,
            psfs=psfs,
            meta={
                schema.KEY_META_PIXEL_SCALE: psf_parameters.pixel_scale.to(u.mas),
                schema.KEY_META_TEL_DIAMETER: psf_parameters.tel_diameter.to(u.m),
                schema.KEY_META_TEL_PUPIL: psf_parameters.tel_pupil,
            },
            diagnostics=dict(self._extract_diagnostics(context)),
        )

    def prepare_psfs_for_stats(
        self,
        psfs: np.ndarray,
        setup: Mapping[str, Any] | SimulationSetup,
        meta: Mapping[str, Any],
    ) -> np.ndarray:
        """Apply the default shared PSF preprocessing path for core stats.

        The base implementation keeps preprocessing limited to the shared
        non-negative clipping and pixel-sum normalization stages. Subclasses
        may override this hook when they need simulation-specific stats
        preprocessing.

        Args:
            psfs: Validated PSF cube with shape ``[M, Ny, Nx]``.
            setup: Bound setup payload used for stats computation.
            meta: Per-simulation PSF metadata mapping.

        Returns:
            Preprocessed PSF cube ready for the core stats stages.
        """
        del setup, meta
        return clip_and_sum_normalize_psfs(psfs)
