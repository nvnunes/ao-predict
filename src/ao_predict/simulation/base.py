"""Shared simulation lifecycle scaffolding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

import numpy as np

from . import schema
from . import atm
from .helpers import _MISSING, get_num_sci, select_mapping_value
from .interfaces import Simulation, SimulationContext, SimulationResult, SimulationSetup, SimulationState
from ..utils import as_array_dict, as_float_vector


@dataclass(frozen=True)
class BaseSimulationSetup(SimulationSetup):
    """Typed setup payload shared by BaseSimulation subclasses.

    This extends the core ``SimulationSetup`` contract with the common
    NGS magnitude zero-point used by simulations that derive WFS photon
    inputs from magnitudes.
    """

    ngs_mag_zeropoint: float


@dataclass(frozen=True)
class PsfParameters:
    """PSF metadata extracted from one completed simulation.

    These values are an internal extraction helper for ``BaseSimulation``
    subclasses. ``finalize()`` flattens them into ``SimulationResult.meta``
    for persistence and core post-processing.
    """

    pixel_scale_mas: float
    tel_diameter_m: float
    tel_pupil: np.ndarray


TBaseSetup = TypeVar("TBaseSetup", bound=BaseSimulationSetup)


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

    KEY_SETUP_LGS_R_ARCSEC = "lgs_r_arcsec"
    KEY_SETUP_LGS_THETA_DEG = "lgs_theta_deg"
    KEY_SETUP_NGS_MAG_ZEROPOINT = "ngs_mag_zeropoint"

    SETUP_KEYS_BASE = (
        schema.KEY_SETUP_ATM_WAVELENGTH_UM,
        schema.KEY_SETUP_ATM_PROFILES,
        KEY_SETUP_LGS_R_ARCSEC,
        KEY_SETUP_LGS_THETA_DEG,
        KEY_SETUP_NGS_MAG_ZEROPOINT,
        schema.KEY_SETUP_SCI_R_ARCSEC,
        schema.KEY_SETUP_SCI_THETA_DEG,
    )
    # Construction and properties

    def __init__(self) -> None:
        """Initialize base simulation state."""
        self._setup: BaseSimulationSetup | None = None

    @property
    def setup(self) -> BaseSimulationSetup:
        """Return bound setup or raise if setup has not been loaded."""
        if self._setup is None:
            raise TypeError(f"{type(self).__name__} setup is not configured. Call load_setup_payload(...) first.")
        return self._setup

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
            *(str(k) for k in exclude_keys or ()),
        }
        payload = {str(k): v for k, v in dict(base_simulation_payload).items()}
        payload.update(
            {
                str(k): v
                for k, v in dict(simulation_cfg).items()
                if str(k) not in exclude_keys
            }
        )
        return payload

    def validate_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Validate simulation-specific persisted ``/simulation`` fields.

        The base implementation is a no-op. Core identity/version checks are
        handled in core validation code before this hook is called.
        """
        del simulation_payload

    # Setup payload lifecycle

    def _build_atm_profiles(
        self,
        base_setup_payload: Mapping[str, Any],
        setup_cfg: Mapping[str, Any],
        atm_wavelength_um: float | None,
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

        profiles = atm.normalize_atm_profiles_with_seeing_alias(profiles, atm_wavelength_um)
        atm.validate_standard_atm_profiles(profiles)
        return profiles

    @classmethod
    def _validate_base_setup(cls, setup: BaseSimulationSetup) -> None:
        """Validate shared semantic constraints on a typed base setup object."""
        lgs_r = as_float_vector(setup.lgs_r_arcsec, label=cls.KEY_SETUP_LGS_R_ARCSEC)
        lgs_theta = as_float_vector(setup.lgs_theta_deg, label=cls.KEY_SETUP_LGS_THETA_DEG)
        cls._validate_base_setup_values(
            ngs_mag_zeropoint=setup.ngs_mag_zeropoint,
            lgs_r_arcsec=lgs_r,
            lgs_theta_deg=lgs_theta,
            atm_profiles=setup.atm_profiles,
        )

    @classmethod
    def _validate_base_setup_values(
        cls,
        ngs_mag_zeropoint: float,
        lgs_r_arcsec: np.ndarray,
        lgs_theta_deg: np.ndarray,
        atm_profiles: Mapping[int, Mapping[str, Any]],
    ) -> None:
        """Validate normalized shared setup values before persistence or binding."""
        ngs_mag_zeropoint = float(ngs_mag_zeropoint)
        if not np.isfinite(ngs_mag_zeropoint) or ngs_mag_zeropoint <= 0.0:
            raise ValueError(f"setup['{cls.KEY_SETUP_NGS_MAG_ZEROPOINT}'] must be a positive finite scalar.")

        if lgs_r_arcsec.shape != lgs_theta_deg.shape:
            raise ValueError(
                f"setup['{cls.KEY_SETUP_LGS_R_ARCSEC}'] and setup['{cls.KEY_SETUP_LGS_THETA_DEG}'] must have identical shape."
            )
        if lgs_r_arcsec.size > 0 and (not np.all(np.isfinite(lgs_r_arcsec)) or not np.all(np.isfinite(lgs_theta_deg))):
            raise ValueError("setup LGS coordinates must be finite.")

        atm.validate_standard_atm_profiles(atm_profiles)

    def _build_setup_payload(
        self,
        base_setup_payload: Mapping[str, Any],
        setup_cfg: Mapping[str, Any],
        *,
        default_atm_wavelength_um: Any = _MISSING,
        default_atm_profile: Mapping[str, Any] | None = None,
        default_lgs_r_arcsec: Any = _MISSING,
        default_lgs_theta_deg: Any = _MISSING,
        default_sci_r_arcsec: Any = _MISSING,
        default_sci_theta_deg: Any = _MISSING,
        default_ngs_mag_zeropoint: Any = None,
    ) -> dict[str, Any]:
        """Build, validate, and serialize setup payload using shared base fields.

        This build path is persistence-oriented and intentionally does not
        require simulation-specific setup subclasses.
        """
        ee_apertures_mas = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            schema.KEY_SETUP_EE_APERTURES_MAS,
        )

        atm_wavelength_um = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            schema.KEY_SETUP_ATM_WAVELENGTH_UM,
            default=default_atm_wavelength_um,
        )
        atm_profiles = self._build_atm_profiles(
            base_setup_payload,
            setup_cfg,
            float(atm_wavelength_um) if atm_wavelength_um is not None else None,
            _default_atm_profile=default_atm_profile,
        )

        lgs_r_arcsec = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            self.KEY_SETUP_LGS_R_ARCSEC,
            default=default_lgs_r_arcsec,
        )
        lgs_theta_deg = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            self.KEY_SETUP_LGS_THETA_DEG,
            default=default_lgs_theta_deg,
        )

        ngs_mag_zeropoint = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            self.KEY_SETUP_NGS_MAG_ZEROPOINT,
            default=default_ngs_mag_zeropoint,
        )
        if ngs_mag_zeropoint is None:
            raise ValueError(f"{type(self).__name__} requires setup['{self.KEY_SETUP_NGS_MAG_ZEROPOINT}'].")

        sci_r_arcsec = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            schema.KEY_SETUP_SCI_R_ARCSEC,
            default=default_sci_r_arcsec,
        )
        sci_theta_deg = select_mapping_value(
            base_setup_payload,
            setup_cfg,
            schema.KEY_SETUP_SCI_THETA_DEG,
            default=default_sci_theta_deg,
        )

        ee_apertures_mas = as_float_vector(ee_apertures_mas, label=schema.KEY_SETUP_EE_APERTURES_MAS)
        atm_wavelength_um_scalar = float(atm_wavelength_um)
        atm_profiles_map = {int(k): dict(v) for k, v in atm_profiles.items()}
        lgs_r_arcsec = as_float_vector(lgs_r_arcsec, label=self.KEY_SETUP_LGS_R_ARCSEC)
        lgs_theta_deg = as_float_vector(lgs_theta_deg, label=self.KEY_SETUP_LGS_THETA_DEG)
        ngs_mag_zeropoint_scalar = float(ngs_mag_zeropoint)
        sci_r_arcsec = as_float_vector(sci_r_arcsec, label=schema.KEY_SETUP_SCI_R_ARCSEC)
        sci_theta_deg = as_float_vector(sci_theta_deg, label=schema.KEY_SETUP_SCI_THETA_DEG)

        self._validate_base_setup_values(
            ngs_mag_zeropoint_scalar,
            lgs_r_arcsec,
            lgs_theta_deg,
            atm_profiles_map,
        )

        return {
            schema.KEY_SETUP_EE_APERTURES_MAS: ee_apertures_mas,
            schema.KEY_SETUP_ATM_WAVELENGTH_UM: atm_wavelength_um_scalar,
            schema.KEY_SETUP_ATM_PROFILES: atm_profiles_map,
            self.KEY_SETUP_LGS_R_ARCSEC: lgs_r_arcsec,
            self.KEY_SETUP_LGS_THETA_DEG: lgs_theta_deg,
            self.KEY_SETUP_NGS_MAG_ZEROPOINT: ngs_mag_zeropoint_scalar,
            schema.KEY_SETUP_SCI_R_ARCSEC: sci_r_arcsec,
            schema.KEY_SETUP_SCI_THETA_DEG: sci_theta_deg,
        }

    def _parse_base_setup_payload(
        self,
        setup_payload: Mapping[str, Any],
        setup_cls: type[TBaseSetup],
    ) -> TBaseSetup:
        """Deserialize and validate shared setup fields into ``setup_cls``."""
        lgs_r_raw = setup_payload.get(self.KEY_SETUP_LGS_R_ARCSEC, [])
        lgs_theta_raw = setup_payload.get(self.KEY_SETUP_LGS_THETA_DEG, [])
        setup = setup_cls(
            ee_apertures_mas=as_float_vector(
                setup_payload[schema.KEY_SETUP_EE_APERTURES_MAS],
                label=schema.KEY_SETUP_EE_APERTURES_MAS,
            ),
            atm_wavelength_um=float(setup_payload[schema.KEY_SETUP_ATM_WAVELENGTH_UM]),
            atm_profiles=atm.parse_atm_profiles(setup_payload[schema.KEY_SETUP_ATM_PROFILES]),
            lgs_r_arcsec=as_float_vector(lgs_r_raw, label=self.KEY_SETUP_LGS_R_ARCSEC),
            lgs_theta_deg=as_float_vector(lgs_theta_raw, label=self.KEY_SETUP_LGS_THETA_DEG),
            ngs_mag_zeropoint=float(setup_payload[self.KEY_SETUP_NGS_MAG_ZEROPOINT]),
            sci_r_arcsec=as_float_vector(
                setup_payload[schema.KEY_SETUP_SCI_R_ARCSEC],
                label=schema.KEY_SETUP_SCI_R_ARCSEC,
            ),
            sci_theta_deg=as_float_vector(
                setup_payload[schema.KEY_SETUP_SCI_THETA_DEG],
                label=schema.KEY_SETUP_SCI_THETA_DEG,
            ),
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
    ) -> dict[str, np.ndarray]:
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

        options_payload: dict[str, np.ndarray] = as_array_dict(base_options_payload, copy_arrays=True)

        for key, value in default_options.items():
            if key not in options_payload:
                value = np.asarray(value)
                dtype = value.dtype if value.ndim == 0 else float
                options_payload[key] = np.full((num_sims,), value, dtype=dtype)

        for key in schema.OPTION_KEYS_NGS:
            if key in options_payload:
                options_payload[key] = np.asarray(options_payload[key], dtype=float)

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

        This hook receives the bound typed setup plus one copied options row
        and should return any transient runtime data needed by ``run()`` and
        ``finalize()``. It may derive simulator-specific state, but it should
        not mutate persisted dataset content.

        Args:
            index: Zero-based simulation index.
            options: Copied per-simulation options mapping.
            setup: Bound typed setup object for this simulation instance.

        Returns:
            Runtime scratch mapping stored in ``SimulationContext.runtime``.

        Raises:
            TypeError: If subclass-specific setup or option types are invalid.
            ValueError: If required runtime inputs are missing or invalid.
        """

    def create(self, index: int, options: Mapping[str, Any]) -> SimulationContext:
        """Create one execution context from bound setup and one options row.

        This shared implementation copies the per-simulation options row,
        creates runtime scratch state via ``_create_runtime_context()``, and
        returns the ``SimulationContext`` consumed by ``run()`` and
        ``finalize()``.

        Args:
            index: Zero-based simulation index.
            options: Per-simulation options mapping.

        Returns:
            Bound simulation context for one simulation.
        """
        setup = self.setup
        options_row = dict(options)
        runtime = self._create_runtime_context(index=int(index), options=options_row, setup=setup)
        return SimulationContext(
            index=int(index),
            setup=setup,
            options=options_row,
            runtime=runtime,
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
                schema.KEY_META_PIXEL_SCALE_MAS: np.float32(psf_parameters.pixel_scale_mas),
                schema.KEY_META_TEL_DIAMETER_M: np.float32(psf_parameters.tel_diameter_m),
                schema.KEY_META_TEL_PUPIL: psf_parameters.tel_pupil,
            },
        )
