"""Simulation base interface and execution context contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping

import numpy as np


# Status model

class SimulationState(IntEnum):
    """Canonical simulation status values persisted in ``/status/state``."""

    PENDING = 0
    SUCCEEDED = 1
    FAILED = 2


@dataclass(frozen=True)
class SimulationSetup:
    """Typed normalized setup payload persisted under `/setup`.

    Core fields are defined here because they are used by ao-predict itself
    (validation, option normalization, and persistence shape checks).
    Simulation subclasses may extend this dataclass to add additional setup
    fields, but should keep these core fields present. Values are normalized
    after setup preparation/loading, so vector fields are concrete arrays
    rather than optional placeholders.

    Attributes:
        ee_apertures_mas: EE aperture diameters (mas).
        sr_method: Dataset-level Strehl selector for PSF statistics.
        fwhm_summary: Dataset-level FWHM contour summary selector.
        atm_wavelength_um: Atmospheric reference wavelength (um).
        atm_profiles: Atmospheric profile mapping keyed by profile id.
        lgs_r_arcsec: Invariant LGS radial coordinates (arcsec).
        lgs_theta_deg: Invariant LGS angular coordinates (deg).
        sci_r_arcsec: Invariant science radial coordinates (arcsec).
        sci_theta_deg: Invariant science angular coordinates (deg).
    """

    ee_apertures_mas: np.ndarray
    sr_method: str
    fwhm_summary: str
    atm_wavelength_um: float
    atm_profiles: dict[int, dict[str, Any]]
    lgs_r_arcsec: np.ndarray
    lgs_theta_deg: np.ndarray
    sci_r_arcsec: np.ndarray
    sci_theta_deg: np.ndarray


# Runtime payloads

@dataclass
class SimulationResult:
    """Output payload for one simulation.

    Attributes:
        state: Final state (`SimulationState.SUCCEEDED` or `SimulationState.FAILED`).
        psfs: Optional PSF cube for this simulation.
        meta: Per-simulation metadata used by persistence.
        stats: Final per-simulation persisted stats assembled by ao-predict.
            This field is initialized empty by default. Successful
            simulations should not populate it directly; ao-predict computes
            the core stats (`sr`, `ee`, and `fwhm_mas`) later from the PSFs
            and merges in simulation-provided declared extra stats returned
            by `build_extra_stats(...)`.
        errors: Failure details when ``state=SimulationState.FAILED``.
        runtime: Non-persisted diagnostics for API-side inspection.

    Notes:
    - `state=SimulationState.SUCCEEDED`: simulation succeeded; `meta` should
      include the required persisted fields. `stats` starts empty and is
      assembled later by ao-predict from core PSF-derived stats plus
      declared extra stats.
    - `state=SimulationState.FAILED`: simulation failed; `errors` should describe the failure.
    """

    state: SimulationState
    psfs: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, np.ndarray] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationContext:
    """Runtime context for one simulation during execution.

    Attributes:
        index: Index of this simulation in the dataset.
        setup: Bound typed setup payload for this simulation class.
        options: Resolved per-simulation options. May include runtime-derived
            non-persisted fields such as ``ngs_used``.
        result: Populated in ``finalize``.
        runtime: Scratch space shared across lifecycle steps.
    """

    index: int
    setup: SimulationSetup
    options: dict[str, Any]
    result: SimulationResult | None = None
    runtime: dict[str, Any] = field(default_factory=dict)


# Simulation contract

class Simulation(ABC):
    """Abstract simulation interface used by ao-predict runners.

    Lifecycle (init path):
    1. `prepare_simulation_payload(...)`
    2. `prepare_setup_payload(...)`
    3. `prepare_options_payload(...)`

    Lifecycle (run/retry path):
    1. `load_simulation_payload(...)`
    2. `load_setup_payload(...)`
    3. for each simulation:
       `create(...)` -> `run(...)` -> `finalize(...)` -> `build_extra_stats(...)`

    Implementers should keep `prepare_*` / `validate_*` methods focused on
    payload construction and validation. Avoid external I/O, simulator runs,
    and mutable runtime state changes in those methods; keep runtime state
    changes in `create/run/finalize`.

    Implementation pattern:
    - `prepare_*`: build payloads (prefer base `_build_*` helpers when available).
    - `validate_*`: call `super()` first, then apply subclass-specific checks.
    - `load_*`: perform subclass-specific deserialization and binding. For
      setup loading, prefer base `_load_base_setup_payload(...)` plus optional
      subclass post-processing.
    """

    _NAME: str = ""
    _VERSION: str = ""

    @property
    def name(self) -> str:
        value = str(getattr(type(self), "_NAME", "")).strip()
        if value:
            return value
        return f"{type(self).__module__}:{type(self).__name__}"

    @property
    def version(self) -> str:
        return str(type(self)._VERSION)

    @property
    def extra_stat_names(self) -> tuple[str, ...]:
        """Return simulation-specific extra stat names persisted under ``/stats``.

        These names are part of the core `/simulation` payload assembled by
        ao-predict before `prepare_simulation_payload(...)` is called.
        """
        return ()

    @abstractmethod
    def prepare_simulation_payload(
        self,
        base_simulation_payload: Mapping[str, Any],
        simulation_cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Build persisted `/simulation` payload from core base fields plus config.

        For file-backed simulations, this is where external config paths are
        converted into serialized content suitable for persistence.

        Args:
            base_simulation_payload: Core-prepared simulation payload containing
                the shared persisted `/simulation` contract fields.
            simulation_cfg: Raw simulation section from API/CLI config.

        Returns:
            Complete persisted mapping for ``/simulation``. Implementations
            should start from ``base_simulation_payload`` and add only
            simulation-specific fields.
        """

    @abstractmethod
    def load_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Load simulation-level metadata from persisted `/simulation` payload.

        Called before setup/options preparation and before run/retry execution.
        Use this to deserialize and cache simulation-level configuration that
        will be needed by later lifecycle methods.

        Args:
            simulation_payload: Persisted simulation mapping read from dataset.
        """

    @abstractmethod
    def validate_simulation_payload(self, simulation_payload: Mapping[str, Any]) -> None:
        """Validate persisted `/simulation` payload without mutating runtime state.

        Args:
            simulation_payload: Candidate persisted simulation payload.

        Raises:
            ValueError: If payload content is invalid.
            TypeError: If payload types are invalid.
        """

    @abstractmethod
    def prepare_setup_payload(self, base_setup_payload: Mapping[str, Any], setup_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
        """Build full persisted `/setup` payload.

        Args:
            base_setup_payload: Normalized core setup values from runner.
            setup_cfg: Raw setup mapping from API/CLI config.

        Returns:
            Complete mapping to persist under ``/setup``.
        """

    @abstractmethod
    def validate_setup_payload(self, setup_payload: Mapping[str, Any]) -> None:
        """Validate persisted `/setup` payload without mutating bound state.

        Args:
            setup_payload: Candidate persisted setup payload.
        """

    @abstractmethod
    def load_setup_payload(self, setup_payload: Mapping[str, Any]) -> None:
        """Bind typed setup from persisted `/setup` payload.

        This is the canonical place to convert the persisted mapping into a
        typed setup object and cache it in the simulation instance.

        Args:
            setup_payload: Persisted setup mapping read from dataset.

        """

    @abstractmethod
    def prepare_options_payload(
        self,
        num_sims: int,
        setup_payload: Mapping[str, Any],
        base_options_payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Complete partial `/options` payload to full per-simulation options.

        `base_options_payload` may omit keys that can be defaulted from
        simulation/base configuration. Return value must include all required
        core option keys for `num_sims` simulations.

        Args:
            num_sims: Number of simulations in this dataset.
            setup_payload: Persisted setup payload for setup-dependent defaults.
            base_options_payload: Partial options payload.

        Returns:
            Complete per-simulation options payload.
        """

    @abstractmethod
    def create(self, index: int, options: Mapping[str, Any]) -> SimulationContext:
        """Create execution context for one simulation.

        Build any per-simulation runtime artifacts needed by `run`.
        Should not mutate persisted data directly.

        Args:
            index: Simulation index.
            options: Per-simulation options.

        Returns:
            Runtime context used by ``run`` and ``finalize``.
        """

    @abstractmethod
    def run(self, context: SimulationContext) -> None:
        """Run the simulation for this context.

        Heavy external calls and simulator execution belong here.
        Intermediate non-persisted artifacts can be stored in `context.runtime`.

        Args:
            context: Runtime context for one simulation.
        """

    @abstractmethod
    def finalize(self, context: SimulationContext) -> None:
        """Finalize outputs by populating `context.result`.

        Must set `context.result` to a `SimulationResult` with:
        - `state=SimulationState.SUCCEEDED` and required output fields on success, or
        - `state=SimulationState.FAILED` plus error information on simulation-level failure.

        Successful results should expose core-readable outputs such as PSFs and
        metadata. Successful simulations must not populate `result.stats`
        directly. ao-predict assembles `result.stats` later from core
        PSF-derived stats plus any declared extra stats returned by
        `build_extra_stats(...)`.

        Args:
            context: Runtime context for one simulation.
        """

    @abstractmethod
    def prepare_psfs_for_stats(
        self,
        psfs: np.ndarray,
        setup: Mapping[str, Any] | SimulationSetup,
        meta: Mapping[str, Any],
    ) -> np.ndarray:
        """Prepare PSFs for core stats computation.

        This hook owns simulation-specific PSF preprocessing used by the core
        stats pipeline. Implementations may apply alternate normalization,
        centering, or compatibility preprocessing, but they must return a PSF
        cube with the same ``[M, Ny, Nx]`` shape as the input.

        Args:
            psfs: Validated PSF cube with shape ``[M, Ny, Nx]``.
            setup: Bound setup payload used for stats computation.
            meta: Per-simulation PSF metadata mapping.

        Returns:
            Preprocessed PSF cube ready for core Strehl, EE, and FWHM stages.
        """

    def build_extra_stats(self, context: SimulationContext) -> Mapping[str, Any]:
        """Build declared simulation-specific extra stats for one completed result.

        This hook runs after `finalize(...)` for successful simulations.
        Implementations should return only declared simulation-owned extra
        stats with per-simulation shape `[M]`. Successful simulations should
        leave `context.result.stats` empty; core stats are computed later by
        ao-predict and must not be returned here.

        Args:
            context: Completed runtime context with a successful result.

        Returns:
            Mapping of declared extra stat name to per-science-target values.
        """
        del context
        return {}
