"""Simulation payload preparation and execution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
from typing import Any, Type

import numpy as np

from ..persistence import SimulationStore
from . import schema
from .config import add_runtime_derived_options
from .validation import (
    validate_psf_cube,
    validate_simulation_payload_core,
    validate_setup_payload_core,
)
from .interfaces import Simulation, SimulationState
from .stats import compute_psf_stats
from ..utils import as_float_vector


# Structures

@dataclass
class RunSummary:
    """Execution counters for one simulation run pass.

    Attributes:
        attempted: Number of simulations attempted.
        succeeded: Number of simulations persisted as succeeded.
        failed: Number of simulations persisted as failed.
    """

    attempted: int
    succeeded: int
    failed: int


# Simulation payload preparation

def _check_simulation_payload(simulation: Simulation, simulation_payload: Mapping[str, Any]) -> None:
    """Ensure persisted ``/simulation`` matches the instantiated simulation.

    Args:
        simulation: Instantiated simulation implementation.
        simulation_payload: Candidate persisted simulation payload.
    """
    validate_simulation_payload_core(
        simulation_payload,
        simulation.name,
        simulation.version,
        simulation.extra_stat_names,
    )
    simulation.validate_simulation_payload(simulation_payload)


def _prepare_base_simulation_payload(simulation: Simulation) -> dict[str, Any]:
    """Build the core persisted ``/simulation`` payload owned by ao-predict."""
    return {
        schema.KEY_SIMULATION_NAME: simulation.name,
        schema.KEY_SIMULATION_VERSION: simulation.version,
        schema.KEY_SIMULATION_EXTRA_STAT_NAMES: np.asarray(
            simulation.extra_stat_names,
            dtype=str,
        ),
    }


def _load_simulation_class(spec: str) -> Type[Simulation]:
    """Resolve and validate a Simulation subclass from a class path string.

    Supported forms:
    - ``pkg.module:ClassName``
    - ``pkg.module.ClassName``
    """
    # Supported forms: "pkg.module:ClassName" or "pkg.module.ClassName"
    if ":" in spec:
        module_name, class_name = spec.split(":", 1)
    else:
        parts = spec.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid simulation path '{spec}'. Use 'module:ClassName' or 'module.ClassName'."
            )
        module_name, class_name = parts[0], parts[1]

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(f"Simulation class '{class_name}' not found in module '{module_name}'.")
    if not isinstance(cls, type) or not issubclass(cls, Simulation):
        raise ValueError(f"'{spec}' does not resolve to a Simulation subclass.")
    return cls


def _create_simulation(simulation_name: str) -> Simulation:
    """Instantiate a Simulation implementation.

    Args:
        simulation_name: Canonical class path or ao-predict short name.

    Returns:
        Instantiated simulation object.

    Raises:
        ValueError: If the name/path cannot be resolved.
    """
    spec = simulation_name.strip()
    if not spec:
        raise ValueError("simulation name must be a non-empty class path.")

    # Canonical path mode supports internal and external simulations.
    if ":" in spec or "." in spec:
        cls = _load_simulation_class(spec)
        return cls()

    # Short-name mode is limited to ao_predict simulations:
    # e.g. "Tiptop" -> "ao_predict.simulation.tiptop:TiptopSimulation"
    short = spec
    class_name = f"{short}Simulation"
    module_name = f"ao_predict.simulation.{short.lower()}"
    class_path = f"{module_name}:{class_name}"
    try:
        cls = _load_simulation_class(class_path)
    except Exception as exc:
        raise ValueError(
            f"Unsupported short simulation name '{simulation_name}'. "
            f"Expected ao-predict short form like 'Tiptop' or canonical class path."
        ) from exc
    return cls()


def create_simulation_from_config(simulation_cfg: Mapping[str, Any]) -> tuple[Simulation, dict[str, Any]]:
    """Create and initialize a simulation from normalized config input.

    Args:
        simulation_cfg: Normalized ``simulation`` config mapping.

    Returns:
        Tuple ``(simulation, simulation_payload)`` where:
        - ``simulation`` is instantiated and loaded with payload state.
        - ``simulation_payload`` is validated and ready for persistence.

    Notes:
        ao-predict assembles the core persisted ``/simulation`` fields
        (`name`, `version`, and `extra_stat_names`) before delegating to
        ``simulation.prepare_simulation_payload(...)`` for simulation-specific
        completion.

    Raises:
        ValueError: If required config fields are missing/invalid.
        TypeError: If simulation payload fields have invalid types.
    """
    simulation_name = simulation_cfg.get("name")
    if not isinstance(simulation_name, str) or not simulation_name.strip():
        raise ValueError("simulation.name must be provided as a non-empty string.")

    simulation = _create_simulation(simulation_name)
    base_simulation_payload = _prepare_base_simulation_payload(simulation)
    simulation_payload = simulation.prepare_simulation_payload(
        base_simulation_payload,
        simulation_cfg,
    )
    _check_simulation_payload(simulation, simulation_payload)
    simulation.load_simulation_payload(simulation_payload)
    return simulation, simulation_payload


def create_simulation_from_payload(simulation_payload: Mapping[str, Any]) -> Simulation:
    """Create and initialize a simulation from persisted ``/simulation``.

    Args:
        simulation_payload: Persisted ``/simulation`` payload.

    Returns:
        Instantiated simulation loaded with payload state.

    Raises:
        ValueError: If required payload fields are missing/invalid.
        TypeError: If payload fields have invalid types.
    """
    simulation_name = simulation_payload.get("name")
    if not isinstance(simulation_name, str) or not simulation_name.strip():
        raise ValueError("Dataset /simulation must include non-empty string field 'name'.")

    simulation = _create_simulation(simulation_name)
    _check_simulation_payload(simulation, simulation_payload)
    simulation.load_simulation_payload(simulation_payload)
    return simulation


# Setup payload preparation

def _check_setup_payload(simulation: Simulation, setup_payload: Mapping[str, Any]) -> None:
    """Ensure persisted ``/setup`` satisfies core and simulation contracts.

    Args:
        simulation: Instantiated simulation implementation.
        setup_payload: Candidate persisted setup payload.
    """
    validate_setup_payload_core(setup_payload)
    simulation.validate_setup_payload(setup_payload)


def _prepare_base_setup_payload(base_setup: dict[str, Any]) -> dict[str, Any]:
    """Normalize core setup fields before simulation-specific preparation.

    Args:
        base_setup: Raw normalized setup mapping.

    Returns:
        Setup mapping with core fields normalized into persistence-ready forms.
    """
    setup = dict(base_setup)
    if "ee_apertures_mas" in setup:
        setup["ee_apertures_mas"] = as_float_vector(setup["ee_apertures_mas"], label="setup.ee_apertures_mas")
    return setup


def prepare_setup_payload(simulation: Simulation, setup_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build and validate the persisted ``/setup`` payload.

    Args:
        simulation: Simulation implementation used for setup completion.
        setup_cfg: Normalized setup config mapping.

    Returns:
        Validated ``/setup`` payload ready for persistence.

    Raises:
        ValueError: If setup values are missing/invalid.
        TypeError: If setup values have invalid types.
    """
    base_setup_payload = _prepare_base_setup_payload(dict(setup_cfg))
    setup_payload = simulation.prepare_setup_payload(base_setup_payload, setup_cfg)
    _check_setup_payload(simulation, setup_payload)
    return setup_payload


# Runtime options preparation

def _prepare_runtime_options(store: SimulationStore, index: int) -> dict[str, Any]:
    """Load and augment one simulation's runtime options.

    Args:
        store: Dataset store.
        index: Zero-based simulation index.

    Returns:
        Runtime options mapping for one simulation.
    """
    return add_runtime_derived_options(store.read_sim_options(index))


# Execution internals


def _filter_execution_indices(
    store: SimulationStore,
    available_indices: np.ndarray,
    requested_indexes: list[int] | None,
) -> np.ndarray:
    """Intersect state-matching indices with an optional requested subset.

    Args:
        store: Dataset store.
        available_indices: Indexes currently in the selected state bucket.
        requested_indexes: Optional user-requested subset.

    Returns:
        Filtered execution indexes.
    """
    if requested_indexes is None:
        return available_indices

    requested = np.asarray(requested_indexes, dtype=np.int64).reshape(-1)
    if requested.size == 0:
        return np.zeros((0,), dtype=np.int64)

    total = int(store.num_sims())
    if np.any(requested < 0) or np.any(requested >= total):
        raise ValueError(f"Requested indexes must be in range [0, {total - 1}].")

    return available_indices[np.isin(available_indices, requested)]


def _populate_result_stats(simulation: Simulation, context: Any) -> None:
    """Populate final result stats from core PSF stats plus simulation extra stats.

    Args:
        simulation: Bound simulation implementation.
        context: Completed simulation context with successful ``result``.

    Raises:
        ValueError: If ``context.result`` or its PSF cube is missing.
    """
    if context.result is None:
        raise ValueError("Cannot populate stats without a successful simulation result.")
    if context.result.psfs is None:
        raise ValueError("Cannot populate stats without a successful result PSF cube.")
    if context.result.stats:
        raise ValueError(
            "Successful simulations must not populate result.stats directly. "
            "Declared extra stats must be returned from build_extra_stats(...)."
        )

    num_sci = int(as_float_vector(context.setup.sci_r_arcsec, label="setup.sci_r_arcsec").shape[0])

    context.result.psfs = validate_psf_cube(
        context.result.psfs,
        num_sci,
        f"{type(context.setup).__name__} PSFs",
    )

    sr, ee, fwhm_mas = compute_psf_stats(
        context.result.psfs,
        context.setup,
        context.result.meta,
    )

    raw_extra_stats = simulation.build_extra_stats(context)
    if not isinstance(raw_extra_stats, Mapping):
        raise TypeError(
            f"{type(simulation).__name__}.build_extra_stats(...) must return a mapping, got {type(raw_extra_stats).__name__}."
        )

    extra_stat_names = tuple(raw_extra_stats.keys())

    provided_core_stat_names = sorted(set(extra_stat_names) & set(schema.CORE_STATS_KEYS))
    if provided_core_stat_names:
        raise ValueError(
            "Simulation built core stats in build_extra_stats(): "
            f"{', '.join(provided_core_stat_names)}. "
            "Core stats are owned by ao-predict and must not be provided by the simulation."
        )

    expected_extra_stat_names = tuple(context.runtime.get("extra_stat_names", ()))

    unexpected_extra_stat_names = sorted(set(extra_stat_names) - set(expected_extra_stat_names))
    if unexpected_extra_stat_names:
        raise ValueError(
            "Simulation built undeclared extra stats in build_extra_stats(): "
            f"{', '.join(unexpected_extra_stat_names)}"
        )

    missing_extra_stat_names = [name for name in expected_extra_stat_names if name not in raw_extra_stats]
    if missing_extra_stat_names:
        raise ValueError(
            "Simulation did not build declared extra stats in build_extra_stats(): "
            f"{', '.join(missing_extra_stat_names)}"
        )

    extra_stats = {
        name: np.asarray(raw_extra_stats[name], dtype=np.float32) for name in expected_extra_stat_names
    }

    context.result.stats = {
        schema.KEY_STATS_SR: sr,
        schema.KEY_STATS_EE: ee,
        schema.KEY_STATS_FWHM_MAS: fwhm_mas,
        **extra_stats,
    }


def _run_simulations_for_indices(
    store: SimulationStore,
    simulation: Simulation,
    indices: np.ndarray,
    *,
    allow_from_failed: bool,
    verbose: bool,
) -> RunSummary:
    """Execute simulations for a fixed index set and persist outcomes.

    Args:
        store: Dataset store.
        simulation: Bound simulation implementation.
        indices: Simulation indexes to run.
        allow_from_failed: Whether store writes may transition from ``FAILED``.
        verbose: If ``True``, print failure details.

    Returns:
        Summary counters for attempted/succeeded/failed simulations.
    """
    attempted = 0
    succeeded = 0
    failed = 0

    for index in indices:
        attempted += 1
        idx = int(index)
        try:
            options = _prepare_runtime_options(store, idx)
            context = simulation.create(idx, options)
            context.runtime["extra_stat_names"] = simulation.extra_stat_names
            simulation.run(context)
            simulation.finalize(context)

            if context.result is None:
                raise ValueError("Simulation did not set context.result.")

            if int(context.result.state) == int(SimulationState.SUCCEEDED):
                _populate_result_stats(simulation, context)
                store.write_simulation_success(idx, context.result, allow_from_failed=allow_from_failed)
                succeeded += 1
            else:
                if verbose:
                    if context.result.errors:
                        msg = "; ".join(str(e) for e in context.result.errors)
                    else:
                        msg = f"non-success state={int(context.result.state)}"
                    print(f"Simulation {idx} failed: {msg}")
                store.write_simulation_failure(idx, allow_from_failed=allow_from_failed)
                failed += 1
        except Exception as exc:
            if verbose:
                print(f"Simulation {idx} failed: {type(exc).__name__}: {exc}")
            store.write_simulation_failure(idx, allow_from_failed=allow_from_failed)
            failed += 1

    return RunSummary(attempted=attempted, succeeded=succeeded, failed=failed)


# Execution entry points

def run_simulations_by_state(
    store: SimulationStore,
    simulation: Simulation,
    state: SimulationState | int,
    *,
    indexes: list[int] | None = None,
    verbose: bool = False,
) -> RunSummary:
    """Run simulations from a selected source state.

    Args:
        store: Dataset store.
        simulation: Bound simulation implementation.
        state: Source state to run from.
            Supported values are ``SimulationState.PENDING`` and
            ``SimulationState.FAILED``.
        indexes: Optional subset of simulation indexes to run.
        verbose: If ``True``, print failure messages.

    Returns:
        Execution counters for attempted/succeeded/failed simulations.

    Raises:
        ValueError: If ``state`` is invalid or unsupported.
    """
    try:
        state_value = SimulationState(int(state))
    except Exception as exc:
        raise ValueError(
            "run_simulations_by_state(..., state, ...) requires a valid SimulationState value."
        ) from exc
    if state_value not in (SimulationState.PENDING, SimulationState.FAILED):
        raise ValueError(
            "run_simulations_by_state(..., state, ...) supports only "
            "SimulationState.PENDING or SimulationState.FAILED."
        )

    candidate_indices = _filter_execution_indices(store, store.indices_with_state(state_value), indexes)
    allow_from_failed = state_value == SimulationState.FAILED
    return _run_simulations_for_indices(
        store,
        simulation,
        candidate_indices,
        allow_from_failed=allow_from_failed,
        verbose=verbose,
    )


def run_pending_simulations(
    store: SimulationStore,
    simulation: Simulation,
    *,
    indexes: list[int] | None = None,
    verbose: bool = False,
) -> RunSummary:
    """Run simulations currently in ``PENDING`` state.

    Args:
        store: Dataset store.
        simulation: Bound simulation implementation.
        indexes: Optional subset of simulation indexes to run.
        verbose: If ``True``, print failure messages.

    Returns:
        Execution counters for attempted/succeeded/failed simulations.
    """
    return run_simulations_by_state(
        store,
        simulation,
        SimulationState.PENDING,
        verbose=verbose,
        indexes=indexes,
    )


def run_failed_simulations(
    store: SimulationStore,
    simulation: Simulation,
    *,
    indexes: list[int] | None = None,
    verbose: bool = False,
) -> RunSummary:
    """Run simulations currently in ``FAILED`` state.

    Args:
        store: Dataset store.
        simulation: Bound simulation implementation.
        indexes: Optional subset of simulation indexes to run.
        verbose: If ``True``, print failure messages.

    Returns:
        Execution counters for attempted/succeeded/failed simulations.
    """
    return run_simulations_by_state(
        store,
        simulation,
        SimulationState.FAILED,
        verbose=verbose,
        indexes=indexes,
    )
