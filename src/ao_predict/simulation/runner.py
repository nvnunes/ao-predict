"""Simulation payload preparation and execution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
from typing import Any, Type

from joblib import Parallel, delayed
import numpy as np
from astropy import units as u

from .._units import quantity_value, unit_string
from ..persistence import SimulationStore
from . import schema
from .config import add_runtime_derived_options
from .validation import (
    resolve_simulation_payload_for_load,
    validate_psf_cube,
    validate_simulation_payload_core,
    validate_setup_payload_core,
)
from .interfaces import Simulation, SimulationResult, SimulationState
from .stats import PsfMetadata, compute_psf_stats


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


@dataclass
class _RunOutcome:
    """Completed single-simulation outcome returned by serial or worker code."""

    index: int
    result: SimulationResult | None = None
    failure_message: str | None = None


# Simulation payload preparation

def _check_simulation_payload(simulation: Simulation, simulation_payload: Mapping[str, Any]) -> None:
    """Ensure persisted ``/simulation`` matches the instantiated simulation.

    Args:
        simulation: Instantiated simulation implementation.
        simulation_payload: Candidate persisted simulation payload.
    """
    validate_simulation_payload_core(
        simulation_payload,
        expected_name=simulation.name,
        expected_version=simulation.version,
        expected_extra_stat_fields=simulation.extra_stat_fields,
        expected_ngs_mag_standard=simulation.ngs_mag_standard,
    )
    simulation.validate_simulation_payload(simulation_payload)


def _prepare_base_simulation_payload(simulation: Simulation) -> dict[str, Any]:
    """Build the core persisted ``/simulation`` payload owned by ao-predict."""
    return {
        schema.KEY_SIMULATION_NAME: simulation.name,
        schema.KEY_SIMULATION_VERSION: simulation.version,
        schema.KEY_SIMULATION_EXTRA_STAT_FIELDS: {
            name: unit_string(unit) for name, unit in simulation.extra_stat_fields.items()
        },
        schema.KEY_SIMULATION_NGS_MAG_STANDARD: simulation.ngs_mag_standard,
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
        (`name`, `version`, `extra_stat_fields`, and `ngs_mag_standard`) before
        delegating to ``simulation.prepare_simulation_payload(...)`` for
        simulation-specific completion.

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
    simulation_payload = resolve_simulation_payload_for_load(
        simulation_payload,
        expected_name=simulation.name,
        expected_version=simulation.version,
        expected_extra_stat_fields=simulation.extra_stat_fields,
        expected_ngs_mag_standard=simulation.ngs_mag_standard,
    )
    simulation.validate_simulation_payload(simulation_payload)
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
    if "ee_apertures" in setup:
        setup["ee_apertures"] = quantity_value(
            setup["ee_apertures"],
            u.mas,
            label="setup.ee_apertures",
            dtype=float,
        ).reshape(-1) * u.mas
    setup.setdefault(schema.KEY_SETUP_SR_METHOD, schema.DEFAULT_SETUP_SR_METHOD)
    setup.setdefault(schema.KEY_SETUP_FWHM_SUMMARY, schema.DEFAULT_SETUP_FWHM_SUMMARY)
    setup.setdefault(schema.KEY_SETUP_EE_GEOMETRY, schema.DEFAULT_SETUP_EE_GEOMETRY)
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


def _validate_parallel_options(num_workers: int, chunk_multiple: int) -> tuple[int, int]:
    """Validate and normalize runner parallelism controls."""
    num_workers = int(num_workers)
    chunk_multiple = int(chunk_multiple)
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1.")
    if chunk_multiple < 1:
        raise ValueError("chunk_multiple must be >= 1.")
    return num_workers, chunk_multiple


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

    num_sci = int(quantity_value(context.setup.sci_r, u.arcsec, label="setup.sci_r").reshape(-1).shape[0])

    context.result.psfs = validate_psf_cube(
        context.result.psfs,
        num_sci,
        f"{type(context.setup).__name__} PSFs",
    )

    psf_metadata = PsfMetadata(
        wavelength=context.options[schema.KEY_OPTION_WAVELENGTH],
        pixel_scale=context.result.meta[schema.KEY_META_PIXEL_SCALE],
        tel_diameter=context.result.meta[schema.KEY_META_TEL_DIAMETER],
        tel_pupil=context.result.meta[schema.KEY_META_TEL_PUPIL],
    )
    sr, ee, fwhm = compute_psf_stats(
        context.result.psfs,
        psf_metadata,
        ee_apertures=context.setup.ee_apertures,
        sr_method=context.setup.sr_method,
        fwhm_summary=context.setup.fwhm_summary,
        ee_geometry=context.setup.ee_geometry,
        preprocess=lambda psfs: simulation.prepare_psfs_for_stats(
            psfs,
            context.setup,
            context.result.meta,
        ),
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

    expected_extra_stat_fields = dict(context.runtime.get("extra_stat_fields", {}))
    expected_extra_stat_names = tuple(expected_extra_stat_fields)

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
        name: quantity_value(
            raw_extra_stats[name],
            expected_extra_stat_fields[name],
            label=f"extra stat {name!r}",
            dtype=np.float32,
        ) * u.Unit(expected_extra_stat_fields[name])
        for name in expected_extra_stat_names
    }

    context.result.stats = {
        schema.KEY_STATS_SR: sr,
        schema.KEY_STATS_EE: ee,
        schema.KEY_STATS_FWHM: fwhm,
        **extra_stats,
    }


def _execute_simulation_index(
    simulation: Simulation,
    index: int,
    options: Mapping[str, Any],
    *,
    verbose: bool,
) -> _RunOutcome:
    """Run one simulation index and return its in-memory outcome.

    Args:
        simulation: Bound simulation implementation.
        index: Zero-based simulation index.
        options: Runtime options for this simulation.
        verbose: Whether failure messages should be included in the outcome.

    Returns:
        Completed run outcome. Successful outcomes contain a populated
        ``SimulationResult``. Failed outcomes contain a failure message when
        ``verbose`` is true.
    """
    try:
        context = simulation.create(int(index), options)
        context.runtime["extra_stat_fields"] = dict(simulation.extra_stat_fields)
        simulation.run(context)
        simulation.finalize(context)

        if context.result is None:
            raise ValueError("Simulation did not set context.result.")

        if int(context.result.state) == int(SimulationState.SUCCEEDED):
            _populate_result_stats(simulation, context)
            return _RunOutcome(index=int(index), result=context.result)

        failure_message = None
        if verbose:
            if context.result.errors:
                failure_message = "; ".join(str(e) for e in context.result.errors)
            else:
                failure_message = f"non-success state={int(context.result.state)}"
        return _RunOutcome(index=int(index), result=context.result, failure_message=failure_message)
    except Exception as exc:
        failure_message = f"{type(exc).__name__}: {exc}" if verbose else None
        return _RunOutcome(index=int(index), failure_message=failure_message)


def _persist_run_outcome(
    store: SimulationStore,
    outcome: _RunOutcome,
    *,
    allow_from_failed: bool,
    verbose: bool,
) -> tuple[int, int]:
    """Persist one completed outcome and return ``(succeeded, failed)``."""
    idx = int(outcome.index)
    result = outcome.result
    if result is not None and int(result.state) == int(SimulationState.SUCCEEDED):
        store.write_simulation_success(idx, result, allow_from_failed=allow_from_failed)
        return 1, 0

    if verbose and outcome.failure_message:
        print(f"Simulation {idx} failed: {outcome.failure_message}")
    store.write_simulation_failure(idx, allow_from_failed=allow_from_failed)
    return 0, 1


def _run_simulations_serial(
    store: SimulationStore,
    simulation: Simulation,
    indices: np.ndarray,
    *,
    allow_from_failed: bool,
    verbose: bool,
) -> RunSummary:
    """Execute simulations serially and persist each outcome immediately."""
    attempted = 0
    succeeded = 0
    failed = 0

    for index in indices:
        attempted += 1
        idx = int(index)
        options = _prepare_runtime_options(store, idx)
        outcome = _execute_simulation_index(simulation, idx, options, verbose=verbose)
        dsucceeded, dfailed = _persist_run_outcome(
            store,
            outcome,
            allow_from_failed=allow_from_failed,
            verbose=verbose,
        )
        succeeded += dsucceeded
        failed += dfailed

    return RunSummary(attempted=attempted, succeeded=succeeded, failed=failed)


def _run_worker_chunk(
    simulation_payload: Mapping[str, Any],
    setup_payload: Mapping[str, Any],
    work_items: tuple[tuple[int, dict[str, Any]], ...],
    *,
    verbose: bool,
) -> list[_RunOutcome]:
    """Run one worker chunk and return ordered in-memory outcomes.

    Args:
        simulation_payload: Persisted ``/simulation`` mapping.
        setup_payload: Persisted ``/setup`` mapping.
        work_items: Ordered ``(index, options)`` pairs assigned to this worker.
        verbose: Whether failure messages should be included in outcomes.

    Returns:
        Ordered list of per-index run outcomes.
    """
    try:
        simulation = create_simulation_from_payload(simulation_payload)
        simulation.load_setup_payload(setup_payload)
        simulation.warmup_worker()
    except Exception as exc:
        failure_message = f"{type(exc).__name__}: {exc}" if verbose else None
        return [
            _RunOutcome(index=int(index), failure_message=failure_message)
            for index, _options in work_items
        ]

    return [
        _execute_simulation_index(simulation, index, options, verbose=verbose)
        for index, options in work_items
    ]


def _chunk_work_items(
    work_items: list[tuple[int, dict[str, Any]]],
    *,
    chunk_multiple: int,
) -> list[tuple[tuple[int, dict[str, Any]], ...]]:
    """Split ordered work items into worker-sized chunks."""
    return [
        tuple(work_items[start : start + chunk_multiple])
        for start in range(0, len(work_items), chunk_multiple)
    ]


def _run_simulations_parallel(
    store: SimulationStore,
    indices: np.ndarray,
    *,
    allow_from_failed: bool,
    verbose: bool,
    num_workers: int,
    chunk_multiple: int,
) -> RunSummary:
    """Execute simulations in joblib workers and persist outcomes in parent."""
    attempted = 0
    succeeded = 0
    failed = 0

    simulation_payload = store.read_simulation()
    setup_payload = store.read_setup()
    parallel_pool = Parallel(
        n_jobs=num_workers,
        backend="loky",
        idle_worker_timeout=None,
    )

    num_indices = int(indices.shape[0])
    outer_chunk_size = min(num_indices, num_workers * chunk_multiple)
    for start in range(0, num_indices, outer_chunk_size):
        outer_indices = indices[start : start + outer_chunk_size]
        work_items = [
            (int(index), _prepare_runtime_options(store, int(index)))
            for index in outer_indices
        ]
        worker_chunks = _chunk_work_items(work_items, chunk_multiple=chunk_multiple)
        chunk_results = parallel_pool(
            delayed(_run_worker_chunk)(
                simulation_payload,
                setup_payload,
                chunk,
                verbose=verbose,
            )
            for chunk in worker_chunks
        )

        for outcomes in chunk_results:
            for outcome in outcomes:
                attempted += 1
                dsucceeded, dfailed = _persist_run_outcome(
                    store,
                    outcome,
                    allow_from_failed=allow_from_failed,
                    verbose=verbose,
                )
                succeeded += dsucceeded
                failed += dfailed

    return RunSummary(attempted=attempted, succeeded=succeeded, failed=failed)


def _run_simulations_for_indices(
    store: SimulationStore,
    simulation: Simulation,
    indices: np.ndarray,
    *,
    allow_from_failed: bool,
    verbose: bool,
    num_workers: int,
    chunk_multiple: int,
) -> RunSummary:
    """Execute simulations for a fixed index set and persist outcomes.

    Args:
        store: Dataset store.
        simulation: Bound simulation implementation.
            Used directly when ``num_workers == 1``. When ``num_workers > 1``,
            worker processes reconstruct simulation instances from the
            persisted ``/simulation`` and ``/setup`` payloads instead of using
            parent-process runtime state.
        indices: Simulation indexes to run.
        allow_from_failed: Whether store writes may transition from ``FAILED``.
        verbose: If ``True``, print failure details.
        num_workers: Number of worker processes. ``1`` executes serially.
        chunk_multiple: Number of simulations assigned to each worker chunk.

    Returns:
        Summary counters for attempted/succeeded/failed simulations.
    """
    num_workers, chunk_multiple = _validate_parallel_options(num_workers, chunk_multiple)
    if indices.shape[0] == 0:
        return RunSummary(attempted=0, succeeded=0, failed=0)
    if num_workers == 1:
        return _run_simulations_serial(
            store,
            simulation,
            indices,
            allow_from_failed=allow_from_failed,
            verbose=verbose,
        )

    return _run_simulations_parallel(
        store,
        indices,
        allow_from_failed=allow_from_failed,
        verbose=verbose,
        num_workers=num_workers,
        chunk_multiple=chunk_multiple,
    )


# Execution entry points

def run_simulations_by_state(
    store: SimulationStore,
    simulation: Simulation,
    state: SimulationState | int,
    *,
    indexes: list[int] | None = None,
    verbose: bool = False,
    num_workers: int = 1,
    chunk_multiple: int = 10,
) -> RunSummary:
    """Run simulations from a selected source state.

    Args:
        store: Dataset store.
        simulation: Bound simulation implementation.
            Used directly when ``num_workers == 1``. When ``num_workers > 1``,
            worker processes reconstruct simulation instances from the
            persisted ``/simulation`` and ``/setup`` payloads instead of using
            parent-process runtime state.
        state: Source state to run from.
            Supported values are ``SimulationState.PENDING`` and
            ``SimulationState.FAILED``.
        indexes: Optional subset of simulation indexes to run.
        verbose: If ``True``, print failure messages.
        num_workers: Number of worker processes. ``1`` preserves serial
            execution.
        chunk_multiple: Number of simulations assigned to each worker chunk
            when ``num_workers > 1``.

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
    num_workers, chunk_multiple = _validate_parallel_options(num_workers, chunk_multiple)

    candidate_indices = _filter_execution_indices(store, store.indices_with_state(state_value), indexes)
    allow_from_failed = state_value == SimulationState.FAILED
    return _run_simulations_for_indices(
        store,
        simulation,
        candidate_indices,
        allow_from_failed=allow_from_failed,
        verbose=verbose,
        num_workers=num_workers,
        chunk_multiple=chunk_multiple,
    )


def run_pending_simulations(
    store: SimulationStore,
    simulation: Simulation,
    *,
    indexes: list[int] | None = None,
    verbose: bool = False,
    num_workers: int = 1,
    chunk_multiple: int = 10,
) -> RunSummary:
    """Run simulations currently in ``PENDING`` state.

    Args:
        store: Dataset store.
        simulation: Bound simulation implementation.
        indexes: Optional subset of simulation indexes to run.
        verbose: If ``True``, print failure messages.
        num_workers: Number of worker processes. ``1`` preserves serial
            execution.
        chunk_multiple: Number of simulations assigned to each worker chunk
            when ``num_workers > 1``.

    Returns:
        Execution counters for attempted/succeeded/failed simulations.
    """
    return run_simulations_by_state(
        store,
        simulation,
        SimulationState.PENDING,
        verbose=verbose,
        indexes=indexes,
        num_workers=num_workers,
        chunk_multiple=chunk_multiple,
    )


def run_failed_simulations(
    store: SimulationStore,
    simulation: Simulation,
    *,
    indexes: list[int] | None = None,
    verbose: bool = False,
    num_workers: int = 1,
    chunk_multiple: int = 10,
) -> RunSummary:
    """Run simulations currently in ``FAILED`` state.

    Args:
        store: Dataset store.
        simulation: Bound simulation implementation.
        indexes: Optional subset of simulation indexes to run.
        verbose: If ``True``, print failure messages.
        num_workers: Number of worker processes. ``1`` preserves serial
            execution.
        chunk_multiple: Number of simulations assigned to each worker chunk
            when ``num_workers > 1``.

    Returns:
        Execution counters for attempted/succeeded/failed simulations.
    """
    return run_simulations_by_state(
        store,
        simulation,
        SimulationState.FAILED,
        verbose=verbose,
        indexes=indexes,
        num_workers=num_workers,
        chunk_multiple=chunk_multiple,
    )
