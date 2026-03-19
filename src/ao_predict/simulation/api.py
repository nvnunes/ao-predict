"""Code-first simulation lifecycle API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from ..persistence import SimulationStore
from .config import (
    normalize_setup_config,
    normalize_simulation_config,
    prepare_options_payload_from_arrays,
    prepare_options_payload_from_table,
)
from .interfaces import Simulation, SimulationState
from . import schema
from . import runner
from .runner import (
    RunSummary,
    create_simulation_from_config,
    create_simulation_from_payload,
    prepare_setup_payload,
)


# Public API type aliases

ConfigMapping = Mapping[str, object]
OptionArrayLike = np.ndarray | list[object] | tuple[object, ...]
OptionArrayMapping = Mapping[str, OptionArrayLike]


# Public API data structures

@dataclass(frozen=True)
class SimulationConfig:
    """Typed simulation configuration for code-driven initialization.

    Attributes:
        name: Simulation class identifier (short name or canonical class path).
        base_path: Optional base directory used by simulations to resolve
            relative paths in ``specific_fields``.
        specific_fields: Additional simulation-specific passthrough fields.
    """

    name: str
    base_path: str | None = None
    specific_fields: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SetupConfig:
    """Typed setup configuration for code-driven initialization.

    Attributes:
        ee_apertures_mas: Core EE aperture list (mas).
        sr_method: Optional dataset-level Strehl selector. Defaults to
            ``pixel_fit`` when omitted.
        fwhm_summary: Optional dataset-level FWHM contour summary selector.
            Defaults to ``geom`` when omitted.
        specific_fields: Additional simulation-specific passthrough fields.
    """

    ee_apertures_mas: list[float]
    sr_method: str | None = None
    fwhm_summary: str | None = None
    specific_fields: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OptionsConfig:
    """Typed columnar per-simulation options.

    Attributes:
        option_arrays: Mapping from option name to per-simulation arrays.
    """

    option_arrays: dict[str, OptionArrayLike]


@dataclass(frozen=True)
class TableOptionsConfig:
    """Typed table/broadcast options input for dataset initialization.

    Attributes:
        broadcast: Scalar/default option values applied to all simulations.
        columns: Optional table column names.
        rows: Optional table rows (each row corresponds to one simulation).
    """

    broadcast: dict[str, object] = field(default_factory=dict)
    columns: list[str] | None = None
    rows: list[list[object]] | None = None


@dataclass(frozen=True)
class InitDatasetRequest:
    """Request payload for dataset initialization via code.

    Attributes:
        dataset_path: Output dataset path.
        simulation: Simulation config (typed or mapping form).
        setup: Setup config (typed or mapping form).
        options: Options input (columnar arrays or config-style table/broadcast).
        overwrite: Whether to overwrite an existing dataset file.
        save_psfs: Whether to allocate and persist ``/psfs/data``.
    """

    dataset_path: str | Path
    simulation: SimulationConfig | ConfigMapping
    setup: SetupConfig | ConfigMapping
    options: OptionsConfig | TableOptionsConfig | OptionArrayMapping
    overwrite: bool = False
    save_psfs: bool = False


@dataclass(frozen=True)
class DatasetStatus:
    """Dataset validation and completion status.

    Attributes:
        dataset_path: Dataset path inspected.
        num_sims: Total number of simulations in the dataset.
        num_pending: Number of simulations in ``SimulationState.PENDING``.
        num_failed: Number of simulations in ``SimulationState.FAILED``.
        num_succeeded: Number of simulations in ``SimulationState.SUCCEEDED``.
        ok: True when schema is valid and there are no pending/failed simulations.
        issues: Human-readable issue list when ``ok`` is false.
    """

    dataset_path: Path
    num_sims: int
    num_pending: int
    num_failed: int
    num_succeeded: int
    ok: bool
    issues: list[str]


class DatasetValidationError(ValueError):
    """Raised when dataset validation fails in strict mode."""

    def __init__(self, issues: list[str]):
        self.issues = list(issues)
        message = "Dataset validation failed:\n- " + "\n- ".join(self.issues)
        super().__init__(message)


# Payload helpers

def _prepare_simulation_payload(simulation_cfg: ConfigMapping) -> tuple[Simulation, dict[str, object]]:
    """Prepare the simulation implementation and persisted ``/simulation`` payload.

    Args:
        simulation_cfg: Normalized simulation configuration mapping.

    Returns:
        Tuple ``(simulation, simulation_payload)`` ready for dataset creation.

    Notes:
        The core ``/simulation`` contract fields are assembled by ao-predict.
        The simulation hook completes that base payload with
        simulation-specific persisted fields.
    """
    return create_simulation_from_config(simulation_cfg)


def _prepare_setup_payload(simulation: Simulation, setup_cfg: ConfigMapping) -> dict[str, object]:
    """Prepare the persisted ``/setup`` payload.

    Args:
        simulation: Simulation implementation used for setup completion.
        setup_cfg: Normalized setup configuration mapping.

    Returns:
        Persisted setup payload ready for dataset creation.
    """
    return prepare_setup_payload(simulation, setup_cfg)


def _prepare_options_payload(
    simulation: Simulation,
    setup_payload: ConfigMapping,
    options: OptionsConfig | TableOptionsConfig | OptionArrayMapping,
) -> dict[str, np.ndarray]:
    """Prepare options payload for dataset initialization.

    Supports typed ``OptionsConfig``, typed ``TableOptionsConfig``, and
    direct columnar mappings.
    """
    if isinstance(options, OptionsConfig):
        return prepare_options_payload_from_arrays(
            simulation,
            setup_payload,
            options.option_arrays,
        )
    if isinstance(options, TableOptionsConfig):
        return prepare_options_payload_from_table(
            simulation,
            setup_payload,
            {
                schema.KEY_CFG_OPTION_BROADCAST: dict(options.broadcast),
                schema.KEY_CFG_OPTION_COLUMNS: options.columns,
                schema.KEY_CFG_OPTION_ROWS: options.rows,
            },
        )

    raw = {str(k): v for k, v in dict(options).items()}
    return prepare_options_payload_from_arrays(simulation, setup_payload, raw)


# Dataset helpers

def _load_dataset(dataset_path: str | Path) -> tuple[SimulationStore, Simulation]:
    """Load, validate, and bind simulation/setup from an existing dataset.

    Args:
        dataset_path: HDF5 dataset path.

    Returns:
        Tuple ``(store, simulation)`` where simulation has setup loaded.

    Raises:
        ValueError: If dataset schema or simulation payload is invalid.
    """
    store = SimulationStore(dataset_path)
    store.validate_schema()
    simulation_payload = store.read_simulation()
    simulation = create_simulation_from_payload(simulation_payload)
    setup_payload = store.read_setup()
    simulation.load_setup_payload(setup_payload)
    return store, simulation


def _collect_dataset_status(dataset_path: str | Path) -> DatasetStatus:
    """Collect dataset schema/state issues into a ``DatasetStatus`` object.

    Args:
        dataset_path: Dataset path to inspect.

    Returns:
        Aggregated dataset status including schema and completion issues.
    """
    store = SimulationStore(dataset_path)
    schema_issues = store.collect_schema_issues()
    issues: list[str] = [f"Schema validation failed: {msg}" for msg in schema_issues]

    num_pending = 0
    num_failed = 0
    total = 0
    try:
        num_pending = int(store.pending_indices().shape[0])
        num_failed = int(store.failed_indices().shape[0])
        total = int(store.num_sims())
    except Exception as exc:
        issues.append(f"Status read failed: {exc}")

    if num_pending > 0:
        issues.append(
            f"{num_pending} simulation(s) are still {SimulationState.PENDING.name.lower()} "
            f"(state={int(SimulationState.PENDING)})."
        )
    if num_failed > 0:
        issues.append(
            f"{num_failed} simulation(s) are {SimulationState.FAILED.name.lower()} "
            f"(state={int(SimulationState.FAILED)})."
        )

    num_succeeded = max(total - num_pending - num_failed, 0)
    ok = len(issues) == 0
    return DatasetStatus(
        dataset_path=store.path,
        num_sims=total,
        num_pending=num_pending,
        num_failed=num_failed,
        num_succeeded=num_succeeded,
        ok=ok,
        issues=issues,
    )


# Public API functions

def init_dataset(request: InitDatasetRequest) -> int:
    """Initialize a simulation dataset from code-provided configuration.

    Args:
        request: Initialization request payload.

    Returns:
        Number of simulations initialized in the dataset.

    Raises:
        ValueError: If simulation/setup/options payloads are invalid.
        FileExistsError: If dataset exists and ``overwrite`` is false.
    """
    dataset_path = Path(request.dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    simulation_cfg = normalize_simulation_config(request.simulation)
    setup_cfg = normalize_setup_config(request.setup)

    simulation, simulation_payload = _prepare_simulation_payload(simulation_cfg)
    setup_payload = _prepare_setup_payload(simulation, setup_cfg)
    options_payload = _prepare_options_payload(simulation, setup_payload, request.options)

    store = SimulationStore(dataset_path)
    store.create(
        simulation_payload,
        setup_payload,
        options_payload,
        overwrite=bool(request.overwrite),
        save_psfs=bool(request.save_psfs),
    )
    return int(store.num_sims())


def run_simulations_by_state(
    dataset_path: str | Path,
    *,
    state: SimulationState | int = SimulationState.PENDING,
    indexes: list[int] | None = None,
    verbose: bool = False,
) -> RunSummary:
    """Run simulations for a selected source state.

    Args:
        dataset_path: Dataset path.
        state: Source state to execute. Supported values:
            - ``SimulationState.PENDING``: run pending simulations
            - ``SimulationState.FAILED``: retry failed simulations
        indexes: Optional subset of 0-based indexes to consider.
        verbose: Print per-simulation failure details.

    Returns:
        Runner execution summary.

    Raises:
        ValueError: If ``state`` is unsupported.
    """
    store, simulation = _load_dataset(dataset_path)
    return runner.run_simulations_by_state(
        store,
        simulation,
        state=state,
        indexes=indexes,
        verbose=verbose,
    )


def reset_simulations(dataset_path: str | Path, indexes: list[int] | None = None) -> int:
    """Reset selected simulations to pending state.

    Args:
        dataset_path: Dataset path.
        indexes: Optional subset of 0-based indexes to reset.

    Returns:
        Number of simulations whose state changed.

    Raises:
        ValueError: If dataset schema is invalid or indexes are invalid.

    Notes:
        Validates dataset schema before mutating ``/status/state`` so malformed
        datasets fail fast with clear validation errors.
    """
    store = SimulationStore(dataset_path)
    store.validate_schema()
    return int(store.reset_to_pending(indexes=indexes))


def check_dataset(dataset_path: str | Path) -> DatasetStatus:
    """Validate schema and completion status for a dataset.

    Args:
        dataset_path: Dataset path.

    Returns:
        Dataset status object that aggregates schema and state checks.
    """
    return _collect_dataset_status(dataset_path)


def validate_dataset(dataset_path: str | Path) -> None:
    """Validate schema and completion status, raising on failure.

    Args:
        dataset_path: Dataset path.

    Raises:
        DatasetValidationError: If schema/state checks fail.
    """
    status = _collect_dataset_status(dataset_path)
    if not status.ok:
        raise DatasetValidationError(status.issues)
