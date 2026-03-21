"""Internal composition helpers for analysis dataset loading."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..persistence import SimulationStore
from ._immutability import freeze_mapping
from .types import (
    AnalysisDataset,
    AnalysisDatasetFactory,
    AnalysisDatasetLoadPayload,
    AnalysisSimulationFactory,
    AnalysisSimulationLoadContext,
    _build_default_analysis_simulation,
)


def _build_psf_loader(store: SimulationStore, sim_idx: int) -> Callable[[], np.ndarray]:
    """Bind one store-backed PSF reader for a simulation index."""

    def _load_psfs() -> np.ndarray:
        try:
            return store.read_simulation_psfs(sim_idx)
        except ValueError as exc:
            if "Missing required dataset '/psfs/data'." in str(exc):
                raise ValueError("PSFs are not available in this dataset.") from None
            raise

    return _load_psfs


def _build_simulation_load_context(
    store: SimulationStore,
    sim_idx: int,
) -> AnalysisSimulationLoadContext:
    """Build one generic lazy-load context for a simulation row."""
    return AnalysisSimulationLoadContext(psf_loader=_build_psf_loader(store, sim_idx))


def _build_default_analysis_dataset(
    payload: AnalysisDatasetLoadPayload,
    simulation_factory: AnalysisSimulationFactory,
) -> AnalysisDataset:
    """Build the default immutable analysis dataset from one loaded payload."""
    return AnalysisDataset(
        path=payload.path,
        simulation_payload=payload.simulation_payload,
        setup=payload.setup,
        options_rows=payload.options_rows,
        meta_rows=payload.meta_rows,
        stats_rows=payload.stats_rows,
        extra_stat_names=payload.extra_stat_names,
        _simulation_contexts=payload.simulation_contexts,
        _simulation_factory=simulation_factory,
    )


def _load_analysis_dataset_from_store(
    store: SimulationStore,
    *,
    dataset_factory: AnalysisDatasetFactory | None = None,
    simulation_factory: AnalysisSimulationFactory | None = None,
) -> AnalysisDataset:
    """Build an immutable analysis dataset from a validated simulation store."""
    num_sims = store.num_sims()
    chosen_simulation_factory = simulation_factory or _build_default_analysis_simulation
    chosen_dataset_factory = dataset_factory or _build_default_analysis_dataset
    payload = AnalysisDatasetLoadPayload(
        path=store.path,
        simulation_payload=freeze_mapping(store.read_simulation()),
        setup=freeze_mapping(store.read_setup()),
        options_rows=tuple(freeze_mapping(store.read_sim_options(sim_idx)) for sim_idx in range(num_sims)),
        meta_rows=tuple(freeze_mapping(store.read_simulation_meta(sim_idx)) for sim_idx in range(num_sims)),
        stats_rows=tuple(freeze_mapping(store.read_simulation_stats(sim_idx)) for sim_idx in range(num_sims)),
        extra_stat_names=store.read_extra_stat_names(),
        simulation_contexts=tuple(_build_simulation_load_context(store, sim_idx) for sim_idx in range(num_sims)),
    )
    return chosen_dataset_factory(payload, chosen_simulation_factory)


def load_analysis_dataset(
    dataset_path: str | Path,
    *,
    dataset_factory: AnalysisDatasetFactory | None = None,
    simulation_factory: AnalysisSimulationFactory | None = None,
) -> AnalysisDataset:
    """Load an immutable analysis dataset from a dataset file path.

    When no factories are passed, this returns the standard
    :class:`AnalysisDataset` and ``dataset.sim(i)`` returns the standard
    :class:`AnalysisSimulation`.

    Optional factories let downstream packages wrap the loaded generic analysis
    payload in custom dataset and simulation types without reimplementing store
    reads or exposing HDF5 handles on the public loaded objects.
    """
    store = SimulationStore(dataset_path)
    store.validate_schema()
    return _load_analysis_dataset_from_store(
        store,
        dataset_factory=dataset_factory,
        simulation_factory=simulation_factory,
    )
