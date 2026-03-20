"""Internal composition helpers for analysis dataset loading."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..persistence import SimulationStore
from ._immutability import freeze_mapping
from .types import AnalysisDataset


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


def load_analysis_dataset(store: SimulationStore) -> AnalysisDataset:
    """Load an immutable analysis dataset from a validated store."""
    store.validate_schema()

    num_sims = store.num_sims()
    simulation_payload = freeze_mapping(store.read_simulation())
    setup = freeze_mapping(store.read_setup())
    extra_stat_names = store.read_extra_stat_names()
    options_rows = tuple(freeze_mapping(store.read_sim_options(sim_idx)) for sim_idx in range(num_sims))
    meta_rows = tuple(freeze_mapping(store.read_simulation_meta(sim_idx)) for sim_idx in range(num_sims))
    stats_rows = tuple(freeze_mapping(store.read_simulation_stats(sim_idx)) for sim_idx in range(num_sims))
    psf_loaders = tuple(_build_psf_loader(store, sim_idx) for sim_idx in range(num_sims))

    return AnalysisDataset(
        path=store.path,
        simulation_payload=simulation_payload,
        setup=setup,
        options_rows=options_rows,
        meta_rows=meta_rows,
        stats_rows=stats_rows,
        extra_stat_names=extra_stat_names,
        _psf_loaders=psf_loaders,
    )
