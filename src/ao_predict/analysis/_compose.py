"""Internal composition helpers for analysis dataset loading."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..persistence import SimulationStore
from .types import (
    AnalysisDataset,
    AnalysisDatasetLoadPayload,
    AnalysisLoadContext,
    AnalysisLoadContribution,
    AnalysisSimulation,
)


AnalysisLoadExtractor = Callable[[AnalysisLoadContext], AnalysisLoadContribution]


def _build_psf_loader(ctx: AnalysisLoadContext, sim_idx: int) -> Callable[[], np.ndarray]:
    """Bind one context-backed PSF reader for a simulation index."""

    def _load_psfs() -> np.ndarray:
        try:
            return ctx.read_sim_array("/psfs/data", sim_idx)
        except ValueError as exc:
            if "Missing required dataset '/psfs/data'." in str(exc):
                raise ValueError("PSFs are not available in this dataset.") from None
            raise

    return _load_psfs


def _build_core_contribution(ctx: AnalysisLoadContext) -> AnalysisLoadContribution:
    """Describe upstream built-in lazy analysis fields through the generic model."""
    return AnalysisLoadContribution(
        simulation_lazy_fields=tuple(
            {"psfs": _build_psf_loader(ctx, sim_idx)}
            for sim_idx in range(ctx.num_sims)
        )
    )


def _merge_load_contributions(
    contributions: Iterable[AnalysisLoadContribution],
    *,
    num_sims: int,
) -> AnalysisLoadContribution:
    """Merge eager/lazy load contributions into one stable combined payload."""
    dataset_fields: dict[str, Any] = {}
    dataset_lazy_fields: dict[str, Callable[[], Any]] = {}
    simulation_fields = [{} for _ in range(num_sims)]
    simulation_lazy_fields = [{} for _ in range(num_sims)]

    for contribution in contributions:
        dataset_fields.update(dict(contribution.dataset_fields))
        dataset_lazy_fields.update(dict(contribution.dataset_lazy_fields))
        _merge_simulation_rows(simulation_fields, contribution.simulation_fields, num_sims=num_sims)
        _merge_simulation_rows(simulation_lazy_fields, contribution.simulation_lazy_fields, num_sims=num_sims)

    return AnalysisLoadContribution(
        dataset_fields=dataset_fields,
        dataset_lazy_fields=dataset_lazy_fields,
        simulation_fields=tuple(simulation_fields),
        simulation_lazy_fields=tuple(simulation_lazy_fields),
    )


def _merge_simulation_rows(
    target_rows: list[dict[str, Any]],
    incoming_rows: tuple[Mapping[str, Any], ...],
    *,
    num_sims: int,
) -> None:
    """Merge one simulation-row contribution family with shared length validation."""
    if not incoming_rows:
        return
    if len(incoming_rows) != num_sims:
        raise ValueError(f"Analysis load contribution rows must match dataset size {num_sims}.")
    for sim_idx, row in enumerate(incoming_rows):
        target_rows[sim_idx].update(dict(row))


def _load_analysis_dataset_from_store(
    store: SimulationStore,
    *,
    dataset_cls: type[AnalysisDataset] = AnalysisDataset,
    simulation_cls: type[AnalysisSimulation] = AnalysisSimulation,
    extra_field_extractors: Iterable[AnalysisLoadExtractor] | None = None,
) -> AnalysisDataset:
    """Build an immutable analysis dataset from a validated simulation store."""
    num_sims = store.num_sims()
    ctx = AnalysisLoadContext(path=store.path, num_sims=num_sims)
    contributions = [_build_core_contribution(ctx)]
    for extractor in extra_field_extractors or ():
        contributions.append(extractor(ctx))
    merged = _merge_load_contributions(contributions, num_sims=num_sims)

    payload = AnalysisDatasetLoadPayload(
        path=store.path,
        simulation_payload=store.read_simulation(),
        setup=store.read_setup(),
        options=store.read_options(),
        meta=store.read_analysis_meta(),
        stats=store.read_analysis_stats(),
        extra_stat_names=store.read_extra_stat_names(),
        dataset_extra_fields=merged.dataset_fields,
        dataset_extra_lazy_fields=merged.dataset_lazy_fields,
        simulation_extra_fields=merged.simulation_fields,
        simulation_extra_lazy_fields=merged.simulation_lazy_fields,
    )
    return dataset_cls.from_load_payload(payload, simulation_cls=simulation_cls)


def load_analysis_dataset(
    dataset_path: str | Path,
    *,
    dataset_cls: type[AnalysisDataset] = AnalysisDataset,
    simulation_cls: type[AnalysisSimulation] = AnalysisSimulation,
    extra_field_extractors: Iterable[AnalysisLoadExtractor] | None = None,
) -> AnalysisDataset:
    """Load an immutable analysis dataset from a dataset file path.

    When no optional hooks are passed, this returns the standard
    :class:`AnalysisDataset` and ``dataset.sim(i)`` returns the standard
    :class:`AnalysisSimulation`.

    Optional subclasses and extractors let downstream packages extend the
    generic loaded analysis surface without reimplementing store reads or
    exposing HDF5 handles on the public loaded objects.
    """
    store = SimulationStore(dataset_path)
    store.validate_schema()
    return _load_analysis_dataset_from_store(
        store,
        dataset_cls=dataset_cls,
        simulation_cls=simulation_cls,
        extra_field_extractors=extra_field_extractors,
    )
