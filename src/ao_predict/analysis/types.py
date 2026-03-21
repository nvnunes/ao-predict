"""Immutable in-memory analysis dataset and simulation views."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from ._immutability import freeze_array, freeze_mapping


_PSF_UNSET = object()
AnalysisSimulationT = TypeVar("AnalysisSimulationT", bound="AnalysisSimulation")
AnalysisDatasetT = TypeVar("AnalysisDatasetT", bound="AnalysisDataset")


@dataclass(frozen=True)
class AnalysisSimulationLoadContext:
    """Generic lazy-load callbacks bound for one loaded simulation row.

    This context keeps analysis objects HDF5-agnostic while allowing loader
    subclasses to attach additional lazy readers during dataset load.

    Public fields are exposed as read-only properties:
    - ``psf_loader``: lazy loader for persisted PSFs, or ``None`` when absent
    - ``extra_loaders``: immutable mapping of additional lazy loaders keyed by
      generic dataset/feature names chosen by the loader implementation
    """

    psf_loader: Callable[[], np.ndarray] | None = field(default=None, repr=False, compare=False)
    extra_loaders: Mapping[str, Callable[[], Any]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_loaders", freeze_mapping(dict(self.extra_loaders)))


@dataclass(frozen=True)
class AnalysisSimulationLoadPayload:
    """Frozen generic payload for constructing one loaded analysis simulation."""

    config: Mapping[str, Any]
    meta: Mapping[str, Any]
    stats: Mapping[str, Any]
    context: AnalysisSimulationLoadContext = field(repr=False, compare=False)


@dataclass(frozen=True)
class AnalysisSimulation:
    """Immutable per-simulation analysis view.

    Subclasses may override :meth:`from_load_payload` to bind additional
    immutable state from the generic load payload/context while preserving the
    public analysis contract.

    Public fields are exposed as read-only properties:
    - ``config``: immutable mapping with exactly ``setup`` and ``options``
    - ``meta``: immutable mapping of persisted scientific metadata
    - ``stats``: immutable mapping of persisted stats
    - ``psfs``: lazily loaded immutable PSF cube
    """

    _config: Mapping[str, Any]
    _meta: Mapping[str, Any]
    _stats: Mapping[str, Any]
    _psf_loader: Callable[[], np.ndarray] | None = field(default=None, repr=False, compare=False)
    _psfs_cache: object = field(default=_PSF_UNSET, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if tuple(self._config.keys()) != ("setup", "options"):
            raise ValueError("AnalysisSimulation.config must contain exactly 'setup' and 'options'.")

    @classmethod
    def from_load_payload(
        cls: type[AnalysisSimulationT],
        payload: AnalysisSimulationLoadPayload,
    ) -> AnalysisSimulationT:
        """Build one immutable loaded simulation view from a generic row payload."""
        return cls(
            _config=payload.config,
            _meta=payload.meta,
            _stats=payload.stats,
            _psf_loader=payload.context.psf_loader,
        )

    @property
    def config(self) -> Mapping[str, Any]:
        return self._config

    @property
    def meta(self) -> Mapping[str, Any]:
        return self._meta

    @property
    def stats(self) -> Mapping[str, Any]:
        return self._stats

    @property
    def psfs(self) -> np.ndarray:
        if self._psfs_cache is _PSF_UNSET:
            if self._psf_loader is None:
                raise ValueError("PSFs are not available in this dataset.")
            object.__setattr__(self, "_psfs_cache", freeze_array(self._psf_loader()))
        return self._psfs_cache  # type: ignore[return-value]


@dataclass(frozen=True)
class AnalysisDatasetLoadPayload:
    """Frozen generic payload for constructing one loaded analysis dataset."""

    path: Path
    simulation_payload: Mapping[str, Any]
    setup: Mapping[str, Any]
    options_rows: tuple[Mapping[str, Any], ...]
    meta_rows: tuple[Mapping[str, Any], ...]
    stats_rows: tuple[Mapping[str, Any], ...]
    extra_stat_names: tuple[str, ...]
    simulation_contexts: tuple[AnalysisSimulationLoadContext, ...] = field(repr=False, compare=False)


@dataclass(frozen=True)
class AnalysisDataset:
    """Immutable dataset-level owner of loaded analysis payloads.

    The default dataset returned by :func:`load_analysis_dataset` stores the
    fully loaded generic analysis payload and builds immutable per-row
    simulations through a configured simulation class. Downstream subclasses may
    override :meth:`from_load_payload` to bind additional immutable state
    without reloading the dataset.
    """

    path: Path
    simulation_payload: Mapping[str, Any]
    setup: Mapping[str, Any]
    options_rows: tuple[Mapping[str, Any], ...]
    meta_rows: tuple[Mapping[str, Any], ...]
    stats_rows: tuple[Mapping[str, Any], ...]
    extra_stat_names: tuple[str, ...]
    _simulation_contexts: tuple[AnalysisSimulationLoadContext, ...] = field(repr=False, compare=False)
    _simulation_cls: type[AnalysisSimulation] = field(
        default=AnalysisSimulation,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_load_payload(
        cls: type[AnalysisDatasetT],
        payload: AnalysisDatasetLoadPayload,
        simulation_cls: type[AnalysisSimulation] = AnalysisSimulation,
    ) -> AnalysisDatasetT:
        """Build one immutable loaded dataset from a generic dataset payload."""
        return cls(
            path=payload.path,
            simulation_payload=payload.simulation_payload,
            setup=payload.setup,
            options_rows=payload.options_rows,
            meta_rows=payload.meta_rows,
            stats_rows=payload.stats_rows,
            extra_stat_names=payload.extra_stat_names,
            _simulation_contexts=payload.simulation_contexts,
            _simulation_cls=simulation_cls,
        )

    def __post_init__(self) -> None:
        num_sims = len(self.options_rows)
        if len(self.meta_rows) != num_sims or len(self.stats_rows) != num_sims:
            raise ValueError("AnalysisDataset rows must have matching per-simulation lengths.")
        if len(self._simulation_contexts) != num_sims:
            raise ValueError("AnalysisDataset simulation contexts must match dataset size.")

    def __len__(self) -> int:
        return len(self.options_rows)

    def sim(self, sim_idx: int) -> AnalysisSimulation:
        """Return one zero-based immutable simulation view."""
        idx = int(sim_idx)
        if idx < 0:
            raise IndexError(f"simulation index must be >= 0, got {idx}.")
        if idx >= len(self):
            raise IndexError(f"simulation index {idx} out of range for dataset of size {len(self)}.")

        return self._simulation_cls.from_load_payload(
            AnalysisSimulationLoadPayload(
                config=freeze_mapping(
                    {
                        "setup": self.setup,
                        "options": self.options_rows[idx],
                    }
                ),
                meta=self.meta_rows[idx],
                stats=self.stats_rows[idx],
                context=self._simulation_contexts[idx],
            )
        )
