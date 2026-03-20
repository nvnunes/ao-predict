"""Immutable in-memory analysis dataset and simulation views."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._immutability import freeze_array, freeze_mapping


_PSF_UNSET = object()


@dataclass(frozen=True)
class AnalysisSimulation:
    """Immutable per-simulation analysis view.

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
class AnalysisDataset:
    """Immutable dataset-level owner of loaded analysis payloads."""

    path: Path
    simulation_payload: Mapping[str, Any]
    setup: Mapping[str, Any]
    options_rows: tuple[Mapping[str, Any], ...]
    meta_rows: tuple[Mapping[str, Any], ...]
    stats_rows: tuple[Mapping[str, Any], ...]
    extra_stat_names: tuple[str, ...]
    _psf_loaders: tuple[Callable[[], np.ndarray] | None, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        num_sims = len(self.options_rows)
        if len(self.meta_rows) != num_sims or len(self.stats_rows) != num_sims:
            raise ValueError("AnalysisDataset rows must have matching per-simulation lengths.")
        if len(self._psf_loaders) != num_sims:
            raise ValueError("AnalysisDataset PSF loaders must match dataset size.")

    def __len__(self) -> int:
        return len(self.options_rows)

    def sim(self, sim_idx: int) -> AnalysisSimulation:
        """Return one zero-based immutable simulation view."""
        idx = int(sim_idx)
        if idx < 0:
            raise IndexError(f"simulation index must be >= 0, got {idx}.")
        if idx >= len(self):
            raise IndexError(f"simulation index {idx} out of range for dataset of size {len(self)}.")

        return AnalysisSimulation(
            _config=freeze_mapping(
                {
                    "setup": self.setup,
                    "options": self.options_rows[idx],
                }
            ),
            _meta=self.meta_rows[idx],
            _stats=self.stats_rows[idx],
            _psf_loader=self._psf_loaders[idx],
        )
