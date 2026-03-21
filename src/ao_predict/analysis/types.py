"""Immutable in-memory analysis dataset and simulation views."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import h5py
import numpy as np

from ._immutability import freeze_array, freeze_mapping


# Constants

_PSFS_FIELD = "psfs"
_MISSING = object()
AnalysisSimulationT = TypeVar("AnalysisSimulationT", bound="AnalysisSimulation")
AnalysisDatasetT = TypeVar("AnalysisDatasetT", bound="AnalysisDataset")
LazyFieldLoader = Callable[[], Any]


# Data structures

@dataclass(frozen=True)
class AnalysisLoadContext:
    """Generic persisted-analysis reader passed to load extractors.

    The context exposes path-based read helpers without leaking an open HDF5
    file handle onto the public loaded analysis objects. Lazy extractors may
    safely capture this context because each read method opens the dataset only
    for the duration of the call.
    """

    path: Path
    num_sims: int

    def has_path(self, path: str) -> bool:
        """Return whether the persisted dataset contains ``path``."""
        with h5py.File(self.path, "r") as f:
            return path in f

    def read_dataset_value(self, path: str) -> Any:
        """Read any persisted node at ``path`` into plain Python objects."""
        with h5py.File(self.path, "r") as f:
            if path not in f:
                raise ValueError(f"Missing required dataset '{path}'.")
            return _read_node(f[path])

    def read_sim_value(self, path: str, sim_index: int) -> Any:
        """Read one per-simulation row from a dataset at ``path``."""
        with h5py.File(self.path, "r") as f:
            ds = _require_dataset(f, path)
            return _read_simulation_dataset_row(ds, sim_index, path=path)

    def read_array(self, path: str) -> np.ndarray:
        """Read a dataset at ``path`` as a standalone NumPy array."""
        with h5py.File(self.path, "r") as f:
            ds = _require_dataset(f, path)
            return np.asarray(ds[...]).copy()

    def read_sim_array(self, path: str, sim_index: int) -> np.ndarray:
        """Read one per-simulation array row from a dataset at ``path``."""
        value = self.read_sim_value(path, sim_index)
        return np.asarray(value).copy()


@dataclass(frozen=True)
class AnalysisLoadContribution:
    """Structured eager/lazy additions contributed during analysis loading."""

    dataset_fields: Mapping[str, Any] = field(default_factory=dict)
    dataset_lazy_fields: Mapping[str, LazyFieldLoader] = field(default_factory=dict, repr=False, compare=False)
    simulation_fields: tuple[Mapping[str, Any], ...] = ()
    simulation_lazy_fields: tuple[Mapping[str, LazyFieldLoader], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_fields", freeze_mapping(dict(self.dataset_fields)))
        object.__setattr__(self, "dataset_lazy_fields", freeze_mapping(dict(self.dataset_lazy_fields)))
        object.__setattr__(self, "simulation_fields", tuple(freeze_mapping(dict(row)) for row in self.simulation_fields))
        object.__setattr__(
            self,
            "simulation_lazy_fields",
            tuple(freeze_mapping(dict(row)) for row in self.simulation_lazy_fields),
        )


@dataclass(frozen=True)
class AnalysisSimulationLoadPayload:
    """Frozen generic payload for constructing one loaded analysis simulation."""

    config: Mapping[str, Any]
    meta: Mapping[str, Any]
    stats: Mapping[str, Any]
    extra_fields: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    extra_lazy_fields: Mapping[str, LazyFieldLoader] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_fields", freeze_mapping(dict(self.extra_fields)))
        object.__setattr__(self, "extra_lazy_fields", freeze_mapping(dict(self.extra_lazy_fields)))


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
    dataset_extra_fields: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    dataset_extra_lazy_fields: Mapping[str, LazyFieldLoader] = field(default_factory=dict, repr=False, compare=False)
    simulation_extra_fields: tuple[Mapping[str, Any], ...] = field(default=(), repr=False, compare=False)
    simulation_extra_lazy_fields: tuple[Mapping[str, LazyFieldLoader], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_extra_fields", freeze_mapping(dict(self.dataset_extra_fields)))
        object.__setattr__(
            self,
            "dataset_extra_lazy_fields",
            freeze_mapping(dict(self.dataset_extra_lazy_fields)),
        )
        object.__setattr__(
            self,
            "simulation_extra_fields",
            tuple(freeze_mapping(dict(row)) for row in self.simulation_extra_fields),
        )
        object.__setattr__(
            self,
            "simulation_extra_lazy_fields",
            tuple(freeze_mapping(dict(row)) for row in self.simulation_extra_lazy_fields),
        )


@dataclass(frozen=True)
class AnalysisSimulation:
    """Immutable per-simulation analysis view.

    Subclasses may expose semantic properties backed by
    :meth:`_require_extra_field` without reimplementing eager/lazy loader and
    cache plumbing. The built-in ``psfs`` property uses the same mechanism.

    Public fields are exposed as read-only properties:
    - ``config``: immutable mapping with exactly ``setup`` and ``options``
    - ``meta``: immutable mapping of persisted scientific metadata
    - ``stats``: immutable mapping of persisted stats
    - ``psfs``: lazily loaded immutable PSF cube
    """

    _config: Mapping[str, Any]
    _meta: Mapping[str, Any]
    _stats: Mapping[str, Any]
    _extra_fields: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _extra_lazy_fields: Mapping[str, LazyFieldLoader] = field(default_factory=dict, repr=False, compare=False)
    _extra_field_cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if tuple(self._config.keys()) != ("setup", "options"):
            raise ValueError("AnalysisSimulation.config must contain exactly 'setup' and 'options'.")
        object.__setattr__(self, "_extra_fields", freeze_mapping(dict(self._extra_fields)))
        object.__setattr__(self, "_extra_lazy_fields", freeze_mapping(dict(self._extra_lazy_fields)))

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
            _extra_fields=payload.extra_fields,
            _extra_lazy_fields=payload.extra_lazy_fields,
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

    def _has_config_field(self, name: str, *, section: str = "setup") -> bool:
        """Return whether one config section contains ``name``."""
        section_mapping = self._config.get(section)
        return isinstance(section_mapping, Mapping) and name in section_mapping

    def _get_config_field(self, name: str, *, section: str = "setup", default: Any = _MISSING) -> Any:
        """Return one config field or ``default`` when provided."""
        section_mapping = self._config.get(section)
        if isinstance(section_mapping, Mapping) and name in section_mapping:
            return section_mapping[name]
        if default is not _MISSING:
            return default
        raise ValueError(f"Missing required config field '{name}' in section '{section}'.")

    def _require_config_field(self, name: str, *, section: str = "setup") -> Any:
        """Return one required config field from ``section``."""
        return self._get_config_field(name, section=section)

    def _has_meta_field(self, name: str) -> bool:
        """Return whether persisted meta contains ``name``."""
        return name in self._meta

    def _get_meta_field(self, name: str, default: Any = _MISSING) -> Any:
        """Return one meta field or ``default`` when provided."""
        if name in self._meta:
            return self._meta[name]
        if default is not _MISSING:
            return default
        raise ValueError(f"Missing required meta field '{name}'.")

    def _require_meta_field(self, name: str) -> Any:
        """Return one required meta field."""
        return self._get_meta_field(name)

    def _get_persisted_field(
        self,
        name: str,
        *,
        setup_first: bool = True,
        default: Any = _MISSING,
    ) -> Any:
        """Return one persisted field from setup/meta using the standard lookup order."""
        if setup_first:
            if self._has_config_field(name, section="setup"):
                return self._require_config_field(name, section="setup")
            if self._has_meta_field(name):
                return self._require_meta_field(name)
        else:
            if self._has_meta_field(name):
                return self._require_meta_field(name)
            if self._has_config_field(name, section="setup"):
                return self._require_config_field(name, section="setup")
        if default is not _MISSING:
            return default
        raise ValueError(f"Missing required persisted field '{name}'.")

    def _require_persisted_field(self, name: str, *, setup_first: bool = True) -> Any:
        """Return one required persisted field using the standard lookup order."""
        return self._get_persisted_field(name, setup_first=setup_first)

    def _require_persisted_string_field(
        self,
        name: str,
        *,
        setup_first: bool = True,
        normalize: bool = False,
    ) -> str:
        """Return one required persisted string field with optional normalization."""
        value = self._require_persisted_field(name, setup_first=setup_first)
        if not isinstance(value, str):
            raise TypeError(f"Persisted field '{name}' must be a string.")
        if not normalize:
            return value
        value = value.strip().lower()
        if not value:
            raise ValueError(f"Persisted field '{name}' must be a non-empty string.")
        return value

    def _has_extra_field(self, name: str) -> bool:
        """Return whether a generic eager or lazy extra field is available."""
        return name in self._extra_fields or name in self._extra_lazy_fields

    def _require_extra_field(self, name: str) -> Any:
        """Return one extra field, loading and caching it on first access."""
        if name in self._extra_fields:
            return self._extra_fields[name]
        if name in self._extra_field_cache:
            return self._extra_field_cache[name]
        if name not in self._extra_lazy_fields:
            raise ValueError(f"Extra field '{name}' is not available in this analysis simulation.")
        value = _freeze_loaded_value(self._extra_lazy_fields[name]())
        self._extra_field_cache[name] = value
        return value

    @property
    def psfs(self) -> np.ndarray:
        try:
            return self._require_extra_field(_PSFS_FIELD)
        except ValueError as exc:
            if "Extra field 'psfs'" in str(exc):
                raise ValueError("PSFs are not available in this dataset.") from None
            raise


@dataclass(frozen=True)
class AnalysisDataset:
    """Immutable dataset-level owner of loaded analysis payloads.

    The default dataset returned by :func:`load_analysis_dataset` stores the
    loaded generic analysis payload and per-simulation extra field definitions.
    Downstream subclasses may override :meth:`from_load_payload` and expose
    semantic properties backed by the generic dataset/simulation field storage.
    """

    path: Path
    simulation_payload: Mapping[str, Any]
    setup: Mapping[str, Any]
    options_rows: tuple[Mapping[str, Any], ...]
    meta_rows: tuple[Mapping[str, Any], ...]
    stats_rows: tuple[Mapping[str, Any], ...]
    extra_stat_names: tuple[str, ...]
    _dataset_extra_fields: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _dataset_extra_lazy_fields: Mapping[str, LazyFieldLoader] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _simulation_extra_fields: tuple[Mapping[str, Any], ...] = field(default=(), repr=False, compare=False)
    _simulation_extra_lazy_fields: tuple[Mapping[str, LazyFieldLoader], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    _dataset_extra_field_cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    _simulation_cls: type[AnalysisSimulation] = field(default=AnalysisSimulation, repr=False, compare=False)

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
            _dataset_extra_fields=payload.dataset_extra_fields,
            _dataset_extra_lazy_fields=payload.dataset_extra_lazy_fields,
            _simulation_extra_fields=payload.simulation_extra_fields,
            _simulation_extra_lazy_fields=payload.simulation_extra_lazy_fields,
            _simulation_cls=simulation_cls,
        )

    def __post_init__(self) -> None:
        num_sims = len(self.options_rows)
        if len(self.meta_rows) != num_sims or len(self.stats_rows) != num_sims:
            raise ValueError("AnalysisDataset rows must have matching per-simulation lengths.")
        if self._simulation_extra_fields and len(self._simulation_extra_fields) != num_sims:
            raise ValueError("AnalysisDataset simulation extra fields must match dataset size.")
        if self._simulation_extra_lazy_fields and len(self._simulation_extra_lazy_fields) != num_sims:
            raise ValueError("AnalysisDataset simulation lazy fields must match dataset size.")
        object.__setattr__(self, "_dataset_extra_fields", freeze_mapping(dict(self._dataset_extra_fields)))
        object.__setattr__(
            self,
            "_dataset_extra_lazy_fields",
            freeze_mapping(dict(self._dataset_extra_lazy_fields)),
        )
        object.__setattr__(
            self,
            "_simulation_extra_fields",
            tuple(freeze_mapping(dict(row)) for row in self._simulation_extra_fields)
            if self._simulation_extra_fields
            else tuple(freeze_mapping({}) for _ in range(num_sims)),
        )
        object.__setattr__(
            self,
            "_simulation_extra_lazy_fields",
            tuple(freeze_mapping(dict(row)) for row in self._simulation_extra_lazy_fields)
            if self._simulation_extra_lazy_fields
            else tuple(freeze_mapping({}) for _ in range(num_sims)),
        )

    def __len__(self) -> int:
        return len(self.options_rows)

    def _has_setup_field(self, name: str) -> bool:
        """Return whether persisted setup contains ``name``."""
        return name in self.setup

    def _get_setup_field(self, name: str, default: Any = _MISSING) -> Any:
        """Return one setup field or ``default`` when provided."""
        if name in self.setup:
            return self.setup[name]
        if default is not _MISSING:
            return default
        raise ValueError(f"Missing required setup field '{name}'.")

    def _require_setup_field(self, name: str) -> Any:
        """Return one required setup field."""
        return self._get_setup_field(name)

    def _require_setup_string_field(self, name: str, normalize: bool = False) -> str:
        """Return one required setup string field with optional normalization."""
        value = self._require_setup_field(name)
        if not isinstance(value, str):
            raise TypeError(f"Setup field '{name}' must be a string.")
        if not normalize:
            return value
        value = value.strip().lower()
        if not value:
            raise ValueError(f"Setup field '{name}' must be a non-empty string.")
        return value

    def _has_simulation_payload_field(self, name: str) -> bool:
        """Return whether persisted simulation payload contains ``name``."""
        return name in self.simulation_payload

    def _get_simulation_payload_field(self, name: str, default: Any = _MISSING) -> Any:
        """Return one simulation-payload field or ``default`` when provided."""
        if name in self.simulation_payload:
            return self.simulation_payload[name]
        if default is not _MISSING:
            return default
        raise ValueError(f"Missing required simulation payload field '{name}'.")

    def _require_simulation_payload_field(self, name: str) -> Any:
        """Return one required simulation-payload field."""
        return self._get_simulation_payload_field(name)

    def _has_extra_field(self, name: str) -> bool:
        """Return whether a dataset-level eager or lazy extra field is available."""
        return name in self._dataset_extra_fields or name in self._dataset_extra_lazy_fields

    def _require_extra_field(self, name: str) -> Any:
        """Return one dataset-level extra field, loading lazily when needed."""
        if name in self._dataset_extra_fields:
            return self._dataset_extra_fields[name]
        if name in self._dataset_extra_field_cache:
            return self._dataset_extra_field_cache[name]
        if name not in self._dataset_extra_lazy_fields:
            raise ValueError(f"Extra field '{name}' is not available in this analysis dataset.")
        value = _freeze_loaded_value(self._dataset_extra_lazy_fields[name]())
        self._dataset_extra_field_cache[name] = value
        return value

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
                extra_fields=self._simulation_extra_fields[idx],
                extra_lazy_fields=self._simulation_extra_lazy_fields[idx],
            )
        )


# Helper primitives

def _freeze_loaded_value(value: Any) -> Any:
    """Freeze one eager/lazy loaded value for read-only public exposure."""
    if isinstance(value, np.ndarray):
        return freeze_array(value)
    if isinstance(value, Mapping):
        return freeze_mapping(dict(value))
    if isinstance(value, list):
        return tuple(_freeze_loaded_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_loaded_value(item) for item in value)
    return value


def _read_node(node: h5py.Group | h5py.Dataset) -> Any:
    """Read an HDF5 node recursively into plain Python objects."""
    if isinstance(node, h5py.Group):
        return {key: _read_node(node[key]) for key in node.keys()}

    data = node[()]
    if isinstance(data, bytes):
        return data.decode("utf-8")
    if isinstance(data, np.ndarray) and data.dtype.kind in {"S", "O"}:
        return data.astype(str)
    return data


def _require_dataset(f: h5py.File | h5py.Group, path: str) -> h5py.Dataset:
    """Return a dataset by path or raise a clear contract error."""
    if path not in f:
        raise ValueError(f"Missing required dataset '{path}'.")
    ds = f[path]
    if not isinstance(ds, h5py.Dataset):
        raise ValueError(f"{path} must be a dataset.")
    return ds


def _read_simulation_dataset_row(ds: h5py.Dataset, sim_idx: int, *, path: str) -> Any:
    """Read one per-simulation dataset row with shared index and shape checks."""
    idx = int(sim_idx)
    if idx < 0:
        raise IndexError(f"simulation index must be >= 0, got {idx}.")
    if ds.ndim == 0:
        raise ValueError(f"{path} must be per-simulation with first dim N.")
    if idx >= ds.shape[0]:
        raise IndexError(f"sim_idx {idx} out of range for {path} shape {ds.shape}")
    value = ds[idx]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O"}:
        return value.astype(str)
    return value
