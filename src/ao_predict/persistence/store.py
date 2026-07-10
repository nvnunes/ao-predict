"""HDF5 storage for simulation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from ..simulation import schema
from ..simulation.validation import (
    normalize_meta_field_names,
    resolve_simulation_payload_for_load,
    validate_atm_profile_ids,
    validate_options_payload_core,
    validate_successful_result,
    validate_setup_payload_core,
    validate_simulation_payload_core,
)
from ..simulation.interfaces import SimulationResult, SimulationState
from ..simulation.helpers import get_ee_apertures, get_num_sci
from ..utils import as_array


# HDF5 conversion helpers
def _write_value(group: h5py.Group, key: str, value: Any) -> None:
    """Write a Python value recursively into an HDF5 group key."""
    if isinstance(value, Mapping):
        sub = group.require_group(key)
        for sub_key, sub_value in value.items():
            _write_value(sub, str(sub_key), sub_value)
        return

    if isinstance(value, str):
        dtype = h5py.string_dtype(encoding="utf-8")
        if key in group:
            del group[key]
        group.create_dataset(key, data=value, dtype=dtype)
        return

    arr = as_array(value)
    if arr.dtype.kind in {"U", "O"}:
        dtype = h5py.string_dtype(encoding="utf-8")
        if key in group:
            del group[key]
        group.create_dataset(key, data=np.asarray(arr, dtype=object), dtype=dtype)
    else:
        if key in group:
            del group[key]
        group.create_dataset(key, data=arr)


def _read_node(node: h5py.Group | h5py.Dataset) -> Any:
    """Read an HDF5 node recursively into plain Python objects."""
    if isinstance(node, h5py.Group):
        return {k: _read_node(node[k]) for k in node.keys()}

    data = node[()]
    if isinstance(data, bytes):
        return data.decode("utf-8")
    if isinstance(data, np.ndarray) and data.dtype.kind in {"S", "O"}:
        return data.astype(str)
    return data


def _ensure_sim_idx(sim_idx: int) -> int:
    """Validate and normalize a zero-based simulation index."""
    idx = int(sim_idx)
    if idx < 0:
        raise IndexError(f"simulation index must be >= 0, got {idx}.")
    return idx


def _ensure_meta_tel_pupil(f: h5py.File, tel_pupil: np.ndarray) -> None:
    """Ensure ``/meta/tel_pupil`` exists and matches the dataset-level shape."""
    meta = f[schema.KEY_META_SECTION]
    expected = tuple(tel_pupil.shape)
    if schema.KEY_META_TEL_PUPIL in meta:
        ds = meta[schema.KEY_META_TEL_PUPIL]
        if ds.shape == (0, 0):
            ds.resize(expected)
            ds[...] = np.nan
            return
        if ds.shape != expected:
            raise ValueError(f"/meta/tel_pupil shape mismatch: expected {expected}, got {ds.shape}")
        return

    meta.create_dataset(
        schema.KEY_META_TEL_PUPIL,
        data=np.full(expected, np.nan, dtype=np.float32),
    )


def _write_dataset_level_telescope_meta(f: h5py.File, result: SimulationResult) -> None:
    """Persist invariant telescope metadata once and enforce consistency."""
    meta = f[schema.KEY_META_SECTION]
    tel_diameter = np.asarray(result.meta[schema.KEY_META_TEL_DIAMETER_M], dtype=np.float32)
    tel_pupil = np.asarray(result.meta[schema.KEY_META_TEL_PUPIL], dtype=np.float32)

    tel_diameter_value = np.float32(tel_diameter.item())
    stored_tel_diameter = np.asarray(meta[schema.KEY_META_TEL_DIAMETER_M][()], dtype=np.float32)
    if np.isnan(stored_tel_diameter):
        meta[schema.KEY_META_TEL_DIAMETER_M][()] = tel_diameter_value
    elif not np.isclose(float(stored_tel_diameter), float(tel_diameter_value), rtol=0.0, atol=0.0):
        raise ValueError(
            "result.meta.tel_diameter_m does not match dataset-level /meta/tel_diameter_m."
        )

    _ensure_meta_tel_pupil(f, tel_pupil)
    stored_tel_pupil = np.asarray(meta[schema.KEY_META_TEL_PUPIL][...], dtype=np.float32)
    if np.all(np.isnan(stored_tel_pupil)):
        meta[schema.KEY_META_TEL_PUPIL][...] = tel_pupil
    elif stored_tel_pupil.shape != tel_pupil.shape or not np.array_equal(stored_tel_pupil, tel_pupil, equal_nan=True):
        raise ValueError(
            "result.meta.tel_pupil does not match dataset-level /meta/tel_pupil."
        )


def _ensure_psfs_data(f: h5py.File, psfs: np.ndarray) -> None:
    """Ensure ``/psfs/data`` exists and matches the expected per-simulation shape."""
    if schema.KEY_PSFS_SECTION not in f:
        f.require_group(schema.KEY_PSFS_SECTION)
    grp = f[schema.KEY_PSFS_SECTION]
    num_sims = int(f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"].shape[0])

    if schema.KEY_PSFS_DATA in grp:
        ds = grp[schema.KEY_PSFS_DATA]
        expected = (num_sims,) + tuple(psfs.shape)
        if ds.shape != expected:
            raise ValueError(f"/psfs/data shape mismatch: expected {expected}, got {ds.shape}")
        return

    shape = (num_sims,) + tuple(psfs.shape)
    grp.create_dataset(schema.KEY_PSFS_DATA, data=np.full(shape, np.nan, dtype=np.float32))


def _read_extra_stat_names(simulation: Mapping[str, Any]) -> tuple[str, ...]:
    """Read declared extra stat names from an already-validated ``/simulation`` payload."""
    return tuple(str(name) for name in np.asarray(simulation[schema.KEY_SIMULATION_EXTRA_STAT_NAMES]).reshape(-1).tolist())


def _read_diagnostic_field_specs(simulation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Read and validate declared diagnostic specs from ``/simulation``.

    The persisted ``diagnostic_fields`` mapping may use flat slash-delimited
    keys or nested mappings that mirror the ``/diagnostics`` group tree. Each
    leaf spec declares a storage ``dtype`` and a shape excluding the leading
    simulation dimension ``N``. Supported dtypes are ``float32``, ``float64``,
    ``int32``, ``int64``, ``bool``, and ``str``. Supported symbolic shape
    tokens are ``num_sci`` and ``num_ngs``; numeric shape values must be
    positive integers.

    Args:
        simulation: Persisted ``/simulation`` payload mapping.

    Returns:
        Flat mapping from slash-delimited ``/diagnostics`` field path to
        normalized dtype/shape spec.

    Raises:
        TypeError: If the diagnostic spec tree is not mapping-shaped.
        ValueError: If field names, dtypes, or shape tokens are invalid.
    """
    raw_specs = simulation.get(schema.KEY_SIMULATION_DIAGNOSTIC_FIELDS, {})
    if raw_specs is None:
        return {}
    if not isinstance(raw_specs, Mapping):
        raise TypeError(f"simulation['{schema.KEY_SIMULATION_DIAGNOSTIC_FIELDS}'] must be a mapping.")
    specs: dict[str, dict[str, Any]] = {}

    def iter_specs(node: Mapping[str, Any], prefix: str = ""):
        for raw_name, raw_spec in node.items():
            name = f"{prefix}/{raw_name}" if prefix else str(raw_name)
            if not isinstance(raw_spec, Mapping):
                raise TypeError(f"diagnostic field spec for '{name}' must be a mapping.")
            if "dtype" in raw_spec or "shape" in raw_spec:
                yield name, raw_spec
            else:
                yield from iter_specs(raw_spec, prefix=name)

    for raw_name, raw_spec in iter_specs(raw_specs):
        name = str(raw_name).strip("/")
        if not name:
            raise ValueError("diagnostic field names must be non-empty.")
        dtype = str(raw_spec.get("dtype", "float32")).strip()
        if dtype not in {"float32", "float64", "int32", "int64", "bool", "str"}:
            raise ValueError(f"diagnostic field '{name}' has unsupported dtype {dtype!r}.")
        shape = raw_spec.get("shape", ())
        shape_values = np.asarray(shape, dtype=object).reshape(-1).tolist()
        normalized_shape: list[str | int] = []
        for value in shape_values:
            if isinstance(value, str):
                if value not in {"num_sci", "num_ngs"}:
                    try:
                        value = int(value)
                    except ValueError as exc:
                        raise ValueError(f"diagnostic field '{name}' has unsupported shape token {value!r}.") from exc
                    if value <= 0:
                        raise ValueError(f"diagnostic field '{name}' shape values must be positive.")
                    normalized_shape.append(value)
                else:
                    normalized_shape.append(value)
            elif int(value) <= 0:
                raise ValueError(f"diagnostic field '{name}' shape values must be positive.")
            else:
                normalized_shape.append(int(value))
        specs[name] = {"dtype": dtype, "shape": tuple(normalized_shape)}
    return specs


def _read_declared_extra_stat_names(f: h5py.File) -> tuple[str, ...]:
    """Read declared extra stat names from the persisted ``/simulation`` payload."""
    return _read_extra_stat_names(_read_node(f[schema.KEY_SIMULATION_SECTION]))


def _read_declared_diagnostic_field_specs(f: h5py.File) -> dict[str, dict[str, Any]]:
    """Read declared diagnostic field specs from the persisted ``/simulation`` payload."""
    return _read_diagnostic_field_specs(_read_node(f[schema.KEY_SIMULATION_SECTION]))


def _read_declared_meta_field_names(f: h5py.File) -> tuple[str, ...]:
    """Read declared extra scalar meta field names from ``/simulation``."""
    simulation = _read_node(f[schema.KEY_SIMULATION_SECTION])
    return normalize_meta_field_names(simulation.get(schema.KEY_SIMULATION_META_FIELDS, ()))


def _read_declared_stat_names(f: h5py.File) -> tuple[str, ...]:
    """Read the full declared stats key set in stable store order."""
    return schema.CORE_STATS_KEYS + _read_declared_extra_stat_names(f)


def _require_dataset(f: h5py.File | h5py.Group, path: str) -> h5py.Dataset:
    """Return a dataset by path or raise a clear contract error."""
    if path not in f:
        raise ValueError(f"Missing required dataset '{path}'.")
    ds = f[path]
    if not isinstance(ds, h5py.Dataset):
        raise ValueError(f"{path} must be a dataset.")
    return ds


def _require_dataset_ndim(ds: h5py.Dataset, *, path: str, ndim: int) -> h5py.Dataset:
    """Validate dataset dimensionality for a specific persisted contract."""
    if ds.ndim != ndim:
        raise ValueError(f"{path} must be {ndim}D.")
    return ds


def _read_simulation_dataset_row(ds: h5py.Dataset, sim_idx: int, *, path: str) -> Any:
    """Read one per-simulation dataset row with shared index and shape checks."""
    sim_idx = _ensure_sim_idx(sim_idx)
    if ds.ndim == 0:
        raise ValueError(f"{path} must be per-simulation with first dim N.")
    if sim_idx >= ds.shape[0]:
        raise IndexError(f"sim_idx {sim_idx} out of range for {path} shape {ds.shape}")
    return ds[sim_idx]


def _read_dataset_value(ds: h5py.Dataset) -> Any:
    """Read one dataset into plain Python/NumPy values preserving array shape."""
    data = ds[()]
    if isinstance(data, bytes):
        return data.decode("utf-8")
    if isinstance(data, np.ndarray) and data.dtype.kind in {"S", "O"}:
        return data.astype(str)
    return data


def _read_meta_values(f: h5py.File) -> dict[str, Any]:
    """Read the full analysis-level ``/meta`` view including declared extras."""
    meta = f[schema.KEY_META_SECTION]
    pixel_scale_path = f"/{schema.KEY_META_SECTION}/{schema.KEY_META_PIXEL_SCALE_MAS}"
    tel_diameter_path = f"/{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_DIAMETER_M}"
    tel_pupil_path = f"/{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_PUPIL}"
    values = {
        schema.KEY_META_PIXEL_SCALE_MAS: _read_dataset_value(
            _require_dataset_ndim(_require_dataset(f, pixel_scale_path), path=pixel_scale_path, ndim=1)
        ),
        schema.KEY_META_TEL_DIAMETER_M: _read_dataset_value(
            _require_dataset_ndim(_require_dataset(f, tel_diameter_path), path=tel_diameter_path, ndim=0)
        ),
        schema.KEY_META_TEL_PUPIL: _read_dataset_value(
            _require_dataset_ndim(_require_dataset(f, tel_pupil_path), path=tel_pupil_path, ndim=2)
        ),
    }
    for name in _read_declared_meta_field_names(f):
        path = f"/{schema.KEY_META_SECTION}/{name}"
        values[name] = _read_dataset_value(_require_dataset_ndim(_require_dataset(f, path), path=path, ndim=1))
    return values


def _write_result_meta_fields(f: h5py.File, sim_idx: int, result: SimulationResult) -> None:
    """Persist declared scalar per-simulation meta fields."""
    meta = f[schema.KEY_META_SECTION]
    for name in _read_declared_meta_field_names(f):
        value = np.asarray(result.meta[name], dtype=np.float32)
        meta[name][sim_idx] = np.float32(value.item())


def _decode_dataset_value(data: Any) -> Any:
    """Decode one already-read HDF5 value into plain Python/NumPy values.

    This mirrors ``_read_dataset_value()`` for values that have already been
    sliced out of a dataset, including per-simulation diagnostic string rows
    returned by h5py as ``bytes``.
    """
    if isinstance(data, bytes):
        return data.decode("utf-8")
    if isinstance(data, np.ndarray) and data.dtype.kind in {"S", "O"}:
        return data.astype(str)
    return data


def _clear_simulation_outputs(f: h5py.File, sim_idx: int) -> None:
    """Reset one simulation's persisted outputs to ``NaN`` values."""
    stats = f[schema.KEY_STATS_SECTION]
    meta = f[schema.KEY_META_SECTION]

    for key in _read_declared_stat_names(f):
        stats[key][sim_idx, ...] = np.nan

    meta[schema.KEY_META_PIXEL_SCALE_MAS][sim_idx] = np.nan
    for key in _read_declared_meta_field_names(f):
        meta[key][sim_idx] = np.nan

    if schema.KEY_PSFS_SECTION in f and schema.KEY_PSFS_DATA in f[schema.KEY_PSFS_SECTION]:
        f[f"{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"][sim_idx, ...] = np.nan

    if schema.KEY_DIAGNOSTICS_SECTION in f:
        _clear_diagnostics_group(f[schema.KEY_DIAGNOSTICS_SECTION], sim_idx)


def _clear_diagnostics_group(group: h5py.Group, sim_idx: int) -> None:
    """Reset one simulation's diagnostics row in a diagnostics group tree."""
    for node in group.values():
        if isinstance(node, h5py.Group):
            _clear_diagnostics_group(node, sim_idx)
        elif isinstance(node, h5py.Dataset):
            node[sim_idx, ...] = _diagnostic_fill_value(node.dtype)


def _diagnostic_fill_value(dtype: np.dtype) -> Any:
    """Return the empty fill value for a diagnostic dataset dtype."""
    if h5py.check_string_dtype(dtype) is not None:
        return ""
    if np.dtype(dtype).kind == "f":
        return np.nan
    if np.dtype(dtype).kind == "b":
        return False
    return 0


def _diagnostic_dtype(dtype_name: str) -> Any:
    """Return the HDF5 dtype for a diagnostic field dtype name."""
    if dtype_name == "str":
        return h5py.string_dtype(encoding="utf-8")
    return {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
        "bool": np.bool_,
    }[dtype_name]


def _num_ngs_slots(options: Mapping[str, Any]) -> int:
    """Return the configured NGS slot count from persisted options."""
    if schema.KEY_OPTION_NGS_R_ARCSEC not in options:
        return 0
    arr = np.asarray(options[schema.KEY_OPTION_NGS_R_ARCSEC])
    if arr.ndim <= 1:
        return 1
    return int(arr.shape[1])


def _resolve_diagnostic_shape(spec: Mapping[str, Any], setup: Mapping[str, Any], options: Mapping[str, Any]) -> tuple[int, ...]:
    """Resolve a diagnostic field shape spec to fixed dimensions excluding ``N``.

    ``num_sci`` is derived from the persisted setup science field and
    ``num_ngs`` is derived from the persisted NGS option slot count. Fixed
    numeric dimensions are passed through as positive integers.

    Args:
        spec: Normalized diagnostic field spec.
        setup: Persisted ``/setup`` payload.
        options: Persisted ``/options`` payload.

    Returns:
        Concrete per-simulation diagnostic shape excluding the leading
        simulation dimension.

    Raises:
        ValueError: If any resolved dimension is non-positive.
    """
    resolved: list[int] = []
    for value in tuple(spec.get("shape", ())):
        if value == "num_sci":
            resolved.append(int(get_num_sci(setup)))
        elif value == "num_ngs":
            resolved.append(_num_ngs_slots(options))
        else:
            resolved.append(int(value))
    if any(dim <= 0 for dim in resolved):
        raise ValueError(f"diagnostic field shape resolved to invalid dimensions {resolved}.")
    return tuple(resolved)


def _require_parent_group(root: h5py.Group, path: str) -> tuple[h5py.Group, str]:
    """Return the parent group and dataset name for a slash-delimited path."""
    parts = [part for part in str(path).strip("/").split("/") if part]
    if not parts:
        raise ValueError("diagnostic field path must be non-empty.")
    group = root
    for part in parts[:-1]:
        group = group.require_group(part)
    return group, parts[-1]


def _create_diagnostic_dataset(
    root: h5py.Group,
    path: str,
    spec: Mapping[str, Any],
    *,
    num_sims: int,
    setup: Mapping[str, Any],
    options: Mapping[str, Any],
) -> None:
    """Create one preallocated ``/diagnostics`` dataset from a spec.

    Dataset paths may include slash-delimited parent groups. The allocated
    shape is ``(num_sims, *resolved_shape)`` where ``resolved_shape`` comes from
    ``_resolve_diagnostic_shape()``. Floating diagnostics are filled with
    ``NaN``, integer diagnostics with ``0``, booleans with ``False``, and string
    diagnostics with empty UTF-8 strings.

    Args:
        root: ``/diagnostics`` HDF5 group.
        path: Slash-delimited diagnostic field path relative to ``root``.
        spec: Normalized dtype/shape spec.
        num_sims: Number of simulation rows to allocate.
        setup: Persisted ``/setup`` payload.
        options: Persisted ``/options`` payload.

    Raises:
        ValueError: If the path or resolved shape is invalid.
    """
    parent, name = _require_parent_group(root, path)
    dtype_name = str(spec["dtype"])
    dtype = _diagnostic_dtype(dtype_name)
    shape = (int(num_sims),) + _resolve_diagnostic_shape(spec, setup, options)
    if dtype_name == "str":
        parent.create_dataset(name, shape=shape, dtype=dtype, fillvalue="")
    else:
        fill_value = _diagnostic_fill_value(np.dtype(dtype))
        parent.create_dataset(name, data=np.full(shape, fill_value, dtype=dtype))


def _flatten_mapping(mapping: Mapping[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten slash-delimited diagnostic mappings without flattening arrays."""
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value, prefix=name))
        else:
            flat[name.strip("/")] = value
    return flat


def _require_diagnostic_dataset(root: h5py.Group, path: str) -> h5py.Dataset:
    """Return one diagnostic dataset by slash-delimited path."""
    if path not in root:
        raise ValueError(f"Missing declared diagnostics dataset '/diagnostics/{path}'.")
    node = root[path]
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"/diagnostics/{path} must be a dataset.")
    return node


def _write_result_diagnostics(f: h5py.File, sim_idx: int, result: SimulationResult) -> None:
    """Persist declared diagnostics for one successful simulation.

    The result diagnostics mapping may be flat or nested, but after flattening
    it must exactly match the fields declared by ``/simulation/diagnostic_fields``.
    Values are written into the preallocated ``/diagnostics`` datasets at
    ``sim_idx`` and must match each dataset's per-simulation shape and dtype.
    If no diagnostics were declared, the result must not provide diagnostics.

    Args:
        f: Open HDF5 dataset file.
        sim_idx: Zero-based simulation row to write.
        result: Successful simulation result containing diagnostics values.

    Raises:
        ValueError: If diagnostics are missing, undeclared, or shape-mismatched.
    """
    specs = _read_declared_diagnostic_field_specs(f)
    result_values = _flatten_mapping(result.diagnostics)
    if not specs:
        if result_values:
            names = ", ".join(sorted(result_values))
            raise ValueError(f"result.diagnostics contains undeclared diagnostics: {names}")
        return

    missing = sorted(set(specs) - set(result_values))
    unexpected = sorted(set(result_values) - set(specs))
    if missing:
        raise ValueError(f"result.diagnostics is missing declared diagnostics: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"result.diagnostics contains undeclared diagnostics: {', '.join(unexpected)}")

    root = f[schema.KEY_DIAGNOSTICS_SECTION]
    for name, spec in specs.items():
        ds = _require_diagnostic_dataset(root, name)
        value = result_values[name]
        if str(spec["dtype"]) == "str":
            if ds.shape[1:] != ():
                arr = np.asarray(value, dtype=object)
                if arr.shape != ds.shape[1:]:
                    raise ValueError(
                        f"result.diagnostics['{name}'] must have shape {ds.shape[1:]}, got {arr.shape}."
                    )
            else:
                arr = str(value)
        else:
            arr = np.asarray(value, dtype=ds.dtype)
            if arr.shape != ds.shape[1:]:
                raise ValueError(
                    f"result.diagnostics['{name}'] must have shape {ds.shape[1:]}, got {arr.shape}."
                )
        ds[sim_idx, ...] = arr


def _read_diagnostics_group(node: h5py.Group | h5py.Dataset, sim_idx: int | None = None, path: str = "") -> Any:
    """Read full-analysis or one-row diagnostics recursively.

    HDF5 groups are returned as nested dictionaries. Dataset values are decoded
    to the same plain Python/NumPy conventions as other store readers: scalar
    UTF-8 strings are returned as ``str`` and string arrays are converted to
    NumPy string arrays. When ``sim_idx`` is provided, only that row/slice is
    read from each diagnostic dataset.

    Args:
        node: ``/diagnostics`` group or one dataset within it.
        sim_idx: Optional simulation row. ``None`` reads full analysis arrays.
        path: Diagnostic path used for error messages.

    Returns:
        Nested diagnostics mapping or decoded dataset value.

    Raises:
        IndexError: If ``sim_idx`` is out of range.
        ValueError: If a diagnostic dataset is malformed.
    """
    if isinstance(node, h5py.Group):
        return {key: _read_diagnostics_group(node[key], sim_idx=sim_idx, path=f"{path}/{key}") for key in node.keys()}
    if sim_idx is None:
        return _read_dataset_value(node)
    return _decode_dataset_value(_read_simulation_dataset_row(node, sim_idx, path=path))


# Store implementation

class SimulationStore:
    """Schema-aware HDF5 store for simulation runs with resume support.

    This class owns dataset creation, schema validation, status transitions,
    and per-simulation read/write operations.
    """

    def __init__(self, path: str | Path):
        """Create a store bound to a dataset file path.

        Args:
            path: Filesystem path to the HDF5 dataset.
        """
        self.path = Path(path)

    # Dataset lifecycle

    def create(
        self,
        simulation: Mapping[str, Any],
        setup: Mapping[str, Any],
        options: Mapping[str, Any],
        *,
        overwrite: bool = False,
        save_psfs: bool = False,
    ) -> None:
        """Create a new simulation dataset and preallocate core arrays.

        Args:
            simulation: Persisted ``/simulation`` payload mapping.
            setup: Persisted ``/setup`` payload mapping.
            options: Persisted ``/options`` payload mapping.
            overwrite: If ``True``, replace an existing dataset at ``path``.
            save_psfs: If ``True``, create the ``/psfs`` group for PSF storage.

        Raises:
            FileExistsError: If dataset exists and ``overwrite`` is ``False``.
            TypeError: If payload arguments are not mappings.
            ValueError: If payload validation fails.
        """

        if simulation is None:
            raise ValueError("create requires non-null simulation mapping.")
        if setup is None:
            raise ValueError("create requires non-null setup mapping.")
        if options is None:
            raise ValueError("create requires non-null options mapping.")
        if not isinstance(simulation, Mapping):
            raise TypeError(f"create expected simulation as Mapping, got {type(simulation).__name__}.")
        if not isinstance(setup, Mapping):
            raise TypeError(f"create expected setup as Mapping, got {type(setup).__name__}.")
        if not isinstance(options, Mapping):
            raise TypeError(f"create expected options as Mapping, got {type(options).__name__}.")

        if self.path.exists() and not overwrite:
            raise FileExistsError(f"Dataset already exists: {self.path}")

        validate_simulation_payload_core(simulation)
        validate_setup_payload_core(setup)
        num_sims = validate_options_payload_core(options)
        validate_atm_profile_ids(setup, options)
        extra_stat_names = _read_extra_stat_names(simulation)
        meta_field_names = normalize_meta_field_names(simulation.get(schema.KEY_SIMULATION_META_FIELDS, ()))
        diagnostic_field_specs = _read_diagnostic_field_specs(simulation)

        m_sci = get_num_sci(setup)
        ee = get_ee_apertures(setup)

        if self.path.exists() and overwrite:
            self.path.unlink()

        with h5py.File(self.path, "w") as f:
            g_simulation = f.require_group(schema.KEY_SIMULATION_SECTION)
            g_setup = f.require_group(schema.KEY_SETUP_SECTION)
            g_options = f.require_group(schema.KEY_OPTION_SECTION)
            g_status = f.require_group(schema.KEY_STATUS_SECTION)
            g_meta = f.require_group(schema.KEY_META_SECTION)
            g_stats = f.require_group(schema.KEY_STATS_SECTION)
            g_diagnostics = f.require_group(schema.KEY_DIAGNOSTICS_SECTION) if diagnostic_field_specs else None
            if save_psfs:
                f.require_group(schema.KEY_PSFS_SECTION)

            for key, value in simulation.items():
                _write_value(g_simulation, str(key), value)

            for key, value in setup.items():
                _write_value(g_setup, str(key), value)

            for key, value in options.items():
                _write_value(g_options, str(key), value)

            g_status.create_dataset(
                schema.KEY_STATUS_STATE,
                data=np.full((num_sims,), int(SimulationState.PENDING), dtype=np.uint8),
            )

            g_meta.create_dataset(schema.KEY_META_PIXEL_SCALE_MAS, data=np.full((num_sims,), np.nan, dtype=np.float32))
            g_meta.create_dataset(schema.KEY_META_TEL_DIAMETER_M, data=np.float32(np.nan))
            g_meta.create_dataset(
                schema.KEY_META_TEL_PUPIL,
                shape=(0, 0),
                maxshape=(None, None),
                chunks=True,
                dtype=np.float32,
            )
            for name in meta_field_names:
                g_meta.create_dataset(name, data=np.full((num_sims,), np.nan, dtype=np.float32))

            g_stats.create_dataset(schema.KEY_STATS_SR, data=np.full((num_sims, m_sci), np.nan, dtype=np.float32))
            g_stats.create_dataset(
                schema.KEY_STATS_EE, data=np.full((num_sims, m_sci, ee.shape[0]), np.nan, dtype=np.float32)
            )
            g_stats.create_dataset(schema.KEY_STATS_FWHM_MAS, data=np.full((num_sims, m_sci), np.nan, dtype=np.float32))
            for name in extra_stat_names:
                g_stats.create_dataset(name, data=np.full((num_sims, m_sci), np.nan, dtype=np.float32))
            if g_diagnostics is not None:
                for name, spec in diagnostic_field_specs.items():
                    _create_diagnostic_dataset(
                        g_diagnostics,
                        name,
                        spec,
                        num_sims=num_sims,
                        setup=setup,
                        options=options,
                    )

    def exists(self) -> bool:
        """Return whether the dataset file currently exists on disk.

        Returns:
            ``True`` if the dataset path exists, else ``False``.
        """
        return self.path.exists()

    # Payload read helpers

    def read_setup(self) -> dict[str, Any]:
        """Read the persisted ``/setup`` group.

        Returns:
            Nested Python mapping decoded from ``/setup``.
        """

        with h5py.File(self.path, "r") as f:
            return _read_node(f[schema.KEY_SETUP_SECTION])

    def read_simulation(self) -> dict[str, Any]:
        """Read ``/simulation`` as a validated current-contract payload.

        Recognized legacy payloads are upgraded in memory and the dataset is
        not rewritten.

        Returns:
            Current-contract mapping decoded from ``/simulation``.
        """

        with h5py.File(self.path, "r") as f:
            simulation = _read_node(f[schema.KEY_SIMULATION_SECTION])
        return resolve_simulation_payload_for_load(simulation)

    def read_extra_stat_names(self) -> tuple[str, ...]:
        """Read declared extra stat names from ``/simulation``."""

        with h5py.File(self.path, "r") as f:
            return _read_declared_extra_stat_names(f)

    def read_options(self) -> dict[str, Any]:
        """Read the persisted ``/options`` group as dataset-level columns."""

        columns: dict[str, Any] = {}
        with h5py.File(self.path, "r") as f:
            g = f[schema.KEY_OPTION_SECTION]
            for key in g.keys():
                path = f"/{schema.KEY_OPTION_SECTION}/{key}"
                ds = _require_dataset(f, path)
                if ds.ndim == 0:
                    raise ValueError(f"{path} must be per-simulation with first dim N.")
                columns[key] = _read_dataset_value(ds)
        return columns

    def read_analysis_meta(self) -> dict[str, Any]:
        """Read the persisted loaded-analysis ``/meta`` view."""

        with h5py.File(self.path, "r") as f:
            return _read_meta_values(f)

    def read_analysis_stats(self) -> dict[str, Any]:
        """Read the persisted loaded-analysis ``/stats`` group as dataset-level columns."""

        columns: dict[str, Any] = {}
        with h5py.File(self.path, "r") as f:
            for key in _read_declared_stat_names(f):
                path = f"/{schema.KEY_STATS_SECTION}/{key}"
                expected_ndim = 3 if key == schema.KEY_STATS_EE else 2
                ds = _require_dataset_ndim(_require_dataset(f, path), path=path, ndim=expected_ndim)
                columns[key] = _read_dataset_value(ds)
        return columns

    def read_analysis_diagnostics(self) -> dict[str, Any]:
        """Read the optional persisted ``/diagnostics`` group.

        Diagnostics are allocated only when the dataset's ``/simulation``
        payload declares diagnostic fields. When absent, this method returns an
        empty mapping. When present, the returned mapping mirrors the
        ``/diagnostics`` HDF5 group tree and contains full-analysis arrays with
        the leading simulation dimension ``N`` preserved. String diagnostics are
        decoded to Python/NumPy string values.

        Returns:
            Nested diagnostics mapping, or an empty mapping when diagnostics
            were not allocated for this dataset.

        Raises:
            ValueError: If the diagnostics group contains malformed datasets.
        """
        with h5py.File(self.path, "r") as f:
            if schema.KEY_DIAGNOSTICS_SECTION not in f:
                return {}
            return _read_diagnostics_group(f[schema.KEY_DIAGNOSTICS_SECTION])

    def read_sim_options(self, sim_idx: int) -> dict[str, Any]:
        """Read one simulation's options from ``/options``.

        Args:
            sim_idx: Zero-based simulation index.

        Returns:
            Mapping of option key to per-simulation value/slice.

        Raises:
            IndexError: If ``sim_idx`` is out of range.
            ValueError: If ``/options`` datasets are malformed.
        """
        row: dict[str, Any] = {}
        with h5py.File(self.path, "r") as f:
            g = f[schema.KEY_OPTION_SECTION]
            for key in g.keys():
                ds = g[key]
                if not isinstance(ds, h5py.Dataset):
                    raise ValueError(f"/options/{key} must be a dataset.")

                value = _read_simulation_dataset_row(ds, sim_idx, path=f"/options/{key}")
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                row[key] = value

        return row

    def read_simulation_meta(self, sim_idx: int) -> dict[str, Any]:
        """Read one simulation's persisted meta view."""

        with h5py.File(self.path, "r") as f:
            pixel_scale_path = f"/{schema.KEY_META_SECTION}/{schema.KEY_META_PIXEL_SCALE_MAS}"
            tel_diameter_path = f"/{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_DIAMETER_M}"
            tel_pupil_path = f"/{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_PUPIL}"

            pixel_scale_mas = _read_simulation_dataset_row(
                _require_dataset_ndim(_require_dataset(f, pixel_scale_path), path=pixel_scale_path, ndim=1),
                sim_idx,
                path=pixel_scale_path,
            )
            tel_diameter_m = _require_dataset_ndim(
                _require_dataset(f, tel_diameter_path),
                path=tel_diameter_path,
                ndim=0,
            )[()]
            tel_pupil = _require_dataset_ndim(
                _require_dataset(f, tel_pupil_path),
                path=tel_pupil_path,
                ndim=2,
            )[...,]
            meta = {
                schema.KEY_META_PIXEL_SCALE_MAS: pixel_scale_mas,
                schema.KEY_META_TEL_DIAMETER_M: tel_diameter_m,
                schema.KEY_META_TEL_PUPIL: tel_pupil,
            }
            for name in _read_declared_meta_field_names(f):
                path = f"/{schema.KEY_META_SECTION}/{name}"
                meta[name] = _read_simulation_dataset_row(
                    _require_dataset_ndim(_require_dataset(f, path), path=path, ndim=1),
                    sim_idx,
                    path=path,
                )

        return meta

    def read_simulation_stats(self, sim_idx: int) -> dict[str, Any]:
        """Read one simulation's persisted stats view."""

        stats_row: dict[str, Any] = {}
        with h5py.File(self.path, "r") as f:
            for key in _read_declared_stat_names(f):
                path = f"/{schema.KEY_STATS_SECTION}/{key}"
                expected_ndim = 3 if key == schema.KEY_STATS_EE else 2
                ds = _require_dataset_ndim(_require_dataset(f, path), path=path, ndim=expected_ndim)
                stats_row[key] = _read_simulation_dataset_row(ds, sim_idx, path=path)

        return stats_row

    def read_simulation_diagnostics(self, sim_idx: int) -> dict[str, Any]:
        """Read one simulation's optional persisted diagnostics view.

        Diagnostics are allocated only when the dataset's ``/simulation``
        payload declares diagnostic fields. When absent, this method returns an
        empty mapping. When present, the returned mapping mirrors the
        ``/diagnostics`` HDF5 group tree and contains only the row/slice for
        ``sim_idx`` from each diagnostic dataset. String diagnostics are decoded
        to ``str`` or NumPy string arrays.

        Args:
            sim_idx: Zero-based simulation index.

        Returns:
            Nested diagnostics mapping for one simulation row, or an empty
            mapping when diagnostics were not allocated for this dataset.

        Raises:
            IndexError: If ``sim_idx`` is out of range.
            ValueError: If the diagnostics group contains malformed datasets.
        """
        sim_idx = _ensure_sim_idx(sim_idx)
        with h5py.File(self.path, "r") as f:
            if schema.KEY_DIAGNOSTICS_SECTION not in f:
                return {}
            return _read_diagnostics_group(f[schema.KEY_DIAGNOSTICS_SECTION], sim_idx=sim_idx, path="/diagnostics")

    def read_simulation_psfs(self, sim_idx: int) -> np.ndarray:
        """Read one simulation's persisted PSF cube from ``/psfs/data``."""

        with h5py.File(self.path, "r") as f:
            path = f"/{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"
            ds = _require_dataset_ndim(_require_dataset(f, path), path=path, ndim=4)
            return np.asarray(_read_simulation_dataset_row(ds, sim_idx, path=path))

    # State and index access

    def num_sims(self) -> int:
        """Return the number of simulations ``N`` in this dataset.

        Returns:
            Number of simulations inferred from ``/status/state`` length.
        """
        with h5py.File(self.path, "r") as f:
            return int(f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"].shape[0])

    def pending_indices(self) -> np.ndarray:
        """Return simulation indexes with state ``PENDING``.

        Returns:
            1D integer numpy array of indexes.
        """
        return self.indices_with_state(SimulationState.PENDING)

    def failed_indices(self) -> np.ndarray:
        """Return simulation indexes with state ``FAILED``.

        Returns:
            1D integer numpy array of indexes.
        """
        return self.indices_with_state(SimulationState.FAILED)

    def indices_with_state(self, state: SimulationState | int) -> np.ndarray:
        """Return simulation indexes matching a specific state value.

        Args:
            state: Desired state as enum or integer value.

        Returns:
            1D integer numpy array of matching indexes.
        """
        state_value = int(SimulationState(int(state)))
        with h5py.File(self.path, "r") as f:
            state_arr = np.asarray(f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"], dtype=np.uint8)
        return np.where(state_arr == state_value)[0]

    def reset_failed_to_pending(self) -> int:
        """Reset all failed simulations to pending.

        Returns:
            Number of simulations whose state changed.
        """

        with h5py.File(self.path, "r+") as f:
            state = f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:]
            mask = state == int(SimulationState.FAILED)
            count = int(np.count_nonzero(mask))
            if count > 0:
                state[mask] = np.uint8(int(SimulationState.PENDING))
                f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:] = state
            return count

    def reset_all_to_pending(self) -> int:
        """Reset all simulations to pending.

        Returns:
            Number of simulations whose state changed.
        """
        return self.reset_to_pending()

    def reset_to_pending(self, indexes: list[int] | np.ndarray | None = None) -> int:
        """Reset selected simulations to pending.

        Args:
            indexes: Optional list/array of zero-based simulation indexes. If
                ``None``, all simulations are considered.

        Returns:
            Number of simulations whose state changed.

        Raises:
            ValueError: If provided indexes contain negative values.
            IndexError: If provided indexes exceed dataset bounds.
        """

        with h5py.File(self.path, "r+") as f:
            state = f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:]
            if indexes is None:
                mask = state != int(SimulationState.PENDING)
            else:
                idx = np.asarray(indexes, dtype=np.int64).reshape(-1)
                if idx.size == 0:
                    return 0
                if np.any(idx < 0):
                    raise ValueError("reset indexes must be non-negative.")
                n = int(state.shape[0])
                if np.any(idx >= n):
                    bad = int(idx[np.argmax(idx >= n)])
                    raise IndexError(f"reset index {bad} out of range for N={n}.")
                mask = np.zeros_like(state, dtype=bool)
                mask[idx] = True
                mask &= state != int(SimulationState.PENDING)
            count = int(np.count_nonzero(mask))
            if count > 0:
                state[mask] = np.uint8(int(SimulationState.PENDING))
                f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"][:] = state
            return count

    # Schema validation

    def validate_schema(self) -> None:
        """Validate required groups/datasets and raise on schema violations.

        Raises:
            ValueError: If one or more schema issues are found.
        """
        issues = self.collect_schema_issues()
        if issues:
            raise ValueError("Schema validation failed:\n- " + "\n- ".join(issues))

    def collect_schema_issues(self) -> list[str]:
        """Collect schema issues without raising exceptions.

        Returns:
            Human-readable schema issue messages. Empty list means valid schema.
        """
        issues: list[str] = []

        try:
            f = h5py.File(self.path, "r")
        except Exception as exc:
            return [f"Unable to open dataset: {exc}"]

        with f:
            for name in schema.REQUIRED_GROUP_KEYS:
                if name not in f or not isinstance(f[name], h5py.Group):
                    issues.append(f"Missing required group '/{name}'.")

            if issues:
                return issues

            status_group = f[schema.KEY_STATUS_SECTION]
            meta_group = f[schema.KEY_META_SECTION]
            stats_group = f[schema.KEY_STATS_SECTION]

            for name in schema.REQUIRED_STATUS_KEYS:
                if name not in status_group:
                    issues.append(f"Missing required dataset '/status/{name}'.")
            for name in schema.REQUIRED_META_KEYS:
                if name not in meta_group:
                    issues.append(f"Missing required dataset '/meta/{name}'.")
            for name in schema.REQUIRED_STATS_KEYS:
                if name not in stats_group:
                    issues.append(f"Missing required dataset '/stats/{name}'.")

            if issues:
                return issues

            state = f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"]
            if state.ndim != 1:
                issues.append("/status/state must be 1D.")
                return issues
            n = int(state.shape[0])
            state_values = np.asarray(state[:], dtype=np.int64).reshape(-1)
            allowed_state_values = {int(s) for s in SimulationState}
            invalid_state_values = sorted({int(v) for v in state_values.tolist()} - allowed_state_values)
            if invalid_state_values:
                issues.append(
                    f"/status/state contains invalid values: {invalid_state_values}. "
                    f"Allowed values: {sorted(allowed_state_values)}."
                )

            try:
                simulation_data = _read_node(f[schema.KEY_SIMULATION_SECTION])
                simulation_data = resolve_simulation_payload_for_load(simulation_data)
                extra_stat_names = _read_extra_stat_names(simulation_data)
                meta_field_names = normalize_meta_field_names(
                    simulation_data.get(schema.KEY_SIMULATION_META_FIELDS, ())
                )
                diagnostic_field_specs = _read_diagnostic_field_specs(simulation_data)
            except Exception as exc:
                issues.append(f"Invalid /simulation payload: {exc}")
                extra_stat_names = ()
                meta_field_names = ()
                diagnostic_field_specs = {}

            try:
                setup_data = _read_node(f[schema.KEY_SETUP_SECTION])
                validate_setup_payload_core(setup_data)
            except Exception as exc:
                issues.append(f"Invalid /setup payload: {exc}")
                setup_data = None

            try:
                options_data = _read_node(f[schema.KEY_OPTION_SECTION])
                validate_options_payload_core(options_data, expected_num_sims=n)
                if setup_data is not None:
                    validate_atm_profile_ids(setup_data, options_data)
            except Exception as exc:
                issues.append(f"Invalid /options payload: {exc}")

            pixel_scale_mas_data = f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_PIXEL_SCALE_MAS}"]
            tel_diameter_m_data = f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_DIAMETER_M}"]
            tel_pupil_data = f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_PUPIL}"]
            extra_meta_data = {
                name: f[f"{schema.KEY_META_SECTION}/{name}"]
                for name in meta_field_names
                if name in meta_group
            }

            sr_data = f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_SR}"]
            ee_data = f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_EE}"]
            fwhm_mas_data = f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_FWHM_MAS}"]
            extra_stat_data = {
                name: f[f"{schema.KEY_STATS_SECTION}/{name}"]
                for name in extra_stat_names
                if name in stats_group
            }

            if sr_data.ndim != 2:
                issues.append("/stats/sr must be 2D [N, M].")
            if ee_data.ndim != 3:
                issues.append("/stats/ee must be 3D [N, M, A].")
            if fwhm_mas_data.ndim != 2:
                issues.append("/stats/fwhm_mas must be 2D [N, M].")
            for name in extra_stat_names:
                if name not in stats_group:
                    issues.append(f"Missing declared extra stats dataset '/stats/{name}'.")
                elif stats_group[name].ndim != 2:
                    issues.append(f"/stats/{name} must be 2D [N, M].")
            for name in meta_field_names:
                if name not in extra_meta_data:
                    issues.append(f"Missing declared meta dataset '/meta/{name}'.")
                elif extra_meta_data[name].ndim != 1:
                    issues.append(f"/meta/{name} must be 1D [N].")
            if pixel_scale_mas_data.ndim != 1:
                issues.append("/meta/pixel_scale_mas must be 1D [N].")
            if tel_diameter_m_data.ndim != 0:
                issues.append("/meta/tel_diameter_m must be a scalar.")
            if tel_pupil_data.ndim != 2:
                issues.append("/meta/tel_pupil must be 2D [Ny, Nx].")

            undeclared_stats = sorted(
                set(stats_group.keys()) - set(schema.CORE_STATS_KEYS) - set(extra_stat_names)
            )
            if undeclared_stats:
                issues.append(f"Undeclared stats datasets found under /stats: {', '.join(undeclared_stats)}.")
            undeclared_meta = sorted(set(meta_group.keys()) - set(schema.CORE_META_KEYS) - set(meta_field_names))
            if undeclared_meta:
                issues.append(f"Undeclared meta datasets found under /meta: {', '.join(undeclared_meta)}.")

            if not issues:
                if (
                    sr_data.shape[0] != n
                    or ee_data.shape[0] != n
                    or fwhm_mas_data.shape[0] != n
                ):
                    issues.append("Stats first dimension must match /status/state length.")
                for name, ds in extra_stat_data.items():
                    if ds.shape[0] != n:
                        issues.append(f"/stats/{name} first dimension must match /status/state length.")
                if pixel_scale_mas_data.shape[0] != n:
                    issues.append("/meta/pixel_scale_mas first dimension must match /status/state length.")
                for name, ds in extra_meta_data.items():
                    if ds.shape[0] != n:
                        issues.append(f"/meta/{name} first dimension must match /status/state length.")
                if (
                    sr_data.shape[1] != ee_data.shape[1]
                    or sr_data.shape[1] != fwhm_mas_data.shape[1]
                ):
                    issues.append("Stats M dimension mismatch between sr/ee/fwhm_mas.")
                for name, ds in extra_stat_data.items():
                    if sr_data.shape[1] != ds.shape[1]:
                        issues.append(f"Stats M dimension mismatch between sr and {name}.")

            if schema.KEY_PSFS_SECTION in f and schema.KEY_PSFS_DATA in f[schema.KEY_PSFS_SECTION]:
                psf_data = f[f"{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"]
                if psf_data.ndim != 4:
                    issues.append("/psfs/data must be 4D [N, M, Ny, Nx].")
                else:
                    if psf_data.shape[0] != n:
                        issues.append("/psfs/data first dimension must match /status/state length.")
                    if sr_data.ndim == 2 and psf_data.shape[1] != sr_data.shape[1]:
                        issues.append("/psfs/data M dimension must match /stats/sr.")

            if diagnostic_field_specs:
                if schema.KEY_DIAGNOSTICS_SECTION not in f or not isinstance(f[schema.KEY_DIAGNOSTICS_SECTION], h5py.Group):
                    issues.append("Missing required group '/diagnostics'.")
                else:
                    diagnostics_group = f[schema.KEY_DIAGNOSTICS_SECTION]
                    for name, spec in diagnostic_field_specs.items():
                        try:
                            ds = _require_diagnostic_dataset(diagnostics_group, name)
                            expected_shape = (n,)
                            if setup_data is not None:
                                options_data = _read_node(f[schema.KEY_OPTION_SECTION])
                                expected_shape = (n,) + _resolve_diagnostic_shape(spec, setup_data, options_data)
                            if ds.shape != expected_shape:
                                issues.append(f"/diagnostics/{name} shape mismatch: expected {expected_shape}, got {ds.shape}.")
                        except Exception as exc:
                            issues.append(str(exc))
            elif schema.KEY_DIAGNOSTICS_SECTION in f:
                diagnostics_group = f[schema.KEY_DIAGNOSTICS_SECTION]
                if isinstance(diagnostics_group, h5py.Group) and len(diagnostics_group.keys()) > 0:
                    issues.append("/diagnostics is present but no diagnostics fields are declared in /simulation.")

        return issues

    # Per-simulation writes

    def write_simulation_success(self, sim_idx: int, result: SimulationResult, *, allow_from_failed: bool = False) -> None:
        """Write one successful simulation and set state to succeeded.

        Args:
            sim_idx: Zero-based simulation index.
            result: Successful simulation result payload.
            allow_from_failed: If ``True``, allow transition from ``FAILED`` to
                ``SUCCEEDED``; otherwise only ``PENDING`` is accepted.

        Raises:
            IndexError: If ``sim_idx`` is invalid.
            ValueError: If state transitions or result payload shapes are invalid.
        """
        sim_idx = _ensure_sim_idx(sim_idx)

        with h5py.File(self.path, "r+") as f:
            state = f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"]
            current_state = int(state[sim_idx])
            allowed_states = (
                {int(SimulationState.PENDING), int(SimulationState.FAILED)}
                if allow_from_failed
                else {int(SimulationState.PENDING)}
            )
            if current_state not in allowed_states:
                raise ValueError(
                    f"Simulation index {sim_idx} has invalid state={current_state} "
                    f"(expected one of {sorted(allowed_states)})."
                )
            if int(result.state) != int(SimulationState.SUCCEEDED):
                raise ValueError(
                    "write_simulation_success requires result.state == "
                    f"{SimulationState.SUCCEEDED.name}, got {int(result.state)}"
                )

            # Derive dataset shape expectations before validating the result payload.
            stats = f[schema.KEY_STATS_SECTION]
            num_sci = int(stats[schema.KEY_STATS_SR].shape[1])
            ee_ds = stats[schema.KEY_STATS_EE]
            if ee_ds.ndim != 3:
                raise ValueError("/stats/ee must be 3D [N, M, A].")
            num_ee = int(ee_ds.shape[2])
            extra_stat_names = _read_declared_extra_stat_names(f)
            meta_field_names = _read_declared_meta_field_names(f)
            require_psfs = schema.KEY_PSFS_SECTION in f

            validate_successful_result(
                result,
                num_sci,
                num_ee,
                extra_stat_names=extra_stat_names,
                meta_field_names=meta_field_names,
                require_psfs=require_psfs,
            )

            # Persist meta values.
            meta = f[schema.KEY_META_SECTION]
            pixel_scale = np.asarray(result.meta[schema.KEY_META_PIXEL_SCALE_MAS], dtype=np.float32)
            meta[schema.KEY_META_PIXEL_SCALE_MAS][sim_idx] = np.float32(pixel_scale.item())
            _write_dataset_level_telescope_meta(f, result)
            _write_result_meta_fields(f, sim_idx, result)

            # Persist stats arrays.
            sr = np.asarray(result.stats[schema.KEY_STATS_SR], dtype=np.float32)
            ee = np.asarray(result.stats[schema.KEY_STATS_EE], dtype=np.float32)
            fwhm = np.asarray(result.stats[schema.KEY_STATS_FWHM_MAS], dtype=np.float32)
            if ee.ndim == 1:
                ee = ee[:, np.newaxis]

            stats[schema.KEY_STATS_SR][sim_idx, :] = sr
            stats[schema.KEY_STATS_EE][sim_idx, :, :] = ee
            stats[schema.KEY_STATS_FWHM_MAS][sim_idx, :] = fwhm
            for name in extra_stat_names:
                stats[name][sim_idx, :] = np.asarray(result.stats[name], dtype=np.float32)

            # Persist PSFs only when the dataset was configured to store them.
            if require_psfs:
                psfs = np.asarray(result.psfs, dtype=np.float32)
                _ensure_psfs_data(f, psfs)
                f[f"{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"][sim_idx, ...] = psfs

            _write_result_diagnostics(f, sim_idx, result)

            state[sim_idx] = np.uint8(int(SimulationState.SUCCEEDED))

    def write_simulation_failure(self, sim_idx: int, *, allow_from_failed: bool = False) -> None:
        """Mark one simulation as failed.

        Args:
            sim_idx: Zero-based simulation index.
            allow_from_failed: If ``True``, allow idempotent failed->failed
                writes; otherwise only ``PENDING`` is accepted.

        Raises:
            IndexError: If ``sim_idx`` is invalid.
            ValueError: If state transition is invalid.
        """
        sim_idx = _ensure_sim_idx(sim_idx)

        with h5py.File(self.path, "r+") as f:
            state = f[f"{schema.KEY_STATUS_SECTION}/{schema.KEY_STATUS_STATE}"]
            current_state = int(state[sim_idx])
            allowed_states = (
                {int(SimulationState.PENDING), int(SimulationState.FAILED)}
                if allow_from_failed
                else {int(SimulationState.PENDING)}
            )
            if current_state not in allowed_states:
                raise ValueError(
                    f"Simulation index {sim_idx} has invalid state={current_state} "
                    f"(expected one of {sorted(allowed_states)})."
                )
            _clear_simulation_outputs(f, sim_idx)
            state[sim_idx] = np.uint8(int(SimulationState.FAILED))
