"""HDF5 storage for simulation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from ..simulation import schema
from ..simulation.validation import (
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


def _require_all_finite(name: str, arr: np.ndarray) -> None:
    """Raise when an array contains ``NaN`` or ``Inf`` values."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")


def _read_extra_stat_names(simulation: Mapping[str, Any]) -> tuple[str, ...]:
    """Read declared extra stat names from an already-validated ``/simulation`` payload."""
    return tuple(str(name) for name in np.asarray(simulation[schema.KEY_SIMULATION_EXTRA_STAT_NAMES]).reshape(-1).tolist())


def _clear_simulation_outputs(f: h5py.File, sim_idx: int) -> None:
    """Reset one simulation's persisted outputs to ``NaN`` values."""
    stats = f[schema.KEY_STATS_SECTION]
    meta = f[schema.KEY_META_SECTION]

    stats[schema.KEY_STATS_SR][sim_idx, ...] = np.nan
    stats[schema.KEY_STATS_EE][sim_idx, ...] = np.nan
    stats[schema.KEY_STATS_FWHM_MAS][sim_idx, ...] = np.nan
    for key in _read_extra_stat_names(_read_node(f[schema.KEY_SIMULATION_SECTION])):
        stats[key][sim_idx, ...] = np.nan

    meta[schema.KEY_META_PIXEL_SCALE_MAS][sim_idx] = np.nan

    if schema.KEY_PSFS_SECTION in f and schema.KEY_PSFS_DATA in f[schema.KEY_PSFS_SECTION]:
        f[f"{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"][sim_idx, ...] = np.nan


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

            g_stats.create_dataset(schema.KEY_STATS_SR, data=np.full((num_sims, m_sci), np.nan, dtype=np.float32))
            g_stats.create_dataset(
                schema.KEY_STATS_EE, data=np.full((num_sims, m_sci, ee.shape[0]), np.nan, dtype=np.float32)
            )
            g_stats.create_dataset(schema.KEY_STATS_FWHM_MAS, data=np.full((num_sims, m_sci), np.nan, dtype=np.float32))
            for name in extra_stat_names:
                g_stats.create_dataset(name, data=np.full((num_sims, m_sci), np.nan, dtype=np.float32))

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
        """Read the persisted ``/simulation`` group.

        Returns:
            Nested Python mapping decoded from ``/simulation``.
        """

        with h5py.File(self.path, "r") as f:
            return _read_node(f[schema.KEY_SIMULATION_SECTION])

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
        sim_idx = _ensure_sim_idx(sim_idx)

        row: dict[str, Any] = {}
        with h5py.File(self.path, "r") as f:
            g = f[schema.KEY_OPTION_SECTION]
            for key in g.keys():
                ds = g[key]
                if not isinstance(ds, h5py.Dataset):
                    raise ValueError(f"/options/{key} must be a dataset.")

                if ds.ndim == 0:
                    raise ValueError(f"/options/{key} must be per-simulation with first dim N.")
                if sim_idx >= ds.shape[0]:
                    raise IndexError(f"sim_idx {sim_idx} out of range for /options/{key} shape {ds.shape}")

                value = ds[sim_idx]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                row[key] = value

        return row

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
                validate_simulation_payload_core(simulation_data)
                extra_stat_names = _read_extra_stat_names(simulation_data)
            except Exception as exc:
                issues.append(f"Invalid /simulation payload: {exc}")
                extra_stat_names = ()

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
            extra_stat_names = _read_extra_stat_names(_read_node(f[schema.KEY_SIMULATION_SECTION]))
            require_psfs = schema.KEY_PSFS_SECTION in f

            validate_successful_result(
                result,
                num_sci,
                num_ee,
                extra_stat_names=extra_stat_names,
                require_psfs=require_psfs,
            )

            # Persist meta values.
            meta = f[schema.KEY_META_SECTION]
            pixel_scale = np.asarray(result.meta[schema.KEY_META_PIXEL_SCALE_MAS], dtype=np.float32)
            meta[schema.KEY_META_PIXEL_SCALE_MAS][sim_idx] = np.float32(pixel_scale.item())
            _write_dataset_level_telescope_meta(f, result)

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
