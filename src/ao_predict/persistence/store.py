"""HDF5 storage for simulation datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from ao_core.io import clear_observation, hdf5, write_observation
from ao_core.io import schema as result_schema
from ao_core.io.results import diagnostic_shape, initialize_layout
from astropy import units as u

from .._precision import real_dtype
from .._units import UNITS_ATTRIBUTE, quantity_from_value, require_quantity, unit_string
from ..simulation import schema
from ..simulation.atm import ATM_PROFILE_FIELD_UNITS
from ..simulation.validation import (
    normalize_meta_fields,
    resolve_setup_payload_for_load,
    resolve_simulation_payload_for_load,
    upgrade_setup_peak_method,
    validate_atm_profile_ids,
    validate_options_payload_core,
    validate_setup_payload_core,
    validate_successful_result,
    validate_simulation_payload_core,
)
from ..simulation.interfaces import SimulationResult, SimulationState
from ..simulation.helpers import get_num_sci


# Stored setup and field declarations


def _read_stored_setup(group: h5py.Group) -> tuple[dict[str, Any], bool]:
    """Resolve stored setup and identify the one supported legacy unit label."""
    setup = hdf5.read_node(group)
    key = schema.KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT
    zeropoint = group.get(key)
    legacy_zeropoint_unit = (
        isinstance(zeropoint, h5py.Dataset)
        and zeropoint.attrs.get(UNITS_ATTRIBUTE) == unit_string(u.photon / u.s)
    )
    if legacy_zeropoint_unit:
        # The stored numbers are flux per area; only their unit label is wrong.
        setup[key] = quantity_from_value(
            setup[key].value,
            schema.SETUP_FIELD_UNITS[key],
        )
    return resolve_setup_payload_for_load(setup), legacy_zeropoint_unit


def _read_declared_extra_stat_names(f: h5py.File) -> tuple[str, ...]:
    """Read declared extra stat names from the persisted ``/simulation`` payload."""
    return tuple(result_schema.extra_stat_fields(hdf5.read_node(f[schema.KEY_SIMULATION_SECTION])))


def _read_declared_extra_stat_fields(f: h5py.File) -> dict[str, u.UnitBase]:
    """Read declared extra-stat units from the persisted simulation payload."""
    return result_schema.extra_stat_fields(hdf5.read_node(f[schema.KEY_SIMULATION_SECTION]))


def _read_declared_meta_fields(f: h5py.File) -> dict[str, u.UnitBase | None]:
    """Read declared extra scalar metadata units from ``/simulation``."""
    simulation = hdf5.read_node(f[schema.KEY_SIMULATION_SECTION])
    return {
        name: None if unit is None else u.Unit(unit)
        for name, unit in normalize_meta_fields(simulation.get(schema.KEY_SIMULATION_META_FIELDS, {})).items()
    }


def _read_declared_meta_field_names(f: h5py.File) -> tuple[str, ...]:
    return tuple(_read_declared_meta_fields(f))


def _read_declared_stat_names(f: h5py.File) -> tuple[str, ...]:
    """Read the full declared stats key set in stable store order."""
    return schema.CORE_STATS_KEYS + _read_declared_extra_stat_names(f)


# Result reads


def _read_meta_values(f: h5py.File) -> dict[str, Any]:
    """Read the full analysis-level ``/meta`` view including declared extras."""
    pixel_scale_path = f"/{schema.KEY_META_SECTION}/{schema.KEY_META_PIXEL_SCALE}"
    tel_diameter_path = f"/{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_DIAMETER}"
    tel_pupil_path = f"/{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_PUPIL}"
    values = {
        schema.KEY_META_PIXEL_SCALE: hdf5.read_dataset_value(
            hdf5.require_dataset_ndim(hdf5.require_dataset(f, pixel_scale_path), path=pixel_scale_path, ndim=1)
        ),
        schema.KEY_META_TEL_DIAMETER: hdf5.read_dataset_value(
            hdf5.require_dataset_ndim(hdf5.require_dataset(f, tel_diameter_path), path=tel_diameter_path, ndim=0)
        ),
        schema.KEY_META_TEL_PUPIL: hdf5.read_dataset_value(
            hdf5.require_dataset_ndim(hdf5.require_dataset(f, tel_pupil_path), path=tel_pupil_path, ndim=2)
        ),
    }
    for name in _read_declared_meta_field_names(f):
        path = f"/{schema.KEY_META_SECTION}/{name}"
        values[name] = hdf5.read_dataset_value(hdf5.require_dataset_ndim(hdf5.require_dataset(f, path), path=path, ndim=1))
    return values


# Simulation store


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
        real_dtype(simulation.get(schema.KEY_SIMULATION_PRECISION, "float32"))
        setup = upgrade_setup_peak_method(setup)
        setup[schema.KEY_SETUP_EE_CENTER] = str(
            setup.get(schema.KEY_SETUP_EE_CENTER, schema.DEFAULT_SETUP_EE_CENTER)
        ).strip()
        validate_setup_payload_core(setup)
        zeropoint_key = schema.KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT
        setup[zeropoint_key] = require_quantity(
            setup[zeropoint_key],
            schema.SETUP_FIELD_UNITS[zeropoint_key],
            label=f"setup['{zeropoint_key}']",
        )
        m_sci = get_num_sci(setup)
        num_sims = validate_options_payload_core(options, expected_num_sci=m_sci)
        validate_atm_profile_ids(setup, options)
        extra_stat_fields = result_schema.extra_stat_fields(simulation)
        meta_fields = result_schema.normalize_meta_fields(
            simulation.get(schema.KEY_SIMULATION_META_FIELDS, {})
        )
        result_schema.diagnostic_field_specs(simulation)
        simulation = dict(simulation)
        simulation[schema.KEY_SIMULATION_EXTRA_STAT_FIELDS] = {
            name: unit_string(unit) for name, unit in extra_stat_fields.items()
        }
        if meta_fields:
            simulation[schema.KEY_SIMULATION_META_FIELDS] = {
                name: unit or "" for name, unit in meta_fields.items()
            }

        if self.path.exists() and overwrite:
            self.path.unlink()

        with h5py.File(self.path, "w") as f:
            initialize_layout(
                f,
                simulation=simulation,
                setup=setup,
                options=options,
                save_psfs=save_psfs,
                standard_stats=schema.CORE_STATS_KEYS,
            )
            f.require_group(schema.KEY_STATUS_SECTION).create_dataset(
                schema.KEY_STATUS_STATE,
                data=np.full((num_sims,), int(SimulationState.PENDING), dtype=np.uint8),
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
            setup, _ = _read_stored_setup(f[schema.KEY_SETUP_SECTION])
            return setup

    def read_simulation(self) -> dict[str, Any]:
        """Read ``/simulation`` as a validated final-contract payload.

        Returns:
            Current-contract mapping decoded from ``/simulation``.
        """

        with h5py.File(self.path, "r") as f:
            simulation = hdf5.read_node(f[schema.KEY_SIMULATION_SECTION])
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
                ds = hdf5.require_dataset(f, path)
                if ds.ndim == 0:
                    raise ValueError(f"{path} must be per-simulation with first dim N.")
                columns[key] = hdf5.read_dataset_value(ds)
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
                ds = hdf5.require_dataset_ndim(hdf5.require_dataset(f, path), path=path, ndim=expected_ndim)
                columns[key] = hdf5.read_dataset_value(ds)
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
            return hdf5.read_diagnostics(f[schema.KEY_DIAGNOSTICS_SECTION])

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

                value = hdf5.read_row(ds, sim_idx, path=f"/options/{key}")
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                row[key] = value

        return row

    def read_simulation_meta(self, sim_idx: int) -> dict[str, Any]:
        """Read one simulation's persisted meta view."""

        with h5py.File(self.path, "r") as f:
            pixel_scale_path = f"/{schema.KEY_META_SECTION}/{schema.KEY_META_PIXEL_SCALE}"
            tel_diameter_path = f"/{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_DIAMETER}"
            tel_pupil_path = f"/{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_PUPIL}"

            pixel_scale = hdf5.read_row(
                hdf5.require_dataset_ndim(hdf5.require_dataset(f, pixel_scale_path), path=pixel_scale_path, ndim=1),
                sim_idx,
                path=pixel_scale_path,
            )
            tel_diameter = hdf5.read_dataset_value(hdf5.require_dataset_ndim(
                hdf5.require_dataset(f, tel_diameter_path),
                path=tel_diameter_path,
                ndim=0,
            ))
            tel_pupil = hdf5.read_dataset_value(hdf5.require_dataset_ndim(
                hdf5.require_dataset(f, tel_pupil_path),
                path=tel_pupil_path,
                ndim=2,
            ))
            meta = {
                schema.KEY_META_PIXEL_SCALE: pixel_scale,
                schema.KEY_META_TEL_DIAMETER: tel_diameter,
                schema.KEY_META_TEL_PUPIL: tel_pupil,
            }
            for name in _read_declared_meta_field_names(f):
                path = f"/{schema.KEY_META_SECTION}/{name}"
                meta[name] = hdf5.read_row(
                    hdf5.require_dataset_ndim(hdf5.require_dataset(f, path), path=path, ndim=1),
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
                ds = hdf5.require_dataset_ndim(hdf5.require_dataset(f, path), path=path, ndim=expected_ndim)
                stats_row[key] = hdf5.read_row(ds, sim_idx, path=path)

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
        sim_idx = hdf5.index_value(sim_idx)
        with h5py.File(self.path, "r") as f:
            if schema.KEY_DIAGNOSTICS_SECTION not in f:
                return {}
            return hdf5.read_diagnostics(f[schema.KEY_DIAGNOSTICS_SECTION], sim_idx=sim_idx, path="/diagnostics")

    def read_simulation_psfs(self, sim_idx: int) -> np.ndarray:
        """Read one simulation's persisted PSF cube from ``/psfs/data``."""

        with h5py.File(self.path, "r") as f:
            path = f"/{schema.KEY_PSFS_SECTION}/{schema.KEY_PSFS_DATA}"
            ds = hdf5.require_dataset_ndim(hdf5.require_dataset(f, path), path=path, ndim=4)
            return np.asarray(hdf5.read_row(ds, sim_idx, path=path))

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
            if schema.KEY_SIMULATION_PRECISION in f[schema.KEY_SIMULATION_SECTION]:
                try:
                    dtype = hdf5.output_dtype(f)
                    for group_name in (schema.KEY_STATS_SECTION, schema.KEY_META_SECTION, schema.KEY_PSFS_SECTION):
                        if group_name in f:
                            for name, dataset in f[group_name].items():
                                if isinstance(dataset, h5py.Dataset) and dataset.dtype.kind == "f" and dataset.dtype != dtype:
                                    issues.append(f"Stored precision mismatch at /{group_name}/{name}")
                except ValueError as error:
                    issues.append(str(error))
                except KeyError:
                    pass  # Required missing datasets are reported below.


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
                simulation_data = hdf5.read_node(f[schema.KEY_SIMULATION_SECTION])
                simulation_data = resolve_simulation_payload_for_load(simulation_data)
                extra_stat_fields = result_schema.extra_stat_fields(simulation_data)
                extra_stat_names = tuple(extra_stat_fields)
                meta_fields = {
                    name: None if unit is None else u.Unit(unit)
                    for name, unit in normalize_meta_fields(
                        simulation_data.get(schema.KEY_SIMULATION_META_FIELDS, {})
                    ).items()
                }
                meta_field_names = tuple(meta_fields)
                diagnostic_field_specs = result_schema.diagnostic_field_specs(simulation_data)
            except Exception as exc:
                issues.append(f"Invalid /simulation payload: {exc}")
                extra_stat_names = ()
                extra_stat_fields = {}
                meta_field_names = ()
                meta_fields = {}
                diagnostic_field_specs = {}

            try:
                setup_data, legacy_zeropoint_unit = _read_stored_setup(f[schema.KEY_SETUP_SECTION])
            except Exception as exc:
                issues.append(f"Invalid /setup payload: {exc}")
                setup_data = None
                legacy_zeropoint_unit = False

            try:
                options_data = hdf5.read_node(f[schema.KEY_OPTION_SECTION])
                validate_options_payload_core(
                    options_data,
                    expected_num_sims=n,
                    expected_num_sci=get_num_sci(setup_data) if setup_data is not None else None,
                )
                if setup_data is not None:
                    validate_atm_profile_ids(setup_data, options_data)
            except Exception as exc:
                issues.append(f"Invalid /options payload: {exc}")

            for name, unit in schema.SETUP_FIELD_UNITS.items():
                if name == schema.KEY_SETUP_NGS_MAGNITUDE_ZEROPOINT and legacy_zeropoint_unit:
                    continue
                hdf5.collect_dataset_unit_issue(
                    f,
                    f"{schema.KEY_SETUP_SECTION}/{name}",
                    unit,
                    issues,
                )
            profiles_path = (
                f"{schema.KEY_SETUP_SECTION}/{schema.KEY_SETUP_ATM_PROFILES}"
            )
            if profiles_path in f and isinstance(f[profiles_path], h5py.Group):
                for profile_id, profile_node in f[profiles_path].items():
                    if not isinstance(profile_node, h5py.Group):
                        continue
                    for name, unit in ATM_PROFILE_FIELD_UNITS.items():
                        hdf5.collect_dataset_unit_issue(
                            f,
                            f"{profiles_path}/{profile_id}/{name}",
                            unit,
                            issues,
                        )
            for name, unit in schema.OPTION_FIELD_UNITS.items():
                if name == schema.KEY_OPTION_SEEING:
                    continue
                hdf5.collect_dataset_unit_issue(
                    f,
                    f"{schema.KEY_OPTION_SECTION}/{name}",
                    unit,
                    issues,
                )

            pixel_scale_mas_data = f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_PIXEL_SCALE}"]
            tel_diameter_m_data = f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_DIAMETER}"]
            tel_pupil_data = f[f"{schema.KEY_META_SECTION}/{schema.KEY_META_TEL_PUPIL}"]
            extra_meta_data = {
                name: f[f"{schema.KEY_META_SECTION}/{name}"]
                for name in meta_field_names
                if name in meta_group
            }

            sr_data = f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_SR}"]
            ee_data = f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_EE}"]
            fwhm_mas_data = f[f"{schema.KEY_STATS_SECTION}/{schema.KEY_STATS_FWHM}"]
            extra_stat_data = {
                name: f[f"{schema.KEY_STATS_SECTION}/{name}"]
                for name in extra_stat_names
                if name in stats_group
            }

            core_meta_data = {
                schema.KEY_META_PIXEL_SCALE: pixel_scale_mas_data,
                schema.KEY_META_TEL_DIAMETER: tel_diameter_m_data,
                schema.KEY_META_TEL_PUPIL: tel_pupil_data,
            }
            for name, dataset in core_meta_data.items():
                expected_unit = unit_string(schema.META_FIELD_UNITS[name])
                if dataset.attrs.get(UNITS_ATTRIBUTE) != expected_unit:
                    issues.append(
                        f"/meta/{name} must declare units={expected_unit!r}."
                    )

            core_stat_data = {
                schema.KEY_STATS_SR: sr_data,
                schema.KEY_STATS_EE: ee_data,
                schema.KEY_STATS_FWHM: fwhm_mas_data,
            }
            for name, dataset in core_stat_data.items():
                expected_unit = unit_string(schema.STATS_FIELD_UNITS[name])
                if dataset.attrs.get(UNITS_ATTRIBUTE) != expected_unit:
                    issues.append(
                        f"/stats/{name} must declare units={expected_unit!r}."
                    )

            if sr_data.ndim != 2:
                issues.append("/stats/sr must be 2D [N, M].")
            if ee_data.ndim != 3:
                issues.append("/stats/ee must be 3D [N, M, A].")
            if fwhm_mas_data.ndim != 2:
                issues.append("/stats/fwhm must be 2D [N, M].")
            for name in extra_stat_names:
                if name not in stats_group:
                    issues.append(f"Missing declared extra stats dataset '/stats/{name}'.")
                elif stats_group[name].ndim != 2:
                    issues.append(f"/stats/{name} must be 2D [N, M].")
                elif stats_group[name].attrs.get(UNITS_ATTRIBUTE) != unit_string(extra_stat_fields[name]):
                    issues.append(f"/stats/{name} must declare units={unit_string(extra_stat_fields[name])!r}.")
            for name in meta_field_names:
                if name not in extra_meta_data:
                    issues.append(f"Missing declared meta dataset '/meta/{name}'.")
                elif extra_meta_data[name].ndim != 1:
                    issues.append(f"/meta/{name} must be 1D [N].")
                elif meta_fields[name] is None and UNITS_ATTRIBUTE in extra_meta_data[name].attrs:
                    issues.append(f"/meta/{name} must not declare units.")
                elif (
                    meta_fields[name] is not None
                    and extra_meta_data[name].attrs.get(UNITS_ATTRIBUTE) != unit_string(meta_fields[name])
                ):
                    issues.append(f"/meta/{name} must declare units={unit_string(meta_fields[name])!r}.")
            if pixel_scale_mas_data.ndim != 1:
                issues.append("/meta/pixel_scale must be 1D [N].")
            if tel_diameter_m_data.ndim != 0:
                issues.append("/meta/tel_diameter must be a scalar.")
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
                    issues.append("/meta/pixel_scale first dimension must match /status/state length.")
                for name, ds in extra_meta_data.items():
                    if ds.shape[0] != n:
                        issues.append(f"/meta/{name} first dimension must match /status/state length.")
                if (
                    sr_data.shape[1] != ee_data.shape[1]
                    or sr_data.shape[1] != fwhm_mas_data.shape[1]
                ):
                    issues.append("Stats M dimension mismatch between sr/ee/fwhm.")
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
                            ds = hdf5.require_diagnostic_dataset(diagnostics_group, name)
                            expected_shape = (n,)
                            if setup_data is not None:
                                options_data = hdf5.read_node(f[schema.KEY_OPTION_SECTION])
                                expected_shape = (n,) + diagnostic_shape(spec, setup_data, options_data)
                            if ds.shape != expected_shape:
                                issues.append(f"/diagnostics/{name} shape mismatch: expected {expected_shape}, got {ds.shape}.")
                            expected_unit = spec.get("unit")
                            actual_unit = ds.attrs.get(UNITS_ATTRIBUTE)
                            if expected_unit is None:
                                if actual_unit is not None:
                                    issues.append(
                                        f"/diagnostics/{name} must not declare units."
                                    )
                            elif actual_unit != expected_unit:
                                issues.append(
                                    f"/diagnostics/{name} must declare units={expected_unit!r}."
                                )
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
        sim_idx = hdf5.index_value(sim_idx)

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
            extra_stat_fields = _read_declared_extra_stat_fields(f)
            meta_fields = _read_declared_meta_fields(f)
            require_psfs = schema.KEY_PSFS_SECTION in f

            dtype = hdf5.output_dtype(f)
            validate_successful_result(
                result,
                num_sci,
                num_ee,
                extra_stat_fields=extra_stat_fields,
                meta_fields=meta_fields,
                require_psfs=require_psfs,
                dtype=dtype,
            )

            statistics = dict(result.stats)
            ee = statistics[schema.KEY_STATS_EE]
            if ee.ndim == 1:
                statistics[schema.KEY_STATS_EE] = ee[:, np.newaxis]
            write_observation(
                f, index=sim_idx, meta=result.meta, stats=statistics,
                psfs=result.psfs, diagnostics=result.diagnostics,
            )

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
        sim_idx = hdf5.index_value(sim_idx)

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
            clear_observation(f, index=sim_idx)
            state[sim_idx] = np.uint8(int(SimulationState.FAILED))
