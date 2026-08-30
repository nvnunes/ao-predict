"""Simulation configuration normalization and options payload helpers.

This module normalizes API/CLI configuration inputs for:
- ``simulation``: simulator identity and base configuration source
- ``setup``: invariant simulation geometry and calibration inputs
- ``options``: per-simulation atmospheric/source overrides

It also converts table/broadcast options into the canonical columnar
``/options`` payload used by the simulation API and persistence layer.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
from astropy import units as u

from .._units import parse_unit, quantity_from_value, quantity_value
from . import schema
from .helpers import get_num_sci
from .interfaces import Simulation
from ..utils import (
    as_array_dict,
    as_float_scalar,
    as_float_vector,
    require_finite_positive_scalar,
    require_lowercase_mapping_keys,
)
from .validation import validate_atm_profile_ids, validate_options_payload_core


# Primitive option value parsing helpers

def _is_null_like(value: object) -> bool:
    """Return ``True`` when a value should be treated as null/empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "null", "none", "nan", "na"}
    return False


def _to_optional_float(value: object, label: str) -> float:
    """Parse numeric-like input to float, mapping null-like values to ``NaN``."""
    if _is_null_like(value):
        return float("nan")
    try:
        return as_float_scalar(value, label=label)
    except Exception as exc:
        raise ValueError(f"Invalid numeric value for {label}: {value!r}") from exc


def _to_optional_quantity_float(
    value: object,
    unit: u.UnitBase,
    label: str,
) -> float:
    """Parse an optional scalar quantity into one canonical numerical value."""
    if _is_null_like(value):
        return float("nan")
    array = quantity_value(value, unit, label=label)
    if array.ndim != 0:
        raise ValueError(f"{label} must be scalar-compatible, got shape {array.shape}.")
    return float(array)


def _option_unit_for_column(name: str) -> u.UnitBase | None:
    """Return the canonical unit for one scalar or expanded-NGS option name."""
    if name in schema.OPTION_FIELD_UNITS:
        return schema.OPTION_FIELD_UNITS[name]
    match = schema.KEY_OPTION_NGS_COLUMN_RE.match(name)
    if match is None:
        return None
    suffix = match.group(2)
    return {
        schema.KEY_OPTION_R_SUFFIX: schema.OPTION_FIELD_UNITS[schema.KEY_OPTION_NGS_R],
        schema.KEY_OPTION_THETA_SUFFIX: schema.OPTION_FIELD_UNITS[schema.KEY_OPTION_NGS_THETA],
        schema.KEY_OPTION_MAGNITUDE_SUFFIX: schema.OPTION_FIELD_UNITS[schema.KEY_OPTION_NGS_MAGNITUDE],
    }[suffix]


def _canonical_option_quantities(values: Mapping[str, object]) -> dict[str, object]:
    """Attach canonical units to prepared numerical option arrays."""
    return {
        key: (
            quantity_from_value(value, schema.OPTION_FIELD_UNITS[key])
            if key in schema.OPTION_FIELD_UNITS
            else value
        )
        for key, value in values.items()
    }


# Broadcast/table option parsing

def _parse_broadcast_defaults(
    options_broadcast: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Split broadcast option input into scalar defaults and explicit NGS input.

    Args:
        options_broadcast: Broadcast options mapping from CLI/API input.

    Returns:
        Tuple ``(scalar_defaults, broadcast_ngs)`` where scalar defaults are
        1D-style option values and explicit broadcast NGS values are per-star
        vectors applied uniformly to every simulation.
    """
    require_lowercase_mapping_keys(
        options_broadcast, label=f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_BROADCAST}"
    )

    scalar_defaults: dict[str, object] = {}
    broadcast_ngs: dict[str, np.ndarray] = {}

    ngs_cfg = options_broadcast.get(schema.KEY_CFG_OPTION_NGS)
    if ngs_cfg is not None:
        if not isinstance(ngs_cfg, list):
            raise ValueError(f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_NGS} must be a list when provided.")
        vals_r: list[float] = []
        vals_t: list[float] = []
        vals_m: list[float] = []
        for i, star in enumerate(ngs_cfg):
            if not isinstance(star, Mapping):
                raise ValueError(f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_NGS}[{i}] must be a mapping/object.")
            require_lowercase_mapping_keys(star, label=f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_NGS}[{i}]")
            vals_r.append(
                _to_optional_quantity_float(
                    star.get(schema.KEY_OPTION_R_SUFFIX),
                    schema.OPTION_FIELD_UNITS[schema.KEY_OPTION_NGS_R],
                    f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_NGS}[{i}].{schema.KEY_OPTION_R_SUFFIX}",
                )
            )
            vals_t.append(
                _to_optional_quantity_float(
                    star.get(schema.KEY_OPTION_THETA_SUFFIX),
                    schema.OPTION_FIELD_UNITS[schema.KEY_OPTION_NGS_THETA],
                    f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_NGS}[{i}].{schema.KEY_OPTION_THETA_SUFFIX}",
                )
            )
            vals_m.append(
                _to_optional_quantity_float(
                    star.get(schema.KEY_OPTION_MAGNITUDE_SUFFIX),
                    schema.OPTION_FIELD_UNITS[schema.KEY_OPTION_NGS_MAGNITUDE],
                    f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_NGS}[{i}].{schema.KEY_OPTION_MAGNITUDE_SUFFIX}",
                )
            )
        broadcast_ngs[schema.KEY_OPTION_NGS_R] = as_float_vector(vals_r, label=schema.KEY_OPTION_NGS_R)
        broadcast_ngs[schema.KEY_OPTION_NGS_THETA] = as_float_vector(
            vals_t, label=schema.KEY_OPTION_NGS_THETA
        )
        broadcast_ngs[schema.KEY_OPTION_NGS_MAGNITUDE] = as_float_vector(vals_m, label=schema.KEY_OPTION_NGS_MAGNITUDE)

    for key, value in options_broadcast.items():
        if key == schema.KEY_CFG_OPTION_NGS:
            continue
        if key in schema.OPTION_KEYS_1D or key == schema.KEY_OPTION_SEEING:
            if isinstance(value, (list, Mapping)):
                raise ValueError(f"options.{key} broadcast default must be a scalar.")
            if key in schema.OPTION_FIELD_UNITS:
                scalar_defaults[key] = _to_optional_quantity_float(
                    value,
                    schema.OPTION_FIELD_UNITS[key],
                    f"options.{key}",
                )
            else:
                scalar_defaults[key] = value
            continue

        m = schema.KEY_OPTION_NGS_COLUMN_RE.match(str(key))
        if m is not None:
            idx = int(m.group(1)) - 1
            attr = m.group(2)
            if idx < 0:
                raise ValueError(f"Invalid NGS column '{key}': index must be >= 1.")
            map_key = f"{schema.KEY_OPTION_NGS_PREFIX}{attr}"
            arr = broadcast_ngs.get(map_key)
            if arr is None:
                arr = np.full((idx + 1,), np.nan, dtype=float)
            elif arr.size <= idx:
                arr = np.pad(arr, (0, idx + 1 - arr.size), constant_values=np.nan)
            canonical_unit = _option_unit_for_column(str(key))
            assert canonical_unit is not None
            arr[idx] = _to_optional_quantity_float(
                value,
                canonical_unit,
                f"options.{key}",
            )
            broadcast_ngs[map_key] = arr
            continue

        raise ValueError(f"Unsupported options broadcast key: '{key}'.")

    return scalar_defaults, broadcast_ngs


def _build_options_from_table(
    columns: list[str],
    rows: list[list[object]],
    units: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], int]:
    """Convert table-form options into typed per-option arrays.

    Args:
        columns: Lowercase table column names.
        rows: Table rows, one row per simulation.

    Returns:
        Tuple ``(options_payload, num_sims)``.
    """
    n = len(rows)
    required_unit_columns = {column for column in columns if _option_unit_for_column(column) is not None}
    declared_unit_columns = set(units)
    missing_units = sorted(required_unit_columns - declared_unit_columns)
    unexpected_units = sorted(declared_unit_columns - set(columns))
    nonphysical_units = sorted(
        column for column in declared_unit_columns if _option_unit_for_column(column) is None
    )
    if missing_units:
        raise ValueError(f"options.table.units is missing physical columns: {', '.join(missing_units)}.")
    if unexpected_units:
        raise ValueError(f"options.table.units contains absent columns: {', '.join(unexpected_units)}.")
    if nonphysical_units:
        raise ValueError(
            "options.table.units must omit nonphysical columns: " + ", ".join(nonphysical_units) + "."
        )
    parsed_units = {
        name: parse_unit(value, label=f"options.table.units[{name!r}]")
        for name, value in units.items()
    }
    raw_scalars: dict[str, np.ndarray] = {}
    ngs_parts: dict[str, dict[int, np.ndarray]] = {key: {} for key in schema.OPTION_KEYS_NGS}
    max_ngs = 0

    for col in columns:
        if col in schema.OPTION_KEYS_1D or col == schema.KEY_OPTION_SEEING:
            raw_scalars[col] = np.full((n,), np.nan, dtype=float)
            continue
        m = schema.KEY_OPTION_NGS_COLUMN_RE.match(col)
        if m is None:
            raise ValueError(f"Unsupported options table column: '{col}'.")
        idx = int(m.group(1))
        attr = m.group(2)
        max_ngs = max(max_ngs, idx)
        ngs_parts[f"{schema.KEY_OPTION_NGS_PREFIX}{attr}"][idx - 1] = np.full((n,), np.nan, dtype=float)

    for i, row in enumerate(rows):
        if not isinstance(row, list):
            raise ValueError(f"options.table row {i} must be a list.")
        if len(row) != len(columns):
            raise ValueError(
                f"options.table row {i} has {len(row)} values, expected {len(columns)} from options.columns."
            )
        for j, col in enumerate(columns):
            value = row[j]
            if col in raw_scalars:
                raw_scalars[col][i] = _to_optional_float(value, f"options.table[{i}].{col}")
                continue
            m = schema.KEY_OPTION_NGS_COLUMN_RE.match(col)
            assert m is not None
            idx = int(m.group(1)) - 1
            attr = m.group(2)
            key = f"{schema.KEY_OPTION_NGS_PREFIX}{attr}"
            ngs_parts[key][idx][i] = _to_optional_float(value, f"options.table[{i}].{col}")

    out: dict[str, np.ndarray] = {}
    for key, arr in raw_scalars.items():
        out[key] = as_float_vector(arr, label=key)

    if max_ngs > 0:
        for key in schema.OPTION_KEYS_NGS:
            mat = np.full((n, max_ngs), np.nan, dtype=float)
            for idx, col_arr in ngs_parts[key].items():
                mat[:, idx] = as_float_vector(col_arr, label=f"{key}[{idx}]")
            out[key] = mat

    for column in columns:
        canonical_unit = _option_unit_for_column(column)
        if canonical_unit is None:
            continue
        declared_unit = parsed_units[column]
        if not declared_unit.is_equivalent(canonical_unit):
            raise ValueError(
                f"options.table.units[{column!r}] must be equivalent to {canonical_unit}."
            )
        if column in out:
            out[column] = np.asarray((out[column] * declared_unit).to_value(canonical_unit))
            continue
        match = schema.KEY_OPTION_NGS_COLUMN_RE.match(column)
        assert match is not None
        index = int(match.group(1)) - 1
        key = f"{schema.KEY_OPTION_NGS_PREFIX}{match.group(2)}"
        out[key][:, index] = np.asarray(
            (out[key][:, index] * declared_unit).to_value(canonical_unit)
        )

    return out, n
# Options payload completion and validation

def _derive_ngs_used(options_row: Mapping[str, object]) -> np.ndarray | None:
    """Derive runtime ``ngs_used`` from one simulation's NGS option vectors.

    Returns ``None`` when one or more NGS option vectors are absent.
    """
    ngs_keys = (
        schema.KEY_OPTION_NGS_R,
        schema.KEY_OPTION_NGS_THETA,
        schema.KEY_OPTION_NGS_MAGNITUDE,
    )
    if not all(k in options_row for k in ngs_keys):
        return None

    ngs_r = quantity_value(options_row[schema.KEY_OPTION_NGS_R], u.arcsec, label=schema.KEY_OPTION_NGS_R, dtype=float).reshape(-1)
    ngs_theta = quantity_value(options_row[schema.KEY_OPTION_NGS_THETA], u.deg, label=schema.KEY_OPTION_NGS_THETA, dtype=float).reshape(-1)
    ngs_magnitude = quantity_value(options_row[schema.KEY_OPTION_NGS_MAGNITUDE], u.mag, label=schema.KEY_OPTION_NGS_MAGNITUDE, dtype=float).reshape(-1)
    if ngs_r.shape != ngs_theta.shape or ngs_r.shape != ngs_magnitude.shape:
        raise ValueError("ngs_r/ngs_theta/ngs_magnitude option vectors must have identical shape.")
    return np.isfinite(ngs_r) & np.isfinite(ngs_theta) & np.isfinite(ngs_magnitude)


def add_runtime_derived_options(options_row: Mapping[str, object]) -> dict[str, object]:
    """Add core runtime-derived option values for one simulation.

    Args:
        options_row: Persisted options row for one simulation.

    Returns:
        Options row augmented with runtime-only derived fields.
    """
    out = dict(options_row)
    ngs_used = _derive_ngs_used(out)
    if ngs_used is not None:
        out[schema.KEY_OPTION_NGS_USED] = ngs_used
    return out


def replace_seeing_with_r0(
    options_payload: Mapping[str, object],
    *,
    atm_wavelength: float,
    num_sims: int,
    has_explicit_r0: bool | None = None,
) -> dict[str, np.ndarray]:
    """Normalize ``seeing`` input into canonical persisted ``r0``.

    Args:
        options_payload: Candidate options payload.
        atm_wavelength: Atmosphere reference wavelength in microns.
        num_sims: Required number of simulations ``N``.
        has_explicit_r0: Optional override indicating whether ``r0`` was
            explicitly supplied by the caller.

    Returns:
        Normalized options payload with ``r0`` present and ``seeing`` removed.
    """
    out = as_array_dict(dict(options_payload), copy_arrays=True)
    n = int(num_sims)
    if schema.KEY_OPTION_SEEING not in out:
        return out

    atm_wavelength = require_finite_positive_scalar(
        atm_wavelength, label=f"{schema.KEY_SETUP_SECTION}.{schema.KEY_SETUP_ATM_WAVELENGTH}"
    )
    seeing = as_float_vector(out[schema.KEY_OPTION_SEEING], label=schema.KEY_OPTION_SEEING, length=n)
    r0 = as_float_vector(
        out.get(schema.KEY_OPTION_R0, np.full((n,), np.nan)),
        label=schema.KEY_OPTION_R0,
        length=n,
    )

    seeing_present = np.isfinite(seeing)
    if np.any(seeing_present):
        if np.any(seeing[seeing_present] <= 0.0):
            raise ValueError(f"{schema.KEY_OPTION_SEEING} values must be > 0 when provided.")
        seeing_rad = seeing[seeing_present] * (math.pi / 648000.0)
        r0_from_seeing = 0.98 * (float(atm_wavelength) * 1e-6) / seeing_rad
        r0_existing = r0[seeing_present]

        explicit_r0 = (schema.KEY_OPTION_R0 in out) if has_explicit_r0 is None else bool(has_explicit_r0)
        if explicit_r0:
            r0_finite = np.isfinite(r0_existing)
            if np.any(r0_finite):
                if not np.allclose(r0_existing[r0_finite], r0_from_seeing[r0_finite], rtol=1e-3, atol=1e-6):
                    raise ValueError(
                        "Inconsistent per-sim atmospheric inputs: r0 and seeing both provided "
                        "but do not match for one or more simulations."
                    )
        else:
            r0_finite = np.zeros_like(r0_existing, dtype=bool)

        r0_existing[~r0_finite] = r0_from_seeing[~r0_finite]
        r0[seeing_present] = r0_existing

    out[schema.KEY_OPTION_R0] = r0
    out.pop(schema.KEY_OPTION_SEEING, None)
    return out


def _finalize_options(
    options_arrays: dict[str, np.ndarray],
    *,
    num_sims: int,
    atm_wavelength: float,
    scalar_defaults: dict[str, object],
    broadcast_ngs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply defaults and final consistency checks to option arrays.

    Args:
        options_arrays: Partially prepared per-option arrays.
        num_sims: Required number of simulations ``N``.
        atm_wavelength: Atmosphere reference wavelength used for seeing->r0 conversion.
        scalar_defaults: Scalar defaults broadcast across simulations.
        broadcast_ngs: Explicit per-star NGS vectors applied uniformly to all
            simulations.

    Returns:
        Fully validated persisted ``/options`` payload.

    Raises:
        ValueError: If defaults, NGS inputs, or final option arrays violate
            the shared persisted ``/options`` contract.
    """
    n = int(num_sims)
    if n <= 0:
        raise ValueError("Number of simulations N must be > 0.")

    out: dict[str, np.ndarray] = as_array_dict(options_arrays, copy_arrays=True)

    for key, val in scalar_defaults.items():
        if key not in schema.OPTION_KEYS_1D and key != schema.KEY_OPTION_SEEING:
            continue
        default_val = _to_optional_float(val, f"options.{key}")
        if key not in out:
            out[key] = np.full((n,), default_val, dtype=float)
        else:
            arr = np.asarray(out[key], dtype=float).reshape(-1)
            if arr.shape[0] != n:
                raise ValueError(f"Option '{key}' length mismatch: expected N={n}, got {arr.shape[0]}.")
            arr[np.isnan(arr)] = default_val
            out[key] = arr

    explicit_ngs_keys = [key for key in schema.OPTION_KEYS_NGS if key in out]
    if explicit_ngs_keys and broadcast_ngs:
        raise ValueError("NGS options must be provided either via table/array input or via options.broadcast.ngs, not both.")

    for key in explicit_ngs_keys:
        arr = np.asarray(out[key], dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{key} must be 2D [N, Kmax].")
        if arr.shape[0] != n:
            raise ValueError(f"{key} first dimension must be N={n}.")

    if broadcast_ngs:
        missing_broadcast_ngs = [key for key in schema.OPTION_KEYS_NGS if key not in broadcast_ngs]
        if missing_broadcast_ngs:
            raise ValueError("Explicit broadcast NGS input must provide ngs_r, ngs_theta, and ngs_magnitude together.")
        kmax = int(max(as_float_vector(broadcast_ngs[key], label=key).shape[0] for key in schema.OPTION_KEYS_NGS))
        broadcast_values: dict[str, np.ndarray] = {}
        for key in schema.OPTION_KEYS_NGS:
            values = as_float_vector(broadcast_ngs[key], label=key)
            if values.shape[0] != kmax:
                raise ValueError("Explicit broadcast NGS arrays must have identical length Kmax.")
            broadcast_values[key] = values

        finite_r = np.isfinite(broadcast_values[schema.KEY_OPTION_NGS_R])
        finite_t = np.isfinite(broadcast_values[schema.KEY_OPTION_NGS_THETA])
        finite_m = np.isfinite(broadcast_values[schema.KEY_OPTION_NGS_MAGNITUDE])
        partial = (finite_r | finite_t | finite_m) & ~(finite_r & finite_t & finite_m)
        if np.any(partial):
            raise ValueError(
                "Each broadcast NGS slot must be either all finite or all NaN across ngs_r/ngs_theta/ngs_magnitude."
            )

        for key, values in broadcast_values.items():
            out[key] = np.broadcast_to(values.reshape(1, kmax), (n, kmax)).copy()

    out = replace_seeing_with_r0(
        out,
        atm_wavelength=atm_wavelength,
        num_sims=n,
        has_explicit_r0=(schema.KEY_OPTION_R0 in options_arrays) or (schema.KEY_OPTION_R0 in scalar_defaults),
    )

    finalized: dict[str, np.ndarray] = {}
    for key in schema.OPTION_KEYS_1D:
        if key not in out:
            continue
        arr = np.asarray(out[key], dtype=float).reshape(-1)
        if arr.shape[0] != n:
            raise ValueError(f"{key} must have length N={n}.")
        if np.all(np.isnan(arr)):
            continue
        if np.any(np.isnan(arr)):
            raise ValueError(f"{key} has missing values after applying options precedence.")
        if key == schema.KEY_OPTION_ATM_PROFILE_ID:
            if not np.all(np.equal(arr, np.round(arr))):
                raise ValueError(f"{schema.KEY_OPTION_ATM_PROFILE_ID} must be integer-valued.")
            finalized[key] = arr.astype(np.int32)
        else:
            finalized[key] = arr.astype(float)

    ngs_present = any(k in out for k in schema.OPTION_KEYS_NGS)
    if ngs_present:
        missing_ngs_keys = [key for key in schema.OPTION_KEYS_NGS if key not in out]
        if missing_ngs_keys:
            raise ValueError("Explicit NGS options must provide ngs_r, ngs_theta, and ngs_magnitude together.")

        ngs_r = np.asarray(out[schema.KEY_OPTION_NGS_R], dtype=float)
        ngs_t = np.asarray(out[schema.KEY_OPTION_NGS_THETA], dtype=float)
        ngs_m = np.asarray(out[schema.KEY_OPTION_NGS_MAGNITUDE], dtype=float)
        if ngs_r.shape != ngs_t.shape or ngs_r.shape != ngs_m.shape:
            raise ValueError("ngs_r/ngs_theta/ngs_magnitude must have identical [N, Kmax] shape.")
        if ngs_r.shape[0] != n:
            raise ValueError("NGS option arrays first dimension must match N.")
        if ngs_r.shape[1] == 0:
            raise ValueError("NGS option arrays must have Kmax >= 1.")
        finite_r = np.isfinite(ngs_r)
        finite_t = np.isfinite(ngs_t)
        finite_m = np.isfinite(ngs_m)
        partial = (finite_r | finite_t | finite_m) & ~(finite_r & finite_t & finite_m)
        if np.any(partial):
            raise ValueError("Each NGS slot must be either all finite or all NaN.")
        finalized[schema.KEY_OPTION_NGS_R] = ngs_r
        finalized[schema.KEY_OPTION_NGS_THETA] = ngs_t
        finalized[schema.KEY_OPTION_NGS_MAGNITUDE] = ngs_m

    return finalized


def _validate_completed_options_payload(
    setup_payload: Mapping[str, object],
    options_payload: Mapping[str, object],
) -> None:
    """Validate one completed persisted ``/options`` payload."""
    if not isinstance(options_payload, Mapping) or len(options_payload) == 0:
        raise ValueError("options payload must be a non-empty mapping.")
    validate_options_payload_core(
        options_payload,
        expected_num_sci=get_num_sci(setup_payload),
    )
    validate_atm_profile_ids(setup_payload, options_payload)


def _normalize_science_offset_options(
    options_payload: Mapping[str, object],
    *,
    num_sims: int,
    num_sci: int,
) -> dict[str, np.ndarray]:
    """Normalize optional science-offset matrices for sparse persistence.

    Present matrices must match ``[N, M]`` and contain finite values. Retained
    values use ``float32``; an all-zero axis is omitted so the zero-offset
    default consumes no persisted or loaded memory.
    """
    normalized = as_array_dict(options_payload, copy_arrays=False)
    expected_shape = (int(num_sims), int(num_sci))
    for key in schema.OPTION_KEYS_SCI_OFFSETS:
        if key not in normalized:
            continue
        try:
            values = np.asarray(normalized[key], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"options['{key}'] must contain real numeric values.") from exc
        if values.ndim != 2 or values.shape != expected_shape:
            raise ValueError(f"options['{key}'] must have shape [N, M]={expected_shape}, got {values.shape}.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"options['{key}'] must be finite.")
        if np.all(values == 0.0):
            normalized.pop(key)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                stored_values = values.astype(np.float32)
            if not np.all(np.isfinite(stored_values)):
                raise ValueError(f"options['{key}'] must be representable as finite float32 values.")
            normalized[key] = stored_values
    return normalized


# Public normalization helpers

def normalize_simulation_config(simulation: object) -> dict[str, object]:
    """Normalize simulation configuration input into a plain mapping.

    Args:
        simulation: Simulation config object or mapping.

    Returns:
        Normalized mapping suitable for simulation payload preparation.

    Raises:
        TypeError: If ``simulation`` has an unsupported type.
        ValueError: If key casing requirements are violated.
    """
    if isinstance(simulation, Mapping):
        require_lowercase_mapping_keys(simulation, label=schema.KEY_SIMULATION_SECTION)
        return {str(k): v for k, v in dict(simulation).items()}

    # API dataclass-like path (SimulationConfig).
    if hasattr(simulation, schema.KEY_SIMULATION_NAME):
        out: dict[str, object] = {schema.KEY_SIMULATION_NAME: getattr(simulation, schema.KEY_SIMULATION_NAME)}
        base_path = getattr(simulation, schema.KEY_CFG_SIMULATION_BASE_PATH, None)
        if base_path is not None:
            out[schema.KEY_CFG_SIMULATION_BASE_PATH] = base_path
        sim_fields = getattr(simulation, "specific_fields", {})
        if isinstance(sim_fields, Mapping):
            require_lowercase_mapping_keys(
                sim_fields, label=f"{schema.KEY_SIMULATION_SECTION}.specific_fields"
            )
            out.update({str(k): v for k, v in dict(sim_fields).items()})
        return {str(k): v for k, v in dict(out).items()}

    raise TypeError("simulation config must be a mapping or SimulationConfig-like object.")


def normalize_setup_config(setup: object) -> dict[str, object]:
    """Normalize setup configuration input into a plain mapping.

    Args:
        setup: Setup config object or mapping.

    Returns:
        Normalized mapping suitable for setup payload preparation.

    Raises:
        TypeError: If ``setup`` has an unsupported type.
        ValueError: If key casing requirements are violated.
    """
    if isinstance(setup, Mapping):
        require_lowercase_mapping_keys(setup, label=schema.KEY_SETUP_SECTION)
        return {str(k): v for k, v in dict(setup).items()}

    # API dataclass-like path (SetupConfig).
    if hasattr(setup, schema.KEY_SETUP_EE_APERTURES):
        out: dict[str, object] = {
            schema.KEY_SETUP_EE_APERTURES: getattr(setup, schema.KEY_SETUP_EE_APERTURES)
        }
        if hasattr(setup, schema.KEY_SETUP_SR_METHOD):
            sr_method = getattr(setup, schema.KEY_SETUP_SR_METHOD)
            if sr_method is not None:
                out[schema.KEY_SETUP_SR_METHOD] = sr_method
        if hasattr(setup, schema.KEY_SETUP_FWHM_SUMMARY):
            fwhm_summary = getattr(setup, schema.KEY_SETUP_FWHM_SUMMARY)
            if fwhm_summary is not None:
                out[schema.KEY_SETUP_FWHM_SUMMARY] = fwhm_summary
        if hasattr(setup, schema.KEY_SETUP_EE_GEOMETRY):
            ee_geometry = getattr(setup, schema.KEY_SETUP_EE_GEOMETRY)
            if ee_geometry is not None:
                out[schema.KEY_SETUP_EE_GEOMETRY] = ee_geometry
        sim_fields = getattr(setup, schema.KEY_CFG_SETUP_SPECIFIC_FIELDS, {})
        if isinstance(sim_fields, Mapping):
            require_lowercase_mapping_keys(
                sim_fields, label=f"{schema.KEY_SETUP_SECTION}.{schema.KEY_CFG_SETUP_SPECIFIC_FIELDS}"
            )
            out.update({str(k): v for k, v in dict(sim_fields).items()})
        return {str(k): v for k, v in dict(out).items()}

    raise TypeError("setup config must be a mapping or SetupConfig-like object.")


def normalize_table_options_config(raw_options_cfg: Mapping[str, object]) -> dict[str, object]:
    """Normalize table-form options config into canonical broadcast/columns/rows.

    Args:
        raw_options_cfg: Raw ``options`` mapping from YAML/API input.

    Returns:
        Canonical mapping with ``broadcast``, ``columns``, and ``rows`` keys.

    Raises:
        ValueError: If table/broadcast structure is invalid.
    """
    if not isinstance(raw_options_cfg, Mapping):
        raise ValueError(f"{schema.KEY_OPTION_SECTION} must be a mapping/object.")

    require_lowercase_mapping_keys(raw_options_cfg, label=schema.KEY_OPTION_SECTION)
    options_cfg = {str(k): v for k, v in dict(raw_options_cfg).items()}
    table_cfg = options_cfg.get(schema.KEY_CFG_OPTION_TABLE)
    columns: list[str] | None = None
    rows: list[list[object]] | None = None
    units: dict[str, object] = {}
    table_path: str | None = None
    if table_cfg is not None:
        if not isinstance(table_cfg, Mapping):
            raise ValueError(f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_TABLE} must be a mapping/object.")
        require_lowercase_mapping_keys(table_cfg, label=f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_TABLE}")
        table = {str(k): v for k, v in dict(table_cfg).items()}
        allowed_table_keys = {
            schema.KEY_CFG_OPTION_TABLE_PATH,
            schema.KEY_CFG_OPTION_COLUMNS,
            schema.KEY_CFG_OPTION_UNITS,
            schema.KEY_CFG_OPTION_ROWS,
        }
        unexpected_table_keys = sorted(set(table) - allowed_table_keys)
        if unexpected_table_keys:
            raise ValueError(
                f"Unsupported options.table keys: {', '.join(unexpected_table_keys)}."
            )
        path_raw = table.get(schema.KEY_CFG_OPTION_TABLE_PATH)
        columns_raw = table.get(schema.KEY_CFG_OPTION_COLUMNS)
        units_raw = table.get(schema.KEY_CFG_OPTION_UNITS, {})
        rows_raw = table.get(schema.KEY_CFG_OPTION_ROWS)
        if not isinstance(units_raw, Mapping):
            raise ValueError("options.table.units must be a mapping.")
        require_lowercase_mapping_keys(units_raw, label="options.table.units")
        units = {str(key): value for key, value in units_raw.items()}
        if path_raw is not None:
            if not isinstance(path_raw, str) or not path_raw.strip():
                raise ValueError("options.table.path must be a non-empty string.")
            if columns_raw is not None or rows_raw is not None:
                raise ValueError("options.table.path is mutually exclusive with columns and rows.")
            table_path = path_raw
        else:
            if not isinstance(columns_raw, list) or len(columns_raw) == 0:
                raise ValueError(
                    f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_TABLE}.{schema.KEY_CFG_OPTION_COLUMNS} must be a non-empty list."
                )
            if not isinstance(rows_raw, list):
                raise ValueError(
                    f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_TABLE}.{schema.KEY_CFG_OPTION_ROWS} must be a list."
                )
        if table_path is not None:
            columns_raw = None
            rows_raw = None
        assert columns_raw is None or isinstance(columns_raw, list)
        if columns_raw is not None and not all(isinstance(col, str) and col.strip() for col in columns_raw):
            raise ValueError(
                f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_TABLE}.{schema.KEY_CFG_OPTION_COLUMNS} entries must be non-empty strings."
            )
        if columns_raw is not None and not all(col == col.lower() for col in columns_raw):
            raise ValueError(
                f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_TABLE}.{schema.KEY_CFG_OPTION_COLUMNS} entries must be lowercase."
            )
        parsed_rows: list[list[object]] = []
        for i, row in enumerate(rows_raw or []):
            if not isinstance(row, list):
                raise ValueError(
                    f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_TABLE}.{schema.KEY_CFG_OPTION_ROWS}[{i}] must be a list."
                )
            parsed_rows.append(list(row))
        columns = None if columns_raw is None else [str(col) for col in columns_raw]
        rows = None if rows_raw is None else parsed_rows

    broadcast_cfg = options_cfg.get(schema.KEY_CFG_OPTION_BROADCAST)
    if broadcast_cfg is None:
        broadcast: dict[str, object] = {}
    else:
        if not isinstance(broadcast_cfg, Mapping):
            raise ValueError(f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_BROADCAST} must be a mapping/object.")
        require_lowercase_mapping_keys(
            broadcast_cfg, label=f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_BROADCAST}"
        )
        broadcast = {str(k): v for k, v in dict(broadcast_cfg).items()}

    for key, value in options_cfg.items():
        if key in {
            schema.KEY_CFG_OPTION_BROADCAST,
            schema.KEY_CFG_OPTION_TABLE,
        }:
            continue
        if key in broadcast:
            raise ValueError(
                f"{schema.KEY_OPTION_SECTION}.{key} cannot be provided both at top level and under "
                f"{schema.KEY_OPTION_SECTION}.{schema.KEY_CFG_OPTION_BROADCAST}."
            )
        broadcast[key] = value

    return {
        schema.KEY_CFG_OPTION_BROADCAST: broadcast,
        schema.KEY_CFG_OPTION_TABLE_PATH: table_path,
        schema.KEY_CFG_OPTION_COLUMNS: columns,
        schema.KEY_CFG_OPTION_UNITS: units,
        schema.KEY_CFG_OPTION_ROWS: rows,
    }


# Public options payload builders

def prepare_options_payload_from_table(
    simulation: Simulation,
    setup_payload: Mapping[str, object],
    options_input: Mapping[str, object],
) -> dict[str, np.ndarray | u.Quantity]:
    """Build persisted ``/options`` payload from table/broadcast options input.

    Args:
        simulation: Bound simulation instance for simulation-specific completion.
        setup_payload: Prepared ``/setup`` payload used by conversions/validation.
        options_input: Canonical or raw table-form options input.

    Returns:
        Fully prepared and validated options payload for persistence.

    Raises:
        ValueError: If options are incomplete, inconsistent, or malformed.
    """
    if (
        schema.KEY_CFG_OPTION_BROADCAST in options_input
        or schema.KEY_CFG_OPTION_COLUMNS in options_input
        or schema.KEY_CFG_OPTION_ROWS in options_input
    ):
        normalized = {
            schema.KEY_CFG_OPTION_BROADCAST: dict(options_input.get(schema.KEY_CFG_OPTION_BROADCAST, {})),
            schema.KEY_CFG_OPTION_COLUMNS: options_input.get(schema.KEY_CFG_OPTION_COLUMNS),
            schema.KEY_CFG_OPTION_UNITS: dict(options_input.get(schema.KEY_CFG_OPTION_UNITS, {})),
            schema.KEY_CFG_OPTION_ROWS: options_input.get(schema.KEY_CFG_OPTION_ROWS),
        }
    else:
        normalized = normalize_table_options_config(options_input)
    table_columns = normalized.get(schema.KEY_CFG_OPTION_COLUMNS)
    table_rows = normalized.get(schema.KEY_CFG_OPTION_ROWS)
    table_units = normalized.get(schema.KEY_CFG_OPTION_UNITS, {})
    options_broadcast = normalized.get(schema.KEY_CFG_OPTION_BROADCAST, {})
    if not isinstance(options_broadcast, Mapping):
        raise ValueError("options broadcast payload must be a mapping.")

    scalar_defaults, broadcast_ngs = _parse_broadcast_defaults(options_broadcast)
    options_arrays: dict[str, np.ndarray] = {}
    num_sims: int | None = None

    columns_empty = table_columns is None or (isinstance(table_columns, list) and len(table_columns) == 0)
    rows_empty = table_rows is None or (isinstance(table_rows, list) and len(table_rows) == 0)

    if not (columns_empty and rows_empty):
        if not isinstance(table_columns, list) or not isinstance(table_rows, list):
            raise ValueError("columns and rows must both be provided as lists.")
        if not isinstance(table_units, Mapping):
            raise ValueError("options table units must be a mapping.")
        options_arrays, num_sims = _build_options_from_table(table_columns, table_rows, table_units)

    if num_sims is None:
        num_sims = 1

    atm_wavelength = float(
        quantity_value(
            setup_payload[schema.KEY_SETUP_ATM_WAVELENGTH],
            schema.SETUP_FIELD_UNITS[schema.KEY_SETUP_ATM_WAVELENGTH],
            label="setup.atm_wavelength",
        )
    )
    base_options_payload = _finalize_options(
        options_arrays,
        num_sims=num_sims,
        atm_wavelength=atm_wavelength,
        scalar_defaults=scalar_defaults,
        broadcast_ngs=broadcast_ngs,
    )
    options_payload = simulation.prepare_options_payload(
        num_sims, setup_payload, _canonical_option_quantities(base_options_payload)
    )
    options_payload = _normalize_science_offset_options(
        options_payload,
        num_sims=num_sims,
        num_sci=get_num_sci(setup_payload),
    )
    options_payload = _canonical_option_quantities(options_payload)
    _validate_completed_options_payload(setup_payload, options_payload)
    return options_payload


def prepare_options_payload_from_arrays(
    simulation: Simulation,
    setup_payload: Mapping[str, object],
    option_arrays: Mapping[str, object],
) -> dict[str, np.ndarray | u.Quantity]:
    """Build persisted ``/options`` payload from explicit per-option arrays.

    Args:
        simulation: Bound simulation instance for simulation-specific completion.
        setup_payload: Prepared ``/setup`` payload used by conversions/validation.
        option_arrays: Mapping of option key to per-simulation arrays.

    Returns:
        Fully prepared and validated options payload for persistence.

    Raises:
        ValueError: If array shapes/values are inconsistent or malformed.
    """
    if not isinstance(option_arrays, Mapping) or len(option_arrays) == 0:
        raise ValueError("options.option_arrays must be a non-empty mapping of per-simulation arrays.")
    require_lowercase_mapping_keys(option_arrays, label="options.option_arrays")

    partial: dict[str, np.ndarray] = {}
    has_explicit_r0 = schema.KEY_OPTION_R0 in option_arrays
    num_sims: int | None = None
    for key, value in option_arrays.items():
        if key in schema.OPTION_FIELD_UNITS:
            arr = quantity_value(
                value,
                schema.OPTION_FIELD_UNITS[key],
                label=f"options.option_arrays[{key!r}]",
            )
        else:
            arr = np.asarray(value)
        if arr.ndim == 0:
            raise ValueError(f"options.option_arrays['{key}'] must be per-simulation and include first dimension N.")
        if num_sims is None:
            num_sims = int(arr.shape[0])
        elif int(arr.shape[0]) != num_sims:
            raise ValueError(
                f"options.option_arrays['{key}'] first dimension must match N={num_sims}, got {arr.shape}."
            )
        partial[str(key)] = arr

    assert num_sims is not None
    completed = simulation.prepare_options_payload(
        int(num_sims), setup_payload, _canonical_option_quantities(partial)
    )
    completed_payload = as_array_dict(dict(completed), copy_arrays=False)

    atm_wavelength = float(
        quantity_value(
            setup_payload[schema.KEY_SETUP_ATM_WAVELENGTH],
            schema.SETUP_FIELD_UNITS[schema.KEY_SETUP_ATM_WAVELENGTH],
            label="setup.atm_wavelength",
        )
    )
    completed_payload = replace_seeing_with_r0(
        completed_payload,
        atm_wavelength=atm_wavelength,
        num_sims=int(num_sims),
        has_explicit_r0=has_explicit_r0,
    )
    completed_payload = _normalize_science_offset_options(
        completed_payload,
        num_sims=int(num_sims),
        num_sci=get_num_sci(setup_payload),
    )

    quantity_payload = _canonical_option_quantities(completed_payload)
    _validate_completed_options_payload(setup_payload, quantity_payload)
    return quantity_payload
