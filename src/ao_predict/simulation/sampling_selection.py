"""Deterministic whole-row selection from realized simulation options.

The supplied population owns parent targets. Callers supply external measurements
and duplicate precision; this module neither evaluates models nor interprets
asterism geometry. Selection is separate from generation and field reassignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from astropy import units as u

from .api import OptionsConfig
from .sampling import _positive_int
from .sampling_balancing import _numeric_array


_FRACTIONS = np.arange(1, 20, dtype=np.float64) / 20
_CANDIDATE_CHUNK = 65536
_IMPROVEMENT_TOLERANCE = 1e-14


@dataclass(frozen=True)
class _SelectionField:
    """Borrowed values and compact owned per-row CDF insertion information."""

    values: np.ndarray
    bounds: tuple[float, float]
    ranks: np.ndarray
    squared: np.ndarray
    target: np.ndarray
    identity: np.ndarray | None = None


def _field_bounds(value: object, values: np.ndarray, label: str) -> tuple[float, float]:
    if value is None:
        dtype = int if values.dtype.kind in "iu" else float
        return dtype(values.min()), dtype(values.max())
    if hasattr(value, "mask"):
        raise ValueError(f"{label} bounds must be unmasked.")
    try:
        bounds = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} bounds must be a finite real pair.") from exc
    if bounds.shape != (2,) or bounds.dtype.kind not in "iuf" or not np.isfinite(bounds).all():
        raise ValueError(f"{label} bounds must be a finite real pair.")
    if bounds[0] >= bounds[1]:
        raise ValueError(f"{label} bounds must be increasing.")
    dtype = int if bounds.dtype.kind in "iu" else float
    return dtype(bounds[0]), dtype(bounds[1])


def _duplicate_bins(value: object, values: np.ndarray, step: object, label: str) -> np.ndarray:
    """Resolve a positive scalar step and return zero-anchored nearest-even bins."""
    if hasattr(step, "mask"):
        raise ValueError(f"{label} dedup_step must be unmasked.")
    if isinstance(value, u.Quantity):
        if not isinstance(step, u.Quantity):
            raise ValueError(f"{label} dedup_step must be a Quantity with compatible units.")
        try:
            step = step.to_value(value.unit)
        except u.UnitConversionError as exc:
            raise ValueError(f"{label} dedup_step must have compatible units.") from exc
    elif isinstance(step, u.Quantity):
        raise ValueError(f"{label} numeric values require a numeric dedup_step.")
    step = np.asarray(step)
    if step.shape != () or step.dtype.kind not in "iuf" or not np.isfinite(step) or step <= 0:
        raise ValueError(f"{label} dedup_step must be a finite positive real scalar.")
    step = step.item()
    if values.dtype.kind in "iu" and step == int(step) and int(step) <= np.iinfo(values.dtype).max:
        divisor = np.asarray(int(step), dtype=values.dtype)
        quotient, remainder = np.divmod(values, divisor)
        complement = divisor - remainder
        increment = (remainder > complement) | ((remainder == complement) & (quotient % 2 != 0))
        return quotient + increment
    if values.dtype.kind in "iu":
        with np.errstate(invalid="ignore"):
            exact = values.astype(np.float64).astype(values.dtype)
        if not np.array_equal(exact, values):
            raise ValueError(f"{label} dedup_step scaling loses integer precision.")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        bins = np.rint(values / step)
    if not np.isfinite(bins).all():
        raise ValueError(f"{label} dedup_step scaling must remain finite.")
    return bins


def _prepare_field(name: str, declaration: Mapping[str, object], options: OptionsConfig, dedup: bool = False) -> _SelectionField:
    """Resolve one measurement, its target and bounded-memory insertion costs."""
    if not isinstance(declaration, Mapping) or set(declaration) - {"target", "values", "bounds", "dedup_step", "dedup_unordered"}:
        raise ValueError(f"Selection field '{name}' requires target, optional values, bounds, dedup_step and dedup_unordered.")
    target = declaration.get("target")
    if not isinstance(target, str) or target not in ("uniform", "parent"):
        raise ValueError(f"Selection field '{name}' target must be 'uniform' or 'parent'.")
    if "bounds" in declaration and target != "uniform":
        raise ValueError(f"Selection field '{name}' bounds are only allowed for uniform targets.")
    if "values" in declaration:
        value = declaration["values"]
    elif name in options.option_arrays:
        value = options.option_arrays[name]
    else:
        raise ValueError(f"Selection field '{name}' is not in options; supply external values.")
    values = _numeric_array(value, f"Selection field '{name}'", name)
    identity = None
    if dedup:
        unordered = declaration.get("dedup_unordered", False)
        if not isinstance(unordered, (bool, np.bool_)):
            raise ValueError(f"Selection field '{name}' dedup_unordered must be a boolean.")
        duplicate_values = values
        if "dedup_step" in declaration:
            duplicate_values = _duplicate_bins(value, values, declaration["dedup_step"], f"Selection field '{name}'")
        if unordered and duplicate_values.ndim == 2:
            duplicate_values = np.sort(duplicate_values, axis=1)
        # Encode each field separately so mixed dtypes and large integers retain
        # their exact identity. Unordered matrices retain value multiplicity.
        _, identity = np.unique(duplicate_values, axis=0, return_inverse=True)
    explicit = declaration.get("bounds")
    if "bounds" in declaration:
        if hasattr(explicit, "mask"):
            raise ValueError(f"Selection field '{name}' bounds must be unmasked.")
        if isinstance(value, u.Quantity):
            try:
                explicit = u.Quantity(explicit).to_value(value.unit)
            except (TypeError, ValueError, u.UnitConversionError) as exc:
                raise ValueError(f"Selection field '{name}' bounds must have compatible units.") from exc
        elif isinstance(explicit, u.Quantity):
            raise ValueError(f"Selection field '{name}' numeric values require numeric bounds.")
        if explicit is None:
            raise ValueError(f"Selection field '{name}' bounds must be a finite real pair.")
    lo, hi = _field_bounds(explicit, values, f"Selection field '{name}'")
    if values.dtype.kind in "iu" and isinstance(lo, int) and isinstance(hi, int):
        # Shift before conversion: nearby large integers must not collapse to
        # one float. Unsigned subtraction also handles the full signed span.
        origin = int(values.min())
        values = (values.astype(np.uint64) - np.uint64(origin % (1 << 64))).astype(np.float64)
        lo, hi = float(lo - origin), float(hi - origin)
    fractions = _FRACTIONS
    with np.errstate(over="ignore"):
        thresholds = lo + (hi - lo) * fractions
    if not np.isfinite(thresholds).all():
        thresholds = lo * (1 - fractions) + hi * fractions
    matrix = values[:, None] if values.ndim == 1 else values
    if not matrix.shape[1]:
        raise ValueError(f"Selection field '{name}' must not have empty columns.")
    ranks = np.searchsorted(thresholds, matrix, side="left").astype(np.uint8)
    if target == "parent" or lo == hi:
        ordered = np.sort(values.ravel())
        cdf = np.searchsorted(ordered, thresholds, side="right") / values.size
    else:
        cdf = _FRACTIONS.copy()
    # In sorted ranks, the jth element is the maximum in 2*j+1 ordered pairs.
    # This integrates a pooled vector's squared CDF without a K-by-K tensor.
    squared = np.empty(len(values), dtype=np.float64)
    weights = 2 * np.arange(matrix.shape[1]) + 1
    for start in range(0, len(values), _CANDIDATE_CHUNK):
        ordered = np.sort(ranks[start:start + _CANDIDATE_CHUNK], axis=1)
        squared[start:start + len(ordered)] = np.sum(
            (len(_FRACTIONS) - ordered) * weights, axis=1,
        ) / matrix.shape[1]**2
    return _SelectionField(values, (lo, hi), ranks, squared, cdf, identity)


def _selected_features(fields: list[_SelectionField], selected: np.ndarray) -> np.ndarray:
    """Materialize CDF contributions only for the small selected population."""
    return np.stack([
        (np.arange(len(_FRACTIONS)) >= field.ranks[selected, :, None]).mean(axis=1)
        for field in fields
    ], axis=1)


def _best_candidate(fields: list[_SelectionField], residual: np.ndarray, count: int, blocked: np.ndarray, groups: np.ndarray | None = None) -> int:
    """Scan exact insertion costs with stable ties and bounded temporary arrays."""
    tails = np.pad(np.cumsum(residual[:, ::-1], axis=1)[:, ::-1], ((0, 0), (0, 1)))
    best, best_cost = -1, np.inf
    for start in range(0, len(fields[0].values), _CANDIDATE_CHUNK):
        stop = min(start + _CANDIDATE_CHUNK, len(fields[0].values))
        costs = np.zeros(stop - start)
        for axis, field in enumerate(fields):
            linear = 2 * tails[axis][field.ranks[start:stop]].mean(axis=1) / count
            costs += (linear + field.squared[start:stop] / count**2) / len(_FRACTIONS)
        costs[blocked[start:stop] if groups is None else blocked[groups[start:stop]]] = np.inf
        index = int(np.argmin(costs))
        if costs[index] < best_cost:
            best, best_cost = start + index, costs[index]
    if best < 0:
        raise ValueError("No distinct eligible candidate remains.")
    return best


def _initialize(fields: list[_SelectionField], count: int, start_axis: int | None, groups: np.ndarray | None = None) -> np.ndarray:
    selected = np.empty(count, dtype=np.int64)
    blocked = np.zeros(len(fields[0].values) if groups is None else int(groups.max()) + 1, dtype=bool)
    target = np.stack([field.target for field in fields])
    if start_axis is None:
        total = np.zeros_like(target)
        for slot in range(count):
            winner = _best_candidate(fields, total / (slot + 1) - target, slot + 1, blocked, groups)
            selected[slot], blocked[winner if groups is None else groups[winner]] = winner, True
            total += _selected_features(fields, np.array([winner]))[0]
    else:
        field = fields[start_axis]
        values, bounds = field.values, field.bounds
        scale = max(abs(bounds[0]), abs(bounds[1]), float(np.max(np.abs(values))))
        if scale > np.finfo(np.float64).max / 2:
            # A common scale preserves nearest-value ordering while preventing
            # interpolation and subtraction overflow for finite extreme inputs.
            values = values / scale
            bounds = (bounds[0] / scale, bounds[1] / scale)
        for slot, point in enumerate(np.linspace(*bounds, count)):
            winner, best_cost = -1, np.inf
            for start in range(0, len(values), _CANDIDATE_CHUNK):
                costs = np.abs(values[start:start + _CANDIDATE_CHUNK] - point)
                stop = start + len(costs)
                costs[blocked[start:stop] if groups is None else blocked[groups[start:stop]]] = np.inf
                index = int(costs.argmin())
                if costs[index] < best_cost:
                    winner, best_cost = start + index, costs[index]
            if winner < 0:
                raise ValueError("No distinct eligible candidate remains.")
            selected[slot], blocked[winner if groups is None else groups[winner]] = winner, True
    return selected


def _select(fields: list[_SelectionField], count: int, start_axis: int | None, max_passes: int, groups: np.ndarray | None = None) -> np.ndarray:
    selected = _initialize(fields, count, start_axis, groups)
    feature = _selected_features(fields, selected)
    target = np.stack([field.target for field in fields])
    blocked = np.zeros(len(fields[0].values) if groups is None else int(groups.max()) + 1, dtype=bool)
    blocked[selected if groups is None else groups[selected]] = True
    loss = float(np.mean((feature.mean(axis=0) - target)**2))
    for _ in range(max_passes):
        changed = False
        for slot in range(count):
            residual = (feature.sum(axis=0) - feature[slot]) / count - target
            blocked[selected[slot] if groups is None else groups[selected[slot]]] = False
            winner = _best_candidate(fields, residual, count, blocked, groups)
            candidate = _selected_features(fields, np.array([winner]))[0]
            trial = (feature.sum(axis=0) - feature[slot] + candidate) / count
            proposed = float(np.mean((trial - target)**2))
            if proposed < loss - _IMPROVEMENT_TOLERANCE:
                selected[slot], feature[slot] = winner, candidate
                loss, changed = proposed, True
            blocked[selected[slot] if groups is None else groups[selected[slot]]] = True
        if not changed:
            break
    return selected


def select_options(
    options: OptionsConfig,
    *,
    count: int,
    fields: Mapping[str, Mapping[str, object]],
    start_field: str | None = None,
    max_passes: int = 8,
    dedup: bool = False,
) -> OptionsConfig:
    """Select complete option rows to approximate equally weighted target distributions.

    Args:
        options: Nonempty realized population. Every array shares its first axis.
            Unmeasured payloads, including inactive-slot NaNs, are carried unchanged.
            Full simulation validation is caller-owned.
        count: Positive number of distinct input rows to select, at most the population size.
        fields: Named declarations with required ``target='uniform'`` or ``'parent'``.
            Omitted ``values`` resolves the name from options; supplied ``values`` is
            an external finite real vector ``(N,)`` or matrix ``(N, K)``. Matrices
            contribute one pooled CDF with one field weight, not per-column targets.
            Optional ``bounds`` overrides uniform min/max and requires compatible
            units. Constant fields without bounds use a point target. Parent targets
            always come from this supplied population; no external reference is used.
            Optional positive scalar ``dedup_step`` rounds this field to the nearest
            zero-anchored multiple for duplicate comparison only, with halfway ties
            to even multiples. Quantity values require a compatible Quantity step;
            numeric values require a numeric step. Omission uses exact values.
            The step is ignored unless ``dedup=True``. Optional boolean
            ``dedup_unordered`` compares matrix rows without column order, retaining
            repeated values. Default false
            retains column order; scalar fields are unaffected. This option is also
            ignored unless ``dedup=True``. Each field is compared independently;
            unordered comparison does not match joint permutations across fields.
        start_field: Optional scalar declared field for evenly spaced initialization.
            Omission greedily builds a selection using all fields equally. Neither
            path locks cases or gives a field extra weight during replacement.
        max_passes: Positive replacement-pass limit, default eight. Stop earlier when
            no replacement improves the combined squared CDF error by more than 1e-14.
        dedup: When true, select at most one row for each combined set of declared
            field values, after optional rounding. Parent targets retain every row;
            objective values and returned options remain unrounded. Default false.

    Returns:
        A new OptionsConfig with owned arrays of complete selected rows, preserving
        dtype, units and trailing shape. Inputs and global random state are unchanged.
        Fixed inputs and ordering replay deterministically. This is best-effort local
        search, not a global optimum or a count-extension guarantee; no diagnostics,
        indices, files, simulation preparation or YAML configuration are returned.

    Raises:
        ValueError: Invalid population, declarations, counts, units, shapes or values.
    """
    if not isinstance(options, OptionsConfig) or not isinstance(options.option_arrays, dict) or not options.option_arrays:
        raise ValueError("options must be a nonempty OptionsConfig array mapping.")
    size = None
    for name, array in options.option_arrays.items():
        if not isinstance(array, (np.ndarray, u.Quantity)) or hasattr(array, "mask") or array.ndim == 0:
            raise ValueError(f"options.{name} must be an unmasked array with a population axis.")
        if size is None:
            size = len(array)
        if not size or len(array) != size:
            raise ValueError("All options arrays must share one nonempty population axis.")
    count = _positive_int(count, "count")
    if not isinstance(dedup, (bool, np.bool_)):
        raise ValueError("dedup must be a boolean.")
    max_passes = _positive_int(max_passes, "max_passes")
    if count > size:
        raise ValueError("count cannot exceed the options population size.")
    if not isinstance(fields, Mapping) or not fields or any(not isinstance(name, str) or not name for name in fields):
        raise ValueError("fields must be a nonempty mapping of named declarations.")
    prepared = [_prepare_field(name, declaration, options, dedup) for name, declaration in fields.items()]
    if any(len(field.values) != size for field in prepared):
        raise ValueError("Every selection field must match the options population axis.")
    start_axis = None
    if start_field is not None:
        if not isinstance(start_field, str) or start_field not in fields:
            raise ValueError("start_field must name a declared selection field.")
        start_axis = list(fields).index(start_field)
        if prepared[start_axis].values.ndim != 1:
            raise ValueError("start_field must have one scalar value per candidate.")
    groups = None
    if dedup:
        _, groups = np.unique(np.column_stack([field.identity for field in prepared]), axis=0, return_inverse=True)
        if count > int(groups.max()) + 1:
            raise ValueError("count cannot exceed the number of distinct combined selection field values.")
    selected = _select(prepared, count, start_axis, max_passes, groups)
    return OptionsConfig({name: array[selected].copy() for name, array in options.option_arrays.items()})
