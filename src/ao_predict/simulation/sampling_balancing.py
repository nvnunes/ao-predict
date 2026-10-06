"""Direct reassignment of one realized simulation-option field.

Population balancing is separate from generation and dataset configuration.
The caller supplies fixed balancing values and owns any scientific evaluation
needed to obtain them. Only the selected field moves; all other options are
borrowed unchanged, and ordinary dataset validation remains downstream.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from astropy import units as u

from .api import OptionsConfig
from .sampling import _field_seed, _positive_int
from .schema import OPTION_FIELD_UNITS


_ROW_SWAP_CHUNK = 2048
_CDF_TOLERANCE = 64 * np.finfo(np.float64).eps


def _numeric_array(value: object, label: str, field: str | None = None) -> np.ndarray:
    """Borrow finite real arrays without discarding masks, units or integer ranks."""
    if not isinstance(value, (np.ndarray, u.Quantity)) or hasattr(value, "mask"):
        raise ValueError(f"{label} must be an unmasked NumPy array or Astropy Quantity.")
    canonical = OPTION_FIELD_UNITS.get(field)
    if canonical is not None:
        if not isinstance(value, u.Quantity) or not value.unit.is_equivalent(canonical):
            raise ValueError(f"{label} must be a Quantity compatible with {canonical}.")
    array = np.asarray(value.value if isinstance(value, u.Quantity) else value)
    if (
        array.ndim not in (1, 2)
        or not array.size
        or array.dtype.kind not in "iuf"
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{label} must be a nonempty finite real numeric vector or matrix.")
    return array


def _group_indices(values: np.ndarray, groups: int, rng: np.random.RandomState) -> list[np.ndarray]:
    """Sort destination values, randomize ties and split into equal-count groups."""
    order = np.lexsort((rng.random_sample(values.size), values))
    return list(np.array_split(order, groups))


def _balance_entries(values: np.ndarray, indices: list[np.ndarray], rng: np.random.RandomState) -> np.ndarray:
    """Spread consecutive sorted batches across groups before shuffling within each."""
    groups = len(indices)
    complete, remainder = divmod(values.size, groups)
    destinations = np.concatenate(
        [rng.permutation(groups) for _ in range(complete)]
        + ([rng.permutation(remainder)] if remainder else [])
    )
    ordered = np.sort(values)
    result = np.empty_like(values)
    for group, members in enumerate(indices):
        result[rng.permutation(members)] = ordered[destinations == group]
    return result


def _row_cdf_residuals(values: np.ndarray, labels: np.ndarray, groups: int) -> np.ndarray:
    """Group-minus-population CDFs at every original column value, including ties."""
    ordered = np.sort(values, axis=0)
    residual = np.empty((groups, values.shape[1], len(values)), dtype=np.float64)
    for column in range(values.shape[1]):
        points = ordered[:, column]
        complete = np.searchsorted(points, points, side="right") / len(values)
        for group in range(groups):
            members = np.sort(values[labels == group, column])
            residual[group, column] = np.searchsorted(members, points, side="right") / len(members) - complete
    return residual


def _row_swap_deltas(
    prefix: np.ndarray,
    ranks: np.ndarray,
    first: int,
    second: np.ndarray,
    labels: np.ndarray,
    sizes: np.ndarray,
) -> np.ndarray:
    """Combined CDF-error change for bounded candidate swaps, without trial copies.

    A swap changes each affected CDF by a constant step between the two values'
    left ranks. Prefix sums integrate that change over all evaluation points;
    repeated values keep their full weight in the objective.
    """
    first_group = labels[first]
    second_groups = labels[second]
    lo = np.minimum(ranks[first], ranks[second])
    hi = np.maximum(ranks[first], ranks[second])
    sign = np.sign(ranks[first] - ranks[second])
    first_step = sign / sizes[first_group]
    second_step = -sign / sizes[second_groups, None]
    columns = np.arange(ranks.shape[1])[None, :]
    first_sum = prefix[first_group, columns, hi] - prefix[first_group, columns, lo]
    second_sum = prefix[second_groups[:, None], columns, hi] - prefix[second_groups[:, None], columns, lo]
    return np.sum(
        2 * first_step * first_sum + (hi - lo) * first_step**2
        + 2 * second_step * second_sum + (hi - lo) * second_step**2,
        axis=1,
    ) / (len(sizes) * ranks.shape[1] * len(ranks))


def _balance_rows(values: np.ndarray, indices: list[np.ndarray], rng: np.random.RandomState) -> np.ndarray:
    """Improve equal-priority column CDFs using intact-row swaps to a local optimum."""
    vector = values.ndim == 1
    matrix = values[:, None] if vector else values
    groups = len(indices)
    # Singleton groups have the same combined objective under every permutation.
    if groups == len(values):
        return values.copy()
    labels = np.empty(len(matrix), dtype=int)
    for group, members in enumerate(indices):
        labels[members] = group
    sizes = np.asarray([len(members) for members in indices])
    assignment = np.arange(len(matrix))
    priority = rng.permutation(len(matrix))
    ordered = np.sort(matrix, axis=0)
    ranks = np.column_stack([
        np.searchsorted(ordered[:, column], matrix[:, column], side="left")
        for column in range(matrix.shape[1])
    ])
    residual = _row_cdf_residuals(matrix, labels, groups)
    current = float(np.mean(residual**2))
    while current > _CDF_TOLERANCE:
        prefix = np.pad(np.cumsum(residual, axis=-1), ((0, 0), (0, 0), (1, 0)))
        best_delta = -_CDF_TOLERANCE
        best_pair = None
        current_ranks = ranks[assignment]
        for position, first in enumerate(priority[:-1]):
            others = priority[position + 1:]
            others = others[labels[others] != labels[first]]
            for start in range(0, len(others), _ROW_SWAP_CHUNK):
                second = others[start:start + _ROW_SWAP_CHUNK]
                deltas = _row_swap_deltas(prefix, current_ranks, first, second, labels, sizes)
                index = int(np.argmin(deltas))
                if deltas[index] < best_delta:
                    best_delta = float(deltas[index])
                    best_pair = (first, second[index])
        if best_pair is None:
            break
        first, second = best_pair
        assignment[first], assignment[second] = assignment[second], assignment[first]
        proposed_residual = _row_cdf_residuals(matrix[assignment], labels, groups)
        proposed = float(np.mean(proposed_residual**2))
        if current - proposed <= _CDF_TOLERANCE:
            raise RuntimeError("Row-swap delta disagrees with the full CDF objective.")
        residual, current = proposed_residual, proposed
    result = matrix[assignment].copy()
    return result[:, 0] if vector else result


def balance_options(
    options: OptionsConfig,
    *,
    field: str,
    balance_field: str | np.ndarray | u.Quantity,
    mode: Literal["entries", "rows"],
    groups: int,
    seed: int = 0,
) -> OptionsConfig:
    """Reassign one field to balance its distribution across destination groups.

    Args:
        options: Realized options. Only the target and balancing fields are
            checked; full simulation-specific validation belongs to the ordinary
            dataset lifecycle. Dense finite arrays are required, including all
            NGS slots; NaN-padded inactive slots are not supported.
        field: Existing numeric target field with shape ``(N,)`` or ``(N, K)``.
            Known physical fields require compatible Astropy quantities.
        balance_field: Another field name resolved from the original options,
            or an external finite real array/quantity describing destinations.
            It must match the target shape in ``entries`` mode, or have shape
            ``(N,)`` in ``rows`` mode. Units need not match the target; zero and
            negative values are allowed. No broadcasting or row reduction occurs.
        mode: ``entries`` redistributes all scalar values together in row-major
            order. ``rows`` moves intact target rows, balancing each column's
            empirical CDF against the complete population with equal priority.
            Only the selected field moves, not complete options records.
        groups: Required positive number of equal-count groups formed by sorted
            balancing values, with seeded tie ordering. Group sizes differ by at
            most one. Limited to the number of entries or rows for the mode.
            One group returns unchanged target values in a new object.
        seed: Integer root seed, default zero, bound to the target field using
            generation's seed convention. No process-global random state changes.

    Returns:
        A new OptionsConfig with a new mapping and owned target array preserving
        shape, dtype, unit and the entry or row multiset. Other fields are borrowed,
        not deep-copied. Inputs are not mutated. Identical inputs replay within
        the producing environment; increasing count may change assignments.
        Row balancing is a best-effort pair-swap local optimum of combined squared
        CDF departure, not exact or global balance; individual columns may worsen.

    Raises:
        ValueError: Missing fields, incompatible units, unsupported types, masks,
            nonfinite values, invalid shapes, mode, group count or seed.

    The caller owns coupled-field validity and any recalculation of derived
    inputs. This operation neither computes balancing values nor draws new values,
    prepares simulation/setup, writes files or participates in YAML configuration.
    """
    if not isinstance(options, OptionsConfig) or not isinstance(options.option_arrays, dict):
        raise ValueError("options must be an OptionsConfig with an option-array mapping.")
    if not isinstance(field, str) or field not in options.option_arrays:
        raise ValueError("field must name an existing options field.")
    if mode not in ("entries", "rows"):
        raise ValueError("mode must be 'entries' or 'rows'.")
    target = options.option_arrays[field]
    values = _numeric_array(target, f"options.{field}", field)
    if isinstance(balance_field, str):
        if balance_field not in options.option_arrays:
            raise ValueError(f"balance_field '{balance_field}' is not an options field.")
        balance = _numeric_array(options.option_arrays[balance_field], f"options.{balance_field}", balance_field)
    else:
        balance = _numeric_array(balance_field, "balance_field")
    expected = values.shape if mode == "entries" else (len(values),)
    if balance.shape != expected:
        raise ValueError(f"balance_field must have shape {expected} for mode '{mode}', got {balance.shape}.")
    groups = _positive_int(groups, "groups")
    limit = values.size if mode == "entries" else len(values)
    if groups > limit:
        raise ValueError(f"groups cannot exceed {limit} for mode '{mode}'.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer, not a Boolean.")
    rng = np.random.RandomState(_field_seed(int(seed), field))
    if groups == 1:
        result = values.copy()
    elif mode == "entries":
        indices = _group_indices(balance.ravel(), groups, rng)
        result = _balance_entries(values.ravel(), indices, rng).reshape(values.shape)
    else:
        indices = _group_indices(balance, groups, rng)
        result = _balance_rows(values, indices, rng)
    arrays = dict(options.option_arrays)
    arrays[field] = u.Quantity(result, unit=target.unit, dtype=result.dtype, copy=False) if isinstance(target, u.Quantity) else result
    return OptionsConfig(arrays)
