from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from itertools import combinations

from ao_predict import OptionsConfig, balance_options


def _objective(values: np.ndarray, labels: np.ndarray, groups: int) -> float:
    """Independent literal empirical-CDF definition, including duplicate points."""
    matrix = values.reshape(len(values), -1)
    departures = []
    for group in range(groups):
        for column in range(matrix.shape[1]):
            complete = matrix[:, column]
            selected = complete[labels == group]
            points = np.sort(complete)
            full_cdf = np.count_nonzero(complete[:, None] <= points, axis=0) / len(complete)
            group_cdf = np.count_nonzero(selected[:, None] <= points, axis=0) / len(selected)
            departures.extend((group_cdf - full_cdf)**2)
    return float(np.mean(departures))


def _labels(count: int, groups: int) -> np.ndarray:
    labels = np.empty(count, dtype=int)
    for group, indices in enumerate(np.array_split(np.arange(count), groups)):
        labels[indices] = group
    return labels


def test_entry_reference_population_balances_both_groups() -> None:
    # e004 reference fixture: all eight scalar values remain, four per group.
    original = np.array([[8, 8], [8, 8], [18, 18], [18, 18]], dtype=np.int64)
    options = OptionsConfig({"values": original})
    result = balance_options(options, field="values", balance_field=np.arange(8).reshape(4, 2), mode="entries", groups=2)
    for group in result.option_arrays["values"].reshape(2, 4):
        np.testing.assert_array_equal(np.sort(group), [8, 8, 18, 18])


@pytest.mark.parametrize("shape", [(1,), (7,), (7, 1), (7, 2), (7, 3), (7, 5)])
@pytest.mark.parametrize("groups", [1, 3])
def test_entries_preserve_values_shape_dtype_and_borrow_other_fields(shape, groups) -> None:
    if groups > np.prod(shape):
        return
    target = np.arange(np.prod(shape), dtype=np.int64).reshape(shape) + 2**60
    other = np.zeros(shape[0])
    target.setflags(write=False)
    other.setflags(write=False)
    options = OptionsConfig({"target": target, "other": other})
    result = balance_options(options, field="target", balance_field=np.ones(shape), mode="entries", groups=groups)
    assert result is not options and result.option_arrays is not options.option_arrays
    assert result.option_arrays["other"] is other
    values = result.option_arrays["target"]
    assert values.shape == target.shape and values.dtype == target.dtype
    assert not np.shares_memory(values, target)
    np.testing.assert_array_equal(np.sort(values.ravel()), np.sort(target.ravel()))
    if groups == 1:
        np.testing.assert_array_equal(values, target)


def test_entries_replay_seed_units_and_named_balance_field() -> None:
    target = np.arange(60, dtype=np.float32).reshape(20, 3) * u.mag
    balance = np.arange(60).reshape(20, 3) * u.mas
    options = OptionsConfig({"ngs_magnitude": target, "width": balance})
    before = np.random.get_state()
    first = balance_options(options, field="ngs_magnitude", balance_field="width", mode="entries", groups=6)
    second = balance_options(options, field="ngs_magnitude", balance_field=balance, mode="entries", groups=6, seed=0)
    third = balance_options(options, field="ngs_magnitude", balance_field=balance.to(u.arcsec), mode="entries", groups=6, seed=1)
    np.testing.assert_array_equal(first.option_arrays["ngs_magnitude"], second.option_arrays["ngs_magnitude"])
    assert not np.array_equal(first.option_arrays["ngs_magnitude"], third.option_arrays["ngs_magnitude"])
    assert first.option_arrays["ngs_magnitude"].unit == u.mag
    assert first.option_arrays["ngs_magnitude"].dtype == np.float32
    assert first.option_arrays["width"] is balance
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(target.value, np.arange(60).reshape(20, 3))


def test_entry_balance_quality_and_unequal_group_sizes() -> None:
    values = np.arange(101)
    options = OptionsConfig({"target": values})
    result = balance_options(options, field="target", balance_field=np.arange(101), mode="entries", groups=6).option_arrays["target"]
    for indices in np.array_split(np.arange(101), 6):
        # Each group receives one value from every complete consecutive batch.
        assert set(result[indices] // 6) >= set(range(16))
        points = np.sort(result[indices]) / 101
        assert np.max(np.abs(np.arange(len(indices)) / len(indices) - points)) < .08


@pytest.mark.parametrize("target", [
    np.array([], dtype=float), np.ones((2, 0)), np.ones((2, 2, 2)), np.array([True]),
    np.array([1j]), np.array(["1"]), np.array([1], dtype=object), np.array([np.nan]),
    np.array([np.inf]), np.ma.array([1.]), [1., 2.],
])
def test_invalid_target_arrays(target) -> None:
    with pytest.raises(ValueError):
        balance_options(OptionsConfig({"target": target}), field="target", balance_field=np.ones(2), mode="entries", groups=1)


@pytest.mark.parametrize("argument,value", [
    ("mode", "within_rows"), ("groups", 0), ("groups", -1), ("groups", True),
    ("groups", 1.5), ("groups", 7), ("seed", True), ("seed", None), ("seed", 2.5),
    ("field", "missing"), ("balance_field", "missing"), ("balance_field", np.ones(3)),
    ("balance_field", np.ones((3, 2))), ("balance_field", np.array([[1, 2, np.nan], [3, 4, 5]])),
    ("balance_field", np.ones((2, 3), dtype=bool)), ("balance_field", np.ma.array(np.ones((2, 3)))),
])
def test_invalid_balancing_arguments(argument, value) -> None:
    arguments = dict(field="target", balance_field=np.ones((2, 3)), mode="entries", groups=2, seed=0)
    arguments[argument] = value
    with pytest.raises(ValueError):
        balance_options(OptionsConfig({"target": np.ones((2, 3))}), **arguments)


@pytest.mark.parametrize("target", [np.ones((2, 3)), np.ones((2, 3)) * u.m])
def test_physical_target_requires_compatible_quantity(target) -> None:
    with pytest.raises(ValueError, match="Quantity compatible"):
        balance_options(OptionsConfig({"ngs_magnitude": target}), field="ngs_magnitude", balance_field=np.ones((2, 3)), mode="entries", groups=2)


def test_physical_named_balance_requires_compatible_quantity() -> None:
    with pytest.raises(ValueError, match="Quantity compatible"):
        balance_options(OptionsConfig({"target": np.ones(4), "ngs_r": np.ones(4)}), field="target", balance_field="ngs_r", mode="entries", groups=2)


def test_equivalent_target_unit_is_preserved_and_nonfinite_other_field_is_untouched() -> None:
    target = np.arange(12, dtype=np.float32).reshape(4, 3) * u.arcmin
    options = OptionsConfig({"ngs_r": target, "unrelated": np.array([np.nan])})
    result = balance_options(options, field="ngs_r", balance_field=-np.arange(12).reshape(4, 3), mode="entries", groups=np.int64(3), seed=-123)
    assert result.option_arrays["ngs_r"].unit == u.arcmin
    assert result.option_arrays["unrelated"] is options.option_arrays["unrelated"]
    np.testing.assert_array_equal(np.sort(result.option_arrays["ngs_r"].value.ravel()), np.sort(target.value.ravel()))


@pytest.mark.parametrize("values,expected", [
    ([[0, 0], [0, 0], [1, 1], [1, 1]], 0.),
    ([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], 1 / 24),
])
def test_rows_reference_fixtures_match_exhaustive_partitions(values, expected) -> None:
    # Fixed e004 reference inputs and objective minima; no executable reference copy.
    values = np.asarray(values)
    labels = _labels(4, 2)
    result = balance_options(OptionsConfig({"target": values}), field="target", balance_field=np.arange(4), mode="rows", groups=2).option_arrays["target"]
    objectives = []
    for first_group in combinations(range(4), 2):
        partition = np.ones(4, dtype=int)
        partition[list(first_group)] = 0
        objectives.append(_objective(values, partition, 2))
    assert min(objectives) == pytest.approx(expected)
    assert _objective(result, labels, 2) == pytest.approx(expected)
    if expected == 0:
        assert _objective(values, labels, 2) == .125


@pytest.mark.parametrize("shape,groups", [((12, 3), 3), ((13, 2), 4), ((7,), 3), ((9, 1), 1), ((8, 5), 3), ((3, 2), 3)])
def test_rows_preservation_pair_optimality_and_chunk_independence(shape, groups, monkeypatch) -> None:
    from ao_predict.simulation import sampling_balancing
    values = np.random.RandomState(3).randint(0, 5, shape)
    options = OptionsConfig({"target": values, "other": np.zeros(len(values))})
    kwargs = dict(field="target", balance_field=np.arange(len(values)), mode="rows", groups=groups, seed=8)
    monkeypatch.setattr(sampling_balancing, "_ROW_SWAP_CHUNK", 2)
    first = balance_options(options, **kwargs)
    monkeypatch.setattr(sampling_balancing, "_ROW_SWAP_CHUNK", 7)
    second = balance_options(options, **kwargs)
    result = first.option_arrays["target"]
    np.testing.assert_array_equal(result, second.option_arrays["target"])
    assert sorted(map(tuple, result.reshape(len(values), -1))) == sorted(map(tuple, values.reshape(len(values), -1)))
    assert result.shape == values.shape and result.dtype == values.dtype
    assert not np.shares_memory(result, values)
    assert first.option_arrays["other"] is options.option_arrays["other"]
    labels = _labels(len(values), groups)
    objective = _objective(result, labels, groups)
    assert objective <= _objective(values, labels, groups)
    for first_row, second_row in combinations(range(len(values)), 2):
        if labels[first_row] == labels[second_row]:
            continue
        swapped = result.copy()
        swapped[[first_row, second_row]] = swapped[[second_row, first_row]]
        assert _objective(swapped, labels, groups) >= objective - 64 * np.finfo(float).eps


def test_row_swap_deltas_match_independent_objective() -> None:
    from ao_predict.simulation.sampling_balancing import _row_cdf_residuals, _row_swap_deltas
    values = np.array([[0, 1], [0, 0], [1, 0], [2, 0], [2, 1]])
    labels = _labels(5, 2)
    ordered = np.sort(values, axis=0)
    ranks = np.column_stack([np.searchsorted(ordered[:, col], values[:, col], side="left") for col in range(2)])
    residuals = _row_cdf_residuals(values, labels, 2)
    assert np.mean(residuals**2) == pytest.approx(_objective(values, labels, 2))
    prefix = np.pad(np.cumsum(residuals, axis=-1), ((0, 0), (0, 0), (1, 0)))
    before = _objective(values, labels, 2)
    for first in range(3):
        second = np.array([3, 4])
        deltas = _row_swap_deltas(prefix, ranks, first, second, labels, np.bincount(labels))
        for index, other in enumerate(second):
            swapped = values.copy()
            swapped[[first, other]] = swapped[[other, first]]
            assert deltas[index] == pytest.approx(_objective(swapped, labels, 2) - before, abs=1e-15)


def test_rows_400_by_3_quality_gate() -> None:
    values = np.random.RandomState(23).normal(size=(400, 3))
    result = balance_options(OptionsConfig({"target": values}), field="target", balance_field=np.arange(400), mode="rows", groups=6).option_arrays["target"]
    labels = _labels(400, 6)
    assert _objective(result, labels, 6) <= .5 * _objective(values, labels, 6)


def test_rows_units_constants_and_large_integer_ranks() -> None:
    values = np.array([[2**60 + i, 2**60 + 8 - i] for i in range(8)], dtype=np.int64)
    balance = np.arange(8)
    result = balance_options(OptionsConfig({"target": values}), field="target", balance_field=balance, mode="rows", groups=2).option_arrays["target"]
    assert result.dtype == values.dtype
    assert sorted(map(tuple, result)) == sorted(map(tuple, values))
    assert _objective(result, _labels(8, 2), 2) < _objective(values, _labels(8, 2), 2)
    for field in (np.full((9, 3), 17.), np.column_stack([np.arange(9), np.ones(9)])):
        result = balance_options(OptionsConfig({"ngs_r": field * u.arcsec}), field="ngs_r", balance_field=np.arange(9), mode="rows", groups=4).option_arrays["ngs_r"]
        scaled = balance_options(OptionsConfig({"ngs_r": (field * u.arcsec).to(u.arcmin)}), field="ngs_r", balance_field=np.arange(9) * u.mas, mode="rows", groups=4).option_arrays["ngs_r"]
        np.testing.assert_allclose(result.value, scaled.to_value(u.arcsec), rtol=1e-15, atol=0)
        if np.all(field == 17):
            np.testing.assert_array_equal(result.value, field)
        else:
            np.testing.assert_array_equal(result.value[:, 1], 1)


def test_rows_equal_column_priority_is_unit_scale_invariant() -> None:
    values = np.random.RandomState(9).normal(size=(20, 3))
    kwargs = dict(field="target", balance_field=np.arange(20), mode="rows", groups=4, seed=2)
    first = balance_options(OptionsConfig({"target": values}), **kwargs).option_arrays["target"]
    second = balance_options(OptionsConfig({"target": values * [1, 1000, .01]}), **kwargs).option_arrays["target"]
    np.testing.assert_allclose(second, first * [1, 1000, .01])


def test_tied_balancing_values_replay_and_use_seed() -> None:
    values = np.arange(30).reshape(10, 3)
    kwargs = dict(field="target", balance_field=np.ones(10), mode="rows", groups=3)
    first = balance_options(OptionsConfig({"target": values}), **kwargs).option_arrays["target"]
    np.testing.assert_array_equal(first, balance_options(OptionsConfig({"target": values}), **kwargs).option_arrays["target"])
    other = balance_options(OptionsConfig({"target": values}), **kwargs, seed=1).option_arrays["target"]
    assert not np.array_equal(first, other)


def test_rows_reject_matrix_balancing_and_too_many_groups() -> None:
    options = OptionsConfig({"target": np.ones((2, 3))})
    for balance, groups in ((np.ones((2, 3)), 2), (np.ones(2), 3)):
        with pytest.raises(ValueError):
            balance_options(options, field="target", balance_field=balance, mode="rows", groups=groups)


@pytest.mark.parametrize("mode,groups", [("entries", 1), ("entries", 2), ("rows", 2)])
def test_integer_quantity_preserves_dtype_and_exact_values(mode, groups) -> None:
    values = (2**60 + np.arange(12, dtype=np.int64)).reshape(4, 3)
    target = u.Quantity(values, unit=u.mag, dtype=np.int64)
    balance = np.arange(12).reshape(4, 3) if mode == "entries" else np.arange(4)
    result = balance_options(OptionsConfig({"ngs_magnitude": target}), field="ngs_magnitude", balance_field=balance, mode=mode, groups=groups).option_arrays["ngs_magnitude"]
    assert result.dtype == target.dtype
    np.testing.assert_array_equal(np.sort(result.value.ravel()), np.sort(values.ravel()))
    np.testing.assert_array_equal(target.value, values)
