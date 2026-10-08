"""Independent objective and complete-row contracts for population selection."""

import numpy as np
import pytest
from astropy import units as u

from ao_predict import OptionsConfig
from ao_predict.simulation import sampling_selection as selection


def options(count):
    return OptionsConfig({"id": np.arange(count, dtype=np.int64)})


def direct_features(values, thresholds):
    matrix = values[:, None] if values.ndim == 1 else values
    return (matrix[..., None] <= thresholds).mean(axis=1)


def direct_problem(values, targets):
    fractions = np.arange(1, 20) / 20
    thresholds, cdfs = [], []
    for array, target in zip(values, targets, strict=True):
        points = array.min() + (array.max() - array.min()) * fractions
        thresholds.append(points)
        cdfs.append(direct_features(array, points).mean(axis=0) if target == "parent" else fractions)
    return np.stack([direct_features(array, points) for array, points in zip(values, thresholds, strict=True)], axis=1), np.stack(cdfs)


def exhaustive_selection(values, targets, count, passes, start=None, identities=None):
    """Small brute-force oracle, with no rank or incremental-cost helpers."""
    feature, target = direct_problem(values, targets)

    def loss(ids):
        return np.mean((feature[ids].mean(axis=0) - target)**2)

    selected = []
    if identities is None:
        identities = list(range(len(feature)))

    def eligible_rows(other):
        return [i for i in range(len(feature)) if identities[i] not in [identities[j] for j in other]]

    for slot in range(count):
        eligible = eligible_rows(selected)
        if start is None:
            winner = min(eligible, key=lambda i: loss(selected + [i]))
        else:
            point = np.linspace(values[start].min(), values[start].max(), count)[slot]
            winner = min(eligible, key=lambda i: abs(values[start][i] - point))
        selected.append(winner)
    for _ in range(passes):
        changed = False
        for slot in range(count):
            eligible = eligible_rows(selected[:slot] + selected[slot + 1:])
            trials = [(loss(selected[:slot] + [i] + selected[slot + 1:]), i) for i in eligible]
            proposed, winner = min(trials)
            if proposed < loss(selected) - 1e-14:
                selected[slot], changed = winner, True
        if not changed:
            break
    return selected


def test_parent_fixture_and_pooled_matrix():
    result = selection.select_options(options(4), count=2, fields={"x": {"values": np.array([0, 0, 1, 1]), "target": "parent"}})
    np.testing.assert_array_equal(result.option_arrays["id"], [0, 2])
    field = selection._prepare_field("x", {"values": np.array([[0, 1], [0, 1]]), "target": "parent"}, options(2))
    np.testing.assert_array_equal(selection._selected_features([field], np.array([0])), 0.5)


@pytest.mark.parametrize("start", [None, 0])
@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("passes", [1, 3, 8])
def test_against_independent_exhaustive_search(start, count, passes):
    rng = np.random.RandomState(41)
    values = [rng.uniform(size=9), rng.uniform(size=(9, 3)), rng.uniform(size=9)]
    targets = ["uniform", "uniform", "parent"]
    fields = {str(i): {"values": array, "target": target} for i, (array, target) in enumerate(zip(values, targets, strict=True))}
    result = selection.select_options(options(9), count=count, fields=fields, start_field=None if start is None else str(start), max_passes=passes)
    assert result.option_arrays["id"].tolist() == exhaustive_selection(values, targets, count, passes, start)


def test_vector_squared_cost_matches_direct_contribution():
    rng = np.random.RandomState(13)
    for width in (1, 2, 3, 17):
        values = rng.uniform(size=(15, width))
        field = selection._prepare_field("x", {"values": values, "target": "parent"}, options(15))
        feature = selection._selected_features([field], np.arange(15))[:, 0]
        np.testing.assert_allclose(field.squared, (feature**2).sum(axis=1), atol=1e-14)


@pytest.mark.parametrize("start", [None, "x"])
def test_constant_fields_and_stable_ties(start):
    result = selection.select_options(options(7), count=4, fields={"x": {"values": np.ones(7), "target": "uniform"}}, start_field=start)
    np.testing.assert_array_equal(result.option_arrays["id"], [0, 1, 2, 3])


def test_complete_rows_units_ownership_and_rng():
    original = OptionsConfig({
        "id": np.arange(8, dtype=np.uint64) + 2**60,
        "ngs_magnitude": np.arange(24).reshape(8, 3).astype(np.int16) * u.mag,
        "extra": np.arange(32).reshape(8, 2, 2).astype(np.float32),
        "inactive": np.full((8, 3), np.nan),
    })
    copies = {key: value.copy() for key, value in original.option_arrays.items()}
    for value in original.option_arrays.values():
        value.flags.writeable = False
    rng_before = np.random.get_state()
    result = selection.select_options(original, count=3, fields={"ngs_magnitude": {"target": "parent"}})
    ids = (result.option_arrays["id"] - 2**60).astype(int)
    assert len(np.unique(ids)) == 3
    for key, value in result.option_arrays.items():
        np.testing.assert_array_equal(value, copies[key][ids])
        np.testing.assert_array_equal(original.option_arrays[key], copies[key])
        assert value.dtype == original.option_arrays[key].dtype
        assert not np.shares_memory(value, original.option_arrays[key])
        if isinstance(value, u.Quantity):
            assert value.unit == original.option_arrays[key].unit
    assert result.option_arrays is not original.option_arrays
    for old, new in zip(rng_before, np.random.get_state(), strict=True):
        np.testing.assert_array_equal(old, new)


def test_field_resolution_external_values_and_unit_equivalence():
    values = np.array([0, 1, 2, 3, 4, 5]) * u.arcsec
    parent = OptionsConfig({"id": np.arange(6), "ngs_r": values[:, None]})
    internal = selection.select_options(parent, count=3, fields={"ngs_r": {"target": "uniform", "bounds": (0 * u.arcsec, 5 * u.arcsec)}})
    external = selection.select_options(parent, count=3, fields={"x": {"values": values.to(u.mas), "target": "uniform", "bounds": (0 * u.arcsec, 5 * u.arcsec)}})
    np.testing.assert_array_equal(internal.option_arrays["id"], external.option_arrays["id"])


def test_chunks_do_not_change_selection(monkeypatch):
    values = np.random.RandomState(52).uniform(size=(11, 5))
    kwargs = dict(count=4, fields={"x": {"values": values, "target": "parent"}})
    baseline = selection.select_options(options(11), **kwargs)
    monkeypatch.setattr(selection, "_CANDIDATE_CHUNK", 3)
    chunked = selection.select_options(options(11), **kwargs)
    np.testing.assert_array_equal(baseline.option_arrays["id"], chunked.option_arrays["id"])


def test_default_replay_and_passes_never_increase_objective():
    rng = np.random.RandomState(6)
    values = [rng.uniform(size=20), rng.uniform(size=(20, 3))]
    targets = ["uniform", "parent"]
    feature, target = direct_problem(values, targets)
    fields = {str(i): {"values": value, "target": t} for i, (value, t) in enumerate(zip(values, targets, strict=True))}
    losses = []
    for passes in (1, 2, 8):
        ids = selection.select_options(options(20), count=4, fields=fields, max_passes=passes).option_arrays["id"]
        losses.append(np.mean((feature[ids].mean(axis=0) - target)**2))
    assert np.all(np.diff(losses) <= 1e-12)
    first = selection.select_options(options(20), count=4, fields=fields)
    second = selection.select_options(options(20), count=4, fields=fields)
    np.testing.assert_array_equal(first.option_arrays["id"], second.option_arrays["id"])


@pytest.mark.parametrize("count,passes", [(0, 8), (-1, 8), (True, 8), (1.5, 8), (4, 8), (1, 0), (1, True)])
def test_bad_count_or_passes(count, passes):
    with pytest.raises(ValueError):
        selection.select_options(options(3), count=count, max_passes=passes, fields={"id": {"target": "parent"}})


@pytest.mark.parametrize("declaration", [
    {}, "uniform", {"target": "bad"}, {"target": []}, {"target": "parent", "bounds": (0, 1)},
    {"target": "uniform", "other": 1}, {"target": "uniform", "values": "id"},
    {"target": "uniform", "values": np.array([0, np.nan, 1])},
    {"target": "uniform", "values": np.array([True, False, True])},
    {"target": "uniform", "values": np.array([1j, 2j, 3j])},
    {"target": "uniform", "values": np.array([1, 2, 3], dtype=object)},
    {"target": "uniform", "values": np.ma.array([1, 2, 3])},
    {"target": "uniform", "values": np.ones((3, 0))},
    {"target": "uniform", "values": np.ones((3, 1, 1))},
    {"target": "uniform", "values": np.ones(2)},
    {"target": "uniform", "values": np.ones(3), "bounds": None},
    {"target": "uniform", "values": np.ones(3), "bounds": (1, 1)},
    {"target": "uniform", "values": np.ones(3), "bounds": (2, 1)},
    {"target": "uniform", "values": np.ones(3), "bounds": (0, np.inf)},
    {"target": "uniform", "values": np.ones(3), "bounds": (0 * u.m, 1 * u.m)},
    {"target": "uniform", "values": np.ones(3) * u.m, "bounds": (0 * u.s, 1 * u.s)},
])
def test_invalid_declarations(declaration):
    with pytest.raises(ValueError):
        selection.select_options(options(3), count=1, fields={"x": declaration})


@pytest.mark.parametrize("fields,start", [({}, None), ({"id": {"target": "parent"}}, "missing"), ({"x": {"values": np.ones((3, 2)), "target": "parent"}}, "x"), ({"missing": {"target": "parent"}}, None)])
def test_bad_fields_or_start(fields, start):
    with pytest.raises(ValueError):
        selection.select_options(options(3), count=1, fields=fields, start_field=start)


@pytest.mark.parametrize("population", [OptionsConfig({}), OptionsConfig({"x": np.array(1)}), OptionsConfig({"x": np.arange(3), "y": np.arange(2)}), OptionsConfig({"x": np.array([])}), OptionsConfig({"x": [1, 2, 3]})])
def test_bad_population(population):
    with pytest.raises(ValueError):
        selection.select_options(population, count=1, fields={"x": {"target": "parent"}})


def test_large_integer_measurements_do_not_collapse_to_constant():
    values = np.arange(4, dtype=np.int64) + 2**60
    result = selection.select_options(options(4), count=2, fields={"x": {"values": values, "target": "uniform"}}, start_field="x")
    ordinary = selection.select_options(options(4), count=2, fields={"x": {"values": np.arange(4), "target": "uniform"}}, start_field="x")
    np.testing.assert_array_equal(result.option_arrays["id"], ordinary.option_arrays["id"])


@pytest.mark.parametrize("values,bounds", [
    (np.arange(3.), (-1e308, 1e308)),
    (np.full(3, -1e308), (1e308, 1.7e308)),
])
def test_extreme_finite_start_values_select_distinct_rows(values, bounds):
    with np.errstate(over="raise", invalid="raise"):
        result = selection.select_options(
            options(3), count=3,
            fields={"x": {"values": values, "target": "uniform", "bounds": bounds}},
            start_field="x",
        )
    np.testing.assert_array_equal(np.sort(result.option_arrays["id"]), np.arange(3))


@pytest.mark.parametrize("start", [None, "x"])
def test_dedup_matches_independent_search_without_changing_parent_weights(start):
    values = [np.array([0.01, 0.02, 0.04, 0.6, 1.01, 1.02, 1.7]), np.array([0, 0, 0, 2, 1, 1, 2])]
    identities = list(zip(np.rint(values[0] / 0.1), values[1], strict=True))
    fields = {
        "x": {"values": values[0], "target": "parent", "dedup_step": 0.1},
        "y": {"values": values[1], "target": "uniform"},
    }
    result = selection.select_options(options(7), count=3, fields=fields, start_field=start, dedup=True)
    assert result.option_arrays["id"].tolist() == exhaustive_selection(values, ["parent", "uniform"], 3, 8, None if start is None else 0, identities)
    ids = result.option_arrays["id"]
    assert len({identities[i] for i in ids}) == 3
    np.testing.assert_array_equal(values[0], [0.01, 0.02, 0.04, 0.6, 1.01, 1.02, 1.7])


def test_dedup_combines_fields_and_preserves_matrix_column_order():
    parent = OptionsConfig({"id": np.arange(4), "x": np.array([[0, 1], [1, 0], [0, 1], [0, 1]]), "y": np.array([0, 0, 1, 1])})
    fields = {name: {"target": "parent"} for name in ("x", "y")}
    result = selection.select_options(parent, count=3, fields=fields, dedup=True)
    assert len(result.option_arrays["id"]) == 3
    with pytest.raises(ValueError, match="distinct combined"):
        selection.select_options(parent, count=4, fields=fields, dedup=True)


def test_exact_dedup_keeps_large_integers_and_unrounded_return_values():
    parent = OptionsConfig({"id": np.arange(4), "x": np.arange(4, dtype=np.uint64) + 2**60, "ngs_r": np.array([0.001, 0.002, 1.001, 1.002]) * u.arcsec})
    exact = selection.select_options(parent, count=4, fields={"x": {"target": "parent"}}, dedup=True)
    np.testing.assert_array_equal(np.sort(exact.option_arrays["id"]), np.arange(4))
    rounded = selection.select_options(parent, count=2, fields={"ngs_r": {"target": "parent", "dedup_step": 0.01 * u.arcsec}}, dedup=True)
    np.testing.assert_array_equal(rounded.option_arrays["ngs_r"], parent.option_arrays["ngs_r"][rounded.option_arrays["id"]])
    with pytest.raises(ValueError, match="distinct combined"):
        selection.select_options(parent, count=3, fields={"ngs_r": {"target": "parent", "dedup_step": 0.01 * u.arcsec}}, dedup=True)


@pytest.mark.parametrize("step", [None, True, 0, -1, np.nan, np.inf, [1], "2", 1j])
def test_dedup_step_is_ignored_when_disabled_and_validated_when_enabled(step):
    fields = {"id": {"target": "parent", "dedup_step": step}}
    result = selection.select_options(options(3), count=3, fields=fields)
    np.testing.assert_array_equal(np.sort(result.option_arrays["id"]), np.arange(3))
    with pytest.raises(ValueError, match="dedup_step"):
        selection.select_options(options(3), count=1, fields=fields, dedup=True)


@pytest.mark.parametrize("dedup", [0, 1, "yes", None])
def test_invalid_dedup(dedup):
    with pytest.raises(ValueError, match="dedup must"):
        selection.select_options(options(3), count=1, fields={"id": {"target": "parent"}}, dedup=dedup)


def test_dedup_rejects_unrepresentable_step_scaling():
    with pytest.raises(ValueError, match="dedup"):
        selection.select_options(
            options(2), count=1, dedup=True,
            fields={"x": {"values": np.array([1., 2.]), "target": "parent", "dedup_step": 1e-320}},
        )


@pytest.mark.parametrize("start", [None, "performance"])
@pytest.mark.parametrize("passes", [1, 8])
def test_unordered_dedup_matches_independent_search(start, passes):
    matrix = np.array([[0.011, 0.201], [0.204, 0.014], [0.7, 0.2], [0.2, 0.7], [0.5, 1.0], [1.0, 0.5]])
    performance = np.array([0., 0., 0.4, 0.4, 1., 1.])
    identities = list(zip(map(tuple, np.sort(np.rint(matrix / 0.01), axis=1)), performance, strict=True))
    fields = {
        "matrix": {"values": matrix, "target": "parent", "dedup_step": 0.01, "dedup_unordered": True},
        "performance": {"values": performance, "target": "uniform"},
    }
    result = selection.select_options(options(6), count=3, fields=fields, start_field=start, max_passes=passes, dedup=True)
    expected = exhaustive_selection([matrix, performance], ["parent", "uniform"], 3, passes, None if start is None else 1, identities)
    assert result.option_arrays["id"].tolist() == expected


def test_unordered_dedup_preserves_complete_star_rows_and_default_order():
    parent = OptionsConfig({
        "id": np.arange(3),
        "ngs_magnitude": np.array([[18.5, 18., 17.], [18.5, 17., 18.], [15., 15., 15.]]) * u.mag,
        "ngs_r": np.array([[43., 60., 58.], [43., 58., 60.], [20., 30., 40.]]) * u.arcsec,
    })
    copies = {name: value.copy() for name, value in parent.option_arrays.items()}
    for value in parent.option_arrays.values():
        value.flags.writeable = False
    ordered = {"ngs_magnitude": {"target": "parent"}}
    result = selection.select_options(parent, count=3, fields=ordered, dedup=True)
    np.testing.assert_array_equal(np.sort(result.option_arrays["id"]), np.arange(3))
    unordered = {"ngs_magnitude": {"target": "parent", "dedup_unordered": True}}
    result = selection.select_options(parent, count=2, fields=unordered, dedup=True)
    ids = result.option_arrays["id"]
    assert len(set(ids) & {0, 1}) == 1 and 2 in ids
    for name, value in result.option_arrays.items():
        np.testing.assert_array_equal(value, copies[name][ids])
        np.testing.assert_array_equal(parent.option_arrays[name], copies[name])
    with pytest.raises(ValueError, match="distinct combined"):
        selection.select_options(parent, count=3, fields=unordered, dedup=True)
    ignored = selection.select_options(parent, count=3, fields=unordered)
    np.testing.assert_array_equal(np.sort(ignored.option_arrays["id"]), np.arange(3))


def test_unordered_dedup_retains_multiplicity_and_exact_large_integers():
    matrix = np.array([[1, 1, 2], [2, 1, 1], [1, 2, 2]], dtype=np.uint64) + 2**60
    fields = {"x": {"values": matrix, "target": "parent", "dedup_unordered": np.bool_(True)}}
    result = selection.select_options(options(3), count=2, fields=fields, dedup=True)
    ids = result.option_arrays["id"]
    assert len(set(ids) & {0, 1}) == 1 and 2 in ids
    with pytest.raises(ValueError, match="distinct combined"):
        selection.select_options(options(3), count=3, fields=fields, dedup=True)


@pytest.mark.parametrize("unordered", [None, 0, 1, "yes"])
def test_dedup_order_is_ignored_when_disabled_and_validated_when_enabled(unordered):
    fields = {"id": {"target": "parent", "dedup_unordered": unordered}}
    result = selection.select_options(options(3), count=3, fields=fields)
    np.testing.assert_array_equal(np.sort(result.option_arrays["id"]), np.arange(3))
    with pytest.raises(ValueError, match="dedup_unordered"):
        selection.select_options(options(3), count=1, fields=fields, dedup=True)


def test_unordered_scalar_field_has_the_same_identity():
    fields = {"x": {"values": np.array([0, 0, 1]), "target": "parent", "dedup_unordered": True}}
    result = selection.select_options(options(3), count=2, fields=fields, dedup=True)
    np.testing.assert_array_equal(result.option_arrays["id"], [0, 2])


def test_dedup_step_accepts_equivalent_units_and_non_decimal_increments():
    radii = np.array([1., 2., 3., 4., 6., 7., 8.]) * u.arcsec
    parent = OptionsConfig({"id": np.arange(7), "ngs_r": radii})
    declared = {"ngs_r": {"target": "parent", "dedup_step": 5 * u.arcsec}}
    result = selection.select_options(parent, count=3, fields=declared, dedup=True)
    external = selection.select_options(parent, count=3, dedup=True, fields={
        "x": {"values": radii.to(u.mas), "target": "parent", "dedup_step": 5 * u.arcsec},
    })
    np.testing.assert_array_equal(result.option_arrays["id"], external.option_arrays["id"])
    ids = result.option_arrays["id"]
    assert len(set(np.rint(radii.value[ids] / 5))) == 3
    np.testing.assert_array_equal(result.option_arrays["ngs_r"], radii[ids])
    with pytest.raises(ValueError, match="distinct combined"):
        selection.select_options(parent, count=4, fields=declared, dedup=True)


def test_dedup_step_preserves_exact_integer_bins():
    values = np.arange(4, dtype=np.uint64) + 2**60
    parent = OptionsConfig({"id": np.arange(4), "x": values})
    fields = {"x": {"target": "parent", "dedup_step": 2}}
    result = selection.select_options(parent, count=3, fields=fields, dedup=True)
    assert len(set(result.option_arrays["id"]) & {0, 1}) == 1
    np.testing.assert_array_equal(np.sort(result.option_arrays["id"])[1:], [2, 3])


@pytest.mark.parametrize("values,step", [
    (np.arange(3.) * u.arcsec, 5),
    (np.arange(3.) * u.arcsec, 5 * u.mag),
    (np.arange(3.), 5 * u.arcsec),
    (np.arange(3.) * u.arcsec, np.array([5]) * u.arcsec),
    (np.arange(3.) * u.arcsec, 0 * u.arcsec),
    (np.arange(3.), np.ma.array(5)),
])
def test_dedup_step_requires_positive_unmasked_scalar_with_matching_units(values, step):
    with pytest.raises(ValueError, match="dedup_step"):
        selection.select_options(options(3), count=1, dedup=True, fields={
            "x": {"values": values, "target": "parent", "dedup_step": step},
        })


def test_dedup_step_rounds_halfway_values_to_even_zero_anchored_bins():
    values = np.array([-7.5, -2.5, 2.5, 7.5])
    result = selection.select_options(options(4), count=3, dedup=True, fields={
        "x": {"values": values, "target": "parent", "dedup_step": 5},
    })
    ids = result.option_arrays["id"]
    assert 0 in ids and 3 in ids and len(set(ids) & {1, 2}) == 1
