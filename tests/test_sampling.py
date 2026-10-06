from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
from astropy import units as u

from ao_predict.simulation.sampling import SamplerRequest, _BUILTINS, _field_seed


def _request(
    name: str,
    *,
    owner: str,
    count: int,
    num_ngs: int = 1,
    parameters: dict[str, object] | None = None,
    unit: u.UnitBase | None = None,
    setup: dict[str, object] | None = None,
    fields: tuple[str, ...] | None = None,
    seed: int = 23,
) -> SamplerRequest:
    return SamplerRequest(
        owner_field=owner,
        fields=fields or (owner,),
        count=count,
        num_ngs=num_ngs,
        seed=seed,
        parameters=parameters or {},
        unit=unit,
        simulation={},
        setup=setup or {},
    )


def test_retained_scalar_ngs_and_polar_sampler_boundaries() -> None:
    cases = [
        ("uniform", "wavelength", 5, 1, {"minimum": -2, "maximum": 5}, u.um,
         [1.6210851869261247, 4.628738226703699, 3.3582183157783483, -0.02322909223021108, -0.4526824571684319]),
        ("uniform_mean_mag", "ngs_magnitude", 5, 1, {"minimum": 8, "maximum": 19}, u.mag,
         [13.690276722312483, 18.416588641962957, 16.420057353365976, 11.10635428363824, 10.431498995878178]),
        ("uniform_mean_mag", "ngs_magnitude", 1, 2, {"minimum": 8, "maximum": 19}, u.mag,
         [18.718386102887585, 8.662167341737382]),
        ("uniform_mean_mag", "ngs_magnitude", 1, 3, {"minimum": 8, "maximum": 19}, u.mag,
         [12.882864595368357, 18.764111603555065, 9.423853968014026]),
        ("weighted_discrete", "ngs_magnitude", 8, 1, {"values": [12, 12.5, 13], "weights": [10, 24, 18]}, u.mag,
         [12.5, 13, 13, 12.5, 12.5, 13, 12, 12.5]),
        ("uniform_area_radius", "ngs_r", 3, 1, {"maximum": 60}, u.arcsec,
         [43.15405406039763, 58.38720213996669, 52.49433430217841]),
        ("uniform_theta", "ngs_theta", 3, 1, {}, u.deg,
         [186.22723818477212, 340.9065373733331, 275.5655133828865]),
    ]
    for name, owner, count, num_ngs, parameters, unit, expected in cases:
        request = _request(name, owner=owner, count=count, num_ngs=num_ngs, parameters=parameters, unit=unit)
        sampler = _BUILTINS[name]
        sampler.validate_declaration(request)
        result = sampler().sample(request)[owner]
        np.testing.assert_array_equal(np.asarray(result).ravel(), expected)


def _science_setup(width: float, points_per_axis: int, margin: float | None = None) -> dict[str, u.Quantity]:
    axis = np.linspace(-width / 2, width / 2, points_per_axis)
    grid_x, grid_y = np.meshgrid(axis, axis)
    radius = np.sqrt(grid_x**2 + grid_y**2).flatten()
    theta = np.rad2deg(np.arctan2(grid_y, grid_x)).flatten()
    if margin is not None:
        cell_width = width / (points_per_axis - 1)
        mask = radius <= width / 2 + margin * cell_width
        radius = radius[mask]
        theta = theta[mask]
    return {"sci_r": radius * u.arcsec, "sci_theta": theta * u.deg}


def test_weighted_uniform_interval_mass_and_conditional_distribution() -> None:
    request = _request(
        "weighted_uniform", owner="ngs_magnitude", count=20000, unit=u.mag,
        parameters={"values": [[8, 16], [16, 18.5]], "weights": [10, 90]},
    )
    sampler = _BUILTINS["weighted_uniform"]()
    sampler.validate_declaration(request)
    before = np.random.get_state()
    values = sampler.sample(request)["ngs_magnitude"].value.ravel()
    np.testing.assert_array_equal(values, sampler.sample(request)["ngs_magnitude"].value.ravel())
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert abs(np.mean(values < 16) - .1) <= .02
    for lo, hi in ((8, 16), (16, 18.5)):
        selected = np.sort(values[(values >= lo) & (values < hi)])
        positions = (selected - lo) / (hi - lo)
        departure = max(
            np.max(np.arange(1, len(selected) + 1) / len(selected) - positions),
            np.max(positions - np.arange(len(selected)) / len(selected)),
        )
        assert departure <= .03


@pytest.mark.parametrize("owner,unit,num_ngs", [("wavelength", u.um, 1), ("ngs_magnitude", u.mag, 3), ("atm_profile_id", None, 1)])
def test_weighted_uniform_single_interval_matches_uniform(owner, unit, num_ngs) -> None:
    request = _request(
        "weighted_uniform", owner=owner, count=20, num_ngs=num_ngs, unit=unit,
        parameters={"values": [[8, 16]], "weights": [7]},
    )
    result = _BUILTINS["weighted_uniform"]().sample(request)[owner]
    expected = _BUILTINS["uniform"]().sample(replace(request, parameters={"minimum": 8, "maximum": 16}))[owner]
    np.testing.assert_array_equal(result, expected)
    assert result.shape == ((20, num_ngs) if owner == "ngs_magnitude" else (20,))


def test_weighted_uniform_preserves_order_gaps_overlap_and_zero_weights() -> None:
    request = _request(
        "weighted_uniform", owner="wavelength", count=500, unit=u.um,
        parameters={"values": [[10, 12], [-100, 100], [1, 2]], "weights": [4, 0, 6]},
    )
    result = _BUILTINS["weighted_uniform"]().sample(request)["wavelength"].value
    draws = np.random.RandomState(23).random_sample(500)
    np.testing.assert_allclose(result, np.where(draws < .4, 10 + 2 * draws / .4, 1 + (draws - .4) / .6), rtol=0, atol=2e-15)
    for intervals in ([[1, 2], [1, 2]], [[1, 3], [2, 4]]):
        result = _BUILTINS["weighted_uniform"]().sample(replace(request, parameters={"values": intervals, "weights": [1, 1]}))["wavelength"].value
        expected = np.where(draws < .5, intervals[0][0] + (intervals[0][1] - intervals[0][0]) * draws / .5,
                            intervals[1][0] + (intervals[1][1] - intervals[1][0]) * (draws - .5) / .5)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("parameters", [
    {}, {"values": [], "weights": []}, {"values": [1, 2], "weights": [1]},
    {"values": [[1, 1]], "weights": [1]}, {"values": [[2, 1]], "weights": [1]},
    {"values": [[1, np.inf]], "weights": [1]}, {"values": [[1, 2]], "weights": [-1]},
    {"values": [[1, 2]], "weights": [0]}, {"values": [[1, 2]], "weights": [np.nan]},
    {"values": [[1, 2]], "weights": [np.inf]}, {"values": [[1, 2]], "weights": [1, 2]},
    {"values": [[1, 2], [2, 3]], "weights": [1e308, 1e308]},
    {"values": [["bad", 2]], "weights": [1]},
    {"values": [[1, 2]], "weights": [1], "extra": 1},
])
def test_weighted_uniform_invalid_inputs(parameters) -> None:
    request = _request("weighted_uniform", owner="wavelength", count=2, unit=u.um, parameters=parameters)
    with pytest.raises(ValueError):
        _BUILTINS["weighted_uniform"].validate_declaration(request)


def test_weighted_uniform_rejects_joint_declaration() -> None:
    request = _request("weighted_uniform", owner="sci_dx", count=2, unit=u.arcsec,
                       fields=("sci_dx", "sci_dy"), parameters={"values": [[1, 2]], "weights": [1]})
    with pytest.raises(ValueError, match="exactly one field"):
        _BUILTINS["weighted_uniform"].validate_declaration(request)


@pytest.mark.parametrize(
    ("width", "points", "margin", "shape", "digest"),
    [
        (120.0, 11, 1.0, (2, 109), "d39d5f4552cf2056640934cf0e34eaa1f109a4024de5530696d80a270f2465a5"),
        (20.0, 9, None, (2, 81), "077e9b10a7c8ebf16e2bd1463dd3244d5a34aeaeedcbd1b0ac5c52feb6797b4f"),
    ],
)
def test_retained_joint_science_offset_boundaries(width, points, margin, shape, digest) -> None:
    request = _request(
        "stratified_science_offsets",
        owner="sci_dx",
        count=2,
        unit=u.arcsec,
        fields=("sci_dx", "sci_dy"),
        setup=_science_setup(width, points, margin),
    )
    sampler = _BUILTINS["stratified_science_offsets"]
    sampler.validate_declaration(request)
    result = sampler().sample(request)
    dx = result["sci_dx"].to_value(u.arcsec)
    dy = result["sci_dy"].to_value(u.arcsec)
    assert dx.shape == dy.shape == shape
    assert dx.dtype == dy.dtype == np.float32
    actual = sha256(np.asarray(dx, dtype="<f4", order="C").tobytes() + np.asarray(dy, dtype="<f4", order="C").tobytes()).hexdigest()
    assert actual == digest
    if width == 120.0:
        np.testing.assert_array_equal(dx[0, :4], [-2.3663101196289062, 2.327198028564453, -0.9318637847900391, 1.6974983215332031])


@pytest.mark.parametrize("name", ["stratified_science_offsets", "stratified_science_offsets_redistributed"])
def test_irregular_science_grid_is_rejected(name) -> None:
    setup = _science_setup(20.0, 3)
    setup["sci_r"][0] += 0.2 * u.arcsec
    request = _request(
        name,
        owner="sci_dx",
        count=2,
        unit=u.arcsec,
        fields=("sci_dx", "sci_dy"),
        setup=setup,
    )
    with pytest.raises(ValueError, match="uniform Cartesian grid spacing"):
        _BUILTINS[name].validate_declaration(request)


def test_full_science_grid_replays_within_inferred_support() -> None:
    setup = _science_setup(10.0, 11)
    request = _request(
        "stratified_science_offsets", owner="sci_dx", count=3, unit=u.arcsec,
        fields=("sci_dx", "sci_dy"), setup=setup,
    )
    sampler = _BUILTINS["stratified_science_offsets"]()
    first = sampler.sample(request)
    second = sampler.sample(request)
    for axis in ("sci_dx", "sci_dy"):
        assert first[axis].shape == (3, 121)
        np.testing.assert_array_equal(first[axis], second[axis])
    theta = setup["sci_theta"].to_value(u.rad)
    x = setup["sci_r"].to_value(u.arcsec) * np.cos(theta) + first["sci_dx"].value
    y = setup["sci_r"].to_value(u.arcsec) * np.sin(theta) + first["sci_dy"].value
    assert np.all(np.abs(x) <= 5.00001)
    assert np.all(np.abs(y) <= 5.00001)


@pytest.mark.parametrize("name", ["stratified_science_offsets", "stratified_science_offsets_redistributed"])
def test_repeating_decimal_grid_spacing_is_supported(name) -> None:
    request = _request(
        name, owner="sci_dx", count=2, unit=u.arcsec,
        fields=("sci_dx", "sci_dy"), setup=_science_setup(20.0, 4),
    )
    sampler = _BUILTINS[name]
    sampler.validate_declaration(request)
    assert sampler().sample(request)["sci_dx"].shape == (2, 16)


@pytest.mark.parametrize("name", ["stratified_science_offsets", "stratified_science_offsets_redistributed"])
@pytest.mark.parametrize(
    ("fields", "parameters", "message"),
    [
        (("sci_dx",), {}, "sci_dx owner"),
        (("sci_dy", "sci_dx"), {}, "sci_dx owner"),
        (("sci_dx", "sci_dy"), {"maximum": 1}, "parameters must be empty"),
    ],
)
def test_science_offset_declarations_are_validated(name, fields, parameters, message) -> None:
    request = _request(name, owner="sci_dx", count=1, fields=fields, parameters=parameters, setup=_science_setup(20, 3))
    with pytest.raises(ValueError, match=message):
        _BUILTINS[name].validate_declaration(request)


@pytest.mark.parametrize(("width", "points", "margin"), [(20, 9, None), (40, 9, 1), (120, 11, 1)])
@pytest.mark.parametrize("seed", [0, 23, 123])
def test_redistributed_science_offsets_preserve_interior_draws_and_replay(width, points, margin, seed) -> None:
    setup = _science_setup(width, points, margin)
    request = _request(
        "stratified_science_offsets_redistributed", owner="sci_dx", count=200,
        fields=("sci_dx", "sci_dy"), unit=u.arcsec, setup=setup, seed=seed,
    )
    sampler = _BUILTINS["stratified_science_offsets_redistributed"]
    before = np.random.get_state()
    sampler.validate_declaration(request)
    first = sampler().sample(request)
    second = sampler().sample(request)
    changed = sampler().sample(replace(request, seed=seed + 1))
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    for axis in ("sci_dx", "sci_dy"):
        assert first[axis].unit == u.arcsec
        assert first[axis].dtype == np.float32
        assert first[axis].shape == (200, len(setup["sci_r"]))
        assert np.all(np.isfinite(first[axis]))
        np.testing.assert_array_equal(first[axis], second[axis])
        assert not np.array_equal(first[axis], changed[axis])

    theta = setup["sci_theta"].to_value(u.rad)
    base_x = np.round(setup["sci_r"].value * np.cos(theta), 6).astype(np.float32)
    base_y = np.round(setup["sci_r"].value * np.sin(theta), 6).astype(np.float32)
    spacing = width / (points - 1)
    rng = np.random.default_rng(seed)
    raw_dx = rng.random(first["sci_dx"].shape, dtype=np.float32) * np.float32(spacing) - np.float32(spacing / 2)
    raw_dy = rng.random(first["sci_dy"].shape, dtype=np.float32) * np.float32(spacing) - np.float32(spacing / 2)
    raw_x, raw_y = base_x + raw_dx, base_y + raw_dy
    interior = (np.abs(raw_x) < width / 2 - 1e-4) & (np.abs(raw_y) < width / 2 - 1e-4)
    x, y = base_x + first["sci_dx"].value, base_y + first["sci_dy"].value
    assert np.all(np.abs(x) <= width / 2 + 1e-5)
    assert np.all(np.abs(y) <= width / 2 + 1e-5)
    if margin is not None:
        diagonal_limit = np.max(np.abs(base_x) + np.abs(base_y))
        interior &= np.abs(raw_x) + np.abs(raw_y) < diagonal_limit - 1e-4
        assert np.all(np.abs(x) + np.abs(y) <= diagonal_limit + 1e-5)
    np.testing.assert_array_equal(first["sci_dx"].value[interior], raw_dx[interior])
    np.testing.assert_array_equal(first["sci_dy"].value[interior], raw_dy[interior])
    assert np.any(np.abs(first["sci_dx"].value) > spacing / 2)


@pytest.mark.parametrize("seed", [0, 23, 123])
@pytest.mark.parametrize("margin", [None, 1])
def test_redistributed_science_positions_have_uniform_area_coverage(seed, margin) -> None:
    request = _request(
        "stratified_science_offsets_redistributed", owner="sci_dx", count=5_000,
        fields=("sci_dx", "sci_dy"), unit=u.arcsec, setup=_science_setup(20, 9, margin), seed=seed,
    )
    result = _BUILTINS["stratified_science_offsets_redistributed"]().sample(request)
    theta = request.setup["sci_theta"].to_value(u.rad)
    x = request.setup["sci_r"].value * np.cos(theta) + result["sci_dx"].value
    y = request.setup["sci_r"].value * np.sin(theta) + result["sci_dy"].value
    cells, _, _ = np.histogram2d(x.ravel(), y.ravel(), bins=8, range=((-10, 10), (-10, 10)))
    assert abs(cells.sum() / x.size - 1) < 1e-5
    areas = np.ones((8, 8))
    if margin is not None:
        areas[0, 0] = areas[0, -1] = areas[-1, 0] = areas[-1, -1] = 0.5
    assert np.max(np.abs(cells / x.size - areas / areas.sum())) < 0.001


@pytest.mark.parametrize("geometry", ["hole", "sloping_gap", "line"])
def test_redistributed_science_offsets_reject_uncovered_or_zero_area_support(geometry) -> None:
    setup = _science_setup(20, 5)
    theta = setup["sci_theta"].to_value(u.rad)
    x = np.round(setup["sci_r"].value * np.cos(theta), 6)
    y = np.round(setup["sci_r"].value * np.sin(theta), 6)
    if geometry == "hole":
        keep = (x != 0) | (y != 0)
    elif geometry == "sloping_gap":
        keep = y <= x / 2 + 5
    else:
        keep = x == y
    setup = {axis: values[keep] for axis, values in setup.items()}
    request = _request(
        "stratified_science_offsets_redistributed", owner="sci_dx", count=2,
        fields=("sci_dx", "sci_dy"), unit=u.arcsec, setup=setup,
    )
    sampler = _BUILTINS["stratified_science_offsets_redistributed"]
    with pytest.raises(ValueError, match="cover.*without gaps|positive area"):
        sampler.validate_declaration(request)
    with pytest.raises(ValueError, match="cover.*without gaps|positive area"):
        sampler().sample(request)


@pytest.mark.parametrize(
    "parameters",
    [
        {"values": [], "weights": []},
        {"values": [1, 2], "weights": [1]},
        {"values": [1, 2], "weights": [-1, 2]},
        {"values": [1, 2], "weights": [0, 0]},
        {"values": [1, float("nan")], "weights": [1, 1]},
    ],
)
def test_weighted_discrete_rejects_invalid_inputs(parameters) -> None:
    request = _request("weighted_discrete", owner="ngs_magnitude", count=1, unit=u.mag, parameters=parameters)
    with pytest.raises(ValueError):
        _BUILTINS["weighted_discrete"].validate_declaration(request)


def _uniform_cdf_error(values: np.ndarray, minimum: float, maximum: float) -> float:
    normalized = np.sort((np.asarray(values).ravel() - minimum) / (maximum - minimum))
    assert np.all((normalized >= 0) & (normalized <= 1))
    n = normalized.size
    return float(max(np.max(np.arange(1, n + 1) / n - normalized), np.max(normalized - np.arange(n) / n)))


def test_predeclared_distribution_properties_and_private_random_state() -> None:
    before = np.random.get_state()
    count = 20_000
    uniform = _BUILTINS["uniform"]().sample(
        _request("uniform", owner="wavelength", count=count, parameters={"minimum": -2, "maximum": 5}, unit=u.um)
    )["wavelength"].value
    assert _uniform_cdf_error(uniform, -2, 5) <= 0.02

    for num_ngs in (1, 2, 3):
        magnitudes = _BUILTINS["uniform_mean_mag"]().sample(
            _request("uniform_mean_mag", owner="ngs_magnitude", count=count, num_ngs=num_ngs,
                     parameters={"minimum": 8, "maximum": 19}, unit=u.mag)
        )["ngs_magnitude"].value
        assert magnitudes.shape == (count, num_ngs)
        assert np.all((magnitudes >= 8) & (magnitudes <= 19))
        assert _uniform_cdf_error(magnitudes.mean(axis=1), 8, 19) <= 0.02
        if num_ngs > 1:
            ranks = np.argsort(magnitudes, axis=1)
            for rank in range(num_ngs):
                for slot in range(num_ngs):
                    assert abs(np.mean(ranks[:, rank] == slot) - 1 / num_ngs) <= 0.03

    weighted = _BUILTINS["weighted_discrete"]().sample(
        _request("weighted_discrete", owner="ngs_magnitude", count=count, unit=u.mag,
                 parameters={"values": [12, 12.5, 13], "weights": [10, 24, 18]})
    )["ngs_magnitude"].value.ravel()
    for value, weight in zip((12, 12.5, 13), (10, 24, 18), strict=True):
        assert abs(np.mean(weighted == value) - weight / 52) <= 0.02

    radii = _BUILTINS["uniform_area_radius"]().sample(
        _request("uniform_area_radius", owner="ngs_r", count=count, num_ngs=2,
                 parameters={"maximum": 60}, unit=u.arcsec, seed=_field_seed(23, "ngs_r"))
    )["ngs_r"].value
    angles = _BUILTINS["uniform_theta"]().sample(
        _request("uniform_theta", owner="ngs_theta", count=count, num_ngs=2, unit=u.deg,
                 seed=_field_seed(23, "ngs_theta"))
    )["ngs_theta"].value
    radial = (radii / 60) ** 2
    angular = (angles % 360) / 360
    assert _uniform_cdf_error(radial, 0, 1) <= 0.02
    assert _uniform_cdf_error(angular, 0, 1) <= 0.02
    for star in range(2):
        cells, _, _ = np.histogram2d(radial[:, star], angular[:, star], bins=(4, 8), range=((0, 1), (0, 1)))
        assert np.max(np.abs(cells / count - 1 / 32)) <= 0.006

    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
