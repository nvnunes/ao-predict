from __future__ import annotations

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


def test_irregular_science_grid_is_rejected() -> None:
    setup = _science_setup(20.0, 3)
    setup["sci_r"][0] += 0.2 * u.arcsec
    request = _request(
        "stratified_science_offsets",
        owner="sci_dx",
        count=2,
        unit=u.arcsec,
        fields=("sci_dx", "sci_dy"),
        setup=setup,
    )
    with pytest.raises(ValueError, match="uniform Cartesian grid spacing"):
        _BUILTINS["stratified_science_offsets"].validate_declaration(request)


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


def test_repeating_decimal_grid_spacing_is_supported() -> None:
    request = _request(
        "stratified_science_offsets", owner="sci_dx", count=2, unit=u.arcsec,
        fields=("sci_dx", "sci_dy"), setup=_science_setup(20.0, 4),
    )
    sampler = _BUILTINS["stratified_science_offsets"]
    sampler.validate_declaration(request)
    assert sampler().sample(request)["sci_dx"].shape == (2, 16)


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
