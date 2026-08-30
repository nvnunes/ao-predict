from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from ao_predict.simulation import schema
from ao_predict.simulation.coordinates import resolve_science_coordinates


def _setup() -> dict[str, u.Quantity]:
    return {
        schema.KEY_SETUP_SCI_R: np.array([0.0, 10.0, 10.0]) * u.arcsec,
        schema.KEY_SETUP_SCI_THETA: np.array([0.0, 0.0, 90.0]) * u.deg,
    }


def test_resolve_science_coordinates_defaults_missing_offsets_to_zero() -> None:
    resolved = resolve_science_coordinates(_setup(), {})

    np.testing.assert_allclose(resolved.r, np.array([0.0, 10.0, 10.0]) * u.arcsec)
    np.testing.assert_allclose(resolved.theta, np.array([0.0, 0.0, 90.0]) * u.deg)
    np.testing.assert_allclose(resolved.x, np.array([0.0, 10.0, 0.0]) * u.arcsec, atol=1.0e-12)
    np.testing.assert_allclose(resolved.y, np.array([0.0, 0.0, 10.0]) * u.arcsec, atol=1.0e-12)


def test_resolve_science_coordinates_applies_each_offset_axis_independently() -> None:
    x_only = resolve_science_coordinates(
        _setup(),
        {schema.KEY_OPTION_SCI_DX: np.array([1.0, -2.0, 3.0]) * u.arcsec},
    )
    y_only = resolve_science_coordinates(
        _setup(),
        {schema.KEY_OPTION_SCI_DY: np.array([1.0, -2.0, 3.0]) * u.arcsec},
    )

    np.testing.assert_allclose(x_only.x, np.array([1.0, 8.0, 3.0]) * u.arcsec, atol=1.0e-12)
    np.testing.assert_allclose(x_only.y, np.array([0.0, 0.0, 10.0]) * u.arcsec, atol=1.0e-12)
    np.testing.assert_allclose(y_only.x, np.array([0.0, 10.0, 0.0]) * u.arcsec, atol=1.0e-12)
    np.testing.assert_allclose(y_only.y, np.array([1.0, -2.0, 13.0]) * u.arcsec, atol=1.0e-12)


def test_resolve_science_coordinates_preserves_tiny_nonzero_direction() -> None:
    setup = {
        schema.KEY_SETUP_SCI_R: np.array([0.0]) * u.arcsec,
        schema.KEY_SETUP_SCI_THETA: np.array([0.0]) * u.deg,
    }

    resolved = resolve_science_coordinates(
        setup,
        {schema.KEY_OPTION_SCI_DY: np.array([1.0e-9]) * u.arcsec},
    )

    np.testing.assert_allclose(resolved.r, np.array([1.0e-9]) * u.arcsec, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(resolved.theta, np.array([90.0]) * u.deg, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(resolved.x, np.array([0.0]) * u.arcsec, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(resolved.y, np.array([1.0e-9]) * u.arcsec, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "offset",
    [
        np.array([[1.0, 2.0, 3.0]]),
        np.array([1.0, 2.0]),
        np.array([1.0, np.nan, 3.0]),
    ],
)
def test_resolve_science_coordinates_rejects_malformed_offset_rows(offset: np.ndarray) -> None:
    with pytest.raises(ValueError):
        resolve_science_coordinates(
            _setup(),
            {schema.KEY_OPTION_SCI_DX: offset * u.arcsec},
        )
