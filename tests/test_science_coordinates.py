from __future__ import annotations

import numpy as np
import pytest

from ao_predict.simulation import schema
from ao_predict.simulation.coordinates import resolve_science_coordinates


def _setup() -> dict[str, np.ndarray]:
    return {
        schema.KEY_SETUP_SCI_R_ARCSEC: np.array([0.0, 10.0, 10.0]),
        schema.KEY_SETUP_SCI_THETA_DEG: np.array([0.0, 0.0, 90.0]),
    }


def test_resolve_science_coordinates_defaults_missing_offsets_to_zero() -> None:
    resolved = resolve_science_coordinates(_setup(), {})

    np.testing.assert_allclose(resolved.r_arcsec, np.array([0.0, 10.0, 10.0]))
    np.testing.assert_allclose(resolved.theta_deg, np.array([0.0, 0.0, 90.0]))
    np.testing.assert_allclose(resolved.x_arcsec, np.array([0.0, 10.0, 0.0]), atol=1.0e-12)
    np.testing.assert_allclose(resolved.y_arcsec, np.array([0.0, 0.0, 10.0]), atol=1.0e-12)


def test_resolve_science_coordinates_applies_each_offset_axis_independently() -> None:
    x_only = resolve_science_coordinates(
        _setup(),
        {schema.KEY_OPTION_SCI_DX_ARCSEC: np.array([1.0, -2.0, 3.0])},
    )
    y_only = resolve_science_coordinates(
        _setup(),
        {schema.KEY_OPTION_SCI_DY_ARCSEC: np.array([1.0, -2.0, 3.0])},
    )

    np.testing.assert_allclose(x_only.x_arcsec, np.array([1.0, 8.0, 3.0]), atol=1.0e-12)
    np.testing.assert_allclose(x_only.y_arcsec, np.array([0.0, 0.0, 10.0]), atol=1.0e-12)
    np.testing.assert_allclose(y_only.x_arcsec, np.array([0.0, 10.0, 0.0]), atol=1.0e-12)
    np.testing.assert_allclose(y_only.y_arcsec, np.array([1.0, -2.0, 13.0]), atol=1.0e-12)


def test_resolve_science_coordinates_preserves_tiny_nonzero_direction() -> None:
    setup = {
        schema.KEY_SETUP_SCI_R_ARCSEC: np.array([0.0]),
        schema.KEY_SETUP_SCI_THETA_DEG: np.array([0.0]),
    }

    resolved = resolve_science_coordinates(
        setup,
        {schema.KEY_OPTION_SCI_DY_ARCSEC: np.array([1.0e-9])},
    )

    np.testing.assert_allclose(resolved.r_arcsec, np.array([1.0e-9]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(resolved.theta_deg, np.array([90.0]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(resolved.x_arcsec, np.array([0.0]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(resolved.y_arcsec, np.array([1.0e-9]), rtol=0.0, atol=0.0)


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
            {schema.KEY_OPTION_SCI_DX_ARCSEC: offset},
        )
