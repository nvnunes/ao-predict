from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from ao_predict.simulation.photometry import (
    WFSPhotometryConfig,
    magnitudes_to_photon_flux,
    magnitudes_to_photons_per_frame,
    photon_flux_to_magnitudes,
    photon_flux_to_photons_per_frame,
    photons_per_frame_to_magnitudes,
    photons_per_frame_to_photon_flux,
)


def test_photometry_preserves_photon_counts_and_corrects_flux_units() -> None:
    flux_unit = u.photon / (u.m**2 * u.s)
    config = WFSPhotometryConfig(
        telescope_diameter=8.0 * u.m,
        n_channels=16,
        frame_rate=500.0 * u.Hz,
        zeropoint=3.0e10 * flux_unit,
    )
    magnitudes = np.array([0.0, 14.0]) * u.mag

    flux = magnitudes_to_photon_flux(magnitudes, config.zeropoint)
    assert flux.unit == flux_unit
    np.testing.assert_allclose(flux.to_value(flux_unit), [3.0e10, 75356.59294528731])
    np.testing.assert_allclose(photon_flux_to_magnitudes(flux, config.zeropoint).to_value(u.mag), [0.0, 14.0])

    photons = magnitudes_to_photons_per_frame(magnitudes, config)
    assert photons.unit == u.photon
    np.testing.assert_allclose(photons.to_value(u.photon), [15000000.0, 37.67829647264365])
    np.testing.assert_allclose(photon_flux_to_photons_per_frame(flux, config), photons)
    np.testing.assert_allclose(photons_per_frame_to_photon_flux(photons, config), flux)
    np.testing.assert_allclose(
        photons.to_value(u.photon) * config.frame_rate.to_value(u.Hz),
        [7500000000.0, 18839.148236321827],
    )
    np.testing.assert_allclose(
        photons_per_frame_to_magnitudes(np.round(photons.to_value(u.photon)) * u.photon, config).to_value(u.mag),
        [0.0, 13.990769156097178],
    )
    assert photon_flux_to_magnitudes(0.0 * flux_unit, config.zeropoint).to_value(u.mag) == pytest.approx(
        101.19280313679916
    )


def test_photometry_rejects_post_aperture_rate_as_flux() -> None:
    with pytest.raises(ValueError, match="photon flux"):
        photon_flux_to_magnitudes(1.0 * u.photon / u.s, 3.0e10 * u.photon / (u.m**2 * u.s))
