"""Shared photometry helpers for simulator-specific photon unit conversions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u

from .._units import quantity_value


_PHOTON_FLUX_UNIT = u.photon / (u.m**2 * u.s)


@dataclass(frozen=True)
class WFSPhotometryConfig:
    """Inputs for converting NGS magnitudes to WFS photon counts.

    ``zeropoint`` is the zero-magnitude flux in ``photon / (m2 s)``.
    ``n_channels`` counts lenslets across the telescope diameter, so the
    nominal subaperture area is ``(telescope_diameter / n_channels)**2``.
    ``frame_rate`` has frequency units.
    """

    telescope_diameter: u.Quantity
    n_channels: float
    frame_rate: u.Quantity
    zeropoint: u.Quantity


def magnitudes_to_photon_flux(
    magnitudes: u.Quantity,
    zeropoint: u.Quantity,
) -> u.Quantity:
    """Convert magnitudes into pre-aperture photon flux per square metre."""
    zeropoint_value = float(quantity_value(zeropoint, _PHOTON_FLUX_UNIT, label="zeropoint").item())
    if zeropoint_value <= 0.0:
        raise ValueError("zeropoint must be > 0 for magnitude conversion.")
    magnitude_values = quantity_value(magnitudes, u.mag, label="magnitudes", dtype=float)
    photon_flux = zeropoint_value * (10.0 ** (-0.4 * magnitude_values))
    return np.asarray(photon_flux, dtype=float) * _PHOTON_FLUX_UNIT


def photon_flux_to_magnitudes(
    photon_flux: u.Quantity,
    zeropoint: u.Quantity,
) -> u.Quantity:
    """Convert pre-aperture photon flux per square metre into magnitudes."""
    zeropoint_value = float(quantity_value(zeropoint, _PHOTON_FLUX_UNIT, label="zeropoint").item())
    if zeropoint_value <= 0.0:
        raise ValueError("zeropoint must be > 0 for magnitude conversion.")
    photon_flux = quantity_value(photon_flux, _PHOTON_FLUX_UNIT, label="photon flux", dtype=float)
    if np.any(photon_flux < 0.0):
        raise ValueError("photon_flux must be >= 0.")
    magnitudes = -2.5 * np.log10(np.clip(photon_flux, 1e-30, None) / zeropoint_value)
    return np.asarray(magnitudes, dtype=float) * u.mag


def photon_flux_to_photons_per_frame(
    photon_flux: u.Quantity,
    photometry: WFSPhotometryConfig,
) -> u.Quantity:
    """Apply nominal subaperture area and frame rate to photon flux."""
    frame_rate = float(quantity_value(photometry.frame_rate, u.Hz, label="frame_rate").item())
    telescope_diameter = float(quantity_value(photometry.telescope_diameter, u.m, label="telescope_diameter").item())
    if frame_rate <= 0.0:
        raise ValueError("frame_rate must be > 0.")
    if telescope_diameter <= 0.0:
        raise ValueError("telescope_diameter must be > 0.")
    if photometry.n_channels <= 0.0:
        raise ValueError("n_channels must be > 0.")
    photon_flux = quantity_value(photon_flux, _PHOTON_FLUX_UNIT, label="photon flux", dtype=float)
    if np.any(photon_flux < 0.0):
        raise ValueError("photon_flux must be >= 0.")
    photons_per_frame = (
        photon_flux
        / frame_rate
        * (telescope_diameter / float(photometry.n_channels)) ** 2
    )
    return photons_per_frame * u.photon


def photons_per_frame_to_photon_flux(
    photons_per_frame: u.Quantity,
    photometry: WFSPhotometryConfig,
) -> u.Quantity:
    """Remove nominal subaperture area and frame time from photon counts."""
    frame_rate = float(quantity_value(photometry.frame_rate, u.Hz, label="frame_rate").item())
    telescope_diameter = float(quantity_value(photometry.telescope_diameter, u.m, label="telescope_diameter").item())
    if frame_rate <= 0.0:
        raise ValueError("frame_rate must be > 0.")
    if telescope_diameter <= 0.0:
        raise ValueError("telescope_diameter must be > 0.")
    if photometry.n_channels <= 0.0:
        raise ValueError("n_channels must be > 0.")
    photons_per_frame = quantity_value(photons_per_frame, u.photon, label="photons_per_frame", dtype=float)
    if np.any(photons_per_frame < 0.0):
        raise ValueError("photons_per_frame must be >= 0.")
    photon_flux = (
        photons_per_frame
        * frame_rate
        / (telescope_diameter / float(photometry.n_channels)) ** 2
    )
    return np.asarray(photon_flux, dtype=float) * _PHOTON_FLUX_UNIT


def magnitudes_to_photons_per_frame(
    magnitudes: u.Quantity,
    photometry: WFSPhotometryConfig,
) -> u.Quantity:
    """Convert magnitudes into photons-per-frame units."""
    magnitudes = quantity_value(magnitudes, u.mag, label="magnitudes", dtype=float).reshape(-1) * u.mag
    photon_flux = magnitudes_to_photon_flux(magnitudes, photometry.zeropoint)
    photons_per_frame = photon_flux_to_photons_per_frame(photon_flux, photometry)
    return photons_per_frame


def photons_per_frame_to_magnitudes(
    photons_per_frame: u.Quantity,
    photometry: WFSPhotometryConfig,
) -> u.Quantity:
    """Convert photons-per-frame units into magnitudes."""
    photons_per_frame = quantity_value(photons_per_frame, u.photon, label="photons_per_frame", dtype=float).reshape(-1) * u.photon
    photon_flux = photons_per_frame_to_photon_flux(photons_per_frame, photometry)
    magnitudes = photon_flux_to_magnitudes(photon_flux, photometry.zeropoint)
    return magnitudes
