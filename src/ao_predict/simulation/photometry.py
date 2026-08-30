"""Shared photometry helpers for simulator-specific photon unit conversions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u

from .._units import quantity_value


@dataclass(frozen=True)
class WFSPhotometryConfig:
    """Photometric conversion inputs derived from simulator configuration."""

    telescope_diameter: u.Quantity
    n_channels: float
    frame_rate: u.Quantity
    zeropoint: u.Quantity


def magnitudes_to_photons_per_second(
    magnitudes: u.Quantity,
    zeropoint: u.Quantity,
) -> u.Quantity:
    """Convert magnitudes into photon flux rate in photons/s."""
    zeropoint_value = float(quantity_value(zeropoint, u.photon / u.s, label="zeropoint").item())
    if zeropoint_value <= 0.0:
        raise ValueError("zeropoint must be > 0 for magnitude conversion.")
    magnitude_values = quantity_value(magnitudes, u.mag, label="magnitudes", dtype=float)
    photons_per_second = zeropoint_value * (10.0 ** (-0.4 * magnitude_values))
    return np.asarray(photons_per_second, dtype=float) * (u.photon / u.s)


def photons_per_second_to_magnitudes(
    photons_per_second: u.Quantity,
    zeropoint: u.Quantity,
) -> u.Quantity:
    """Convert photon flux rate in photons/s into magnitudes."""
    zeropoint_value = float(quantity_value(zeropoint, u.photon / u.s, label="zeropoint").item())
    if zeropoint_value <= 0.0:
        raise ValueError("zeropoint must be > 0 for magnitude conversion.")
    photons_per_second = quantity_value(photons_per_second, u.photon / u.s, label="photon flux", dtype=float)
    if np.any(photons_per_second < 0.0):
        raise ValueError("photons_per_second must be >= 0.")
    magnitudes = -2.5 * np.log10(np.clip(photons_per_second, 1e-30, None) / zeropoint_value)
    return np.asarray(magnitudes, dtype=float) * u.mag


def photons_per_second_to_photons_per_frame(
    photons_per_second: u.Quantity,
    photometry: WFSPhotometryConfig,
) -> u.Quantity:
    """Convert photon flux rate in photons/s into photons-per-frame units."""
    frame_rate = float(quantity_value(photometry.frame_rate, u.Hz, label="frame_rate").item())
    telescope_diameter = float(quantity_value(photometry.telescope_diameter, u.m, label="telescope_diameter").item())
    if frame_rate <= 0.0:
        raise ValueError("frame_rate must be > 0.")
    if telescope_diameter <= 0.0:
        raise ValueError("telescope_diameter must be > 0.")
    if photometry.n_channels <= 0.0:
        raise ValueError("n_channels must be > 0.")
    photons_per_second = quantity_value(photons_per_second, u.photon / u.s, label="photon flux", dtype=float)
    if np.any(photons_per_second < 0.0):
        raise ValueError("photons_per_second must be >= 0.")
    photons_per_frame = (
        photons_per_second
        / frame_rate
        * (telescope_diameter / float(photometry.n_channels)) ** 2
    )
    return photons_per_frame * u.photon


def photons_per_frame_to_photons_per_second(
    photons_per_frame: u.Quantity,
    photometry: WFSPhotometryConfig,
) -> u.Quantity:
    """Convert photons-per-frame units into photon flux rate in photons/s."""
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
    photons_per_second = (
        photons_per_frame
        * frame_rate
        / (telescope_diameter / float(photometry.n_channels)) ** 2
    )
    return np.asarray(photons_per_second, dtype=float) * (u.photon / u.s)


def magnitudes_to_photons_per_frame(
    magnitudes: u.Quantity,
    photometry: WFSPhotometryConfig,
) -> u.Quantity:
    """Convert magnitudes into photons-per-frame units."""
    magnitudes = quantity_value(magnitudes, u.mag, label="magnitudes", dtype=float).reshape(-1) * u.mag
    photons_per_second = magnitudes_to_photons_per_second(magnitudes, photometry.zeropoint)
    photons_per_frame = photons_per_second_to_photons_per_frame(photons_per_second, photometry)
    return photons_per_frame


def photons_per_frame_to_magnitudes(
    photons_per_frame: u.Quantity,
    photometry: WFSPhotometryConfig,
) -> u.Quantity:
    """Convert photons-per-frame units into magnitudes."""
    photons_per_frame = quantity_value(photons_per_frame, u.photon, label="photons_per_frame", dtype=float).reshape(-1) * u.photon
    photons_per_second = photons_per_frame_to_photons_per_second(photons_per_frame, photometry)
    magnitudes = photons_per_second_to_magnitudes(photons_per_second, photometry.zeropoint)
    return magnitudes
