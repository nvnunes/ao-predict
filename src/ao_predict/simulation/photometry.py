"""Shared photometry helpers for simulator-specific photon unit conversions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WFSPhotometryConfig:
    """Photometric conversion inputs derived from simulator configuration."""

    telescope_diameter_m: float
    n_channels: float
    frame_rate_hz: float
    zeropoint: float


def magnitudes_to_photons_per_second(
    magnitudes: np.ndarray,
    zeropoint: float,
) -> np.ndarray:
    """Convert magnitudes into photon flux rate in photons/s."""
    if float(zeropoint) <= 0.0:
        raise ValueError("zeropoint must be > 0 for magnitude conversion.")
    photons_per_second = float(zeropoint) * (10.0 ** (-0.4 * np.asarray(magnitudes, dtype=float)))
    return np.asarray(photons_per_second, dtype=float)


def photons_per_second_to_magnitudes(
    photons_per_second: np.ndarray,
    zeropoint: float,
) -> np.ndarray:
    """Convert photon flux rate in photons/s into magnitudes."""
    if float(zeropoint) <= 0.0:
        raise ValueError("zeropoint must be > 0 for magnitude conversion.")
    photons_per_second = np.asarray(photons_per_second, dtype=float)
    if np.any(photons_per_second < 0.0):
        raise ValueError("photons_per_second must be >= 0.")
    magnitudes = -2.5 * np.log10(np.clip(photons_per_second, 1e-30, None) / float(zeropoint))
    return np.asarray(magnitudes, dtype=float)


def photons_per_second_to_photons_per_frame(
    photons_per_second: np.ndarray,
    photometry: WFSPhotometryConfig,
) -> np.ndarray:
    """Convert photon flux rate in photons/s into photons-per-frame units."""
    if photometry.frame_rate_hz <= 0.0:
        raise ValueError("frame_rate_hz must be > 0.")
    if photometry.telescope_diameter_m <= 0.0:
        raise ValueError("telescope_diameter_m must be > 0.")
    if photometry.n_channels <= 0.0:
        raise ValueError("n_channels must be > 0.")
    photons_per_second = np.asarray(photons_per_second, dtype=float)
    if np.any(photons_per_second < 0.0):
        raise ValueError("photons_per_second must be >= 0.")
    photons_per_frame = (
        photons_per_second
        / float(photometry.frame_rate_hz)
        * (float(photometry.telescope_diameter_m) / float(photometry.n_channels)) ** 2
    )
    return photons_per_frame


def photons_per_frame_to_photons_per_second(
    photons_per_frame: np.ndarray,
    photometry: WFSPhotometryConfig,
) -> np.ndarray:
    """Convert photons-per-frame units into photon flux rate in photons/s."""
    if photometry.frame_rate_hz <= 0.0:
        raise ValueError("frame_rate_hz must be > 0.")
    if photometry.telescope_diameter_m <= 0.0:
        raise ValueError("telescope_diameter_m must be > 0.")
    if photometry.n_channels <= 0.0:
        raise ValueError("n_channels must be > 0.")
    photons_per_frame = np.asarray(photons_per_frame, dtype=float)
    if np.any(photons_per_frame < 0.0):
        raise ValueError("photons_per_frame must be >= 0.")
    photons_per_second = (
        photons_per_frame
        * float(photometry.frame_rate_hz)
        / (float(photometry.telescope_diameter_m) / float(photometry.n_channels)) ** 2
    )
    return np.asarray(photons_per_second, dtype=float)


def magnitudes_to_photons_per_frame(
    magnitudes: np.ndarray,
    photometry: WFSPhotometryConfig,
) -> np.ndarray:
    """Convert magnitudes into photons-per-frame units."""
    magnitudes = np.asarray(magnitudes, dtype=float).reshape(-1)
    photons_per_second = magnitudes_to_photons_per_second(magnitudes, photometry.zeropoint)
    photons_per_frame = photons_per_second_to_photons_per_frame(photons_per_second, photometry)
    return photons_per_frame


def photons_per_frame_to_magnitudes(
    photons_per_frame: np.ndarray,
    photometry: WFSPhotometryConfig,
) -> np.ndarray:
    """Convert photons-per-frame units into magnitudes."""
    photons_per_frame = np.asarray(photons_per_frame, dtype=float).reshape(-1)
    photons_per_second = photons_per_frame_to_photons_per_second(photons_per_frame, photometry)
    magnitudes = photons_per_second_to_magnitudes(photons_per_second, photometry.zeropoint)
    return magnitudes
