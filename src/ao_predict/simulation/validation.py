"""Core validation utilities for persisted simulation payloads and results.

These validators enforce ao-predict's schema-level contracts that are shared
across simulation implementations.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from . import schema
from .interfaces import SimulationResult, SimulationState
from ..utils import (
    as_array,
    as_float_matrix,
    as_float_scalar,
    as_float_vector,
    require_keys,
)


#
# Payload validators
#

def validate_simulation_payload_core(
    simulation: Mapping[str, Any],
    expected_name: str | None = None,
    expected_version: str | None = None,
) -> None:
    """Validate core persisted ``/simulation`` payload constraints.

    Args:
        simulation: Candidate ``/simulation`` payload mapping.
        expected_name: Optional simulation class ``name`` expected by the caller.
        expected_version: Optional simulation class ``version`` expected by the caller.

    Raises:
        ValueError: If required keys are missing or values are invalid.
    """
    require_keys(simulation, schema.REQUIRED_SIMULATION_KEYS, label="simulation")

    simulation_name = str(simulation[schema.KEY_SIMULATION_NAME]).strip()
    simulation_version = str(simulation[schema.KEY_SIMULATION_VERSION]).strip()
    if not simulation_name:
        raise ValueError(f"simulation['{schema.KEY_SIMULATION_NAME}'] must be a non-empty string.")
    if not simulation_version:
        raise ValueError(f"simulation['{schema.KEY_SIMULATION_VERSION}'] must be a non-empty string.")
    if expected_name is not None:
        normalized_expected_name = str(expected_name).strip()
        if not normalized_expected_name:
            raise ValueError("Simulation implementation name must be a non-empty string.")
        if simulation_name != normalized_expected_name:
            raise ValueError(
                f"Simulation payload name mismatch: payload has '{simulation_name}', "
                f"but instantiated simulation is '{normalized_expected_name}'."
            )
    if expected_version is not None:
        normalized_expected_version = str(expected_version).strip()
        if not normalized_expected_version:
            raise ValueError("Simulation implementation version must be a non-empty string.")
        if simulation_version != normalized_expected_version:
            raise ValueError(
                f"Simulation payload version mismatch: payload has '{simulation_version}', "
                f"but instantiated simulation expects '{normalized_expected_version}'."
            )


def validate_setup_payload_core(setup: Mapping[str, Any]) -> None:
    """Validate core ``/setup`` payload constraints.

    Args:
        setup: Candidate setup mapping.

    Raises:
        ValueError: If required keys are missing or setup values/shapes are invalid.
    """
    require_keys(setup, schema.REQUIRED_SETUP_KEYS, label="setup")

    ee_apertures_mas = as_float_vector(
        setup[schema.KEY_SETUP_EE_APERTURES_MAS],
        label=f"setup['{schema.KEY_SETUP_EE_APERTURES_MAS}']",
    )
    if ee_apertures_mas.size == 0 or not np.all(np.isfinite(ee_apertures_mas)):
        raise ValueError("setup['ee_apertures_mas'] must be a non-empty 1D finite array.")

    atm_wavelength_um = as_float_scalar(
        setup[schema.KEY_SETUP_ATM_WAVELENGTH_UM],
        label=f"setup['{schema.KEY_SETUP_ATM_WAVELENGTH_UM}']",
    )
    if not np.isfinite(atm_wavelength_um) or atm_wavelength_um <= 0.0:
        raise ValueError(f"setup['{schema.KEY_SETUP_ATM_WAVELENGTH_UM}'] must be a positive finite scalar.")

    sci_r_arcsec = as_float_vector(
        setup[schema.KEY_SETUP_SCI_R_ARCSEC],
        label=f"setup['{schema.KEY_SETUP_SCI_R_ARCSEC}']",
    )
    sci_theta_deg = as_float_vector(
        setup[schema.KEY_SETUP_SCI_THETA_DEG],
        label=f"setup['{schema.KEY_SETUP_SCI_THETA_DEG}']",
    )
    if sci_r_arcsec.size == 0 or sci_theta_deg.size == 0:
        raise ValueError(
            f"setup['{schema.KEY_SETUP_SCI_R_ARCSEC}'] and setup['{schema.KEY_SETUP_SCI_THETA_DEG}'] must be non-empty 1D arrays."
        )
    if sci_r_arcsec.shape != sci_theta_deg.shape:
        raise ValueError(
            f"setup['{schema.KEY_SETUP_SCI_R_ARCSEC}'] and setup['{schema.KEY_SETUP_SCI_THETA_DEG}'] must have identical shape."
        )
    if not np.all(np.isfinite(sci_r_arcsec)) or not np.all(np.isfinite(sci_theta_deg)):
        raise ValueError("setup science coordinates must be finite.")


#
# Shared validation primitives
#


def validate_ngs_options(options: Mapping[str, Any], expected_num_sims: int | None = None) -> None:
    """Validate the shared NGS option triplet when present.

    The shared NGS options are the coupled triplet
    ``ngs_r_arcsec``/``ngs_theta_deg``/``ngs_mag``. Persisted ``/options``
    requires the triplet, but this helper is also reused in earlier options
    preparation flows where the triplet may not yet be materialized. When
    present, all three keys must exist with identical ``[N, Kmax]`` shape,
    where each star slot is either all finite or all ``NaN`` across the
    triplet. This allows ragged per-simulation star counts after rectangular
    normalization.

    Args:
        options: Candidate options mapping.
        expected_num_sims: Optional expected number of simulations ``N``.

    Raises:
        ValueError: If the NGS triplet is partially present or violates the
            shared shape/slot contract.
    """
    ngs_keys_present = [key for key in schema.OPTION_KEYS_NGS if key in options]
    if not ngs_keys_present:
        return

    missing_ngs_keys = [key for key in schema.OPTION_KEYS_NGS if key not in options]
    if missing_ngs_keys:
        raise ValueError("Explicit NGS options must provide ngs_r_arcsec, ngs_theta_deg, and ngs_mag together.")

    ngs_r = as_float_matrix(options[schema.KEY_OPTION_NGS_R_ARCSEC], label=schema.KEY_OPTION_NGS_R_ARCSEC)
    ngs_theta = as_float_matrix(options[schema.KEY_OPTION_NGS_THETA_DEG], label=schema.KEY_OPTION_NGS_THETA_DEG)
    ngs_mag = as_float_matrix(options[schema.KEY_OPTION_NGS_MAG], label=schema.KEY_OPTION_NGS_MAG)
    if ngs_r.ndim != 2 or ngs_theta.ndim != 2 or ngs_mag.ndim != 2:
        raise ValueError("NGS option arrays must be 2D [N, Kmax].")
    if ngs_r.shape != ngs_theta.shape or ngs_r.shape != ngs_mag.shape:
        raise ValueError("NGS option arrays must have identical shape [N, Kmax].")
    if expected_num_sims is not None and int(ngs_r.shape[0]) != int(expected_num_sims):
        raise ValueError("NGS option arrays first dimension must match N.")
    if int(ngs_r.shape[1]) == 0:
        raise ValueError("NGS option arrays must have Kmax >= 1.")

    finite_r = np.isfinite(ngs_r)
    finite_theta = np.isfinite(ngs_theta)
    finite_mag = np.isfinite(ngs_mag)
    any_finite = finite_r | finite_theta | finite_mag
    all_finite = finite_r & finite_theta & finite_mag
    partial = any_finite & ~all_finite
    if np.any(partial):
        count = int(np.count_nonzero(partial))
        raise ValueError(
            f"Invalid NGS options: found {count} entries with partial NaN values. "
            "Each star slot must be either all finite or all NaN across ngs_r_arcsec/ngs_theta_deg/ngs_mag."
        )


def validate_options_payload_core(options: Mapping[str, Any], expected_num_sims: int | None = None) -> int:
    """Validate core ``/options`` payload and return simulation count ``N``.

    Args:
        options: Candidate options mapping.
        expected_num_sims: Optional expected first-dimension size.

    Returns:
        Number of simulations ``N`` inferred from options arrays.

    Raises:
        ValueError: If option keys, dtypes, or shapes violate the core contract.
    """
    allowed_keys = set(schema.OPTION_KEYS_1D + schema.OPTION_KEYS_NGS)
    unknown_keys = sorted(set(options.keys()) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unsupported options keys: {', '.join(unknown_keys)}")
    missing_keys = [key for key in schema.REQUIRED_OPTION_KEYS if key not in options]
    if missing_keys:
        raise ValueError(f"Missing required options keys: {', '.join(missing_keys)}")

    num_sims: int | None = None
    for key, value in options.items():
        arr = as_array(value)
        if arr.ndim == 0:
            raise ValueError(f"Option '{key}' must be per-simulation and include first dimension N.")
        n_key = int(arr.shape[0])
        if num_sims is None:
            num_sims = n_key
        elif n_key != num_sims:
            raise ValueError(f"Option '{key}' first dimension must match N={num_sims}, got {arr.shape}.")
    assert num_sims is not None
    if num_sims == 0:
        raise ValueError("options must contain at least one simulation.")
    if expected_num_sims is not None and num_sims != int(expected_num_sims):
        raise ValueError(f"options N={num_sims} does not match expected N={int(expected_num_sims)}.")

    for key in schema.OPTION_KEYS_1D:
        arr = np.asarray(options[key])
        if arr.ndim != 1:
            raise ValueError(f"options['{key}'] must be 1D [N].")
        if int(arr.shape[0]) != num_sims:
            raise ValueError(f"options['{key}'] length must match N={num_sims}.")
        if key != schema.KEY_OPTION_ATM_PROFILE_ID:
            if not np.all(np.isfinite(np.asarray(arr, dtype=float))):
                raise ValueError(f"options['{key}'] must be finite.")
        else:
            if not np.issubdtype(arr.dtype, np.integer):
                arrf = np.asarray(arr, dtype=float)
                if not np.all(np.isfinite(arrf)):
                    raise ValueError(f"options['{schema.KEY_OPTION_ATM_PROFILE_ID}'] must be finite.")
                if not np.all(np.equal(arrf, np.round(arrf))):
                    raise ValueError(f"options['{schema.KEY_OPTION_ATM_PROFILE_ID}'] must be integer-valued.")

    validate_ngs_options(options, num_sims)

    return num_sims


def validate_atm_profile_ids(setup: Mapping[str, Any], options: Mapping[str, Any]) -> None:
    """Validate that ``atm_profile_id`` values exist in ``setup['atm_profiles']``.

    Args:
        setup: Candidate setup payload containing ``atm_profiles``.
        options: Candidate options payload containing ``atm_profile_id``.

    Raises:
        ValueError: If profile ids are missing from ``setup['atm_profiles']``.
    """
    if schema.KEY_OPTION_ATM_PROFILE_ID not in options:
        return

    atm_profiles = setup.get(schema.KEY_SETUP_ATM_PROFILES)
    if not isinstance(atm_profiles, Mapping):
        raise ValueError(
            f"setup['{schema.KEY_SETUP_ATM_PROFILES}'] must be a mapping when options['{schema.KEY_OPTION_ATM_PROFILE_ID}'] is provided."
        )
    if len(atm_profiles) == 0:
        return
    allowed_ids = {int(k) for k in atm_profiles.keys()}
    profile_ids = np.asarray(options[schema.KEY_OPTION_ATM_PROFILE_ID], dtype=np.int64).reshape(-1)
    missing = sorted({int(v) for v in profile_ids.tolist()} - allowed_ids)
    if missing:
        raise ValueError(
            f"options['{schema.KEY_OPTION_ATM_PROFILE_ID}'] references unknown profile ids: {missing}. "
            f"Available ids: {sorted(allowed_ids)}."
        )

def validate_psf_cube(psfs: np.ndarray, num_sci: int, label: str = "PSFs") -> np.ndarray:
    """Validate and return a PSF cube with shape ``[M, Ny, Nx]`` and finite values."""
    cube = np.asarray(psfs, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"{label} must be [M, Ny, Nx], got shape {cube.shape}")
    if cube.shape[0] != int(num_sci):
        raise ValueError(f"{label} science dimension mismatch: expected {int(num_sci)}, got {cube.shape[0]}")
    if not np.all(np.isfinite(cube)):
        raise ValueError(f"{label} contain non-finite values.")
    return cube


#
# Result validators
#


def validate_successful_result(
    result: SimulationResult,
    num_sci: int,
    num_ee: int,
    *,
    require_psfs: bool = False,
) -> None:
    """Validate one successful simulation result against core persistence rules.

    Args:
        result: Successful per-simulation result payload.
        num_sci: Required science-target count ``M``.
        num_ee: Required EE-aperture count ``A``.
        require_psfs: Whether a valid PSF cube must be present.

    Raises:
        ValueError: If the result state, stats, metadata, or PSF cube violates
            the core successful-result contract.
    """
    if int(result.state) != int(SimulationState.SUCCEEDED):
        raise ValueError(
            "Successful result validation requires result.state == "
            f"{SimulationState.SUCCEEDED.name}, got {int(result.state)}."
        )

    missing_stats_keys = [key for key in schema.REQUIRED_STATS_KEYS if key not in result.stats]
    if missing_stats_keys:
        raise ValueError(
            "result.stats must include sr, ee, and fwhm_mas for successful results."
        )

    for key in (
        schema.KEY_STATS_SR,
        schema.KEY_STATS_FWHM_MAS,
    ):
        value = np.asarray(result.stats[key], dtype=np.float32)
        if value.shape != (int(num_sci),):
            raise ValueError(f"result.{key} must have shape ({int(num_sci)},), got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"result.{key} must contain only finite values.")

    ee = np.asarray(result.stats[schema.KEY_STATS_EE], dtype=np.float32)

    if ee.ndim == 1:
        if int(num_ee) != 1 or ee.shape != (int(num_sci),):
            raise ValueError(f"result.ee shape incompatible with EE dimension A={int(num_ee)}.")
    elif ee.ndim == 2:
        if ee.shape != (int(num_sci), int(num_ee)):
            raise ValueError(f"result.ee must have shape ({int(num_sci)}, {int(num_ee)}), got {ee.shape}")
    else:
        raise ValueError("result.ee must be 1D or 2D.")

    if not np.all(np.isfinite(ee)):
        raise ValueError(f"result.{schema.KEY_STATS_EE} must contain only finite values.")

    missing_meta_keys = [key for key in schema.REQUIRED_META_KEYS if key not in result.meta]
    if missing_meta_keys:
        raise ValueError(
            "result.meta must include pixel_scale_mas, tel_diameter_m, and tel_pupil for successful results."
        )

    pixel_scale = np.asarray(result.meta[schema.KEY_META_PIXEL_SCALE_MAS], dtype=np.float32)
    tel_diameter = np.asarray(result.meta[schema.KEY_META_TEL_DIAMETER_M], dtype=np.float32)
    tel_pupil = np.asarray(result.meta[schema.KEY_META_TEL_PUPIL], dtype=np.float32)
    if pixel_scale.ndim != 0:
        raise ValueError("result.meta.pixel_scale_mas must be a scalar.")
    if tel_diameter.ndim != 0:
        raise ValueError("result.meta.tel_diameter_m must be a scalar.")
    if tel_pupil.ndim != 2:
        raise ValueError("result.meta.tel_pupil must be 2D [Ny, Nx].")
    if not np.all(np.isfinite(pixel_scale.reshape(1))):
        raise ValueError("result.meta.pixel_scale_mas must contain only finite values.")
    if not np.all(np.isfinite(tel_diameter.reshape(1))):
        raise ValueError("result.meta.tel_diameter_m must contain only finite values.")
    if not np.all(np.isfinite(tel_pupil)):
        raise ValueError("result.meta.tel_pupil must contain only finite values.")

    if result.psfs is None:
        if require_psfs:
            raise ValueError("result.psfs must be provided when PSFs are required.")
        return

    validate_psf_cube(result.psfs, int(num_sci), label="result.psfs")
