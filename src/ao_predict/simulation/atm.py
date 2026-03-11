"""Atmospheric profile parsing and validation helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


KEY_SETUP_ATM_PROFILE_NAME = "name"
KEY_SETUP_ATM_PROFILE_R0_M = "r0_m"
KEY_SETUP_ATM_PROFILE_L0_M = "L0_m"
KEY_SETUP_ATM_PROFILE_CN2_HEIGHTS_M = "cn2_heights_m"
KEY_SETUP_ATM_PROFILE_CN2_WEIGHTS = "cn2_weights"
KEY_SETUP_ATM_PROFILE_WIND_SPEED_MPS = "wind_speed_mps"
KEY_SETUP_ATM_PROFILE_WIND_DIRECTION_DEG = "wind_direction_deg"
KEY_SETUP_ATM_PROFILE_SEEING_ARCSEC = "seeing_arcsec"

REQUIRED_ATM_PROFILE_KEYS = (
    KEY_SETUP_ATM_PROFILE_NAME,
    KEY_SETUP_ATM_PROFILE_R0_M,
    KEY_SETUP_ATM_PROFILE_L0_M,
    KEY_SETUP_ATM_PROFILE_CN2_HEIGHTS_M,
    KEY_SETUP_ATM_PROFILE_CN2_WEIGHTS,
    KEY_SETUP_ATM_PROFILE_WIND_SPEED_MPS,
    KEY_SETUP_ATM_PROFILE_WIND_DIRECTION_DEG,
)


def parse_atm_profiles(atm_profiles: Any) -> dict[int, dict[str, Any]]:
    """Parse atmospheric profile mappings into normalized numeric forms."""
    if not isinstance(atm_profiles, Mapping):
        return {}

    parsed: dict[int, dict[str, Any]] = {}
    for profile_id_raw, profile_raw in atm_profiles.items():
        profile_id = int(profile_id_raw)
        if not isinstance(profile_raw, Mapping):
            raise ValueError(f"Atmospheric profile '{profile_id}' must be a mapping.")

        profile: dict[str, Any] = {}
        for key_raw, value in profile_raw.items():
            key = str(key_raw)
            if key.lower() == KEY_SETUP_ATM_PROFILE_L0_M.lower():
                key = KEY_SETUP_ATM_PROFILE_L0_M
            if key == KEY_SETUP_ATM_PROFILE_NAME:
                profile[key] = str(value)
                continue
            value = np.asarray(value)
            if value.ndim == 0:
                profile[key] = float(value)
            else:
                profile[key] = np.asarray(value, dtype=float).reshape(-1)
        parsed[profile_id] = profile
    return parsed


def normalize_atm_profiles_with_seeing_alias(
    atm_profiles: Mapping[int, Mapping[str, Any]],
    atm_wavelength_um: float | None,
) -> dict[int, dict[str, Any]]:
    """Normalize ``seeing_arcsec`` aliases into canonical ``r0_m`` values."""
    normalized: dict[int, dict[str, Any]] = {}
    for profile_id_raw, profile_raw in atm_profiles.items():
        profile_id = int(profile_id_raw)
        profile = dict(profile_raw)
        has_r0 = (
            KEY_SETUP_ATM_PROFILE_R0_M in profile
            and np.asarray(profile[KEY_SETUP_ATM_PROFILE_R0_M]).ndim == 0
            and np.isfinite(float(profile[KEY_SETUP_ATM_PROFILE_R0_M]))
        )
        has_seeing = (
            KEY_SETUP_ATM_PROFILE_SEEING_ARCSEC in profile
            and np.asarray(profile[KEY_SETUP_ATM_PROFILE_SEEING_ARCSEC]).ndim == 0
            and np.isfinite(float(profile[KEY_SETUP_ATM_PROFILE_SEEING_ARCSEC]))
        )
        if has_seeing:
            seeing_arcsec = float(profile[KEY_SETUP_ATM_PROFILE_SEEING_ARCSEC])
            if seeing_arcsec <= 0.0:
                raise ValueError(f"atm_profiles[{profile_id}]['{KEY_SETUP_ATM_PROFILE_SEEING_ARCSEC}'] must be > 0.")
            if atm_wavelength_um is None or not np.isfinite(float(atm_wavelength_um)) or float(atm_wavelength_um) <= 0.0:
                raise ValueError(
                    f"atm_wavelength_um must be finite and > 0 when using "
                    f"atm_profiles[*]['{KEY_SETUP_ATM_PROFILE_SEEING_ARCSEC}']."
                )
            seeing_rad = seeing_arcsec * (math.pi / 648000.0)
            r0_from_seeing = 0.98 * (float(atm_wavelength_um) * 1e-6) / seeing_rad
            if has_r0:
                r0_value = float(profile[KEY_SETUP_ATM_PROFILE_R0_M])
                if not np.isclose(r0_value, r0_from_seeing, rtol=1e-3, atol=1e-6):
                    raise ValueError(
                        f"Inconsistent atmospheric profile {profile_id}: both '{KEY_SETUP_ATM_PROFILE_R0_M}' "
                        f"and '{KEY_SETUP_ATM_PROFILE_SEEING_ARCSEC}' are provided "
                        "but do not match."
                    )
            else:
                profile[KEY_SETUP_ATM_PROFILE_R0_M] = float(r0_from_seeing)
        profile.pop(KEY_SETUP_ATM_PROFILE_SEEING_ARCSEC, None)
        normalized[profile_id] = profile
    return normalized


def validate_standard_atm_profiles(atm_profiles: Mapping[int, Mapping[str, Any]]) -> None:
    """Validate the shared atmospheric profile structure and numeric content."""
    if not atm_profiles:
        raise ValueError("atm_profiles must be non-empty.")
    if 0 not in {int(k) for k in atm_profiles.keys()}:
        raise ValueError("atm_profiles must include profile id 0.")

    for profile_id_raw, profile in atm_profiles.items():
        profile_id = int(profile_id_raw)
        if not isinstance(profile, Mapping):
            raise ValueError(f"atm_profiles[{profile_id}] must be a mapping.")
        missing = [k for k in REQUIRED_ATM_PROFILE_KEYS if k not in profile]
        if missing:
            raise ValueError(f"atm_profiles[{profile_id}] missing required keys: {', '.join(missing)}.")

        name = str(profile[KEY_SETUP_ATM_PROFILE_NAME]).strip()
        if not name:
            raise ValueError(f"atm_profiles[{profile_id}]['{KEY_SETUP_ATM_PROFILE_NAME}'] must be non-empty.")

        r0 = np.asarray(profile[KEY_SETUP_ATM_PROFILE_R0_M], dtype=float)
        l0 = np.asarray(profile[KEY_SETUP_ATM_PROFILE_L0_M], dtype=float)
        if r0.ndim != 0 or not np.isfinite(float(r0)):
            raise ValueError(f"atm_profiles[{profile_id}]['{KEY_SETUP_ATM_PROFILE_R0_M}'] must be a finite scalar.")
        if float(r0) <= 0.0:
            raise ValueError(f"atm_profiles[{profile_id}]['{KEY_SETUP_ATM_PROFILE_R0_M}'] must be > 0.")
        if l0.ndim != 0 or not np.isfinite(float(l0)):
            raise ValueError(f"atm_profiles[{profile_id}]['{KEY_SETUP_ATM_PROFILE_L0_M}'] must be a finite scalar.")
        if float(l0) <= 0.0:
            raise ValueError(f"atm_profiles[{profile_id}]['{KEY_SETUP_ATM_PROFILE_L0_M}'] must be > 0.")

        cn2_heights = np.asarray(profile[KEY_SETUP_ATM_PROFILE_CN2_HEIGHTS_M], dtype=float).reshape(-1)
        cn2_weights = np.asarray(profile[KEY_SETUP_ATM_PROFILE_CN2_WEIGHTS], dtype=float).reshape(-1)
        wind_speed = np.asarray(profile[KEY_SETUP_ATM_PROFILE_WIND_SPEED_MPS], dtype=float).reshape(-1)
        wind_dir = np.asarray(profile[KEY_SETUP_ATM_PROFILE_WIND_DIRECTION_DEG], dtype=float).reshape(-1)
        lengths = {cn2_heights.size, cn2_weights.size, wind_speed.size, wind_dir.size}
        if 0 in lengths or len(lengths) != 1:
            raise ValueError(f"atm_profiles[{profile_id}] layer vectors must be non-empty and have equal length.")
        if (
            not np.all(np.isfinite(cn2_heights))
            or not np.all(np.isfinite(cn2_weights))
            or not np.all(np.isfinite(wind_speed))
            or not np.all(np.isfinite(wind_dir))
        ):
            raise ValueError(f"atm_profiles[{profile_id}] layer vectors must be finite.")


def select_atm_profile(
    atm_profiles: Mapping[int, Mapping[str, Any]],
    profile_id: int,
) -> Mapping[str, Any]:
    """Return one atmospheric profile by id."""
    if not atm_profiles:
        raise ValueError("atm_profiles is empty.")
    if int(profile_id) not in atm_profiles:
        available = ", ".join(str(k) for k in sorted(atm_profiles))
        raise ValueError(f"atm_profile_id={int(profile_id)} not found. Available profiles: {available}")
    return atm_profiles[int(profile_id)]
