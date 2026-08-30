"""Atmospheric profile parsing and validation helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from astropy import units as u

from .._units import quantity_value


KEY_SETUP_ATM_PROFILE_NAME = "name"
KEY_SETUP_ATM_PROFILE_R0 = "r0"
KEY_SETUP_ATM_PROFILE_L0 = "L0"
KEY_SETUP_ATM_PROFILE_CN2_HEIGHTS = "cn2_heights"
KEY_SETUP_ATM_PROFILE_CN2_WEIGHTS = "cn2_weights"
KEY_SETUP_ATM_PROFILE_WIND_SPEED = "wind_speed"
KEY_SETUP_ATM_PROFILE_WIND_DIRECTION = "wind_direction"
KEY_SETUP_ATM_PROFILE_SEEING = "seeing"

REQUIRED_ATM_PROFILE_KEYS = (
    KEY_SETUP_ATM_PROFILE_NAME,
    KEY_SETUP_ATM_PROFILE_R0,
    KEY_SETUP_ATM_PROFILE_L0,
    KEY_SETUP_ATM_PROFILE_CN2_HEIGHTS,
    KEY_SETUP_ATM_PROFILE_CN2_WEIGHTS,
    KEY_SETUP_ATM_PROFILE_WIND_SPEED,
    KEY_SETUP_ATM_PROFILE_WIND_DIRECTION,
)

ATM_PROFILE_FIELD_UNITS: dict[str, u.UnitBase] = {
    KEY_SETUP_ATM_PROFILE_R0: u.m,
    KEY_SETUP_ATM_PROFILE_L0: u.m,
    KEY_SETUP_ATM_PROFILE_CN2_HEIGHTS: u.m,
    KEY_SETUP_ATM_PROFILE_CN2_WEIGHTS: u.dimensionless_unscaled,
    KEY_SETUP_ATM_PROFILE_WIND_SPEED: u.m / u.s,
    KEY_SETUP_ATM_PROFILE_WIND_DIRECTION: u.deg,
    KEY_SETUP_ATM_PROFILE_SEEING: u.arcsec,
}


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
            if key.lower() == KEY_SETUP_ATM_PROFILE_L0.lower():
                key = KEY_SETUP_ATM_PROFILE_L0
            if key == KEY_SETUP_ATM_PROFILE_NAME:
                profile[key] = str(value)
                continue
            if key not in ATM_PROFILE_FIELD_UNITS:
                raise ValueError(f"Atmospheric profile '{profile_id}' has unsupported field '{key}'.")
            values = quantity_value(
                value,
                ATM_PROFILE_FIELD_UNITS[key],
                label=f"atm_profiles[{profile_id}]['{key}']",
                dtype=float,
            )
            profile[key] = values.item() * ATM_PROFILE_FIELD_UNITS[key] if values.ndim == 0 else values.reshape(-1) * ATM_PROFILE_FIELD_UNITS[key]
        parsed[profile_id] = profile
    return parsed


def normalize_atm_profiles_with_seeing_alias(
    atm_profiles: Mapping[int, Mapping[str, Any]],
    atm_wavelength: u.Quantity | None,
) -> dict[int, dict[str, Any]]:
    """Normalize ``seeing`` aliases into canonical ``r0`` values."""
    normalized: dict[int, dict[str, Any]] = {}
    for profile_id_raw, profile_raw in atm_profiles.items():
        profile_id = int(profile_id_raw)
        profile = dict(profile_raw)
        has_r0 = (
            KEY_SETUP_ATM_PROFILE_R0 in profile
            and np.asarray(profile[KEY_SETUP_ATM_PROFILE_R0].value).ndim == 0
            and np.isfinite(float(profile[KEY_SETUP_ATM_PROFILE_R0].to_value(u.m)))
        )
        has_seeing = (
            KEY_SETUP_ATM_PROFILE_SEEING in profile
            and np.asarray(profile[KEY_SETUP_ATM_PROFILE_SEEING].value).ndim == 0
            and np.isfinite(float(profile[KEY_SETUP_ATM_PROFILE_SEEING].to_value(u.arcsec)))
        )
        if has_seeing:
            seeing = float(profile[KEY_SETUP_ATM_PROFILE_SEEING].to_value(u.arcsec))
            if seeing <= 0.0:
                raise ValueError(f"atm_profiles[{profile_id}]['{KEY_SETUP_ATM_PROFILE_SEEING}'] must be > 0.")
            if atm_wavelength is None or not np.isfinite(float(atm_wavelength.to_value(u.um))) or float(atm_wavelength.to_value(u.um)) <= 0.0:
                raise ValueError(
                    f"atm_wavelength must be finite and > 0 when using "
                    f"atm_profiles[*]['{KEY_SETUP_ATM_PROFILE_SEEING}']."
                )
            seeing_rad = seeing * (math.pi / 648000.0)
            r0_from_seeing = 0.98 * atm_wavelength.to_value(u.m) / seeing_rad
            if has_r0:
                r0_value = float(profile[KEY_SETUP_ATM_PROFILE_R0].to_value(u.m))
                if not np.isclose(r0_value, r0_from_seeing, rtol=1e-3, atol=1e-6):
                    raise ValueError(
                        f"Inconsistent atmospheric profile {profile_id}: both '{KEY_SETUP_ATM_PROFILE_R0}' "
                        f"and '{KEY_SETUP_ATM_PROFILE_SEEING}' are provided "
                        "but do not match."
                    )
            else:
                profile[KEY_SETUP_ATM_PROFILE_R0] = float(r0_from_seeing) * u.m
        profile.pop(KEY_SETUP_ATM_PROFILE_SEEING, None)
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

        r0 = quantity_value(profile[KEY_SETUP_ATM_PROFILE_R0], u.m, label=f"atm_profiles[{profile_id}]['r0']", dtype=float)
        l0 = quantity_value(profile[KEY_SETUP_ATM_PROFILE_L0], u.m, label=f"atm_profiles[{profile_id}]['L0']", dtype=float)
        if r0.ndim != 0 or not np.isfinite(float(r0)):
            raise ValueError(f"atm_profiles[{profile_id}]['{KEY_SETUP_ATM_PROFILE_R0}'] must be a finite scalar.")
        if float(r0) <= 0.0:
            raise ValueError(f"atm_profiles[{profile_id}]['{KEY_SETUP_ATM_PROFILE_R0}'] must be > 0.")
        if l0.ndim != 0 or not np.isfinite(float(l0)):
            raise ValueError(f"atm_profiles[{profile_id}]['{KEY_SETUP_ATM_PROFILE_L0}'] must be a finite scalar.")
        if float(l0) <= 0.0:
            raise ValueError(f"atm_profiles[{profile_id}]['{KEY_SETUP_ATM_PROFILE_L0}'] must be > 0.")

        cn2_heights = quantity_value(profile[KEY_SETUP_ATM_PROFILE_CN2_HEIGHTS], u.m, label=f"atm_profiles[{profile_id}]['cn2_heights']", dtype=float).reshape(-1)
        cn2_weights = quantity_value(profile[KEY_SETUP_ATM_PROFILE_CN2_WEIGHTS], u.dimensionless_unscaled, label=f"atm_profiles[{profile_id}]['cn2_weights']", dtype=float).reshape(-1)
        wind_speed = quantity_value(profile[KEY_SETUP_ATM_PROFILE_WIND_SPEED], u.m / u.s, label=f"atm_profiles[{profile_id}]['wind_speed']", dtype=float).reshape(-1)
        wind_dir = quantity_value(profile[KEY_SETUP_ATM_PROFILE_WIND_DIRECTION], u.deg, label=f"atm_profiles[{profile_id}]['wind_direction']", dtype=float).reshape(-1)
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
