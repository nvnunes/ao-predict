"""Shared schema/key constants for simulation config and payloads."""

from __future__ import annotations

import re

# Top-level sections

KEY_SIMULATION_SECTION = "simulation"
KEY_SETUP_SECTION = "setup"
KEY_OPTION_SECTION = "options"
KEY_STATUS_SECTION = "status"
KEY_META_SECTION = "meta"
KEY_STATS_SECTION = "stats"
KEY_PSFS_SECTION = "psfs"

# Simulation config grammar keys (YAML/API-facing)

KEY_CFG_SIMULATION_BASE_PATH = "base_path"
KEY_CFG_SIMULATION_SPECIFIC_FIELDS = "specific_fields"

# Simulation payload keys (persisted `/simulation`)

KEY_SIMULATION_NAME = "name"
KEY_SIMULATION_VERSION = "version"

# Setup config grammar keys (YAML/API-facing)

KEY_CFG_SETUP_SPECIFIC_FIELDS = "specific_fields"

# Setup payload keys

KEY_SETUP_EE_APERTURES_MAS = "ee_apertures_mas"
KEY_SETUP_ATM_WAVELENGTH_UM = "atm_wavelength_um"
KEY_SETUP_ATM_PROFILES = "atm_profiles"
KEY_SETUP_SCI_R_ARCSEC = "sci_r_arcsec"
KEY_SETUP_SCI_THETA_DEG = "sci_theta_deg"

# Options config grammar keys (YAML/CSV-facing)

KEY_CFG_OPTION_BROADCAST = "broadcast"
KEY_CFG_OPTION_TABLE = "table"
KEY_CFG_OPTION_TABLE_PATH = "table_path"
KEY_CFG_OPTION_COLUMNS = "columns"
KEY_CFG_OPTION_ROWS = "rows"
KEY_CFG_OPTION_NGS = "ngs"

# Options payload keys (persisted `/options`)

KEY_OPTION_WAVELENGTH_UM = "wavelength_um"
KEY_OPTION_ZENITH_ANGLE_DEG = "zenith_angle_deg"
KEY_OPTION_ATM_PROFILE_ID = "atm_profile_id"
KEY_OPTION_R0_M = "r0_m"
KEY_OPTION_SEEING = "seeing_arcsec"
KEY_OPTION_NGS_R_ARCSEC = "ngs_r_arcsec"
KEY_OPTION_NGS_THETA_DEG = "ngs_theta_deg"
KEY_OPTION_NGS_MAG = "ngs_mag"

# Runtime-derived option keys (not persisted in `/options`)

KEY_OPTION_NGS_USED = "ngs_used"

# NGS column parsing helpers

KEY_OPTION_NGS_PREFIX = "ngs_"
KEY_OPTION_R_ARCSEC_SUFFIX = "r_arcsec"
KEY_OPTION_THETA_DEG_SUFFIX = "theta_deg"
KEY_OPTION_MAG_SUFFIX = "mag"
KEY_OPTION_NGS_COLUMN_RE = re.compile(
    rf"^ngs(\d+)_({KEY_OPTION_R_ARCSEC_SUFFIX}|{KEY_OPTION_THETA_DEG_SUFFIX}|{KEY_OPTION_MAG_SUFFIX})$"
)

# Status/meta/stats/psfs payload keys (persisted HDF5 datasets)

KEY_STATUS_STATE = "state"

KEY_META_PIXEL_SCALE_MAS = "pixel_scale_mas"
KEY_META_TEL_DIAMETER_M = "tel_diameter_m"
KEY_META_TEL_PUPIL = "tel_pupil"

KEY_STATS_SR = "sr"
KEY_STATS_EE = "ee"
KEY_STATS_FWHM_MAS = "fwhm_mas"
KEY_STATS_JITTER_MAS = "jitter_mas"

KEY_PSFS_DATA = "data"

# Key collections

SIMULATION_KEYS_CORE = (
    KEY_SIMULATION_NAME,
    KEY_SIMULATION_VERSION,
)

SETUP_KEYS_CORE = (KEY_SETUP_EE_APERTURES_MAS,)

OPTION_KEYS_1D = (
    KEY_OPTION_WAVELENGTH_UM,
    KEY_OPTION_ZENITH_ANGLE_DEG,
    KEY_OPTION_ATM_PROFILE_ID,
    KEY_OPTION_R0_M,
)

OPTION_KEYS_NGS = (
    KEY_OPTION_NGS_R_ARCSEC,
    KEY_OPTION_NGS_THETA_DEG,
    KEY_OPTION_NGS_MAG,
)

# Contract key collections

REQUIRED_SIMULATION_KEYS = SIMULATION_KEYS_CORE
REQUIRED_SETUP_KEYS = (
    KEY_SETUP_EE_APERTURES_MAS,
    KEY_SETUP_ATM_WAVELENGTH_UM,
    KEY_SETUP_SCI_R_ARCSEC,
    KEY_SETUP_SCI_THETA_DEG,
)
REQUIRED_OPTION_KEYS = OPTION_KEYS_1D + OPTION_KEYS_NGS

REQUIRED_GROUP_KEYS = (
    KEY_SIMULATION_SECTION,
    KEY_SETUP_SECTION,
    KEY_OPTION_SECTION,
    KEY_STATUS_SECTION,
    KEY_META_SECTION,
    KEY_STATS_SECTION,
)

REQUIRED_STATUS_KEYS = (
    KEY_STATUS_STATE,
)

REQUIRED_META_KEYS = (
    KEY_META_PIXEL_SCALE_MAS,
    KEY_META_TEL_DIAMETER_M,
    KEY_META_TEL_PUPIL,
)

REQUIRED_STATS_KEYS = (
    KEY_STATS_SR,
    KEY_STATS_EE,
    KEY_STATS_FWHM_MAS,
    KEY_STATS_JITTER_MAS,
)
