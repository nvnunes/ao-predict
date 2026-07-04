"""Simulation runtime interfaces and execution helpers."""

# Public re-exports

from .base import BaseSimulation, BaseSimulationSetup
from .config_backed import ConfigBackedSimulation
from .interfaces import Simulation, SimulationContext, SimulationResult, SimulationSetup, SimulationState
from .stats import PsfMetadata, clip_and_sum_normalize_psfs, compute_psf_ee, compute_psf_fwhm, compute_psf_sr, compute_psf_stats
from .tiptop import TiptopSimulation
from .tiptop_config_backed import TiptopBaseConfig, TiptopConfigBackedSimulation

__all__ = [
    "BaseSimulation",
    "BaseSimulationSetup",
    "ConfigBackedSimulation",
    "PsfMetadata",
    "clip_and_sum_normalize_psfs",
    "compute_psf_ee",
    "compute_psf_fwhm",
    "compute_psf_sr",
    "compute_psf_stats",
    "Simulation",
    "SimulationContext",
    "SimulationSetup",
    "SimulationResult",
    "SimulationState",
    "TiptopBaseConfig",
    "TiptopConfigBackedSimulation",
    "TiptopSimulation",
]
