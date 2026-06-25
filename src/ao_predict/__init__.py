"""ao-predict package."""

# Public re-exports

from .analysis import (
    AnalysisDataset,
    AnalysisDatasetLoadPayload,
    AnalysisLoadContext,
    AnalysisLoadContribution,
    AnalysisSimulation,
    AnalysisSimulationLoadPayload,
    load_analysis_dataset,
)
from .simulation import (
    BaseSimulation,
    Simulation,
    SimulationContext,
    SimulationResult,
    SimulationState,
    TiptopBaseConfig,
    TiptopSimulation,
    PsfMetadata,
    clip_and_sum_normalize_psfs,
    compute_psf_ee,
    compute_psf_fwhm,
    compute_psf_sr,
    compute_psf_stats,
)
from .simulation.api import (
    DatasetConfigMismatchError,
    DatasetValidationError,
    DatasetStatus,
    InitDatasetRequest,
    OptionsConfig,
    TableOptionsConfig,
    SetupConfig,
    SimulationConfig,
    check_dataset,
    init_dataset,
    reset_simulations,
    resume_simulations,
    run_simulations_by_state,
    validate_dataset,
    validate_dataset_matches_request,
)
from .simulation.runner import RunSummary

# Package export surface

__all__ = [
    "__version__",
    "check_dataset",
    "PsfMetadata",
    "clip_and_sum_normalize_psfs",
    "compute_psf_ee",
    "compute_psf_fwhm",
    "compute_psf_sr",
    "compute_psf_stats",
    "DatasetStatus",
    "DatasetConfigMismatchError",
    "DatasetValidationError",
    "InitDatasetRequest",
    "init_dataset",
    "OptionsConfig",
    "TableOptionsConfig",
    "reset_simulations",
    "resume_simulations",
    "RunSummary",
    "run_simulations_by_state",
    "validate_dataset",
    "validate_dataset_matches_request",
    "SetupConfig",
    "BaseSimulation",
    "Simulation",
    "SimulationConfig",
    "SimulationContext",
    "SimulationResult",
    "SimulationState",
    "TiptopBaseConfig",
    "TiptopSimulation",
    "AnalysisDataset",
    "AnalysisDatasetLoadPayload",
    "AnalysisLoadContext",
    "AnalysisLoadContribution",
    "AnalysisSimulation",
    "AnalysisSimulationLoadPayload",
    "load_analysis_dataset",
]

# Package metadata

__version__ = "0.0.1"
