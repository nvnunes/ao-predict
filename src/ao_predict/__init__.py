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
from .prediction import ModelEvaluationResult, ModelPredictor, load_model_predictor
from .simulation import (
    BaseSimulation,
    HybridSimulation,
    Simulation,
    SimulationContext,
    SimulationResult,
    SimulationState,
    TiptopBaseConfig,
    TiptopSimulation,
)
from .simulation.api import (
    DatasetConfigMismatchError,
    DatasetStatus,
    DatasetValidationError,
    InitDatasetRequest,
    OptionsConfig,
    SetupConfig,
    SimulationConfig,
    TableOptionsConfig,
    check_dataset,
    init_dataset,
    reset_simulations,
    resume_simulations,
    run_simulations_by_state,
    validate_dataset,
    validate_dataset_matches_request,
)
from .simulation.runner import RunSummary
from .training import (
    ModelTrainingDataConfig,
    ModelTrainingValidationError,
    TrainingRecoveryMismatchError,
    TrainingTerminationReason,
    TrainingValidationRecord,
    TrainModelRequest,
    TrainModelResult,
    train_model,
)

# Package export surface

__all__ = [
    "AnalysisDataset",
    "AnalysisDatasetLoadPayload",
    "AnalysisLoadContext",
    "AnalysisLoadContribution",
    "AnalysisSimulation",
    "AnalysisSimulationLoadPayload",
    "BaseSimulation",
    "DatasetConfigMismatchError",
    "DatasetStatus",
    "DatasetValidationError",
    "HybridSimulation",
    "InitDatasetRequest",
    "ModelEvaluationResult",
    "ModelPredictor",
    "ModelTrainingDataConfig",
    "ModelTrainingValidationError",
    "OptionsConfig",
    "RunSummary",
    "SetupConfig",
    "Simulation",
    "SimulationConfig",
    "SimulationContext",
    "SimulationResult",
    "SimulationState",
    "TableOptionsConfig",
    "TiptopBaseConfig",
    "TiptopSimulation",
    "TrainModelRequest",
    "TrainModelResult",
    "TrainingRecoveryMismatchError",
    "TrainingTerminationReason",
    "TrainingValidationRecord",
    "__version__",
    "check_dataset",
    "init_dataset",
    "load_analysis_dataset",
    "load_model_predictor",
    "reset_simulations",
    "resume_simulations",
    "run_simulations_by_state",
    "train_model",
    "validate_dataset",
    "validate_dataset_matches_request",
]

# Package metadata

__version__ = "0.0.1"
