"""Run adaptive-optics simulations and build surrogate models.

Pass simulation, setup, and option configurations in an ``InitDatasetRequest``
to ``init_dataset``. Options can come from arrays, a table, or
``generate_options``. Use ``run_simulations_by_state`` to run selected
simulations and ``resume_simulations`` to continue an existing dataset.
``load_analysis_dataset`` loads saved results for analysis.

Pass a ``TrainModelRequest`` with named feature and target arrays or Astropy
quantities to ``train_model``. Use ``load_model_predictor(result.model_path)``
to predict or evaluate with the published model. Simulation classes, sampler
contracts, and analysis-loading hooks support downstream extensions.
"""

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
from .simulation.tiptop import tiptop_full_flux_denominators
from .simulation.sampling import (
    GenerateOptionsConfig,
    GenerateOptionsRequest,
    Sampler,
    SamplerRequest,
    generate_options,
)
from .simulation.sampling_balancing import balance_options
from .simulation.sampling_selection import select_options
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

# List exports in workflow order.

__all__ = [  # noqa: RUF022
    # Simulation options
    "OptionsConfig",
    "TableOptionsConfig",
    "GenerateOptionsConfig",
    "GenerateOptionsRequest",
    "generate_options",
    "balance_options",
    "select_options",

    # Dataset setup and validation
    "SimulationConfig",
    "SetupConfig",
    "InitDatasetRequest",
    "init_dataset",
    "DatasetStatus",
    "DatasetValidationError",
    "DatasetConfigMismatchError",
    "check_dataset",
    "validate_dataset",
    "validate_dataset_matches_request",

    # Simulation execution
    "SimulationState",
    "RunSummary",
    "run_simulations_by_state",
    "resume_simulations",
    "reset_simulations",

    # Analysis
    "AnalysisDataset",
    "AnalysisSimulation",
    "load_analysis_dataset",

    # Model training
    "ModelTrainingDataConfig",
    "TrainModelRequest",
    "train_model",
    "TrainModelResult",
    "TrainingValidationRecord",
    "TrainingTerminationReason",
    "ModelTrainingValidationError",
    "TrainingRecoveryMismatchError",

    # Model prediction and evaluation
    "load_model_predictor",
    "ModelPredictor",
    "ModelEvaluationResult",

    # Simulation and sampler extensions
    "Simulation",
    "BaseSimulation",
    "SimulationContext",
    "SimulationResult",
    "TiptopBaseConfig",
    "TiptopSimulation",
    "tiptop_full_flux_denominators",
    "HybridSimulation",
    "Sampler",
    "SamplerRequest",

    # Analysis loading extensions
    "AnalysisLoadContext",
    "AnalysisLoadContribution",
    "AnalysisDatasetLoadPayload",
    "AnalysisSimulationLoadPayload",

    # Package metadata
    "__version__",
]

# Package metadata

__version__ = "0.0.1"
