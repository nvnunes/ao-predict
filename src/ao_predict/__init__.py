"""ao-predict package."""

# Public re-exports

from .simulation import (
    BaseSimulation,
    Simulation,
    SimulationContext,
    SimulationResult,
    SimulationState,
    TiptopBaseConfig,
    TiptopSimulation,
)
from .simulation.api import (
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
    run_simulations_by_state,
    validate_dataset,
)
from .simulation.runner import RunSummary

# Package export surface

__all__ = [
    "__version__",
    "check_dataset",
    "DatasetStatus",
    "DatasetValidationError",
    "InitDatasetRequest",
    "init_dataset",
    "OptionsConfig",
    "TableOptionsConfig",
    "reset_simulations",
    "RunSummary",
    "run_simulations_by_state",
    "validate_dataset",
    "SetupConfig",
    "BaseSimulation",
    "Simulation",
    "SimulationConfig",
    "SimulationContext",
    "SimulationResult",
    "SimulationState",
    "TiptopBaseConfig",
    "TiptopSimulation",
]

# Package metadata

__version__ = "0.0.1"
