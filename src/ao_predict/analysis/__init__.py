"""Immutable analysis views built on top of persisted simulation datasets."""

# Public re-exports

from ._compose import load_analysis_dataset
from .types import (
    AnalysisDataset,
    AnalysisDatasetFactory,
    AnalysisDatasetLoadPayload,
    AnalysisSimulation,
    AnalysisSimulationFactory,
    AnalysisSimulationLoadContext,
    AnalysisSimulationLoadPayload,
)

__all__ = [
    "AnalysisDataset",
    "AnalysisDatasetFactory",
    "AnalysisDatasetLoadPayload",
    "AnalysisSimulation",
    "AnalysisSimulationFactory",
    "AnalysisSimulationLoadContext",
    "AnalysisSimulationLoadPayload",
    "load_analysis_dataset",
]
