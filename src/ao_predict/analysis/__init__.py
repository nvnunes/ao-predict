"""Immutable analysis views built on top of persisted simulation datasets."""

# Public re-exports

from ._compose import load_analysis_dataset
from .types import (
    AnalysisDataset,
    AnalysisDatasetLoadPayload,
    AnalysisSimulation,
    AnalysisSimulationLoadContext,
    AnalysisSimulationLoadPayload,
)

__all__ = [
    "AnalysisDataset",
    "AnalysisDatasetLoadPayload",
    "AnalysisSimulation",
    "AnalysisSimulationLoadContext",
    "AnalysisSimulationLoadPayload",
    "load_analysis_dataset",
]
