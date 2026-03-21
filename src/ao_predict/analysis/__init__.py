"""Immutable analysis views built on top of persisted simulation datasets."""

# Public re-exports

from ._compose import load_analysis_dataset
from .types import (
    AnalysisDataset,
    AnalysisDatasetLoadPayload,
    AnalysisLoadContext,
    AnalysisLoadContribution,
    AnalysisSimulation,
    AnalysisSimulationLoadPayload,
)

__all__ = [
    "AnalysisDataset",
    "AnalysisDatasetLoadPayload",
    "AnalysisLoadContext",
    "AnalysisLoadContribution",
    "AnalysisSimulation",
    "AnalysisSimulationLoadPayload",
    "load_analysis_dataset",
]
