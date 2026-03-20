"""Immutable analysis views built on top of persisted simulation datasets."""

# Public re-exports

from ._compose import load_analysis_dataset
from .types import AnalysisDataset, AnalysisSimulation

__all__ = ["AnalysisDataset", "AnalysisSimulation", "load_analysis_dataset"]
