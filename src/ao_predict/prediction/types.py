"""Public result types for model evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Aggregate physical relative-error measurements for one population.

    All error values are dimensionless ratios, not percentages. The overall
    values pool every example and target; ``target_relative_rmse`` reports one
    complete-population value for each target name.
    """

    example_count: int
    relative_mse: float
    relative_rmse: float
    target_relative_rmse: Mapping[str, float]

    def __post_init__(self) -> None:
        """Make the per-target mapping immutable with the result."""

        object.__setattr__(
            self,
            "target_relative_rmse",
            MappingProxyType(dict(self.target_relative_rmse)),
        )
