"""Public extension contract for seeded simulation-option samplers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Mapping

import numpy as np
from astropy import units as u


@dataclass(frozen=True)
class SamplerRequest:
    """One immutable invocation supplied to a sampler implementation.

    ``fields`` contains the owner followed by its direct referring fields.
    Physical outputs must be quantities; nonphysical outputs must be arrays.
    Neither prepared context nor parameters may be mutated by a sampler.

    Attributes:
        owner_field: Canonical field whose definition owns this invocation.
        fields: All declared canonical output fields for the invocation.
        count: Number of simulation-option rows to draw.
        num_ngs: Shared width of NGS fields.
        seed: Stable owner-specific integer invocation seed.
        parameters: Sampler-owned, read-only parameter mapping.
        unit: Owner field input unit, or ``None`` for a nonphysical field.
        simulation: Read-only prepared simulation payload.
        setup: Read-only prepared setup payload.
    """

    owner_field: str
    fields: tuple[str, ...]
    count: int
    num_ngs: int
    seed: int
    parameters: Mapping[str, object]
    unit: u.UnitBase | None
    simulation: Mapping[str, object]
    setup: Mapping[str, object]


class Sampler(ABC):
    """Stateless extension point for one seeded option-field draw.

    Implementations declare a positive integer ``version`` and return exactly
    the canonical fields listed in ``request.fields``. They must use only the
    supplied seed for randomness and leave process-global random state alone.
    """

    version: ClassVar[int]

    @classmethod
    def validate_declaration(cls, request: SamplerRequest) -> None:
        """Validate declared fields and parameters before any draw."""

    @abstractmethod
    def sample(self, request: SamplerRequest) -> Mapping[str, np.ndarray | u.Quantity]:
        """Draw one complete mapping of declared option fields."""
