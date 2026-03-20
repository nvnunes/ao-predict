"""Helpers for constructing immutable analysis payload views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np


def freeze_array(value: Any) -> np.ndarray:
    """Return a detached non-writeable numpy array copy."""
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def freeze_value(value: Any) -> Any:
    """Recursively freeze analysis payload values."""
    if isinstance(value, np.ndarray):
        return freeze_array(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Mapping):
        return freeze_mapping(value)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(freeze_value(item) for item in value)

    return value


def freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively freeze a mapping into an immutable mapping view."""
    return MappingProxyType({str(key): freeze_value(value) for key, value in mapping.items()})
