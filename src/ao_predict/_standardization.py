"""Private numerical contract shared by model training and prediction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def validate_float32_scaler(
    mean: float,
    scale: float,
    *,
    label: str,
) -> None:
    """Require one scaler to remain finite with positive scale as float32."""

    if not math.isfinite(mean) or not math.isfinite(scale):
        raise ValueError(f"{label} scaler values must be finite.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = np.asarray([mean, scale], dtype=np.float32)
    if not bool(np.all(np.isfinite(converted))):
        raise ValueError(
            f"{label} scaler values must be representable as finite float32."
        )
    if float(converted[1]) <= 0.0:
        raise ValueError(f"{label} scale must remain positive as float32.")


def standardize_to_float32(
    values: np.ndarray,
    means: float | Sequence[float] | np.ndarray,
    scales: float | Sequence[float] | np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """Standardize physical values in float64, then return contiguous float32."""

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        physical = np.asarray(values, dtype=np.float64)
        standardized = (physical - np.asarray(means, dtype=np.float64)) / np.asarray(
            scales, dtype=np.float64
        )
        output = np.ascontiguousarray(standardized, dtype=np.float32)
    if not bool(np.all(np.isfinite(output))):
        raise ValueError(f"{label} cannot be standardized as finite float32 values.")
    return output
