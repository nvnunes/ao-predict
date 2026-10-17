"""Generic simulation-output precision, independent of scientific engines."""

import numpy as np
from numpy.typing import DTypeLike


def real_dtype(value: DTypeLike) -> np.dtype:
    """Validate the float32/float64 simulation output contract."""
    if value is None:
        raise ValueError("precision must be float32 or float64")
    try:
        dtype = np.dtype(value)
    except (ValueError, TypeError) as error:
        raise ValueError("precision must be float32 or float64") from error
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("precision must be float32 or float64")
    return dtype


def diagnostic_precision(fields, dtype):
    """Copy declarations, converting floating fields while retaining other types."""
    return {
        name: {
            **spec,
            **(
                {"dtype": dtype.name}
                if spec.get("dtype") in ("float32", "float64")
                else {}
            ),
        }
        for name, spec in fields.items()
    }
