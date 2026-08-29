"""Private shared PyTorch runtime selection."""

from __future__ import annotations

import torch


def select_device(name: str, cpu_threads: int | None) -> torch.device:
    """Resolve one explicitly requested and available execution device."""

    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"device is invalid: {exc}") from exc
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device type must be cpu, cuda, or mps.")
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("CPU devices must not include an index.")
    elif cpu_threads is not None:
        raise ValueError("cpu_threads is accepted only with a CPU device.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA device {name!r} is unavailable.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable.")
        device = torch.device("cuda", index)
    if device.type == "mps":
        if device.index is not None:
            raise ValueError("MPS devices must not include an index.")
        if not torch.backends.mps.is_available():
            raise ValueError("MPS device is unavailable.")
    return device
