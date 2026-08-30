"""Private ownership of AO Predict's single dense-regression model family."""

from __future__ import annotations

import math
from itertools import pairwise

import torch
from torch import nn


class _DenseRegressionModel(nn.Module):
    """The single AO Predict dense-regression family."""

    def __init__(
        self,
        input_width: int,
        hidden_widths: tuple[int, ...],
        output_width: int,
    ) -> None:
        super().__init__()
        widths = (input_width, *hidden_widths, output_width)
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(pairwise(widths)):
            layers.append(nn.Linear(source, target, bias=True))
            if index < len(widths) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _initialize_linear_layers(model: nn.Module, generator: torch.Generator) -> None:
    """Apply the explicit fan-in-scaled PyTorch Linear initialization rule."""

    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(module.bias, -bound, bound, generator=generator)


def build_dense_model(
    input_width: int,
    hidden_widths: tuple[int, ...],
    output_width: int,
    *,
    initialization_seed: int,
) -> tuple[_DenseRegressionModel, torch.Tensor]:
    """Construct and initialize a model without changing global random state."""

    with torch.random.fork_rng(devices=[]):
        model = _DenseRegressionModel(input_width, hidden_widths, output_width)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(initialization_seed)
    _initialize_linear_layers(model, generator)
    return model, generator.get_state().clone()


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy one model state to independent CPU tensors."""

    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
