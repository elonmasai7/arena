from __future__ import annotations

import copy

import torch


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.shadow = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_parameter, parameter in zip(self.shadow.parameters(), model.parameters(), strict=True):
            ema_parameter.data.mul_(self.decay).add_(parameter.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow.state_dict()
