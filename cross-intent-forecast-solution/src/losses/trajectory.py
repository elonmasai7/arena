from __future__ import annotations

import torch
from torch import nn


def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred - target.unsqueeze(1), dim=-1).mean(dim=-1)


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred[:, :, -1] - target[:, -1].unsqueeze(1), dim=-1)


class HeatmapKLLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_expanded = target.unsqueeze(1).expand_as(pred)
        ratio = (target_expanded + 1e-8).log() - (pred + 1e-8).log()
        return (target_expanded * ratio).mean()
