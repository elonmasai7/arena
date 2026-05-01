from __future__ import annotations

import torch


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.sigmoid(logits / max(temperature, 1e-3))
