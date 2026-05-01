from __future__ import annotations

import torch


def average_ensemble(outputs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = outputs[0].keys()
    return {key: torch.stack([output[key] for output in outputs], dim=0).mean(dim=0) for key in keys}
