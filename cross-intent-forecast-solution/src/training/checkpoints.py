from __future__ import annotations

from pathlib import Path

import torch


def average_checkpoints(paths: list[str | Path], destination: str | Path) -> Path:
    if not paths:
        raise ValueError("At least one checkpoint path is required.")
    checkpoints = [torch.load(Path(path), map_location="cpu") for path in paths]
    state_dicts = [checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint for checkpoint in checkpoints]
    keys = state_dicts[0].keys()
    averaged: dict[str, torch.Tensor] = {}
    for key in keys:
        averaged[key] = sum(state_dict[key] for state_dict in state_dicts) / len(state_dicts)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": averaged}, destination)
    return destination
