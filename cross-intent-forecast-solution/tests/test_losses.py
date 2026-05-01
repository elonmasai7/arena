from __future__ import annotations

import torch

from src.losses import JointForecastingLoss


def test_joint_loss_runs() -> None:
    loss_fn = JointForecastingLoss()
    outputs = {
        "intent_logits": torch.randn(2),
        "intent_prob": torch.sigmoid(torch.randn(2)),
        "mode_logits": torch.randn(2, 6),
        "waypoints": torch.rand(2, 6, 8, 2),
        "heatmaps": torch.softmax(torch.rand(2, 6, 8, 16 * 16), dim=-1).view(2, 6, 8, 16, 16),
        "aleatoric": torch.rand(2, 6, 8, 2) + 0.1,
    }
    batch = {
        "intent": torch.tensor([1.0, 0.0]),
        "waypoints": torch.rand(2, 8, 2),
        "heatmaps": torch.softmax(torch.rand(2, 8, 16 * 16), dim=-1).view(2, 8, 16, 16),
    }
    result = loss_fn(outputs, batch, stage="joint")
    assert result["loss"].item() > 0
