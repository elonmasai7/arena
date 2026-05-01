from __future__ import annotations

import torch

from src.models.network import CrossingIntentModel, ModelConfig


def test_model_forward_shapes() -> None:
    cfg = ModelConfig(
        backbone_name="convnext_tiny",
        fpn_channels=32,
        transformer_dim=64,
        transformer_heads=4,
        transformer_layers=2,
        dropout=0.1,
        num_modes=6,
        num_waypoints=8,
        heatmap_size=(16, 16),
        interaction_dim=32,
        label_smoothing=0.02,
    )
    model = CrossingIntentModel(cfg)
    outputs = model(torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64), torch.randn(2, 10))
    assert outputs["intent_prob"].shape == (2,)
    assert outputs["mode_logits"].shape == (2, 6)
    assert outputs["waypoints"].shape == (2, 6, 8, 2)
    assert outputs["heatmaps"].shape == (2, 6, 8, 16, 16)
