from __future__ import annotations

import torch

from src.inference.predictor import ForecastPredictor
from src.models.network import ModelConfig


def test_predictor_batch() -> None:
    cfg = ModelConfig(
        backbone_name="convnext_tiny",
        fpn_channels=32,
        transformer_dim=64,
        transformer_heads=4,
        transformer_layers=2,
        dropout=0.1,
        num_modes=6,
        num_waypoints=8,
        heatmap_size=(8, 8),
        interaction_dim=32,
        label_smoothing=0.02,
    )
    predictor = ForecastPredictor(cfg, device="cpu", temperature=1.2)
    outputs = predictor.predict_batch(torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64), torch.randn(2, 10), tta=True)
    assert outputs["intent_prob"].shape == (2,)
