from __future__ import annotations

import torch

from src.export.onnx_export import export_onnx
from src.models.network import CrossingIntentModel, ModelConfig


def test_onnx_export(tmp_path) -> None:
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
    model = CrossingIntentModel(cfg).eval()
    path = export_onnx(model, tmp_path / "model.onnx", torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64), torch.randn(1, 10))
    assert path.exists()
