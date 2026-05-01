from __future__ import annotations

from pathlib import Path

import torch

from src.models import CrossingIntentModel


class ONNXWrapper(torch.nn.Module):
    def __init__(self, model: CrossingIntentModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, scene: torch.Tensor, pedestrian: torch.Tensor, interaction: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.model(scene, pedestrian, interaction)
        return (
            outputs["intent_prob"],
            outputs["mode_logits"],
            outputs["waypoints"],
            outputs["heatmaps"],
            outputs["aleatoric"],
        )


def export_onnx(
    model: CrossingIntentModel,
    destination: str | Path,
    scene: torch.Tensor,
    pedestrian: torch.Tensor,
    interaction: torch.Tensor,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ONNXWrapper(model).eval()
    torch.onnx.export(
        wrapper,
        (scene, pedestrian, interaction),
        destination,
        input_names=["scene", "pedestrian", "interaction"],
        output_names=["intent_prob", "mode_logits", "waypoints", "heatmaps", "aleatoric"],
        dynamic_axes={"scene": {0: "batch"}, "pedestrian": {0: "batch"}, "interaction": {0: "batch"}},
        opset_version=17,
    )
    return destination
