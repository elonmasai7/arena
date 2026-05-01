from __future__ import annotations

from pathlib import Path

import torch

from src.inference.calibration import temperature_scale
from src.models import CrossingIntentModel
from src.models.network import ModelConfig


class ForecastPredictor:
    def __init__(
        self,
        model_cfg: ModelConfig,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        temperature: float = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.model = CrossingIntentModel(model_cfg).to(self.device)
        self.temperature = temperature
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            cleaned = {key.replace("model.", "", 1): value for key, value in state_dict.items()}
            self.model.load_state_dict(cleaned, strict=False)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(
        self,
        scene: torch.Tensor,
        pedestrian: torch.Tensor,
        interaction: torch.Tensor,
        tta: bool = False,
    ) -> dict[str, torch.Tensor]:
        scene = scene.to(self.device)
        pedestrian = pedestrian.to(self.device)
        interaction = interaction.to(self.device)
        outputs = self.model(scene, pedestrian, interaction)
        if tta:
            flipped_scene = torch.flip(scene, dims=[3])
            flipped_ped = torch.flip(pedestrian, dims=[3])
            flipped_outputs = self.model(flipped_scene, flipped_ped, interaction)
            flipped_outputs["waypoints"][..., 0] = 1.0 - flipped_outputs["waypoints"][..., 0]
            outputs = {
                key: (outputs[key] + flipped_outputs[key]) / 2.0
                for key in outputs
            }
        outputs["intent_prob"] = temperature_scale(outputs["intent_logits"], self.temperature)
        return outputs

    def predict_single(
        self,
        scene: torch.Tensor,
        pedestrian: torch.Tensor,
        interaction: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.predict_batch(scene.unsqueeze(0), pedestrian.unsqueeze(0), interaction.unsqueeze(0), tta=False)
