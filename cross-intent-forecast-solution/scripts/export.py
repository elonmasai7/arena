from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from src.export import export_onnx
from src.models import CrossingIntentModel
from src.models.network import ModelConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model_cfg = ModelConfig(**cfg.model)
    model = CrossingIntentModel(model_cfg).eval()
    scene = torch.randn(1, 3, *cfg.data.image_size)
    pedestrian = torch.randn(1, 3, cfg.data.ped_crop_size[0], cfg.data.ped_crop_size[1])
    interaction = torch.randn(1, 10)
    export_onnx(model, Path("artifacts/model.onnx"), scene, pedestrian, interaction)


if __name__ == "__main__":
    main()
