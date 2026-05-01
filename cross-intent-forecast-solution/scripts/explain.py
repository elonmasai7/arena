from __future__ import annotations

from pathlib import Path

import cv2
import hydra
import torch
from omegaconf import DictConfig

from src.data.datamodule import CrossingDataModule, DataConfig
from src.models import CrossingIntentModel
from src.models.network import ModelConfig
from src.utils.visualization import grad_cam, save_attention_map


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data_cfg = DataConfig(**cfg.data)
    model_cfg = ModelConfig(**cfg.model)
    datamodule = CrossingDataModule(data_cfg, heatmap_size=tuple(cfg.model.heatmap_size))
    datamodule.setup("predict")
    batch = next(iter(datamodule.val_dataloader()))
    model = CrossingIntentModel(model_cfg).eval()
    cam = grad_cam(
        model,
        batch["scene"][:1].requires_grad_(True),
        batch["pedestrian"][:1],
        batch["interaction"][:1],
    )
    save_attention_map(cam, Path("artifacts/gradcam.png"))
    image = batch["scene"][0].permute(1, 2, 0).detach().cpu().numpy()
    image = ((image - image.min()) / max(float(image.max() - image.min()), 1e-6) * 255).astype("uint8")
    cv2.imwrite("artifacts/scene.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
