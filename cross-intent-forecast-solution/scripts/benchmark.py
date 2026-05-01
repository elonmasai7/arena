from __future__ import annotations

import json

import hydra
import torch
from omegaconf import DictConfig

from src.inference.benchmark import benchmark_predictor
from src.inference.predictor import ForecastPredictor
from src.models.network import ModelConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    model_cfg = ModelConfig(**cfg.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = ForecastPredictor(model_cfg=model_cfg, device=device, temperature=float(cfg.inference.temperature))
    scene = torch.randn(cfg.inference.max_batch_size, 3, *cfg.data.image_size)
    pedestrian = torch.randn(cfg.inference.max_batch_size, 3, cfg.data.ped_crop_size[0], cfg.data.ped_crop_size[1])
    interaction = torch.randn(cfg.inference.max_batch_size, 10)
    metrics = benchmark_predictor(predictor, scene, pedestrian, interaction, iterations=int(cfg.inference.benchmark_iterations))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
