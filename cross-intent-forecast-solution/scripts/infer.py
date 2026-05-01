from __future__ import annotations

import json

import hydra
import torch
from omegaconf import DictConfig

from src.data.datamodule import CrossingDataModule, DataConfig
from src.inference.predictor import ForecastPredictor
from src.models.network import ModelConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data_cfg = DataConfig(**cfg.data)
    model_cfg = ModelConfig(**cfg.model)
    datamodule = CrossingDataModule(data_cfg, heatmap_size=tuple(cfg.model.heatmap_size))
    datamodule.setup("predict")
    batch = next(iter(datamodule.val_dataloader()))
    predictor = ForecastPredictor(
        model_cfg=model_cfg,
        checkpoint_path=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        temperature=float(cfg.inference.temperature),
    )
    outputs = predictor.predict_batch(
        batch["scene"],
        batch["pedestrian"],
        batch["interaction"],
        tta=bool(cfg.inference.tta),
    )
    result = {
        "intent_prob": outputs["intent_prob"][0].item(),
        "best_mode": int(outputs["mode_logits"][0].argmax().item()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
