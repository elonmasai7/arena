from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from src.data.datamodule import CrossingDataModule, DataConfig
from src.inference.analyzer import analyze_failures
from src.inference.predictor import ForecastPredictor
from src.models.network import ModelConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data_cfg = DataConfig(**cfg.data)
    model_cfg = ModelConfig(**cfg.model)
    datamodule = CrossingDataModule(data_cfg, heatmap_size=tuple(cfg.model.heatmap_size))
    datamodule.setup("predict")
    batch = next(iter(datamodule.val_dataloader()))
    predictor = ForecastPredictor(model_cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    outputs = predictor.predict_batch(batch["scene"], batch["pedestrian"], batch["interaction"], tta=bool(cfg.inference.tta))
    report = analyze_failures(outputs, batch)
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    report.to_csv("artifacts/failure_cases.csv", index=False)


if __name__ == "__main__":
    main()
