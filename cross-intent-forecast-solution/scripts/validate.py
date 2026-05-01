from __future__ import annotations

import hydra
import lightning as L
from omegaconf import DictConfig

from src.data.datamodule import CrossingDataModule, DataConfig
from src.models.network import ModelConfig
from src.training.module import CrossingForecastModule, TrainingConfig
from src.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    data_cfg = DataConfig(**cfg.data)
    model_cfg = ModelConfig(**cfg.model)
    training_cfg = TrainingConfig(
        **{
            **cfg.training,
            "curriculum_stage_epochs": tuple(cfg.training.curriculum_stage_epochs),
        }
    )
    datamodule = CrossingDataModule(data_cfg, heatmap_size=tuple(cfg.model.heatmap_size))
    module = CrossingForecastModule(model_cfg, training_cfg)
    trainer = L.Trainer(accelerator="auto", devices=1, logger=False)
    trainer.validate(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
