from __future__ import annotations

import hydra
import lightning as L
from omegaconf import DictConfig

from src.data.datamodule import CrossingDataModule, DataConfig
from src.models.network import ModelConfig
from src.training.callbacks import build_callbacks, build_logger
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
    trainer = L.Trainer(
        accelerator=training_cfg.accelerator,
        devices=training_cfg.devices,
        strategy=training_cfg.strategy,
        precision=training_cfg.precision,
        max_epochs=training_cfg.max_epochs,
        gradient_clip_val=training_cfg.gradient_clip_val,
        accumulate_grad_batches=training_cfg.accumulate_grad_batches,
        callbacks=build_callbacks(training_cfg),
        logger=build_logger(training_cfg.wandb_project, cfg.experiment_name),
        deterministic=True,
        default_root_dir=cfg.output_dir,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
