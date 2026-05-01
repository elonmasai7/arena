from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import WandbLogger

from .module import TrainingConfig


def build_callbacks(cfg: TrainingConfig) -> list[L.Callback]:
    return [
        ModelCheckpoint(
            monitor="val/challenge_composite",
            mode="min",
            save_top_k=cfg.checkpoint_top_k,
            filename="epoch{epoch:02d}",
        ),
        EarlyStopping(
            monitor="val/challenge_composite",
            mode="min",
            patience=cfg.patience,
        ),
        StochasticWeightAveraging(swa_lrs=cfg.swa_lrs),
    ]


def build_logger(project: str, name: str) -> WandbLogger:
    return WandbLogger(project=project, name=name, log_model=False)
