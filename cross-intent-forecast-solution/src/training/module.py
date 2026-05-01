from __future__ import annotations

from dataclasses import dataclass

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel

from src.losses import JointForecastingLoss
from src.metrics import ChallengeCompositeMeter
from src.models import CrossingIntentModel
from src.models.network import ModelConfig
from src.utils.ema import ExponentialMovingAverage


@dataclass(slots=True)
class TrainingConfig:
    accelerator: str
    devices: int | str
    strategy: str
    precision: str
    max_epochs: int
    gradient_clip_val: float
    accumulate_grad_batches: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    ema_decay: float
    swa_lrs: float
    curriculum_stage_epochs: tuple[int, int, int]
    hard_negative_ratio: float
    patience: int
    k_folds: int
    checkpoint_top_k: int
    wandb_project: str


class CrossingForecastModule(L.LightningModule):
    def __init__(self, model_cfg: ModelConfig, training_cfg: TrainingConfig) -> None:
        super().__init__()
        self.model = CrossingIntentModel(model_cfg)
        self.loss_fn = JointForecastingLoss()
        self.training_cfg = training_cfg
        self.val_meter = ChallengeCompositeMeter()
        self.ema = ExponentialMovingAverage(self.model, decay=training_cfg.ema_decay)
        self.swa_model = AveragedModel(self.model)
        self.save_hyperparameters(ignore=["model"])

    def current_stage(self) -> str:
        boundary1, boundary2, _boundary3 = self.training_cfg.curriculum_stage_epochs
        if self.current_epoch < boundary1:
            return "trajectory"
        if self.current_epoch < boundary1 + boundary2:
            return "intent"
        return "joint"

    def forward(
        self,
        scene: torch.Tensor,
        pedestrian: torch.Tensor,
        interaction: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.model(scene, pedestrian, interaction)

    def _shared_step(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        outputs = self(batch["scene"], batch["pedestrian"], batch["interaction"])
        loss_dict = self.loss_fn(outputs, batch, stage=self.current_stage())
        self.log(f"{prefix}/loss", loss_dict["loss"], prog_bar=True, batch_size=batch["scene"].shape[0])
        self.log(f"{prefix}/bce", loss_dict["bce"], batch_size=batch["scene"].shape[0])
        self.log(f"{prefix}/ade", loss_dict["ade"], batch_size=batch["scene"].shape[0])
        if prefix == "val":
            self.val_meter.update(
                outputs["intent_prob"], batch["intent"], loss_dict["minade_per_sample"], loss_dict["minfde_per_sample"]
            )
        return loss_dict["loss"]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        loss = self._shared_step(batch, prefix="train")
        return loss

    def on_after_backward(self) -> None:
        self.ema.update(self.model)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        del batch_idx
        self._shared_step(batch, prefix="val")

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_meter.compute()
        for name, value in metrics.items():
            self.log(f"val/{name}", value, prog_bar=name == "challenge_composite")
        self.val_meter = ChallengeCompositeMeter()
        self.swa_model.update_parameters(self.model)

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = AdamW(self.parameters(), lr=self.training_cfg.lr, weight_decay=self.training_cfg.weight_decay)

        def lr_lambda(epoch: int) -> float:
            if epoch < self.training_cfg.warmup_epochs:
                return float(epoch + 1) / max(1, self.training_cfg.warmup_epochs)
            progress = (epoch - self.training_cfg.warmup_epochs) / max(
                1, self.training_cfg.max_epochs - self.training_cfg.warmup_epochs
            )
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
