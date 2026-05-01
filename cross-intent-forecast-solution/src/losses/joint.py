from __future__ import annotations

import torch
from torch import nn

from .trajectory import HeatmapKLLoss, ade, fde


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class JointForecastingLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal_bce = FocalBCELoss()
        self.heatmap_kl = HeatmapKLLoss()
        self.log_vars = nn.Parameter(torch.zeros(4))

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        stage: str,
    ) -> dict[str, torch.Tensor]:
        target_intent = batch["intent"]
        target_waypoints = batch["waypoints"]
        target_heatmaps = batch["heatmaps"]
        pred_waypoints = outputs["waypoints"]
        aleatoric = outputs["aleatoric"]

        smooth_intent = target_intent * 0.98 + 0.01
        bce_loss = self.bce(outputs["intent_logits"], smooth_intent)
        focal_loss = self.focal_bce(outputs["intent_logits"], target_intent)
        ade_all = ade(pred_waypoints, target_waypoints)
        fde_all = fde(pred_waypoints, target_waypoints)
        best_mode = torch.argmin(ade_all, dim=1)
        gather_index = best_mode[:, None, None, None].expand(-1, 1, target_waypoints.shape[1], 2)
        best_pred = pred_waypoints.gather(1, gather_index).squeeze(1)
        best_aleatoric = aleatoric.gather(1, gather_index).squeeze(1)
        regression = ((best_pred - target_waypoints) ** 2 / best_aleatoric + best_aleatoric.log()).mean()
        heatmap_loss = self.heatmap_kl(outputs["heatmaps"], target_heatmaps)
        ade_loss = ade_all.min(dim=1).values.mean()
        fde_loss = fde_all.min(dim=1).values.mean()

        stage_weights = {
            "trajectory": torch.tensor([0.0, 0.0, 1.0, 1.0], device=target_intent.device),
            "intent": torch.tensor([1.0, 1.0, 0.2, 0.0], device=target_intent.device),
            "joint": torch.tensor([1.0, 1.0, 1.0, 1.0], device=target_intent.device),
        }[stage]
        task_losses = torch.stack([bce_loss + focal_loss, heatmap_loss, ade_loss + fde_loss, regression])
        precision = torch.exp(-self.log_vars)
        total = (stage_weights * (precision * task_losses + self.log_vars)).sum()
        return {
            "loss": total,
            "bce": bce_loss,
            "focal_bce": focal_loss,
            "heatmap_kl": heatmap_loss,
            "ade": ade_loss,
            "fde": fde_loss,
            "regression": regression,
            "minade_per_sample": ade_all.min(dim=1).values.detach(),
            "minfde_per_sample": fde_all.min(dim=1).values.detach(),
        }
