from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def binary_cross_entropy(prob: torch.Tensor, target: torch.Tensor) -> float:
    clamped = prob.clamp(1e-6, 1 - 1e-6)
    loss = -(target * clamped.log() + (1 - target) * (1 - clamped).log()).mean()
    return float(loss.item())


@dataclass
class ChallengeCompositeMeter:
    do_nothing_bce: float = 0.69314718056
    do_nothing_ade: float = 60.0
    probs: list[float] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    minade: list[float] = field(default_factory=list)
    minfde: list[float] = field(default_factory=list)

    def update(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        minade: torch.Tensor,
        minfde: torch.Tensor,
    ) -> None:
        self.probs.extend(probs.detach().cpu().flatten().tolist())
        self.targets.extend(targets.detach().cpu().int().flatten().tolist())
        self.minade.extend(minade.detach().cpu().flatten().tolist())
        self.minfde.extend(minfde.detach().cpu().flatten().tolist())

    def compute(self) -> dict[str, float]:
        probs = torch.tensor(self.probs, dtype=torch.float32)
        targets = torch.tensor(self.targets, dtype=torch.float32)
        bce = binary_cross_entropy(probs, targets)
        auroc = roc_auc_score(self.targets, self.probs) if len(set(self.targets)) > 1 else 0.5
        f1 = f1_score(self.targets, (np.asarray(self.probs) >= 0.5).astype(np.int32), zero_division=0)
        ade = float(np.mean(self.minade)) if self.minade else 0.0
        fde = float(np.mean(self.minfde)) if self.minfde else 0.0
        normalized_bce = bce / self.do_nothing_bce
        normalized_ade = ade / self.do_nothing_ade
        composite = 0.5 * (normalized_bce + normalized_ade)
        return {
            "bce": bce,
            "auroc": float(auroc),
            "f1": float(f1),
            "ade": ade,
            "fde": fde,
            "minADE@6": ade,
            "minFDE@6": fde,
            "normalized_bce": normalized_bce,
            "normalized_ade": normalized_ade,
            "challenge_composite": composite,
        }
