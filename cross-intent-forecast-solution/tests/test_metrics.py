from __future__ import annotations

import torch

from src.metrics import ChallengeCompositeMeter


def test_challenge_metric_computes() -> None:
    meter = ChallengeCompositeMeter()
    meter.update(
        probs=torch.tensor([0.9, 0.1]),
        targets=torch.tensor([1.0, 0.0]),
        minade=torch.tensor([3.0, 5.0]),
        minfde=torch.tensor([4.0, 6.0]),
    )
    metrics = meter.compute()
    assert metrics["challenge_composite"] < 1.0
    assert metrics["auroc"] == 1.0
