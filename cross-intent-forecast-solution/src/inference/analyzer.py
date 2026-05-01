from __future__ import annotations

import pandas as pd
import torch


def analyze_failures(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> pd.DataFrame:
    intent_error = (outputs["intent_prob"] - batch["intent"]).abs().detach().cpu().numpy()
    best_mode = torch.argmin(torch.norm(outputs["waypoints"] - batch["waypoints"].unsqueeze(1), dim=-1).mean(dim=-1), dim=1)
    rows: list[dict[str, float]] = []
    for index in range(batch["intent"].shape[0]):
        mode = int(best_mode[index].item())
        ade = torch.norm(outputs["waypoints"][index, mode] - batch["waypoints"][index], dim=-1).mean().item()
        rows.append({"sample_index": index, "intent_abs_error": float(intent_error[index]), "ade": float(ade)})
    return pd.DataFrame(rows).sort_values(["intent_abs_error", "ade"], ascending=False).reset_index(drop=True)
