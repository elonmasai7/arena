from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .backbone import ConvNeXtFPN
from .blocks import CrossAttentionBlock, TemporalEncoder


@dataclass(slots=True)
class ModelConfig:
    backbone_name: str
    fpn_channels: int
    transformer_dim: int
    transformer_heads: int
    transformer_layers: int
    dropout: float
    num_modes: int
    num_waypoints: int
    heatmap_size: tuple[int, int]
    interaction_dim: int
    label_smoothing: float


class CrossingIntentModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.scene_backbone = ConvNeXtFPN(out_channels=cfg.fpn_channels)
        self.ped_backbone = ConvNeXtFPN(out_channels=cfg.fpn_channels)
        self.scene_proj = nn.Linear(cfg.fpn_channels, cfg.transformer_dim)
        self.ped_proj = nn.Linear(cfg.fpn_channels, cfg.transformer_dim)
        self.interaction_proj = nn.Sequential(
            nn.Linear(10, cfg.interaction_dim),
            nn.GELU(),
            nn.Linear(cfg.interaction_dim, cfg.transformer_dim),
        )
        self.temporal = TemporalEncoder(
            dim=cfg.transformer_dim,
            heads=cfg.transformer_heads,
            layers=cfg.transformer_layers,
            dropout=cfg.dropout,
        )
        self.cross_attention = CrossAttentionBlock(
            dim=cfg.transformer_dim,
            heads=cfg.transformer_heads,
            dropout=cfg.dropout,
        )
        self.intent_head = nn.Sequential(
            nn.Linear(cfg.transformer_dim, cfg.transformer_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.transformer_dim, 1),
        )
        self.mode_head = nn.Linear(cfg.transformer_dim, cfg.num_modes)
        self.waypoint_head = nn.Linear(cfg.transformer_dim, cfg.num_modes * cfg.num_waypoints * 2)
        heatmap_dim = cfg.num_modes * cfg.num_waypoints * cfg.heatmap_size[0] * cfg.heatmap_size[1]
        self.heatmap_head = nn.Linear(cfg.transformer_dim, heatmap_dim)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(cfg.transformer_dim, cfg.num_modes * cfg.num_waypoints * 2),
            nn.Softplus(),
        )

    @staticmethod
    def _pool_fpn(features: list[torch.Tensor]) -> torch.Tensor:
        pooled = [feature.flatten(2).mean(-1) for feature in features]
        return torch.stack(pooled, dim=1)

    def forward(
        self,
        scene: torch.Tensor,
        pedestrian: torch.Tensor,
        interaction: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        scene_tokens = self._pool_fpn(self.scene_backbone(scene))
        ped_tokens = self._pool_fpn(self.ped_backbone(pedestrian))
        scene_tokens = self.scene_proj(scene_tokens)
        ped_tokens = self.ped_proj(ped_tokens)
        interaction_token = self.interaction_proj(interaction).unsqueeze(1)
        tokens = torch.cat([scene_tokens, ped_tokens, interaction_token], dim=1)
        encoded = self.temporal(tokens)
        query = encoded[:, -1:, :]
        fused = self.cross_attention(query, encoded).squeeze(1)
        intent_logits = self.intent_head(fused).squeeze(-1)
        mode_logits = self.mode_head(fused)
        waypoint_raw = self.waypoint_head(fused)
        heatmap_raw = self.heatmap_head(fused)
        uncertainty = self.uncertainty_head(fused)

        batch = scene.shape[0]
        waypoints = waypoint_raw.view(batch, self.cfg.num_modes, self.cfg.num_waypoints, 2).sigmoid()
        heatmaps = heatmap_raw.view(
            batch,
            self.cfg.num_modes,
            self.cfg.num_waypoints,
            self.cfg.heatmap_size[0],
            self.cfg.heatmap_size[1],
        )
        heatmaps = torch.softmax(heatmaps.flatten(-2), dim=-1).view_as(heatmaps)
        aleatoric = uncertainty.view(batch, self.cfg.num_modes, self.cfg.num_waypoints, 2) + 1e-4
        return {
            "intent_logits": intent_logits,
            "intent_prob": torch.sigmoid(intent_logits),
            "mode_logits": mode_logits,
            "waypoints": waypoints,
            "heatmaps": heatmaps,
            "aleatoric": aleatoric,
        }
