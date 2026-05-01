from __future__ import annotations

import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(query, context, context, need_weights=False)
        query = self.norm1(query + attended)
        fed = self.ffn(query)
        return self.norm2(query + fed)


class TemporalEncoder(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)
