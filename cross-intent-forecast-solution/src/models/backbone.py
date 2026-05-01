from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torchvision.models import convnext_tiny


class ConvNeXtFPN(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        backbone = convnext_tiny(weights=None)
        self.stem = backbone.features[0]
        self.stage1 = backbone.features[1]
        self.stage2 = backbone.features[2:4]
        self.stage3 = backbone.features[4:6]
        self.stage4 = backbone.features[6:8]
        channels = [96, 192, 384, 768]
        self.lateral = nn.ModuleList(nn.Conv2d(c, out_channels, kernel_size=1) for c in channels)
        self.output = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm2d(out_channels),
            )
            for _ in channels
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        feats = [c1, c2, c3, c4]
        pyramid: list[torch.Tensor] = [self.lateral[-1](feats[-1])]
        for idx in range(len(feats) - 2, -1, -1):
            upsampled = nn.functional.interpolate(pyramid[0], size=feats[idx].shape[-2:], mode="nearest")
            fused = self.lateral[idx](feats[idx]) + upsampled
            pyramid.insert(0, fused)
        return [layer(feature) for layer, feature in zip(self.output, pyramid)]
