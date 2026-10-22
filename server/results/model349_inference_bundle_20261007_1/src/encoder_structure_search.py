# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import torch

import torch.nn as nn

import torch.nn.functional as F

from train_models_2_to_6 import UNetPlusPlusResNet34

class ResidualASPP(nn.Module):
    """ASPP whose residual scale starts at zero for checkpoint compatibility."""

    def __init__(self, channels: int = 512, branch_channels: int = 128) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            self._branch(channels, branch_channels, 1),
            self._branch(channels, branch_channels, 3, dilation=6),
            self._branch(channels, branch_channels, 3, dilation=12),
            self._branch(channels, branch_channels, 3, dilation=18),
        ])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 5, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _branch(channels: int, branch_channels: int, kernel: int, dilation: int = 1) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, branch_channels, kernel, padding=dilation * (kernel // 2),
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        values = [branch(x) for branch in self.branches]
        pooled = self.pool(x)
        pooled = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
        values.append(pooled)
        return x + self.scale * self.project(torch.cat(values, dim=1))

class ASPPUNetPP(UNetPlusPlusResNet34):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__(pretrained=pretrained)
        self.context = ResidualASPP(512, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x0_0 = self.stem(x)
        x1_0 = self.enc1(self.pool(x0_0))
        x2_0 = self.enc2(x1_0)
        x3_0 = self.enc3(x2_0)
        x4_0 = self.context(self.enc4(x3_0))

        x3_1 = self.x3_1(torch.cat([self.up(x4_0, x3_0), x3_0], dim=1))
        x2_1 = self.x2_1(torch.cat([self.up(x3_0, x2_0), x2_0], dim=1))
        x1_1 = self.x1_1(torch.cat([self.up(x2_0, x1_0), x1_0], dim=1))
        x0_1 = self.x0_1(torch.cat([self.up(x1_0, x0_0), x0_0], dim=1))
        x2_2 = self.x2_2(torch.cat([self.up(x3_1, x2_0), x2_0, x2_1], dim=1))
        x1_2 = self.x1_2(torch.cat([self.up(x2_1, x1_0), x1_0, x1_1], dim=1))
        x0_2 = self.x0_2(torch.cat([self.up(x1_1, x0_0), x0_0, x0_1], dim=1))
        x1_3 = self.x1_3(torch.cat([self.up(x2_2, x1_0), x1_0, x1_1, x1_2], dim=1))
        x0_3 = self.x0_3(torch.cat([self.up(x1_2, x0_0), x0_0, x0_1, x0_2], dim=1))
        y = F.interpolate(x0_3, size=input_size, mode="bilinear", align_corners=False)
        stem_skip = F.interpolate(x0_0, size=input_size, mode="bilinear", align_corners=False)
        return self.head(torch.cat([y, stem_skip], dim=1))

def build_model(candidate: str, pretrained: bool, checkpoint: str | None):
    if candidate != 'aspp':
        raise ValueError('This Model349 bundle includes only its ASPP architecture')
    model = ASPPUNetPP(pretrained=False)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model'], strict=True)
    return model
