# R ensemble inference dependency.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision.models import ResNet34_Weights

from torchvision.models import resnet34

from unet_resnet34 import ConvBlock

from unet_resnet34 import NUM_CLASSES

class UNetPlusPlusResNet34(nn.Module):
    """Compact nested U-Net++ decoder using the ResNet34 feature pyramid."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = resnet34(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.enc1 = encoder.layer1
        self.enc2 = encoder.layer2
        self.enc3 = encoder.layer3
        self.enc4 = encoder.layer4

        self.x3_1 = ConvBlock(512 + 256, 256)
        self.x2_1 = ConvBlock(256 + 128, 128)
        self.x1_1 = ConvBlock(128 + 64, 64)
        self.x0_1 = ConvBlock(64 + 64, 64)
        self.x2_2 = ConvBlock(256 + 128 + 128, 128)
        self.x1_2 = ConvBlock(128 + 64 + 64, 64)
        self.x0_2 = ConvBlock(64 + 64 + 64, 64)
        self.x1_3 = ConvBlock(128 + 64 + 64 + 64, 64)
        self.x0_3 = ConvBlock(64 + 64 + 64 + 64, 64)
        self.head = nn.Sequential(ConvBlock(64 + 64, 32), nn.Conv2d(32, NUM_CLASSES, 1))

    @staticmethod
    def up(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x0_0 = self.stem(x)
        x1_0 = self.enc1(self.pool(x0_0))
        x2_0 = self.enc2(x1_0)
        x3_0 = self.enc3(x2_0)
        x4_0 = self.enc4(x3_0)

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
