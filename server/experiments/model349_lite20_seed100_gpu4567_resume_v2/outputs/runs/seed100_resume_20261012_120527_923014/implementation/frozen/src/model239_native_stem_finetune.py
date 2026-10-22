# Model349 inference dependency; extracted from the recorded source snapshot.

# Unused training/search entrypoints are intentionally omitted.

from __future__ import annotations


import torch

import torch.nn as nn

import torch.nn.functional as F

from model181_raw_zcdf_aspp import TeamRepresentationASPP

class Stride1NativeStemASPP(TeamRepresentationASPP):
    """Keep the pretrained path intact; inject full-resolution features at the head."""

    def __init__(self) -> None:
        super().__init__(pretrained=False)
        self.native_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.native_to_head = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(self.native_to_head.weight)

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

        decoded = F.interpolate(x0_3, size=input_size, mode="bilinear", align_corners=False)
        stem_skip = F.interpolate(x0_0, size=input_size, mode="bilinear", align_corners=False)
        joined = torch.cat([decoded, stem_skip], dim=1)
        head_block = self.head[0].block
        hidden = head_block[0](joined) + self.native_to_head(self.native_stem(x))
        hidden = head_block[2](head_block[1](hidden))
        hidden = head_block[5](head_block[4](head_block[3](hidden)))
        return self.head[1](hidden)
