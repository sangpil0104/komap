import torch
from torch import nn
import torch.nn.functional as F

from alphase.models.hybrid import DetailEncoder, ResidualBlock


class PhaseCNN(nn.Module):
    """README E01 baseline, stride-1 skip retained through the output head."""
    def __init__(self, in_channels=3):
        super().__init__()
        self.detail = DetailEncoder(in_channels)
        self.up8 = ResidualBlock(384+256, 256)
        self.up4 = ResidualBlock(256+128, 128)
        self.up2 = ResidualBlock(128+64, 64)
        self.up1 = ResidualBlock(64+32, 32)
        self.classifier = nn.Conv2d(32, 4, 1)
        self.aux_head = nn.Conv2d(128, 4, 1)
        self.boundary_head = nn.Conv2d(32, 1, 1)

    def forward(self, detail, dino=None, box=None):
        if any(s % 32 for s in detail.shape[-2:]):
            raise ValueError("Pad detail inputs to multiples of 32")
        c1, c2, c4, c8, c16 = self.detail(detail)
        def up(x, skip, block):
            return block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        d8 = up(c16, c8, self.up8)
        d4 = up(d8, c4, self.up4)
        d2 = up(d4, c2, self.up2)
        d1 = up(d2, c1, self.up1)
        return dict(logits=self.classifier(d1), aux_logits=self.aux_head(d4), boundary_logits=self.boundary_head(d1))
