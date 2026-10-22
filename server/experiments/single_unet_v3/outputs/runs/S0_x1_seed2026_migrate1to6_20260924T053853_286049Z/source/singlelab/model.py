from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

from .common import ROOT


def pretrained_state():
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1
    return torch.hub.load_state_dict_from_url(weights.url, model_dir=str(ROOT / "weights"),
                                            progress=True, check_hash=True, map_location="cpu")


class DecoderBlock(nn.Module):
    def __init__(self, incoming, skip, outgoing):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(incoming + skip, outgoing, 3, padding=1, bias=False),
            nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True))

    def forward(self, x, skip, size):
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.convs(torch.cat((x, skip), 1) if skip is not None else x)


class EfficientUNet(nn.Module):
    """Torchvision B4 + ordinary five-stage U-Net; one segmentation head only."""
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        backbone = efficientnet_b4(weights=None)
        if pretrained:
            backbone.load_state_dict(pretrained_state())
        # Endpoints at strides 2/4/8/16/32, channels 24/32/56/160/448.
        self.encoder = nn.Sequential(*list(backbone.features.children())[:8])
        self.freeze_bn = cfg["model"]["freeze_encoder_bn_stats"]
        self.checkpoint_encoder = cfg["model"]["checkpoint_encoder"]
        if self.checkpoint_encoder and not self.freeze_bn:
            raise ValueError("Checkpoint recomputation requires frozen encoder BN statistics")
        widths = cfg["model"]["decoder_channels"]
        incoming = 448
        blocks = []
        for skip, outgoing in zip((160, 56, 32, 24, 0), widths):
            blocks.append(DecoderBlock(incoming, skip, outgoing))
            incoming = outgoing
        self.decoder = nn.ModuleList(blocks)
        self.head = nn.Conv2d(incoming, 4, 1)
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "freeze_bn", False):
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, x):
        size = x.shape[-2:]
        x = (x - self.mean) / self.std
        endpoints = []
        for i, stage in enumerate(self.encoder):
            x = checkpoint(stage, x, use_reentrant=False) if self.training and self.checkpoint_encoder else stage(x)
            if i in (1, 2, 3, 5, 7):
                endpoints.append(x)
        for block, skip in zip(self.decoder, list(reversed(endpoints[:-1])) + [None]):
            x = block(x, skip, skip.shape[-2:] if skip is not None else size)
        return self.head(x)
