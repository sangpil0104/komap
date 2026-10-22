from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .common import ROOT, digest


def prepare_weights(download=False):
    weights = ResNet18_Weights.IMAGENET1K_V1
    path = ROOT / "weights" / Path(weights.url).name
    if not path.exists() and download:
        torch.hub.load_state_dict_from_url(weights.url, model_dir=str(path.parent),
                                          check_hash=True, map_location="cpu")
    if not path.exists():
        raise FileNotFoundError(f"Run prepare first to download pretrained weights: {path}")
    expected_prefix = path.stem.split("-")[-1]
    checksum = digest(path)
    if not checksum.startswith(expected_prefix):
        raise ValueError("ResNet18 ImageNet weight checksum mismatch")
    return path, checksum


class Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        net = resnet18(weights=None)
        if pretrained:
            path, _ = prepare_weights()
            net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])

    def forward(self, x):
        x = self.stem(x)
        features = [x]
        x = self.pool(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


def convs(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(8, out_channels), nn.ReLU(inplace=True),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(8, out_channels), nn.ReLU(inplace=True))


class SmallUNet(nn.Module):
    """One ResNet18 encoder, native-resolution skip, and one output head."""
    def __init__(self, outputs=4, pretrained=False, freeze_bn=True):
        super().__init__()
        self.encoder = Encoder(pretrained)
        self.full_resolution = convs(3, 16)
        self.decoder = nn.ModuleList([convs(512 + 256, 128), convs(128 + 128, 64),
                                      convs(64 + 64, 48), convs(48 + 64, 32), convs(32 + 16, 16)])
        self.head = nn.Conv2d(16, outputs, 1)
        self.freeze_bn = freeze_bn
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for layer in self.encoder.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def forward(self, image):
        x = (image - self.mean) / self.std
        features = self.encoder(x)
        x = features[-1]
        for block, skip in zip(self.decoder, list(reversed(features[:-1])) + [self.full_resolution(image)]):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


def transfer_encoder(destination, source_state):
    state = {key[len("encoder."):]: value for key, value in source_state.items() if key.startswith("encoder.")}
    destination.encoder.load_state_dict(state, strict=True)
