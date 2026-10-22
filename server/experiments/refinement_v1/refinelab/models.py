from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from alphase.models.cnn import PhaseCNN
from alphase.data.palette import PALETTE
from alphase.data.splits import select_rows
from alphase.utils import sha256


class ResidualASPP(nn.Module):
    """Lightweight parallel atrous context, initialized to an exact identity."""
    def __init__(self, channels=384, width=64):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(channels, width, 1, bias=False), nn.GroupNorm(8, width), nn.GELU())
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width, 3, padding=d, dilation=d, groups=width, bias=False),
                          nn.Conv2d(width, width, 1, bias=False), nn.GroupNorm(8, width), nn.GELU())
            for d in (1, 2, 4, 8)])
        self.project = nn.Conv2d(4 * width, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        reduced = self.reduce(x)
        return x + self.project(torch.cat([branch(reduced) for branch in self.branches], dim=1))


class ContextPhaseCNN(PhaseCNN):
    def __init__(self, in_channels=3):
        super().__init__(in_channels)
        self.context = ResidualASPP()

    def forward(self, detail, dino=None, box=None):
        if any(s % 32 for s in detail.shape[-2:]):
            raise ValueError("Pad detail inputs to multiples of 32")
        c1, c2, c4, c8, c16 = self.detail(detail)
        def up(x, skip, block):
            return block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        d8 = up(self.context(c16), c8, self.up8)
        d4 = up(d8, c4, self.up4)
        d2 = up(d4, c2, self.up2)
        d1 = up(d2, c1, self.up1)
        return dict(logits=self.classifier(d1), aux_logits=self.aux_head(d4), boundary_logits=self.boundary_head(d1))


def validate_parent(state, cfg):
    train, valid = select_rows(cfg, "official")
    train_ids, valid_ids = {r["image_id"] for r in train}, {r["image_id"] for r in valid}
    if set(state["train_ids"]) != train_ids or set(state["validation_ids"]) != valid_ids or train_ids & valid_ids:
        raise ValueError("Parent checkpoint must use the same official train/validation IDs")
    if state["palette"] != PALETTE.tolist() or state["resolved_config"]["data"]["classes"] != cfg["data"]["classes"]:
        raise ValueError("Parent class/color mapping mismatch")
    if state["resolved_config"]["model"]["name"] != "cnn_unet":
        raise ValueError("Parent must be the preserved CNN")


def build_model(cfg, load_pretrained=True):
    name = cfg["model"]["name"]
    if name not in ("cnn_unet", "cnn_aspp_v1"):
        raise ValueError(f"Unknown refinement model: {name}")
    model = (ContextPhaseCNN if name == "cnn_aspp_v1" else PhaseCNN)(len(cfg["data"]["image_channels"]))
    if load_pretrained:
        ref = cfg["refinement"]
        if sha256(ref["parent_checkpoint"]) != ref["parent_sha256"]:
            raise ValueError("Parent weights changed")
        state = torch.load(ref["parent_checkpoint"], map_location="cpu", weights_only=False)
        validate_parent(state, cfg)
        selected = state["ema_model"] if state["selected_weights"] == "ema" else state["model"]
        missing, unexpected = model.load_state_dict(selected, strict=False)
        expected = {n for n in model.state_dict() if n.startswith("context.")}
        if set(missing) != expected or unexpected:
            raise ValueError(f"Unexpected parent transfer: missing={missing}, unexpected={unexpected}")
    if not cfg["loss"]["auxiliary"]:
        model.aux_head.requires_grad_(False)
    if not cfg["loss"]["boundary"]:
        model.boundary_head.requires_grad_(False)
    return model
