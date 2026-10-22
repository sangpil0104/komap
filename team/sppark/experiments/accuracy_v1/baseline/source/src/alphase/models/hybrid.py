from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 8:
            raise ValueError("out_channels must be divisible by 8")
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.body(x) + self.skip(x))


class DetailEncoder(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        widths = (32, 64, 128, 256, 384)
        self.stem = nn.Sequential(
            ResidualBlock(in_channels, widths[0]),
            ResidualBlock(widths[0], widths[0]),
        )
        self.down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=2, stride=2),
                ResidualBlock(cout, cout),
                ResidualBlock(cout, cout),
            )
            for cin, cout in zip(widths[:-1], widths[1:])
        ])

    def forward(self, x: Tensor) -> list[Tensor]:
        result = [self.stem(x)]
        for block in self.down:
            result.append(block(result[-1]))
        return result  # strides 1,2,4,8,16


def sample_context(
    feature: Tensor,
    boxes_xyxy: Tensor,
    global_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> Tensor:
    """Sample per-item boxes in padded global-image edge coordinates."""
    batch = feature.shape[0]
    if feature.ndim != 4 or boxes_xyxy.shape != (batch, 4):
        raise ValueError("Expected feature [B,C,Hf,Wf] and boxes [B,4]")
    gh, gw = global_hw
    oh, ow = output_hw
    if min(gh, gw, oh, ow) <= 0:
        raise ValueError("Spatial dimensions must be positive")

    device = feature.device
    box = boxes_xyxy.to(device=device, dtype=torch.float32)
    if not torch.isfinite(box).all() or (box[:, 2:] <= box[:, :2]).any():
        raise ValueError("Boxes must be finite and have positive width/height")

    # Keep coordinates and interpolation in float32, even with AMP enabled.
    with torch.autocast(device_type=device.type, enabled=False):
        ux = (torch.arange(ow, device=device, dtype=torch.float32) + 0.5) / ow
        uy = (torch.arange(oh, device=device, dtype=torch.float32) + 0.5) / oh
        x = box[:, 0, None] + (box[:, 2] - box[:, 0])[:, None] * ux[None]
        y = box[:, 1, None] + (box[:, 3] - box[:, 1])[:, None] * uy[None]
        gx = (2.0 * x / gw - 1.0)[:, None, :].expand(batch, oh, ow)
        gy = (2.0 * y / gh - 1.0)[:, :, None].expand(batch, oh, ow)
        grid = torch.stack((gx, gy), dim=-1)
        return F.grid_sample(
            feature.float(), grid, mode="bilinear",
            padding_mode="border", align_corners=False,
        )


class PhaseHybrid(nn.Module):
    """V1 core. Inputs must already be normalized, aligned and padded."""
    def __init__(
        self,
        backbone: nn.Module,
        layer_indices: Sequence[int],
        detail_in_channels: int = 3,
        context_channels: int = 256,
    ) -> None:
        super().__init__()
        indices = tuple(layer_indices)
        if len(indices) != 4 or tuple(sorted(set(indices))) != indices:
            raise ValueError("Provide four distinct ascending block indices")
        if indices[0] < 0 or indices[-1] >= len(backbone.blocks):
            raise ValueError("Block index out of bounds")
        if context_channels % 8:
            raise ValueError("context_channels must be divisible by 8")
        if int(backbone.patch_size) != 16:
            raise ValueError("This reference implementation assumes patch_size=16")

        self.backbone = backbone
        self.indices = indices
        self.detail = DetailEncoder(detail_in_channels)
        dim = int(backbone.embed_dim)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, context_channels, 1, bias=False),
                nn.GroupNorm(8, context_channels),
                nn.GELU(),
            ) for _ in indices
        ])
        self.context_mix = nn.ModuleDict({
            str(stride): nn.Sequential(
                nn.Conv2d(4 * context_channels, context_channels, 1, bias=False),
                nn.GroupNorm(8, context_channels),
                nn.GELU(),
            ) for stride in (4, 8, 16)
        })
        self.fusion = nn.ModuleDict({
            str(stride): ResidualBlock(width + context_channels, width)
            for stride, width in ((4, 128), (8, 256), (16, 384))
        })
        self.up8 = ResidualBlock(384 + 256, 256)
        self.up4 = ResidualBlock(256 + 128, 128)
        self.up2 = ResidualBlock(128 + 64, 64)
        self.up1 = ResidualBlock(64 + 32, 32)
        self.classifier = nn.Conv2d(32, 4, 1)
        self.aux_head = nn.Conv2d(128, 4, 1)
        self.boundary_head = nn.Conv2d(32, 1, 1)
        self.set_backbone_trainable(last_n=0)

    def train(self, mode: bool = True) -> PhaseHybrid:
        super().train(mode)
        # Deterministic backbone mode; parameter gradients are controlled below.
        self.backbone.eval()
        return self

    def set_backbone_trainable(self, last_n: int) -> None:
        depth = len(self.backbone.blocks)
        if last_n < -1 or last_n > depth:
            raise ValueError("last_n must be -1 or within [0, depth]")
        self.backbone.requires_grad_(False)
        if last_n == -1:
            self.backbone.requires_grad_(True)
        elif last_n > 0:
            for block in self.backbone.blocks[-last_n:]:
                block.requires_grad_(True)
            for name in ("norm", "cls_norm"):
                norm = getattr(self.backbone, name, None)
                if isinstance(norm, nn.Module):
                    norm.requires_grad_(True)
        self.backbone.eval()

    def encode_context(self, image_dino: Tensor) -> list[Tensor]:
        if image_dino.ndim != 4 or image_dino.shape[1] != 3:
            raise ValueError("DINO input must be [B,3,H,W]")
        h, w = image_dino.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError("Pad global input dimensions to multiples of 32")
        needs_grad = any(p.requires_grad for p in self.backbone.parameters())
        guard = nullcontext() if needs_grad else torch.no_grad()
        with guard:
            features = self.backbone.get_intermediate_layers(
                image_dino, n=list(self.indices), reshape=True, norm=True,
                return_class_token=False, return_extra_tokens=False,
            )
        if len(features) != 4:
            raise RuntimeError("Backbone returned an unexpected number of features")
        for f in features:
            if f.shape[-2:] != (h // 16, w // 16):
                raise RuntimeError("Unexpected DINO feature grid")
        # Deliberately outside no_grad: projection parameters must learn in S0.
        return [project(f) for project, f in zip(self.projections, features)]

    @staticmethod
    def _up(x: Tensor, skip: Tensor) -> Tensor:
        return F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False,
        )

    def decode_local(
        self,
        image_detail: Tensor,
        projected_context: list[Tensor],
        boxes_xyxy: Tensor,
        global_hw: tuple[int, int],
    ) -> dict[str, Tensor]:
        if image_detail.ndim != 4:
            raise ValueError("Detail input must be [B,C,H,W]")
        if any(size % 32 for size in image_detail.shape[-2:]):
            raise ValueError("Pad local input dimensions to multiples of 32")
        if len(projected_context) != 4 or any(
            f.shape[0] != image_detail.shape[0] for f in projected_context
        ):
            raise ValueError("Context and detail batches must match")
        c1, c2, c4, c8, c16 = self.detail(image_detail)
        fused = {}
        for stride, local_feature in ((4, c4), (8, c8), (16, c16)):
            aligned = [
                sample_context(p, boxes_xyxy, global_hw, local_feature.shape[-2:])
                for p in projected_context
            ]
            context = self.context_mix[str(stride)](torch.cat(aligned, dim=1))
            fused[stride] = self.fusion[str(stride)](
                torch.cat((local_feature, context), dim=1)
            )
        d8 = self.up8(torch.cat((self._up(fused[16], fused[8]), fused[8]), dim=1))
        d4 = self.up4(torch.cat((self._up(d8, fused[4]), fused[4]), dim=1))
        d2 = self.up2(torch.cat((self._up(d4, c2), c2), dim=1))
        d1 = self.up1(torch.cat((self._up(d2, c1), c1), dim=1))
        return {
            "logits": self.classifier(d1),
            "aux_logits": self.aux_head(d4),
            "boundary_logits": self.boundary_head(d1),
        }

    def forward(
        self,
        image_detail: Tensor,
        image_dino: Tensor,
        boxes_xyxy: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if image_detail.shape[0] != image_dino.shape[0]:
            raise ValueError("Context and detail batches must match")
        gh, gw = image_dino.shape[-2:]
        if boxes_xyxy is None:
            if image_detail.shape[-2:] != (gh, gw):
                raise ValueError("Local mode requires explicit global-coordinate boxes")
            boxes_xyxy = image_dino.new_tensor([0, 0, gw, gh], dtype=torch.float32)
            boxes_xyxy = boxes_xyxy.unsqueeze(0).repeat(image_dino.shape[0], 1)
        projected = self.encode_context(image_dino)
        return self.decode_local(image_detail, projected, boxes_xyxy, (gh, gw))
