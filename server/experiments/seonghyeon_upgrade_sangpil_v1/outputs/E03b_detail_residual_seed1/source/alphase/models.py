import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .contracts import align_context_features, d4_forward, d4_inverse, relative_features
from .io import local_path


def conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.SiLU(inplace=True),
    )


def base_model():
    return smp.Unet(
        encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=4
    )


class TextureStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.full = conv(3, 16)
        self.half_block = conv(16, 32)
        self.quarter = conv(32, 64)

    def forward(self, x):
        a = self.full(x)
        b = self.half_block(F.avg_pool2d(a, 2))
        c = self.quarter(F.avg_pool2d(b, 2))
        return a, b, c


class ContextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [conv(1, 32), conv(32, 64), conv(64, 128), conv(128, 256)]
        )
        self.project8 = nn.Conv2d(128, 32, 1)
        self.project16 = nn.Conv2d(256, 32, 1)

    def forward(self, x):
        out = []
        for block in self.blocks:
            x = block(F.avg_pool2d(x, 2))
            out.append(x)
        return self.project8(out[2]), self.project16(out[3])


class EnhancedB4(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.base = base_model()
        self.flags = cfg["model"]["new_branches"].copy()
        self.freeze_base = False
        self.freeze_bn = cfg["training"]["freeze_bn"]
        self.texture = TextureStem() if self.flags["texture"] else None
        self.context = ContextCNN() if self.flags["context"] else None
        if self.flags["residual_head"]:
            detail_channels = self.base.segmentation_head[0].in_channels
            self.detail_projection = nn.Conv2d(detail_channels, 16, 1)
            self.fuse_quarter = conv(
                16
                + 64 * int(self.texture is not None)
                + 64 * int(self.context is not None),
                64,
            )
            self.fuse_half = conv(64 + 16 + 32 * int(self.texture is not None), 32)
            self.fuse_full = conv(32 + 16 + 16 * int(self.texture is not None), 16)
            self.residual = nn.Conv2d(16, 4, 1)
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)
        else:
            self.residual = None

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base:
            self.base.eval()
        elif self.freeze_bn:
            for m in self.base.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def set_base_frozen(self, value):
        self.freeze_base = value
        for p in self.base.parameters():
            p.requires_grad_(not value)
        self.train(self.training)

    def forward_features(self, x):
        return self.base.decoder(self.base.encoder(x))

    def forward_logits(self, x):
        return self.base.segmentation_head(self.forward_features(x))

    def forward(
        self,
        detail_rgb,
        texture_detail=None,
        context_gray=None,
        detail_rects=None,
        context_rects=None,
        d4=None,
    ):
        features = self.forward_features(detail_rgb)
        base = self.base.segmentation_head(features)
        if self.residual is None:
            return {
                "logits_detail": base,
                "base_logits_detail": base,
                "delta_logits_detail": None,
                "aux": {},
            }
        detail = self.detail_projection(features)
        h, w = detail.shape[-2:]
        half = (h // 2, w // 2)
        quarter = (h // 4, w // 4)
        pieces = [
            F.interpolate(detail, size=quarter, mode="bilinear", align_corners=False)
        ]
        tx = None
        if self.texture is not None:
            if texture_detail is None:
                raise ValueError("texture_detail required")
            tx = self.texture(texture_detail)
            pieces.append(tx[2])
        if self.context is not None:
            if context_gray is None or detail_rects is None or context_rects is None:
                raise ValueError("Context and rectangles required")
            # Undo shared D4 on feature maps before native alignment, then apply it
            # back. This also works for off-center D inside C, unlike center-crop hacks.
            for feature in self.context(context_gray):
                aligned = []
                for i, (k, flip) in enumerate(d4 or [(0, False)] * len(detail_rgb)):
                    canonical = d4_inverse(feature[i : i + 1], k, flip)
                    f = align_context_features(
                        canonical,
                        detail_rects[i : i + 1],
                        context_rects[i : i + 1],
                        quarter,
                    )
                    aligned.append(d4_forward(f, k, flip))
                pieces.append(torch.cat(aligned))
        f = self.fuse_quarter(torch.cat(pieces, 1))
        pieces = [
            F.interpolate(f, size=half, mode="bilinear", align_corners=False),
            F.interpolate(detail, size=half, mode="bilinear", align_corners=False),
        ]
        if tx is not None:
            pieces.append(tx[1])
        f = self.fuse_half(torch.cat(pieces, 1))
        pieces = [
            F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False),
            detail,
        ]
        if tx is not None:
            pieces.append(tx[0])
        delta = self.residual(self.fuse_full(torch.cat(pieces, 1)))
        return {
            "logits_detail": base + delta,
            "base_logits_detail": base,
            "delta_logits_detail": delta,
            "aux": {},
        }


class NativeRefiner(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.enc = nn.ModuleList(
            [conv(8, 32), conv(32, 64), conv(64, 128), conv(128, 256)]
        )
        self.dec = nn.ModuleList(
            [conv(256 + 128, 128), conv(128 + 64, 64), conv(64 + 32, 32)]
        )
        self.head = nn.Conv2d(32, 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, gray, prob, relative=None):
        safe = prob.float().clamp_min(self.eps)
        safe = safe / safe.sum(1, keepdim=True)
        if relative is None:
            relative = relative_features(gray)[:, 1:]
        entropy = -(safe * safe.log()).sum(1, keepdim=True) / 1.3862943611198906
        x = torch.cat((gray, safe, relative, entropy), 1)
        skips = []
        for i, block in enumerate(self.enc):
            x = block(F.avg_pool2d(x, 2) if i else x)
            skips.append(x)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            x = block(
                torch.cat(
                    (
                        F.interpolate(
                            x,
                            size=skip.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                        skip,
                    ),
                    1,
                )
            )
        return safe.log() + self.head(x).float()


def load_model(cfg, checkpoint=None, device="cpu", imagenet=False):
    model = EnhancedB4(cfg)
    if checkpoint:
        saved = torch.load(
            local_path(checkpoint), map_location="cpu", weights_only=True
        )
        state = saved.get("model", saved)
        if local_path(checkpoint).name.startswith("best_ema"):
            if saved.get("ema") is None:
                raise ValueError("EMA checkpoint lacks EMA weights")
            state = saved["ema"]
        if "config" in saved and saved["config"]["classes"] != cfg["classes"]:
            raise ValueError("Checkpoint class metadata mismatch")
        if state and all(k.startswith("base.") for k in state):
            # An audited base-only wrapper can initialize new zero-init modules.
            model.base.load_state_dict(
                {k.removeprefix("base."): v for k, v in state.items()}, strict=True
            )
        elif any(k.startswith("base.") for k in state):
            model.load_state_dict(state, strict=True)
        else:
            model.base.load_state_dict(state, strict=True)
    elif imagenet:
        model.base.encoder.load_state_dict(
            torch.load(
                local_path(cfg["paths"]["encoder_weights"]),
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
    return model.to(device)
