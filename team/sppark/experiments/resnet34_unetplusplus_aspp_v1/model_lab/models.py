from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    """Convolution block with GroupNorm and GELU for stable training across micro-batch sizes."""
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        groups = min(num_groups, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualASPP(nn.Module):
    """Residual Atrous Spatial Pyramid Pooling (residual ASPP) bottleneck module.
    
    Provides multi-scale receptive field contextual feature extraction at the lowest resolution,
    with an explicit residual connection to preserve backbone feature fidelity.
    """
    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 512,
        width: int = 128,
        rates: tuple[int, ...] = (1, 2, 4, 8),
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        groups = min(num_groups, width)
        while width % groups != 0 and groups > 1:
            groups -= 1

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.GELU(),
        )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(groups, width),
                nn.GELU(),
            )
            for d in rates
        ])

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, width, kernel_size=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.GELU(),
        )

        total_width = (len(rates) + 1) * width
        proj_groups = min(num_groups, out_channels)
        while out_channels % proj_groups != 0 and proj_groups > 1:
            proj_groups -= 1

        self.project = nn.Sequential(
            nn.Conv2d(total_width, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(proj_groups, out_channels),
            nn.GELU(),
        )

        self.res_connection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        pool = self.global_pool(reduced)
        pool = F.interpolate(pool, size=x.shape[-2:], mode="bilinear", align_corners=False)
        branch_outs = [b(reduced) for b in self.branches] + [pool]
        concatenated = torch.cat(branch_outs, dim=1)
        return self.res_connection(x) + self.project(concatenated)


class ResNet34UNetPlusPlusASPP(nn.Module):
    """ResNet34 U-Net++ with Residual ASPP bottleneck.
    
    Structure:
    - Backbone: Pretrained ResNet-34 encoder with multi-scale skip stages.
    - Stride-1 Stem: High-resolution stem retaining native pixel-level details (stride 1).
    - Bottleneck Context: Residual ASPP module on stage 5 (stride 32).
    - Nested Decoder: Dense nested skip pathways (U-Net++ X_{i, j} grid) spanning strides 32 -> 1.
    - Head: 1x1 classification head outputting 4-class logits at native resolution.
    - Deep Supervision: Optional intermediate classification heads at X_{0, 1}, X_{0, 2}, X_{0, 3}, X_{0, 4}.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        stem_channels: int = 32,
        aspp_width: int = 128,
        aspp_rates: tuple[int, ...] = (1, 2, 4, 8),
        pretrained: bool = True,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # Load ResNet-34 backbone
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Level 0 (Stride 1, H x W): Native-resolution Stem
        self.stem = ConvBlock(in_channels, stem_channels)

        # Encoder Levels from ResNet34:
        # Level 1 (Stride 2, H/2 x W/2): conv1 (64ch)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        # Level 2 (Stride 4, H/4 x W/4): maxpool + layer1 (64ch)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        # Level 3 (Stride 8, H/8 x W/8): layer2 (128ch)
        self.enc3 = resnet.layer2
        # Level 4 (Stride 16, H/16 x W/16): layer3 (256ch)
        self.enc4 = resnet.layer3
        # Level 5 (Stride 32, H/32 x W/32): layer4 (512ch)
        self.enc5 = resnet.layer4

        # Residual ASPP at Level 5 bottleneck
        self.aspp = ResidualASPP(
            in_channels=512,
            out_channels=512,
            width=aspp_width,
            rates=aspp_rates,
        )

        c0 = stem_channels  # 32
        c1 = 64
        c2 = 64
        c3 = 128
        c4 = 256
        c5 = 512

        # U-Net++ Dense Nested Decoder Nodes:
        # Column j = 1
        self.conv4_1 = ConvBlock(c4 + c5, c4)
        self.conv3_1 = ConvBlock(c3 + c4, c3)
        self.conv2_1 = ConvBlock(c2 + c3, c2)
        self.conv1_1 = ConvBlock(c1 + c2, c1)
        self.conv0_1 = ConvBlock(c0 + c1, c0)

        # Column j = 2
        self.conv3_2 = ConvBlock(c3 * 2 + c4, c3)
        self.conv2_2 = ConvBlock(c2 * 2 + c3, c2)
        self.conv1_2 = ConvBlock(c1 * 2 + c2, c1)
        self.conv0_2 = ConvBlock(c0 * 2 + c1, c0)

        # Column j = 3
        self.conv2_3 = ConvBlock(c2 * 3 + c3, c2)
        self.conv1_3 = ConvBlock(c1 * 3 + c2, c1)
        self.conv0_3 = ConvBlock(c0 * 3 + c1, c0)

        # Column j = 4
        self.conv1_4 = ConvBlock(c1 * 4 + c2, c1)
        self.conv0_4 = ConvBlock(c0 * 4 + c1, c0)

        # Column j = 5
        self.conv0_5 = ConvBlock(c0 * 5 + c1, c0)

        # Final Segmentation Head
        self.final_head = nn.Conv2d(c0, num_classes, kernel_size=1)

        # Deep Supervision Heads
        if self.deep_supervision:
            self.head0_1 = nn.Conv2d(c0, num_classes, kernel_size=1)
            self.head0_2 = nn.Conv2d(c0, num_classes, kernel_size=1)
            self.head0_3 = nn.Conv2d(c0, num_classes, kernel_size=1)
            self.head0_4 = nn.Conv2d(c0, num_classes, kernel_size=1)

    def _up(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        # Encoder passes
        x0_0 = self.stem(x)                # (B, 32, H, W)
        x1_0 = self.enc1(x)                # (B, 64, H/2, W/2)
        x2_0 = self.enc2(x1_0)             # (B, 64, H/4, W/4)
        x3_0 = self.enc3(x2_0)             # (B, 128, H/8, W/8)
        x4_0 = self.enc4(x3_0)             # (B, 256, H/16, W/16)
        x5_0 = self.aspp(self.enc5(x4_0))  # (B, 512, H/32, W/32) with residual ASPP

        # Nested Decoder Column j = 1
        x4_1 = self.conv4_1(torch.cat([x4_0, self._up(x5_0, x4_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))

        # Nested Decoder Column j = 2
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self._up(x4_1, x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))

        # Nested Decoder Column j = 3
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self._up(x3_2, x2_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))

        # Nested Decoder Column j = 4
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self._up(x2_3, x1_0)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1))

        # Nested Decoder Column j = 5
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self._up(x1_4, x0_0)], dim=1))

        # Final Logits
        logits = self.final_head(x0_5)

        result: dict[str, torch.Tensor | list[torch.Tensor]] = {"logits": logits}
        if self.deep_supervision and self.training:
            result["aux_logits"] = [
                self.head0_1(x0_1),
                self.head0_2(x0_2),
                self.head0_3(x0_3),
                self.head0_4(x0_4),
            ]
        return result

    def get_parameter_groups(
        self,
        encoder_lr: float,
        decoder_lr: float,
        weight_decay: float = 1e-4,
    ) -> list[dict]:
        """Separates backbone encoder parameters and new decoder/ASPP parameters for differential learning rates."""
        encoder_params = []
        decoder_params = []

        encoder_modules = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]
        encoder_param_ids = set()
        for mod in encoder_modules:
            for p in mod.parameters():
                encoder_param_ids.add(id(p))
                encoder_params.append(p)

        for p in self.parameters():
            if id(p) not in encoder_param_ids:
                decoder_params.append(p)

        return [
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": decoder_params, "lr": decoder_lr, "weight_decay": weight_decay},
        ]


def build_model(cfg: dict) -> ResNet34UNetPlusPlusASPP:
    """Instantiate ResNet34 U-Net++ + residual ASPP from config."""
    m_cfg = cfg.get("model", {})
    return ResNet34UNetPlusPlusASPP(
        in_channels=m_cfg.get("in_channels", 3),
        num_classes=m_cfg.get("num_classes", 4),
        stem_channels=m_cfg.get("stem_channels", 32),
        aspp_width=m_cfg.get("aspp_width", 128),
        aspp_rates=tuple(m_cfg.get("aspp_rates", (1, 2, 4, 8))),
        pretrained=m_cfg.get("pretrained", True),
        deep_supervision=m_cfg.get("deep_supervision", False),
    )

