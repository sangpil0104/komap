#!/usr/bin/env python3
"""KoMaP guide section 7.2-7.4 experiments.

72: DeepLabV3+ with a pretrained ResNet34 encoder (SMP)
73: HRNet-W18 multi-resolution encoder with an FPN-style decoder (timm)
74: Mamba-UNet-style model: ResNet34 U-Net with gated state-space blocks
    at the bottleneck. The portable block avoids fused CUDA kernels, which
    keeps it portable across GPU/CUDA versions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models import ResNet34_Weights, resnet34

from unet_resnet34 import (
    CLASS_COLORS,
    CLASS_NAMES,
    NUM_CLASSES,
    ConvBlock,
    KoMaPDataset,
    combined_loss,
    encode_mask,
    evaluate,
    seed_everything,
)


class DeepLabV3Plus(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class HRNetW18FPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import timm

        self.encoder = timm.create_model("hrnet_w18", pretrained=True, features_only=True)
        channels = list(self.encoder.feature_info.channels())
        self.projections = nn.ModuleList([nn.Conv2d(c, 64, 1, bias=False) for c in channels])
        self.fuse = ConvBlock(64, 64)
        self.head = nn.Conv2d(64, NUM_CLASSES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        target_size = features[0].shape[-2:]
        fused = 0.0
        for feature, projection in zip(features, self.projections):
            fused = fused + F.interpolate(
                projection(feature), size=target_size, mode="bilinear", align_corners=False
            )
        logits = self.head(self.fuse(fused))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


class PortableMambaBlock(nn.Module):
    """A fused-kernel-free gated state-space-style vision block.

    It uses channel-last normalization, gated projections, and depthwise local
    mixing. This is deliberately portable across GPUs and is labeled as a
    Mamba-UNet-style baseline rather than claiming the fused mamba_ssm kernel.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, channels * 2)
        self.depthwise = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.out_proj = nn.Linear(channels, channels)
        self.scale = nn.Parameter(torch.ones(channels) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.permute(0, 2, 3, 1)
        y = self.norm(y)
        y = self.in_proj(y).permute(0, 3, 1, 2)
        y = self.depthwise(y)
        value, gate = y.chunk(2, dim=1)
        y = (value * torch.sigmoid(gate)).permute(0, 2, 3, 1)
        y = self.out_proj(y).permute(0, 3, 1, 2)
        return residual + y * self.scale.view(1, -1, 1, 1)


class MambaUNetStyle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.enc1 = encoder.layer1
        self.enc2 = encoder.layer2
        self.enc3 = encoder.layer3
        self.enc4 = encoder.layer4
        self.mamba1 = PortableMambaBlock(512)
        self.mamba2 = PortableMambaBlock(512)
        self.dec4 = ConvBlock(512 + 256, 256)
        self.dec3 = ConvBlock(256 + 128, 128)
        self.dec2 = ConvBlock(128 + 64, 64)
        self.dec1 = ConvBlock(64 + 64, 64)
        self.head = nn.Sequential(ConvBlock(64 + 64, 32), nn.Conv2d(32, NUM_CLASSES, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        s0 = self.stem(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        y = self.mamba2(self.mamba1(s4))
        y = F.interpolate(y, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec4(torch.cat([y, s3], dim=1))
        y = F.interpolate(y, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec3(torch.cat([y, s2], dim=1))
        y = F.interpolate(y, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(torch.cat([y, s1], dim=1))
        y = F.interpolate(y, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec1(torch.cat([y, s0], dim=1))
        y = F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        stem_skip = F.interpolate(s0, size=input_size, mode="bilinear", align_corners=False)
        return self.head(torch.cat([y, stem_skip], dim=1))


def build_model(model_id: int) -> nn.Module:
    if model_id == 72:
        return DeepLabV3Plus()
    if model_id == 73:
        return HRNetW18FPN()
    if model_id == 74:
        return MambaUNetStyle()
    raise ValueError(model_id)


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    count = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = combined_loss(logits, masks)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.item()) * images.shape[0]
        count += images.shape[0]
    return total_loss / max(count, 1)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_ds = KoMaPDataset(args.data_dir, "train", args.image_size, augment=True)
    valid_ds = KoMaPDataset(args.data_dir, "valid", args.image_size, augment=False)
    batch_size = args.batch_size or 4
    loader_args = dict(
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_args)
    valid_loader = DataLoader(valid_ds, shuffle=False, drop_last=False, **loader_args)
    print(
        f"model={args.model_id} device={device} logical_gpu="
        f"{torch.cuda.current_device() if device.type == 'cuda' else 'cpu'} "
        f"amp={use_amp} train={len(train_ds)} valid={len(valid_ds)} batch={batch_size}",
        flush=True,
    )
    model = build_model(args.model_id).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"parameters={sum(p.numel() for p in model.parameters()):,} "
        f"trainable={sum(p.numel() for p in trainable):,}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: List[dict] = []
    best_miou = -1.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        valid = evaluate(model, valid_loader, device, use_amp)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "valid": valid,
            "epoch_seconds": time.time() - start,
        }
        history.append(record)
        with (out_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": valid,
            "args": vars(args),
            "class_names": CLASS_NAMES,
            "class_colors": CLASS_COLORS,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        miou = float(valid["miou"])
        print(
            f"epoch={epoch:03d}/{args.epochs} train_loss={train_loss:.4f} "
            f"valid_loss={float(valid['loss']):.4f} miou={miou:.4f} "
            f"iou=" + ", ".join(f"{v:.3f}" for v in valid["iou"].values()) +
            f" time={record['epoch_seconds']:.1f}s",
            flush=True,
        )
        if miou > best_miou:
            best_miou = miou
            torch.save(checkpoint, out_dir / "best.pt")
            print(f"saved best.pt (mIoU={best_miou:.6f})", flush=True)
    summary = {
        "model_id": args.model_id,
        "best_miou": best_miou,
        "epochs_completed": len(history),
        "elapsed_seconds": time.time() - start_time,
        "best_checkpoint": str(out_dir / "best.pt"),
        "class_names": CLASS_NAMES,
        "class_colors": CLASS_COLORS,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


@torch.no_grad()
def predict_test(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(args.model_id).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    out_dir = Path(args.output_dir) / "test_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        resized = image.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with autocast(enabled=use_amp):
            pred = model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
        encode_mask(pred).resize(original_size, Image.Resampling.NEAREST).save(
            out_dir / image_path.name.replace("_image.png", "_mask.png")
        )
    print(f"wrote {len(list(out_dir.glob('*.png')))} test masks to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=int, required=True, choices=[72, 73, 74])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.predict_test and not args.checkpoint:
        parser.error("--predict-test requires --checkpoint")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.predict_test:
        predict_test(parsed)
    else:
        train(parsed)
