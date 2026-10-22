#!/usr/bin/env python3
"""KoMaP models 2-6 under the same fixed train/valid protocol as model 1.

Model mapping:
  2: U-Net + ConvNeXt-Tiny
  3: SegFormer-B0
  4: DINOv2-Small frozen encoder + segmentation decoder
  5: SAM2.1 Hiera-Tiny frozen image encoder + segmentation decoder
  6: U-Net++ + ResNet34
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet34_Weights, convnext_tiny, resnet34

from model1_unet_resnet34 import (
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


class ConvNeXtTinyUNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = convnext_tiny(weights=weights)
        self.features = self.encoder.features
        self.dec4 = ConvBlock(768 + 384, 384)
        self.dec3 = ConvBlock(384 + 192, 192)
        self.dec2 = ConvBlock(192 + 96, 96)
        self.dec1 = ConvBlock(96, 64)
        self.head = nn.Conv2d(64, NUM_CLASSES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x0 = self.features[0](x)
        x0 = self.features[1](x0)
        x1 = self.features[2](x0)
        x1 = self.features[3](x1)
        x2 = self.features[4](x1)
        x2 = self.features[5](x2)
        x3 = self.features[6](x2)
        x3 = self.features[7](x3)
        y = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec4(torch.cat([y, x2], dim=1))
        y = F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec3(torch.cat([y, x1], dim=1))
        y = F.interpolate(y, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(torch.cat([y, x0], dim=1))
        y = F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        y = self.dec1(y)
        return self.head(y)


class SegFormerB0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class DINOv2Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from transformers import Dinov2Model

        self.encoder = Dinov2Model.from_pretrained("facebook/dinov2-small")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        hidden = int(self.encoder.config.hidden_size)
        self.decoder = nn.Sequential(
            ConvBlock(hidden, 192),
            ConvBlock(192, 96),
            nn.Conv2d(96, NUM_CLASSES, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder(pixel_values=x).last_hidden_state
        tokens = features[:, 1:, :]
        grid = int(tokens.shape[1] ** 0.5)
        if grid * grid != tokens.shape[1]:
            raise RuntimeError(f"DINOv2 token grid is not square: {tokens.shape}")
        y = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid, grid)
        y = self.decoder(y)
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)


class SAM2Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from transformers import Sam2Model

        self.encoder = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.decoder = nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 64),
            nn.Conv2d(64, NUM_CLASSES, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transformers 5.17's Sam2Model.get_image_embeddings uses view() on
        # non-contiguous FPN tensors. We reproduce the official transformation
        # with reshape(), which is safe for the current SAM2 checkpoint.
        with torch.no_grad():
            output = self.encoder.get_image_features(x, return_dict=True)
            feature = output.fpn_hidden_states[-1]
            feature = feature + self.encoder.no_memory_embedding
            # The FPN tensors returned by this checkpoint are flattened as
            # (H*W, batch, channels). The public helper pairs them with
            # backbone_feature_sizes, but the checkpoint's flattened sizes do
            # not correspond to that list for a 512px input. Infer the square
            # spatial grid directly from the sequence length.
            grid = int(feature.shape[0] ** 0.5)
            if grid * grid != feature.shape[0]:
                raise RuntimeError(f"SAM2 FPN grid is not square: {feature.shape}")
            feature = feature.permute(1, 2, 0).reshape(x.shape[0], -1, grid, grid)
        y = self.decoder(feature)
        return F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)


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


def build_model(model_id: int) -> nn.Module:
    if model_id == 2:
        return ConvNeXtTinyUNet(pretrained=True)
    if model_id == 3:
        return SegFormerB0()
    if model_id == 4:
        return DINOv2Decoder()
    if model_id == 5:
        return SAM2Decoder()
    if model_id == 6:
        return UNetPlusPlusResNet34(pretrained=True)
    raise ValueError(f"model_id must be 2-6, got {model_id}")


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


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "class_names": CLASS_NAMES,
            "class_colors": CLASS_COLORS,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_ds = KoMaPDataset(args.data_dir, "train", args.image_size, augment=True)
    valid_ds = KoMaPDataset(args.data_dir, "valid", args.image_size, augment=False)
    batch_size = args.batch_size or (2 if args.model_id == 5 else 4)
    workers = args.num_workers
    loader_args = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
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
    lr = args.lr if args.lr is not None else (1e-3 if args.model_id in (4, 5) else 3e-4)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, object]] = []
    best_miou = -1.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        valid = evaluate(model, valid_loader, device, use_amp)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "valid": valid,
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(record)
        with (out_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch, valid, args)
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
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch, valid, args)
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
    test_dir = Path(args.data_dir) / "test" / "images"
    output_dir = Path(args.output_dir) / "test_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(test_dir.glob("*_image.png")):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        resized = image.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
        image_np = np.asarray(resized, dtype=np.float32) / 255.0
        image_np = (image_np - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with autocast(enabled=use_amp):
            prediction = model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
        encode_mask(prediction).resize(original_size, Image.Resampling.NEAREST).save(
            output_dir / image_path.name.replace("_image.png", "_mask.png")
        )
    print(f"wrote {len(list(output_dir.glob('*.png')))} test masks to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=int, required=True, choices=[2, 3, 4, 5, 6])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
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
