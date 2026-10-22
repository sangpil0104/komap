#!/usr/bin/env python3
"""Additional KoMaP candidates beyond the first nine experiments.

Candidates:
  effb4_unet       EfficientNet-B4 + U-Net
  effb4_unetpp     EfficientNet-B4 + U-Net++
  effb5_unetpp     EfficientNet-B5 + U-Net++
  effv2s_unetpp    EfficientNetV2-S + a multi-scale decoder
  hrnet_w32        HRNet-W32 + high-resolution context head
  hrnet_w48        HRNet-W48 + high-resolution context head
  segformer_b2     SegFormer-B2
  segformer_b3     SegFormer-B3
  segnext_style_t  portable SegNeXt-style MSCA head over ConvNeXt-Tiny
  segnext_style_s  portable SegNeXt-style MSCA head over ConvNeXt-Small

The SegNeXt-style variants are explicitly portable adaptations: the official
SegNeXt/MMseg stack is not installed in the P100 environment, so these use the
same multi-scale convolutional-attention idea with pretrained timm encoders.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

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


MODEL_NAMES = [
    "effb4_unet",
    "effb4_unetpp",
    "effb5_unetpp",
    "effv2s_unetpp",
    "hrnet_w32",
    "hrnet_w48",
    "segformer_b2",
    "segformer_b3",
    "segnext_style_t",
    "segnext_style_s",
]


def make_smp_model(architecture: str, encoder: str, pretrained: bool) -> nn.Module:
    import segmentation_models_pytorch as smp

    constructor = getattr(smp, architecture)
    return constructor(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )


class TimmMultiScaleDecoder(nn.Module):
    def __init__(self, encoder_name: str, decoder_channels: int = 96, msca: bool = False, pretrained: bool = True):
        super().__init__()
        import timm

        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        channels = list(self.encoder.feature_info.channels())
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, decoder_channels, 1, bias=False),
                    nn.BatchNorm2d(decoder_channels),
                    nn.ReLU(inplace=True),
                )
                for c in channels
            ]
        )
        self.msca = MSCA(decoder_channels) if msca else nn.Identity()
        self.fuse = ConvBlock(decoder_channels, decoder_channels)
        self.head = nn.Conv2d(decoder_channels, NUM_CLASSES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        target_size = features[0].shape[-2:]
        fused = None
        for feature, projection in zip(features, self.projections):
            value = F.interpolate(
                projection(feature), size=target_size, mode="bilinear", align_corners=False
            )
            fused = value if fused is None else fused + value
        fused = self.msca(fused)
        logits = self.head(self.fuse(fused))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


class MSCA(nn.Module):
    """Portable multi-scale convolutional attention block."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.pre = nn.Conv2d(channels, channels, 1, bias=False)
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.dw7 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.merge = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.pre(self.norm(x))
        attention = self.dw3(y) + self.dw5(y) + self.dw7(y)
        attention = self.merge(attention)
        return residual + y * self.gate(attention)


class SegformerModel(nn.Module):
    def __init__(self, size: str, pretrained: bool):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        checkpoint = f"nvidia/segformer-{size}-finetuned-ade-512-512"
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint,
                num_labels=NUM_CLASSES,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig.from_pretrained(checkpoint)
            config.num_labels = NUM_CLASSES
            config.id2label = {str(i): name for i, name in enumerate(CLASS_NAMES)}
            config.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "effb4_unet":
        return make_smp_model("Unet", "efficientnet-b4", pretrained)
    if name == "effb4_unetpp":
        return make_smp_model("UnetPlusPlus", "efficientnet-b4", pretrained)
    if name == "effb5_unetpp":
        return make_smp_model("UnetPlusPlus", "efficientnet-b5", pretrained)
    if name == "effv2s_unetpp":
        return TimmMultiScaleDecoder("tf_efficientnetv2_s", decoder_channels=96, pretrained=pretrained)
    if name == "hrnet_w32":
        return TimmMultiScaleDecoder("hrnet_w32", decoder_channels=96, pretrained=pretrained)
    if name == "hrnet_w48":
        return TimmMultiScaleDecoder("hrnet_w48", decoder_channels=96, pretrained=pretrained)
    if name == "segformer_b2":
        return SegformerModel("b2", pretrained)
    if name == "segformer_b3":
        return SegformerModel("b3", pretrained)
    if name == "segnext_style_t":
        return TimmMultiScaleDecoder("convnext_tiny", decoder_channels=96, msca=True, pretrained=pretrained)
    if name == "segnext_style_s":
        return TimmMultiScaleDecoder("convnext_small", decoder_channels=96, msca=True, pretrained=pretrained)
    raise ValueError(f"unknown model: {name}")


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    count = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            loss = combined_loss(model(images), masks)
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


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds = KoMaPDataset(args.data_dir, "train", args.image_size, augment=True)
    valid_ds = KoMaPDataset(args.data_dir, "valid", args.image_size, augment=False)
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_args)
    valid_loader = DataLoader(valid_ds, shuffle=False, drop_last=False, **loader_args)
    print(
        f"model={args.model} device={device} logical_gpu="
        f"{torch.cuda.current_device() if device.type == 'cuda' else 'cpu'} "
        f"amp={use_amp} train={len(train_ds)} valid={len(valid_ds)} batch={args.batch_size}",
        flush=True,
    )

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
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
    history = []
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
        (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2))
        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch, valid, args)
        miou = float(valid["miou"])
        print(
            f"epoch={epoch:03d}/{args.epochs} train_loss={train_loss:.4f} "
            f"valid_loss={float(valid['loss']):.4f} miou={miou:.4f} "
            + ", ".join(f"{v:.3f}" for v in valid["iou"].values())
            + f" time={record['epoch_seconds']:.1f}s",
            flush=True,
        )
        if miou > best_miou:
            best_miou = miou
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch, valid, args)
            print(f"saved best.pt (mIoU={best_miou:.6f})", flush=True)

    summary = {
        "model": args.model,
        "best_miou": best_miou,
        "epochs_completed": len(history),
        "elapsed_seconds": time.time() - start_time,
        "best_checkpoint": str(out_dir / "best.pt"),
        "class_names": CLASS_NAMES,
        "class_colors": CLASS_COLORS,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


@torch.no_grad()
def predict_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(args.model, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    out_dir = Path(args.output_dir) / "test_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir = Path(args.data_dir) / "test" / "images"
    for image_path in sorted(test_dir.glob("*_image.png")):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        image = image.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with autocast(enabled=use_amp):
            pred = model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
        output = encode_mask(pred).resize(original_size, Image.Resampling.NEAREST)
        output.save(out_dir / image_path.name.replace("_image.png", "_mask.png"))
    print(f"wrote {len(list(out_dir.glob('*.png')))} test masks to {out_dir}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
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
