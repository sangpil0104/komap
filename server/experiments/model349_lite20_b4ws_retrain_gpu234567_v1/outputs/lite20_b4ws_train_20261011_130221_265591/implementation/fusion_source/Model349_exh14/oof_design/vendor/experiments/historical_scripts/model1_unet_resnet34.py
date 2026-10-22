#!/usr/bin/env python3
"""KoMaP model 1: U-Net with an ImageNet-pretrained ResNet34 encoder.

The first experiment intentionally follows the competition guide closely:
fixed train/valid split, basic geometry augmentation, Cross Entropy + Dice,
and per-class IoU/mIoU logging.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34


NUM_CLASSES = 4
# The mask palette is fixed by the competition data.  Keep the trained
# channel IDs aligned with that palette; the official semantic names for the
# four existing channels are therefore [Al3Ni, Primary Si, Al, Eutectic Si].
# Do not reorder CLASS_COLORS without remapping every checkpoint's output.
CLASS_NAMES = ["Al3Ni", "Primary Si", "Al", "Eutectic Si"]
# The provided masks are RGBA PNGs. Alpha is constant 255; RGB identifies the
# four semantic classes. This mapping is kept explicit for reproducible output.
CLASS_COLORS: Tuple[Tuple[int, int, int], ...] = (
    # blue  -> official Al3Ni
    (25, 76, 153),
    # green -> official Primary Si
    (76, 178, 76),
    # brown -> official Al
    (153, 127, 76),
    # gray  -> official Eutectic Si
    (204, 204, 204),
)
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decode_mask(mask: Image.Image) -> np.ndarray:
    rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
    result = np.full(rgb.shape[:2], -1, dtype=np.int64)
    for class_id, color in enumerate(CLASS_COLORS):
        result[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = class_id
    if (result < 0).any():
        unknown = np.unique(rgb[result < 0].reshape(-1, 3), axis=0)
        raise ValueError(f"Unknown mask colors: {unknown[:10].tolist()}")
    return result


def encode_mask(mask: np.ndarray) -> Image.Image:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        output[mask == class_id] = color
    return Image.fromarray(output, mode="RGB")


class KoMaPDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.root = Path(root) / split
        self.image_size = int(image_size)
        self.augment = augment
        self.image_paths = sorted((self.root / "images").glob("*_image.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {self.root / 'images'}")
        self.mask_paths = []
        for image_path in self.image_paths:
            mask_path = self.root / "masks" / image_path.name.replace("_image.png", "_mask.png")
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
            self.mask_paths.append(mask_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        image_np = np.asarray(image, dtype=np.uint8)
        mask_np = decode_mask(mask)

        if self.augment:
            if random.random() < 0.5:
                image_np = np.flip(image_np, axis=1).copy()
                mask_np = np.flip(mask_np, axis=1).copy()
            if random.random() < 0.5:
                image_np = np.flip(image_np, axis=0).copy()
                mask_np = np.flip(mask_np, axis=0).copy()
            quarter_turns = random.randrange(4)
            if quarter_turns:
                image_np = np.rot90(image_np, quarter_turns).copy()
                mask_np = np.rot90(mask_np, quarter_turns).copy()

        image_float = image_np.astype(np.float32) / 255.0
        image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image_float.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_np).long()
        return image_tensor, mask_tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetResNet34(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            encoder = resnet34(weights=weights)
        except Exception as exc:
            if not pretrained:
                raise
            print(f"Pretrained ResNet34 unavailable ({exc}); using random initialization.", flush=True)
            encoder = resnet34(weights=None)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.enc1 = encoder.layer1
        self.enc2 = encoder.layer2
        self.enc3 = encoder.layer3
        self.enc4 = encoder.layer4

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.head = nn.Sequential(
            ConvBlock(64 + 64, 32),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        s0 = self.stem(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        d4 = self.dec4(s4, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)
        x = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)
        # The stem feature is at half input resolution; use it as a final skip.
        stem_skip = F.interpolate(s0, size=input_size, mode="bilinear", align_corners=False)
        return self.head(torch.cat([x, stem_skip], dim=1))


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * target_one_hot).sum(dims)
    denominator = probs.sum(dims) + target_one_hot.sum(dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, object]:
    model.eval()
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    image_scores = []
    total_loss = 0.0
    count = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = combined_loss(logits, masks)
        predictions = logits.argmax(dim=1).cpu()
        targets = masks.cpu()
        for prediction, target in zip(predictions, targets):
            bins = target.reshape(-1) * NUM_CLASSES + prediction.reshape(-1)
            current = torch.bincount(bins, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
            current_intersection = torch.diag(current).float()
            current_union = current.sum(0).float() + current.sum(1).float() - current_intersection
            image_scores.append(float((current_intersection / current_union.clamp_min(1.0)).mean()))
            confusion += current
        total_loss += float(loss.item()) * images.shape[0]
        count += images.shape[0]

    intersection = torch.diag(confusion).float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - intersection
    iou = (intersection / union.clamp_min(1.0)).tolist()
    global_miou = float(np.mean(iou))
    image_mean_miou = float(np.mean(image_scores))
    return {
        "loss": total_loss / max(count, 1),
        "iou": {name: float(score) for name, score in zip(CLASS_NAMES, iou)},
        "global_miou": global_miou,
        "image_mean_miou": image_mean_miou,
        "miou": image_mean_miou,
        "confusion": confusion.tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, object],
    args: argparse.Namespace,
) -> None:
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


def run_training(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device={device} gpu={torch.cuda.current_device() if device.type == 'cuda' else 'cpu'} amp={use_amp}", flush=True)

    train_ds = KoMaPDataset(args.data_dir, "train", args.image_size, augment=True)
    valid_ds = KoMaPDataset(args.data_dir, "valid", args.image_size, augment=False)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, drop_last=False, **loader_kwargs)
    print(f"train={len(train_ds)} valid={len(valid_ds)} image_size={args.image_size}", flush=True)

    model = UNetResNet34(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, object]] = []
    best_miou = -1.0
    stale_epochs = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        valid_metrics = evaluate(model, valid_loader, device, use_amp)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "valid": valid_metrics,
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(record)
        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch, valid_metrics, args)

        miou = float(valid_metrics["miou"])
        print(
            f"epoch={epoch:03d}/{args.epochs} train_loss={train_loss:.4f} "
            f"valid_loss={float(valid_metrics['loss']):.4f} image_miou={miou:.4f} "
            f"global_miou={float(valid_metrics['global_miou']):.4f} "
            f"iou=" + ", ".join(f"{v:.3f}" for v in valid_metrics["iou"].values()) +
            f" time={record['epoch_seconds']:.1f}s",
            flush=True,
        )
        if miou > best_miou:
            best_miou = miou
            stale_epochs = 0
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch, valid_metrics, args)
            print(f"saved best.pt (mIoU={best_miou:.6f})", flush=True)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {args.patience} stale epochs", flush=True)
                break

    summary = {
        "best_miou": best_miou,
        "epochs_completed": len(history),
        "elapsed_seconds": time.time() - start_time,
        "best_checkpoint": str(out_dir / "best.pt"),
        "class_names": CLASS_NAMES,
        "class_colors": CLASS_COLORS,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


@torch.no_grad()
def predict_test(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = UNetResNet34(pretrained=False).to(device)
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
        image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with autocast(enabled=use_amp):
            pred = model(tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
        pred_image = encode_mask(pred).resize(original_size, Image.Resampling.NEAREST)
        out_path = output_dir / image_path.name.replace("_image.png", "_mask.png")
        pred_image.save(out_path)
    print(f"wrote {len(list(output_dir.glob('*.png')))} test masks to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    if args.predict_test and not args.checkpoint:
        parser.error("--predict-test requires --checkpoint")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.predict_test:
        predict_test(parsed)
    else:
        run_training(parsed)
