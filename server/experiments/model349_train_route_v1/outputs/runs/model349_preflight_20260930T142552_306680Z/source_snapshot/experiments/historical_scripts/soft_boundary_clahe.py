#!/usr/bin/env python3
"""Fine-tune with soft boundary targets and microscopy local-contrast augmentation."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from advanced_inference import predict_probabilities
from hard_negative_finetune import HARD_CLASS_WEIGHTS, load_model
from improve_five_causes import CLASS_NAMES, NUM_CLASSES, RareAwarePatchDataset, collect_split, decode_mask
from calibrate_al import score_prediction


def boundary_map(target):
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    dilated = F.max_pool2d(one_hot, 3, 1, 1)
    eroded = -F.max_pool2d(-one_hot, 3, 1, 1)
    return (dilated - eroded).amax(dim=1).clamp(0.0, 1.0)


def dice_loss(logits, target):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def local_contrast(images):
    mean = F.avg_pool2d(images, kernel_size=15, stride=1, padding=7)
    mean_sq = F.avg_pool2d(images * images, kernel_size=15, stride=1, padding=7)
    std = torch.sqrt((mean_sq - mean * mean).clamp_min(1e-4))
    enhanced = (images - mean) / (std + 0.35)
    return 0.70 * images + 0.30 * enhanced


def loss_fn(logits, target, smoothing):
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    boundaries = boundary_map(target).unsqueeze(1)
    soft_target = one_hot * (1.0 - smoothing * boundaries) + (1.0 - one_hot) * (smoothing / (NUM_CLASSES - 1.0)) * boundaries
    log_probs = F.log_softmax(logits, dim=1)
    pixel_loss = -(soft_target * log_probs).sum(dim=1)
    class_weights = HARD_CLASS_WEIGHTS.to(logits.device)[target]
    return (pixel_loss * class_weights).mean() + dice_loss(logits, target)


@torch.no_grad()
def evaluate(model, samples, args, device, use_amp):
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    image_scores = []
    for image_path, mask_path in _local_progress(samples, 'soft_boundary_clahe'):
        probabilities = predict_probabilities(model, image_path, args.patch_size, args.val_stride, device,
                                               use_amp, args.tile_batch_size, "d4", "gaussian")
        probabilities = np.log(np.maximum(probabilities, 1e-7))
        probabilities += np.asarray([0.0, 0.0, 0.10, -0.20], dtype=np.float32)[:, None, None]
        prediction = probabilities.argmax(axis=0).astype(np.uint8)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        bins = target.reshape(-1) * NUM_CLASSES + prediction.reshape(-1)
        current = np.bincount(bins, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
        confusion += current
        image_scores.append(score_prediction(prediction, target))
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = intersection / np.maximum(union, 1.0)
    return {"image_miou_compat": float(iou.mean()),
            "image_class_iou": {name: float(value) for name, value in zip(CLASS_NAMES, iou)},
            "image_mean_miou": float(np.mean(image_scores)), "confusion": confusion.tolist()}


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    dataset = RareAwarePatchDataset(train_samples, args.patch_size, args.patches_per_image, True, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    model = load_model(args.checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=use_amp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    start = time.time()
    for epoch in _local_progress(range(1, args.epochs + 1), 'soft_boundary_clahe'):
        model.train()
        total, count = 0.0, 0
        epoch_start = time.time()
        for images, masks in _local_progress(loader, 'soft_boundary_clahe'):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            if random.random() < 0.5:
                images = local_contrast(images)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                loss = loss_fn(model(images), masks, args.smoothing)
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
            total += float(loss.item()) * images.shape[0]
            count += images.shape[0]
        validation = evaluate(model, valid_samples, args, device, use_amp)
        record = {"epoch": epoch, "train_loss": total / max(count, 1), "valid": validation,
                  "seconds": time.time() - epoch_start}
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": validation, "args": vars(args)}, output_dir / "last.pt")
        print(f"epoch={epoch} loss={record['train_loss']:.4f} global={validation['image_miou_compat']:.4f} "
              f"image={validation['image_mean_miou']:.4f} iou=" +
              ",".join(f"{v:.3f}" for v in validation['image_class_iou'].values()), flush=True)
        if validation["image_miou_compat"] > best:
            best = validation["image_miou_compat"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": validation, "args": vars(args)}, output_dir / "best.pt")
    summary = {"best_image_miou_compat": best, "epochs": len(history), "elapsed_seconds": time.time() - start}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--val-stride", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--patches-per-image", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--smoothing", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    train(parser.parse_args())
