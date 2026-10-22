#!/usr/bin/env python3
"""Fine-tune the ASPP ResNet34 model with wide 512x768 native patches.

The current square-patch model is strong globally but fails on a few wide
validation images where phase boundaries span a tile seam.  This experiment
keeps the native scale while exposing more horizontal context per crop.
"""

from __future__ import annotations

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
from torch.utils.data import DataLoader, Dataset

from advanced_inference import inverse_view, transform_view, views
from encoder_structure_search import build_model
from hard_negative_finetune import HARD_CLASS_WEIGHTS
from improve_five_causes import (
    CLASS_NAMES, FOCUS_PROBS, NUM_CLASSES, collect_split, decode_mask,
    encode_mask, load_pair, normalize_image, scaled_pair,
)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pad_pair(image, mask, height, width):
    pad_h = max(height, image.shape[0]); pad_w = max(width, image.shape[1])
    image = np.pad(image, ((0, pad_h - image.shape[0]), (0, pad_w - image.shape[1]), (0, 0)), mode="reflect")
    mask = np.pad(mask, ((0, pad_h - mask.shape[0]), (0, pad_w - mask.shape[1])), mode="edge")
    return image, mask


def crop_bounds_rect(height, width, crop_h, crop_w, center_y, center_x):
    top = int(np.clip(center_y - crop_h // 2, 0, max(height - crop_h, 0)))
    left = int(np.clip(center_x - crop_w // 2, 0, max(width - crop_w, 0)))
    return top, left


class RectPatchDataset(Dataset):
    def __init__(self, samples, crop_h, crop_w, patches_per_image, augment, seed):
        self.samples = list(samples); self.crop_h = crop_h; self.crop_w = crop_w
        self.patches_per_image = patches_per_image; self.augment = augment; self.seed = seed

    def __len__(self): return len(self.samples) * self.patches_per_image

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index % len(self.samples)]
        image, mask = load_pair(image_path, mask_path)
        image, mask = scaled_pair(image, mask, self.crop_h)
        image, mask = pad_pair(image, mask, self.crop_h, self.crop_w)
        height, width = mask.shape
        rng = random.Random(self.seed + index * 1009 + random.randrange(1_000_000))
        focus = None
        if rng.random() < 0.9:
            draw = rng.random(); cumulative = 0.0
            for class_id, probability in FOCUS_PROBS:
                cumulative += probability
                if draw <= cumulative:
                    focus = class_id; break
        ys, xs = np.where(mask == focus) if focus is not None else np.where(mask != 2)
        if len(xs):
            chosen = rng.randrange(len(xs)); cy, cx = int(ys[chosen]), int(xs[chosen])
        else:
            cy, cx = height // 2, width // 2
        top, left = crop_bounds_rect(height, width, self.crop_h, self.crop_w, cy, cx)
        image = image[top:top + self.crop_h, left:left + self.crop_w]
        mask = mask[top:top + self.crop_h, left:left + self.crop_w]
        if self.augment:
            if rng.random() < 0.5: image, mask = np.flip(image, 1).copy(), np.flip(mask, 1).copy()
            if rng.random() < 0.5: image, mask = np.flip(image, 0).copy(), np.flip(mask, 0).copy()
            if rng.random() < 0.45:
                contrast = rng.uniform(0.94, 1.06); brightness = rng.uniform(-0.025, 0.025) * 255.0
                image = np.clip((image.astype(np.float32) - 127.5) * contrast + 127.5 + brightness,
                                0.0, 255.0).astype(np.uint8)
        return normalize_image(image), torch.from_numpy(mask).long()


def rect_starts(length, patch, stride):
    if length <= patch: return [0]
    values = list(range(0, length - patch + 1, stride)); last = length - patch
    if values[-1] != last: values.append(last)
    return values


def rect_window(height, width):
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    value = np.exp(-(yy * yy + xx * xx) / (2.0 * 0.55 * 0.55)).astype(np.float32)
    return 0.15 + 0.85 * value


@torch.no_grad()
def predict_rect(model, image_path, crop_h, crop_w, stride_h, stride_w, device, amp, tile_batch):
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), crop_h)
    height, width = scaled.shape[:2]
    ys, xs = rect_starts(height, crop_h, stride_h), rect_starts(width, crop_w, stride_w)
    padded_h, padded_w = max(height, ys[-1] + crop_h), max(width, xs[-1] + crop_w)
    padded = np.pad(scaled, ((0, padded_h - height), (0, padded_w - width), (0, 0)), mode="reflect")
    total = np.zeros((NUM_CLASSES, padded_h, padded_w), np.float32)
    normalizer = np.zeros((padded_h, padded_w), np.float32)
    window = rect_window(crop_h, crop_w)
    locations = [(top, left) for top in ys for left in xs]
    for start in range(0, len(locations), tile_batch):
        current = locations[start:start + tile_batch]
        inputs = torch.stack([normalize_image(padded[t:t + crop_h, l:l + crop_w]) for t, l in current]).to(device)
        probability_batch = None
        for rotation, flip in views("d4"):
            view = transform_view(inputs, rotation, flip)
            with torch.autocast(device_type="cuda", enabled=amp):
                value = torch.softmax(model(view), dim=1)
            value = inverse_view(value, rotation, flip)
            probability_batch = value if probability_batch is None else probability_batch + value
        probability_batch = (probability_batch / 8.0).float().cpu().numpy()
        for i, (top, left) in enumerate(current):
            total[:, top:top + crop_h, left:left + crop_w] += probability_batch[i] * window[None]
            normalizer[top:top + crop_h, left:left + crop_w] += window
    probability = total / np.maximum(normalizer[None], 1e-6)
    probability = probability[:, :height, :width]
    original_size = image.shape[1], image.shape[0]
    restored = np.stack([np.asarray(Image.fromarray(probability[c]).resize(original_size, Image.Resampling.BILINEAR), np.float32)
                         for c in range(NUM_CLASSES)])
    return restored / np.maximum(restored.sum(0, keepdims=True), 1e-7)


def dice_loss(logits, target):
    probs = torch.softmax(logits, 1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    intersection = (probs * one_hot).sum((0, 2, 3))
    denominator = probs.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def metrics(predictions, targets):
    total = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64); scores = []
    for prediction, target in zip(predictions, targets):
        cm = np.bincount(target.reshape(-1) * NUM_CLASSES + prediction.reshape(-1), minlength=16).reshape(4, 4)
        total += cm
        inter = np.diag(cm).astype(float); union = cm.sum(0) + cm.sum(1) - inter
        scores.append(float(np.mean(inter / np.maximum(union, 1.0))))
    inter = np.diag(total).astype(float); union = total.sum(0) + total.sum(1) - inter
    iou = inter / np.maximum(union, 1.0)
    return {"global_miou": float(iou.mean()), "global_iou": dict(zip(CLASS_NAMES, iou.tolist())),
            "image_mean_miou": float(np.mean(scores)), "image_scores": scores, "confusion": total.tolist()}


def evaluate(model, samples, args, device, amp, bias):
    predictions, targets = [], []
    model.eval()
    for image_path, mask_path in samples:
        value = predict_rect(model, image_path, args.crop_h, args.crop_w, args.stride_h, args.stride_w,
                             device, amp, args.tile_batch)
        logits = np.log(np.maximum(value, 1e-7)); logits[3] += bias
        predictions.append(logits.argmax(0).astype(np.uint8))
        targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
    return metrics(predictions, targets)


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    train_samples, valid_samples = collect_split(args.data_dir, "train"), collect_split(args.data_dir, "valid")
    dataset = RectPatchDataset(train_samples, args.crop_h, args.crop_w, args.patches_per_image, True, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    model = build_model("aspp", pretrained=False, checkpoint=args.checkpoint).to(device)
    weights = HARD_CLASS_WEIGHTS.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=amp); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    best = -1.0; history = []; start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; count = 0
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                logits = model(images)
                loss = F.cross_entropy(logits, masks, weight=weights) + dice_loss(logits, masks)
            if amp:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.item()) * images.shape[0]; count += images.shape[0]
        scheduler.step()
        value = evaluate(model, valid_samples, args, device, amp, args.eval_bias)
        record = {"epoch": epoch, "train_loss": total / max(count, 1), "valid": value}
        history.append(record); (out / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": value, "args": vars(args)}, out / "last.pt")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if value["image_mean_miou"] > best:
            best = value["image_mean_miou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": value, "args": vars(args)}, out / "best.pt")
    payload = torch.load(out / "best.pt", map_location="cpu"); model.load_state_dict(payload["model"]); model.eval()
    sweep = [evaluate(model, valid_samples, args, device, amp, float(bias)) for bias in np.arange(-0.8, 0.21, 0.05)]
    calibrated = max(sweep, key=lambda x: x["image_mean_miou"])
    summary = {"best_fixed": best, "best_calibrated": calibrated, "elapsed_seconds": time.time() - start_time}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    test_out = out / "test_masks"; test_out.mkdir(exist_ok=True)
    bias = float(np.arange(-0.8, 0.21, 0.05)[int(np.argmax([x["image_mean_miou"] for x in sweep]))])
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        value = predict_rect(model, image_path, args.crop_h, args.crop_w, args.stride_h, args.stride_w,
                             device, amp, args.tile_batch)
        logits = np.log(np.maximum(value, 1e-7)); logits[3] += bias
        encode_mask(logits.argmax(0).astype(np.uint8)).save(test_out / image_path.name.replace("_image.png", "_mask.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--crop-h", type=int, default=512)
    parser.add_argument("--crop-w", type=int, default=768); parser.add_argument("--stride-h", type=int, default=384)
    parser.add_argument("--stride-w", type=int, default=576); parser.add_argument("--patches-per-image", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tile-batch", type=int, default=2); parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-6); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-bias", type=float, default=-0.55); parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-amp", action="store_true"); main(parser.parse_args())
