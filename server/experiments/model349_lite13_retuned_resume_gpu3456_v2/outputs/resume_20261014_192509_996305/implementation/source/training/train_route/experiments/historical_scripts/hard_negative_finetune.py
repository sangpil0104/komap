#!/usr/bin/env python3
"""Mine non-Eutectic-Si pixels predicted as Eutectic Si and fine-tune a model."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from calibrate_al import predict_probabilities, score_prediction
from improve_five_causes import (
    CLASS_COLORS,
    CLASS_NAMES,
    NUM_CLASSES,
    RareAwarePatchDataset,
    collect_split,
    decode_mask,
    encode_mask,
    load_pair,
    normalize_image,
    scaled_pair,
    crop_bounds,
)
from train_models_2_to_6 import UNetPlusPlusResNet34


# Internal channel order: [Al3Ni, Primary Si, Al, Eutectic Si].  Eutectic Si
# is the rare, difficult class under the official palette, so it receives the
# strongest loss weight in this corrected fine-tuning run.
HARD_CLASS_WEIGHTS = torch.tensor([1.15, 1.45, 0.70, 2.00], dtype=torch.float32)


def load_model(checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = UNetPlusPlusResNet34(pretrained=False).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def mine_hard_negatives(args: argparse.Namespace, device: torch.device, use_amp: bool):
    model = load_model(args.checkpoint, device)
    hard = []
    report = []
    for image_path, mask_path in collect_split(args.data_dir, "train"):
        probabilities = predict_probabilities(model, image_path, args.patch_size, args.mine_stride, device,
                                               use_amp, args.tile_batch_size, False)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        adjusted = np.log(np.maximum(probabilities, 1e-7))
        adjusted[3] += args.eutectic_bias
        prediction = adjusted.argmax(axis=0)
        hard_class = int(args.hard_class)
        false_negative = np.argwhere((prediction != hard_class) & (target == hard_class))
        rng = np.random.default_rng(args.seed + len(report) * 17)
        if len(false_negative) > args.centers_per_image:
            chosen = false_negative[rng.choice(len(false_negative), args.centers_per_image, replace=False)]
        else:
            chosen = false_negative
        original_height, original_width = target.shape
        for y, x in chosen:
            hard.append((image_path, mask_path, float(y) / max(original_height - 1, 1),
                         float(x) / max(original_width - 1, 1)))
        report.append({"image": image_path.name, "hard_class": hard_class,
                       "false_negative_pixels": int(len(false_negative)),
                       "hard_centers": int(len(chosen))})
        print(f"mined {image_path.name}: fn_class{hard_class}={len(false_negative)} centers={len(chosen)}", flush=True)
    return hard, report


class HardNegativeDataset(Dataset):
    def __init__(self, base_samples, hard_centers, patch_size, base_patches_per_image, hard_repeats, seed):
        self.base = RareAwarePatchDataset(base_samples, patch_size, base_patches_per_image, True, seed)
        self.hard_centers = list(hard_centers) * int(hard_repeats)
        self.patch_size = int(patch_size)
        self.seed = int(seed)

    def __len__(self):
        return len(self.base) + len(self.hard_centers)

    def __getitem__(self, index):
        if index < len(self.base):
            return self.base[index]
        image_path, mask_path, y_fraction, x_fraction = self.hard_centers[index - len(self.base)]
        image, mask = load_pair(image_path, mask_path)
        image, mask = scaled_pair(image, mask, self.patch_size)
        height, width = mask.shape
        center_y = int(round(y_fraction * max(height - 1, 1)))
        center_x = int(round(x_fraction * max(width - 1, 1)))
        top, left = crop_bounds(height, width, self.patch_size, center_y, center_x)
        image = image[top:top + self.patch_size, left:left + self.patch_size]
        mask = mask[top:top + self.patch_size, left:left + self.patch_size]
        rng = random.Random(self.seed + index * 997 + random.randrange(1_000_000))
        if rng.random() < 0.5:
            image, mask = np.flip(image, axis=1).copy(), np.flip(mask, axis=1).copy()
        if rng.random() < 0.5:
            image, mask = np.flip(image, axis=0).copy(), np.flip(mask, axis=0).copy()
        if rng.random() < 0.5:
            contrast = rng.uniform(0.96, 1.04)
            brightness = rng.uniform(-0.015, 0.015) * 255.0
            image = np.clip((image.astype(np.float32) - 127.5) * contrast + 127.5 + brightness,
                            0.0, 255.0).astype(np.uint8)
        return normalize_image(image), torch.from_numpy(mask).long()


def dice_loss(logits, target):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denominator = probs.sum(dims) + one_hot.sum(dims)
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def loss_fn(logits, target):
    return F.cross_entropy(logits, target, weight=HARD_CLASS_WEIGHTS.to(logits.device)) + dice_loss(logits, target)


@torch.no_grad()
def evaluate_calibrated(model, samples, args, device, use_amp):
    from local_image_metrics import summary
    model.eval()
    predictions, targets = [], []
    for image_path, mask_path in _local_progress(samples, 'hard_negative_finetune'):
        probabilities = predict_probabilities(model, image_path, args.patch_size, args.val_stride, device,
                                              use_amp, args.tile_batch_size, True)
        adjusted = np.log(np.maximum(probabilities, 1e-7))
        adjusted[3] += args.eutectic_bias
        predictions.append(adjusted.argmax(0).astype(np.uint8))
        targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
    return summary(predictions, targets, CLASS_NAMES)


def split_parameters(model):
    prefixes = ("stem.", "enc1.", "enc2.", "enc3.", "enc4.")
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith(prefixes) else decoder).append(parameter)
    return encoder, decoder


def train(args):
    global HARD_CLASS_WEIGHTS
    if args.class_weights is not None:
        HARD_CLASS_WEIGHTS = torch.tensor(args.class_weights, dtype=torch.float32)
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
    hard_centers, mining_report = mine_hard_negatives(args, device, use_amp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hard_mining.json").write_text(json.dumps(mining_report, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset = HardNegativeDataset(train_samples, hard_centers, args.patch_size, args.base_patches_per_image,
                                  args.hard_repeats, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    model = load_model(args.checkpoint, device)
    encoder, decoder = split_parameters(model)
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": args.encoder_lr},
                                   {"params": decoder, "lr": args.decoder_lr}], weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)
    best = -1.0
    history = []
    start = time.time()
    print(f"device={device} base={len(dataset.base)} hard={len(hard_centers)} total={len(dataset)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        epoch_start = time.time()
        for images, masks in _local_progress(loader, 'hard_negative_finetune'):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                loss = loss_fn(model(images), masks)
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
        validation = evaluate_calibrated(model, valid_samples, args, device, use_amp)
        record = {"epoch": epoch, "train_loss": total / max(count, 1), "valid": validation,
                  "seconds": time.time() - epoch_start}
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": validation, "args": vars(args)},
                   output_dir / "last.pt")
        score = validation["image_mean_miou"]
        print(f"epoch={epoch:02d} loss={record['train_loss']:.4f} global={score:.4f} "
              f"image={validation['image_mean_miou']:.4f} "
              f"iou=" + ",".join(f"{v:.3f}" for v in validation["image_class_iou"].values()) +
              f" time={record['seconds']:.1f}s", flush=True)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": validation, "args": vars(args)},
                       output_dir / "best.pt")
    summary = {"best_image_mean_miou": best, "epochs": len(history), "hard_centers": len(hard_centers),
               "elapsed_seconds": time.time() - start}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--mine-stride", type=int, default=448)
    parser.add_argument("--val-stride", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--base-patches-per-image", type=int, default=4)
    parser.add_argument("--centers-per-image", type=int, default=8)
    parser.add_argument("--hard-repeats", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-6)
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eutectic-bias", type=float, default=0.0)
    parser.add_argument("--class-weights", type=float, nargs=4, default=None,
                        metavar=("AL3NI", "PRIMARY", "AL", "EUTECTIC"))
    parser.add_argument("--hard-class", type=int, default=3, choices=range(4),
                        help="class whose false negatives are mined for extra crops")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
