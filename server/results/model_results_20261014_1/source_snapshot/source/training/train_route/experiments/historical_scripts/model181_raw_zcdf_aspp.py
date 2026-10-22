#!/usr/bin/env python3
"""Train/evaluate a team-style raw/z-score/CDF ASPP branch.

This is a controlled follow-up to the team's ``resnet34_unetplusplus_aspp_v1``
description.  The decoder and class mapping stay compatible with the current
ResNet34-ASPP checkpoints, while the input is represented by three channels:
raw grayscale, per-image z-score, and empirical CDF.  The branch writes
validation probabilities so it can be tested as an additional member of the
model-173 route without retraining the ensemble.
"""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from advanced_inference import blend_window, inverse_view, transform_view, views
from encoder_structure_search import ASPPUNetPP
from hard_negative_finetune import HARD_CLASS_WEIGHTS
from improve_five_causes import (
    CLASS_NAMES,
    FOCUS_PROBS,
    NUM_CLASSES,
    collect_split,
    crop_bounds,
    decode_mask,
    encode_mask,
    load_pair,
    scaled_pair,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TeamResidualASPP(nn.Module):
    """Residual ASPP variant described in the team report: d=1,2,4,8."""

    def __init__(self, channels: int = 512, branch_channels: int = 128) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            self._branch(channels, branch_channels, 1, 1),
            self._branch(channels, branch_channels, 3, 2),
            self._branch(channels, branch_channels, 3, 4),
            self._branch(channels, branch_channels, 3, 8),
        ])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 5, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _branch(channels: int, branch_channels: int, kernel: int, dilation: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, branch_channels, kernel,
                      padding=dilation * (kernel // 2), dilation=dilation, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        values = [branch(x) for branch in self.branches]
        pooled = self.pool(x)
        pooled = F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)
        values.append(pooled)
        return x + self.scale * self.project(torch.cat(values, dim=1))


class TeamRepresentationASPP(ASPPUNetPP):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(pretrained=pretrained)
        self.context = TeamResidualASPP()


def representation(image: np.ndarray) -> np.ndarray:
    """Return CxHxW approximately standardized raw/z/CDF channels."""
    gray = image[..., 0].astype(np.float32) / 255.0
    z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
    z = np.clip(z, -4.0, 4.0) / 3.0
    flat = gray.reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(order.size, dtype=np.float32)
    ranks[order] = np.arange(order.size, dtype=np.float32)
    cdf = (ranks / max(order.size - 1, 1)).reshape(gray.shape)
    values = np.stack([
        (gray - 0.5) / 0.25,
        z,
        (cdf - 0.5) / 0.288675,
    ], axis=0)
    return values.astype(np.float32)


class RepresentationPatchDataset(Dataset):
    def __init__(self, samples, patch_size: int, patches_per_image: int, augment: bool, seed: int):
        self.samples = list(samples)
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.augment = bool(augment)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index % len(self.samples)]
        image, mask = load_pair(image_path, mask_path)
        image, mask = scaled_pair(image, mask, self.patch_size)
        height, width = mask.shape
        rng = random.Random(self.seed + index * 1009 + random.randrange(1_000_000))
        focus_class = None
        if rng.random() < 0.90:
            draw, cumulative = rng.random(), 0.0
            for class_id, probability in FOCUS_PROBS:
                cumulative += probability
                if draw <= cumulative:
                    focus_class = class_id
                    break
        ys, xs = np.where(mask == focus_class) if focus_class is not None else np.where(mask != 2)
        if len(xs):
            selected = rng.randrange(len(xs))
            center_y, center_x = int(ys[selected]), int(xs[selected])
        else:
            center_y, center_x = height // 2, width // 2
        top, left = crop_bounds(height, width, self.patch_size, center_y, center_x)
        image = image[top:top + self.patch_size, left:left + self.patch_size]
        mask = mask[top:top + self.patch_size, left:left + self.patch_size]
        if self.augment:
            if rng.random() < 0.5:
                image, mask = np.flip(image, 1).copy(), np.flip(mask, 1).copy()
            if rng.random() < 0.5:
                image, mask = np.flip(image, 0).copy(), np.flip(mask, 0).copy()
            turns = rng.randrange(4)
            if turns:
                image, mask = np.rot90(image, turns).copy(), np.rot90(mask, turns).copy()
            if rng.random() < 0.45:
                contrast = rng.uniform(0.94, 1.06)
                brightness = rng.uniform(-0.025, 0.025) * 255.0
                image = np.clip((image.astype(np.float32) - 127.5) * contrast
                                + 127.5 + brightness, 0.0, 255.0).astype(np.uint8)
        return torch.from_numpy(representation(image)), torch.from_numpy(mask).long()


def load_checkpoint(model: nn.Module, path: str) -> None:
    payload = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    print(f"loaded={path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)


@torch.no_grad()
def predict_probabilities(model: nn.Module, image_path: Path, patch_size: int, stride: int,
                          device: torch.device, use_amp: bool, tile_batch_size: int) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), patch_size)
    height, width = scaled.shape[:2]
    features = representation(scaled)
    ys = [0] if height <= patch_size else list(range(0, height - patch_size + 1, stride))
    xs = [0] if width <= patch_size else list(range(0, width - patch_size + 1, stride))
    last_y, last_x = max(height - patch_size, 0), max(width - patch_size, 0)
    if ys[-1] != last_y:
        ys.append(last_y)
    if xs[-1] != last_x:
        xs.append(last_x)
    padded_height = max(height, ys[-1] + patch_size)
    padded_width = max(width, xs[-1] + patch_size)
    padded = np.pad(features, ((0, 0), (0, padded_height - height),
                               (0, padded_width - width)), mode="reflect")
    probability_sum = np.zeros((NUM_CLASSES, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    window = blend_window(patch_size, "gaussian")
    locations = [(top, left) for top in ys for left in xs]
    for start in range(0, len(locations), tile_batch_size):
        batch_locations = locations[start:start + tile_batch_size]
        inputs = torch.stack([
            torch.from_numpy(padded[:, top:top + patch_size, left:left + patch_size])
            for top, left in batch_locations
        ]).to(device, non_blocking=True)
        probabilities = None
        for rotation, flip in views("d4"):
            value = transform_view(inputs, rotation, flip)
            with autocast(enabled=use_amp):
                value = torch.softmax(model(value), dim=1)
            value = inverse_view(value, rotation, flip)
            probabilities = value if probabilities is None else probabilities + value
        probabilities = (probabilities / 8.0).float().cpu().numpy()
        for index, (top, left) in enumerate(batch_locations):
            probability_sum[:, top:top + patch_size, left:left + patch_size] += (
                probabilities[index] * window[None]
            )
            weight_sum[top:top + patch_size, left:left + patch_size] += window
    value = probability_sum / np.maximum(weight_sum[None], 1e-7)
    value = value[:, :height, :width]
    original_size = image.shape[1], image.shape[0]
    restored = np.stack([
        np.asarray(Image.fromarray(value[class_id]).resize(original_size, Image.Resampling.BILINEAR),
                   dtype=np.float32)
        for class_id in range(NUM_CLASSES)
    ])
    return restored / np.maximum(restored.sum(axis=0, keepdims=True), 1e-7)


def score(predictions, targets):
    from local_image_metrics import summary
    return summary(predictions, targets, CLASS_NAMES)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    intersection = (probabilities * one_hot).sum((0, 2, 3))
    denominator = probabilities.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def train_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total, count = 0.0, 0
    weights = HARD_CLASS_WEIGHTS.to(device)
    for images, masks in _local_progress(loader, 'model181_raw_zcdf_aspp'):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, masks, weight=weights) + dice_loss(logits, masks)
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
    return total / max(count, 1)


def evaluate(model, samples, args, device, use_amp, output_probs: bool = False):
    model.eval()
    probabilities, targets = [], []
    for image_path, mask_path in samples:
        probabilities.append(predict_probabilities(model, image_path, args.patch_size,
                                                   args.val_stride, device, use_amp,
                                                   args.tile_batch_size))
        targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
        print(f"validated {image_path.name}", flush=True)
    predictions = [value.argmax(axis=0).astype(np.uint8) for value in probabilities]
    result = score(predictions, targets)
    if output_probs:
        result["probabilities"] = probabilities
    return result


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    dataset = RepresentationPatchDataset(train_samples, args.patch_size, args.patches_per_image,
                                          True, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=(args.num_workers > 0))
    # The supplied checkpoint already contains the ImageNet-initialized
    # encoder; avoid an unnecessary network lookup on the server.
    model = TeamRepresentationASPP(pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint)
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith(("stem.", "enc1.", "enc2.", "enc3.", "enc4.")) else decoder).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": args.encoder_lr},
        {"params": decoder, "lr": args.decoder_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_score, best_state = -1.0, None
    history = []
    start_time = time.time()
    print(f"device={device} patches={len(dataset)} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, scaler, device, use_amp)
        valid = evaluate(model, valid_samples, args, device, use_amp)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "valid": valid,
                  "seconds": time.time() - start_time}
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        selection = float(valid[args.selection_metric])
        print(json.dumps({"epoch": epoch, "train_loss": train_loss,
                          "selection": selection, **{k: valid[k] for k in
                          ("image_miou_compat", "image_mean_miou", "image_mean_miou_exclude")}},
                         ensure_ascii=False), flush=True)
        if selection > best_score:
            best_score = selection
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model": best_state, "epoch": epoch, "metrics": valid,
                        "args": vars(args)}, output / "best.pt")
    model.load_state_dict(best_state)
    final = evaluate(model, valid_samples, args, device, use_amp, output_probs=True)
    probabilities = final.pop("probabilities")
    valid_payload = {path.stem: value.astype(np.float16)
                     for (path, _), value in zip(valid_samples, probabilities)}
    np.savez_compressed(output / "valid_probs.npz", **valid_payload)
    summary = {"selection_metric": args.selection_metric, "best_selection": best_score,
               "final": final, "elapsed_seconds": time.time() - start_time,
               "representation": "raw_gray+per_image_zscore+empirical_cdf",
               "aspp_dilation": [1, 2, 4, 8]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    test_output = output / "test_masks"
    test_output.mkdir(exist_ok=True)
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        value = predict_probabilities(model, image_path, args.patch_size, args.val_stride,
                                      device, use_amp, args.tile_batch_size)
        encode_mask(value.argmax(axis=0).astype(np.uint8)).save(
            test_output / image_path.name.replace("_image.png", "_mask.png"))
    print(f"wrote test_masks={len(list(test_output.glob('*.png')))}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--val-stride", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selection-metric", choices=("image_mean_miou", "image_mean_miou_exclude"),
                        default="image_mean_miou_exclude")
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
