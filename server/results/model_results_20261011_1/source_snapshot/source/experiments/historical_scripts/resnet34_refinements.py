#!/usr/bin/env python3
"""ResNet34-focused input, structure, Al-boundary and stability experiments.

All candidates start from model42/model109 (ResNet34 U-Net++ + residual ASPP)
and use the native-patch protocol. The candidates are deliberately small:

  input_filters : RGB + grayscale/local-contrast/Sobel/LoG learnable stem
  scse          : residual scSE gates on encoder skip features
  sdf           : auxiliary signed-distance regression for the Al class
  raw_highres   : residual raw-input stride-2 high-resolution skip
  lovasz        : late Lovasz-Softmax phase after CE+Dice warm-up
  ema           : exponential moving-average weights for validation/test
"""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_laplace
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from advanced_inference import blend_window, inverse_view, transform_view, views
from encoder_structure_search import ASPPUNetPP
from hard_negative_finetune import HARD_CLASS_WEIGHTS
from improve_five_causes import (
    CLASS_NAMES,
    FOCUS_PROBS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    RareAwarePatchDataset,
    collect_split,
    crop_bounds,
    decode_mask,
    encode_mask,
    load_pair,
    normalize_image,
    scaled_pair,
)


CANDIDATES = ("input_filters", "scse", "sdf", "raw_highres", "lovasz", "ema")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_compatible(model: nn.Module, checkpoint: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    source = payload["model"]
    target = model.state_dict()
    compatible = {key: value for key, value in source.items()
                  if key in target and target[key].shape == value.shape}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if "stem.0.weight" in source and "stem.0.weight" in target and source["stem.0.weight"].shape != target["stem.0.weight"].shape:
        old_weight = source["stem.0.weight"]
        with torch.no_grad():
            model.stem[0].weight[:, :old_weight.shape[1]].copy_(old_weight)
    print(f"loaded={checkpoint} compatible={len(compatible)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def standardize(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.float32)
    value = (value - float(value.mean())) / max(float(value.std()), 1e-4)
    return np.clip(value, -6.0, 6.0)


def input_features(image: np.ndarray) -> torch.Tensor:
    """Return 8 channels: normalized RGB plus five local morphology signals."""
    raw = image.astype(np.float32) / 255.0
    rgb = (raw - IMAGENET_MEAN) / IMAGENET_STD
    gray = np.clip(np.dot(raw, np.asarray([0.299, 0.587, 0.114], dtype=np.float32)), 0.0, 1.0)
    local_mean = gaussian_filter(gray, sigma=3.0)
    local_contrast = gray - local_mean
    gy, gx = np.gradient(gaussian_filter(gray, sigma=0.7))
    sobel = np.sqrt(gx * gx + gy * gy)
    log1 = gaussian_laplace(gray, sigma=1.0)
    log2 = gaussian_laplace(gray, sigma=2.0)
    extra = np.stack([
        standardize(gray), standardize(local_contrast), standardize(sobel),
        standardize(log1), standardize(log2),
    ], axis=0)
    return torch.from_numpy(np.concatenate([rgb.transpose(2, 0, 1), extra], axis=0)).float()


class InputFeaturePatchDataset(Dataset):
    def __init__(self, samples, patch_size: int, patches_per_image: int, augment: bool, seed: int):
        self.samples = list(samples)
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.augment = augment
        self.seed = int(seed)

    def __len__(self):
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, index):
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
                image = np.clip((image.astype(np.float32) - 127.5) * contrast + 127.5 + brightness,
                                0.0, 255.0).astype(np.uint8)
        return input_features(image), torch.from_numpy(mask).long()


class SDFDataset(RareAwarePatchDataset):
    def __getitem__(self, index):
        image, mask = super().__getitem__(index)
        mask_np = mask.numpy()
        inside = mask_np == 3
        sdf = (distance_transform_edt(inside) - distance_transform_edt(~inside)).astype(np.float32)
        sdf /= max(mask_np.shape)
        return image, mask, torch.from_numpy(np.clip(sdf, -1.0, 1.0)).float()


class ResidualSCSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        gated = x * self.channel(x) * self.spatial(x)
        return x + self.scale * (gated - x)


class SCSEASPP(ASPPUNetPP):
    def __init__(self):
        super().__init__(pretrained=False)
        self.skip_gates = nn.ModuleList([ResidualSCSE(c) for c in (64, 64, 128, 256, 512)])

    def forward(self, x):
        input_size = x.shape[-2:]
        x0_0 = self.skip_gates[0](self.stem(x))
        x1_0 = self.skip_gates[1](self.enc1(self.pool(x0_0)))
        x2_0 = self.skip_gates[2](self.enc2(x1_0))
        x3_0 = self.skip_gates[3](self.enc3(x2_0))
        x4_0 = self.skip_gates[4](self.context(self.enc4(x3_0)))
        x3_1 = self.x3_1(torch.cat([self.up(x4_0, x3_0), x3_0], 1))
        x2_1 = self.x2_1(torch.cat([self.up(x3_0, x2_0), x2_0], 1))
        x1_1 = self.x1_1(torch.cat([self.up(x2_0, x1_0), x1_0], 1))
        x0_1 = self.x0_1(torch.cat([self.up(x1_0, x0_0), x0_0], 1))
        x2_2 = self.x2_2(torch.cat([self.up(x3_1, x2_0), x2_0, x2_1], 1))
        x1_2 = self.x1_2(torch.cat([self.up(x2_1, x1_0), x1_0, x1_1], 1))
        x0_2 = self.x0_2(torch.cat([self.up(x1_1, x0_0), x0_0, x0_1], 1))
        x1_3 = self.x1_3(torch.cat([self.up(x2_2, x1_0), x1_0, x1_1, x1_2], 1))
        x0_3 = self.x0_3(torch.cat([self.up(x1_2, x0_0), x0_0, x0_1, x0_2], 1))
        y = F.interpolate(x0_3, size=input_size, mode="bilinear", align_corners=False)
        stem_skip = F.interpolate(x0_0, size=input_size, mode="bilinear", align_corners=False)
        return self.head(torch.cat([y, stem_skip], 1))


class RawHighResASPP(ASPPUNetPP):
    def __init__(self):
        super().__init__(pretrained=False)
        self.raw_branch = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.raw_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        input_size = x.shape[-2:]
        x0_0 = self.stem(x)
        x1_0 = self.enc1(self.pool(x0_0))
        x2_0 = self.enc2(x1_0)
        x3_0 = self.enc3(x2_0)
        x4_0 = self.context(self.enc4(x3_0))
        x3_1 = self.x3_1(torch.cat([self.up(x4_0, x3_0), x3_0], 1))
        x2_1 = self.x2_1(torch.cat([self.up(x3_0, x2_0), x2_0], 1))
        x1_1 = self.x1_1(torch.cat([self.up(x2_0, x1_0), x1_0], 1))
        x0_1 = self.x0_1(torch.cat([self.up(x1_0, x0_0), x0_0], 1))
        x2_2 = self.x2_2(torch.cat([self.up(x3_1, x2_0), x2_0, x2_1], 1))
        x1_2 = self.x1_2(torch.cat([self.up(x2_1, x1_0), x1_0, x1_1], 1))
        x0_2 = self.x0_2(torch.cat([self.up(x1_1, x0_0), x0_0, x0_1], 1))
        x1_3 = self.x1_3(torch.cat([self.up(x2_2, x1_0), x1_0, x1_1, x1_2], 1))
        x0_3 = self.x0_3(torch.cat([self.up(x1_2, x0_0), x0_0, x0_1, x0_2], 1))
        x0_3 = x0_3 + self.raw_scale * self.raw_branch(x)
        y = F.interpolate(x0_3, size=input_size, mode="bilinear", align_corners=False)
        stem_skip = F.interpolate(x0_0, size=input_size, mode="bilinear", align_corners=False)
        return self.head(torch.cat([y, stem_skip], 1))


class SDFASPP(ASPPUNetPP):
    def __init__(self):
        super().__init__(pretrained=False)
        self.sdf_head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))

    def forward(self, x, return_aux: bool = False):
        input_size = x.shape[-2:]
        x0_0 = self.stem(x)
        x1_0 = self.enc1(self.pool(x0_0))
        x2_0 = self.enc2(x1_0)
        x3_0 = self.enc3(x2_0)
        x4_0 = self.context(self.enc4(x3_0))
        x3_1 = self.x3_1(torch.cat([self.up(x4_0, x3_0), x3_0], 1))
        x2_1 = self.x2_1(torch.cat([self.up(x3_0, x2_0), x2_0], 1))
        x1_1 = self.x1_1(torch.cat([self.up(x2_0, x1_0), x1_0], 1))
        x0_1 = self.x0_1(torch.cat([self.up(x1_0, x0_0), x0_0], 1))
        x2_2 = self.x2_2(torch.cat([self.up(x3_1, x2_0), x2_0, x2_1], 1))
        x1_2 = self.x1_2(torch.cat([self.up(x2_1, x1_0), x1_0, x1_1], 1))
        x0_2 = self.x0_2(torch.cat([self.up(x1_1, x0_0), x0_0, x0_1], 1))
        x1_3 = self.x1_3(torch.cat([self.up(x2_2, x1_0), x1_0, x1_1, x1_2], 1))
        x0_3 = self.x0_3(torch.cat([self.up(x1_2, x0_0), x0_0, x0_1, x0_2], 1))
        y = F.interpolate(x0_3, size=input_size, mode="bilinear", align_corners=False)
        stem_skip = F.interpolate(x0_0, size=input_size, mode="bilinear", align_corners=False)
        logits = self.head(torch.cat([y, stem_skip], 1))
        if not return_aux:
            return logits
        sdf = F.interpolate(self.sdf_head(x0_3), size=input_size, mode="bilinear", align_corners=False)
        return logits, sdf


class InputFilterASPP(ASPPUNetPP):
    def __init__(self):
        super().__init__(pretrained=False)
        old = self.stem[0]
        self.stem[0] = nn.Conv2d(8, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)


def build_model(candidate: str) -> nn.Module:
    if candidate == "input_filters":
        return InputFilterASPP()
    if candidate == "scse":
        return SCSEASPP()
    if candidate == "sdf":
        return SDFASPP()
    if candidate == "raw_highres":
        return RawHighResASPP()
    return ASPPUNetPP(pretrained=False)


def dice_loss(logits, target):
    probs = torch.softmax(logits, 1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    inter = (probs * one_hot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - inter / union.clamp_min(1e-6)
    if p > 1:
        jaccard = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]])
    return jaccard


def lovasz_softmax(probs, labels):
    losses = []
    flat_labels = labels.reshape(-1)
    flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)
    for class_id in range(NUM_CLASSES):
        foreground = (flat_labels == class_id).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - flat_probs[:, class_id]).abs()
        errors_sorted, order = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(foreground[order])))
    return torch.stack(losses).mean() if losses else probs.sum() * 0.0


def segmentation_loss(logits, target, mode: str, epoch: int, warmup_epochs: int):
    ce = F.cross_entropy(logits, target, weight=HARD_CLASS_WEIGHTS.to(logits.device))
    dice = dice_loss(logits, target)
    if mode == "lovasz" and epoch > warmup_epochs:
        return 0.5 * ce + 0.5 * dice + 0.35 * lovasz_softmax(torch.softmax(logits, 1), target)
    return ce + dice


def confusion(prediction, target):
    bins = target.reshape(-1) * NUM_CLASSES + prediction.reshape(-1)
    return np.bincount(bins, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics(predictions, targets):
    from local_image_metrics import summary
    return summary(predictions, targets, CLASS_NAMES)


def tensor_input(image: np.ndarray, candidate: str) -> torch.Tensor:
    return input_features(image) if candidate == "input_filters" else normalize_image(image)


@torch.no_grad()
def predict_probabilities(model, image_path, candidate, patch_size, stride, device, use_amp, tile_batch_size,
                          tta_mode="d4", blend_mode="gaussian"):
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    scale = max(1.0, float(patch_size) / min(height, width))
    if scale > 1.0:
        new_size = (max(patch_size, int(round(width * scale))), max(patch_size, int(round(height * scale))))
        scaled = np.asarray(Image.fromarray(image).resize(new_size, Image.Resampling.BICUBIC), dtype=np.uint8)
    else:
        scaled = image
    height, width = scaled.shape[:2]
    ys = [0] if height <= patch_size else list(range(0, height - patch_size + 1, stride))
    xs = [0] if width <= patch_size else list(range(0, width - patch_size + 1, stride))
    if ys[-1] != max(height - patch_size, 0): ys.append(max(height - patch_size, 0))
    if xs[-1] != max(width - patch_size, 0): xs.append(max(width - patch_size, 0))
    padded_h, padded_w = max(height, ys[-1] + patch_size), max(width, xs[-1] + patch_size)
    padded = np.pad(scaled, ((0, padded_h - height), (0, padded_w - width), (0, 0)), mode="reflect")
    probability_sum = np.zeros((NUM_CLASSES, padded_h, padded_w), np.float32)
    weight_sum = np.zeros((padded_h, padded_w), np.float32)
    window = blend_window(patch_size, blend_mode)
    locations = [(top, left) for top in ys for left in xs]
    transforms = views(tta_mode)
    for start in range(0, len(locations), tile_batch_size):
        batch_locations = locations[start:start + tile_batch_size]
        inputs = torch.stack([tensor_input(padded[top:top + patch_size, left:left + patch_size], candidate)
                              for top, left in batch_locations]).to(device, non_blocking=True)
        probability_batch = None
        for rotation, flip in transforms:
            view = transform_view(inputs, rotation, flip)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                value = torch.softmax(model(view), 1)
            value = inverse_view(value, rotation, flip)
            probability_batch = value if probability_batch is None else probability_batch + value
        probability_batch = (probability_batch / len(transforms)).float().cpu().numpy()
        for index, (top, left) in enumerate(batch_locations):
            probability_sum[:, top:top + patch_size, left:left + patch_size] += probability_batch[index] * window[None]
            weight_sum[top:top + patch_size, left:left + patch_size] += window
    probabilities = probability_sum / np.maximum(weight_sum[None], 1e-6)
    probabilities = probabilities[:, :height, :width]
    original_size = image.shape[1], image.shape[0]
    restored = np.stack([np.asarray(Image.fromarray(probabilities[c]).resize(original_size, Image.Resampling.BILINEAR), np.float32)
                         for c in range(NUM_CLASSES)])
    return restored / np.maximum(restored.sum(0, keepdims=True), 1e-7)


@torch.no_grad()
def collect_probabilities(model, samples, candidate, args, device, use_amp):
    probabilities, targets = [], []
    model.eval()
    for image_path, mask_path in _local_progress(samples, 'resnet34_refinements'):
        probabilities.append(predict_probabilities(model, image_path, candidate, args.patch_size, args.val_stride,
                                                   device, use_amp, args.tile_batch_size, args.tta, args.blend))
        targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
    return probabilities, targets


def score_cached(probabilities, targets, bias):
    predictions = []
    bias_array = np.asarray([0.0, 0.0, 0.0, bias], np.float32)[:, None, None]
    for value in probabilities:
        predictions.append((np.log(np.maximum(value, 1e-7)) + bias_array).argmax(0).astype(np.uint8))
    result = metrics(predictions, targets)
    result["al_bias"] = float(bias)
    return result


def update_ema(ema, model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda": torch.backends.cudnn.benchmark = True
    train_samples, valid_samples = collect_split(args.data_dir, "train"), collect_split(args.data_dir, "valid")
    if args.candidate == "input_filters":
        dataset = InputFeaturePatchDataset(train_samples, args.patch_size, args.patches_per_image, True, args.seed)
    elif args.candidate == "sdf":
        dataset = SDFDataset(train_samples, args.patch_size, args.patches_per_image, True, args.seed)
    else:
        dataset = RareAwarePatchDataset(train_samples, args.patch_size, args.patches_per_image, True, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    model = build_model(args.candidate)
    load_compatible(model, args.checkpoint)
    model.to(device)
    ema = copy.deepcopy(model).to(device).eval() if args.candidate == "ema" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    best_score, best_state, history = -1.0, None, []
    start = time.time()
    print(f"candidate={args.candidate} device={device} patches={len(dataset)} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in _local_progress(range(1, args.epochs + 1), 'resnet34_refinements'):
        model.train(); total, count = 0.0, 0
        for batch in _local_progress(loader, 'resnet34_refinements'):
            if args.candidate == "sdf":
                images, masks, sdf_target = [value.to(device, non_blocking=True) for value in batch]
            else:
                images, masks = [value.to(device, non_blocking=True) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                if args.candidate == "sdf":
                    logits, sdf = model(images, return_aux=True)
                    loss = segmentation_loss(logits, masks, args.candidate, epoch, args.warmup_epochs)
                    loss = loss + args.sdf_weight * F.smooth_l1_loss(torch.tanh(sdf[:, 0]), sdf_target)
                else:
                    logits = model(images)
                    loss = segmentation_loss(logits, masks, args.candidate, epoch, args.warmup_epochs)
            if use_amp:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            if ema is not None: update_ema(ema, model, args.ema_decay)
            total += float(loss.item()) * images.shape[0]; count += images.shape[0]
        scheduler.step()
        eval_model = ema if ema is not None else model
        probabilities, targets = collect_probabilities(eval_model, valid_samples, args.candidate, args, device, use_amp)
        valid = score_cached(probabilities, targets, args.eval_bias)
        history.append({"epoch": epoch, "train_loss": total / max(count, 1), "valid": valid,
                        "seconds": time.time() - start})
        (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        score = valid["image_miou_compat"]
        print(f"epoch={epoch:02d}/{args.epochs} loss={total/max(count,1):.4f} miou={score:.5f} "
              f"image={valid['image_mean_miou']:.5f} "
              + ",".join(f"{v:.4f}" for v in valid["image_class_iou"].values()), flush=True)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(eval_model.state_dict())
            torch.save({"model": best_state, "epoch": epoch, "metrics": valid, "args": vars(args)}, output / "best.pt")
    model.load_state_dict(best_state)
    model.eval()
    probabilities, targets = collect_probabilities(model, valid_samples, args.candidate, args, device, use_amp)
    sweep = [score_cached(probabilities, targets, float(bias)) for bias in np.arange(-0.80, 0.21, 0.05)]
    best = max(sweep, key=lambda item: item["image_miou_compat"])
    summary = {"candidate": args.candidate, "best_fixed_miou": best_score, "best_calibrated": best,
               "calibration_sweep": sweep, "elapsed_seconds": time.time() - start}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate": args.candidate, "best_fixed_miou": best_score, "best_calibrated": best}, ensure_ascii=False), flush=True)
    test_output = output / "test_masks"
    test_output.mkdir(exist_ok=True)
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        value = predict_probabilities(model, image_path, args.candidate, args.patch_size, args.val_stride,
                                      device, use_amp, args.tile_batch_size, args.tta, args.blend)
        bias = np.asarray([0.0, 0.0, 0.0, best["al_bias"]], np.float32)[:, None, None]
        encode_mask((np.log(np.maximum(value, 1e-7)) + bias).argmax(0).astype(np.uint8)).save(
            test_output / image_path.name.replace("_image.png", "_mask.png"))
    print(f"wrote test_masks={len(list(test_output.glob('*.png')))} dir={test_output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
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
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--eval-bias", type=float, default=-0.35)
    parser.add_argument("--sdf-weight", type=float, default=0.20)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--tta", choices=("legacy", "d4"), default="d4")
    parser.add_argument("--blend", choices=("uniform", "gaussian"), default="gaussian")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
