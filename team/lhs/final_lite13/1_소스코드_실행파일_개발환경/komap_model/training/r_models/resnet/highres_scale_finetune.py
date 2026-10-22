#!/usr/bin/env python3
"""Scale-controlled fine-tuning for the native ResNet34-ASPP branch.

Supports both crop-local and inference-matched full-image z-score/CDF
representations. The scale/patch controls can reproduce true multi-scale
training while keeping an approximately matched source-image field of view.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import affine_transform, gaussian_filter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from advanced_inference import blend_window, inverse_view, transform_view, views
from improve_five_causes import (
    FOCUS_PROBS,
    NUM_CLASSES,
    collect_split,
    crop_bounds,
    decode_mask,
    encode_mask,
    load_pair,
)
from raw_zcdf_aspp import (
    HARD_CLASS_WEIGHTS,
    dice_loss,
    representation,
    score,
    seed_everything,
)
from native_stem_finetune import Stride1NativeStemASPP


def resize_pair(image: np.ndarray, mask: np.ndarray, scale: float,
                minimum_side: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    factor = max(float(scale), float(minimum_side) / min(height, width))
    if abs(factor - 1.0) < 1e-6:
        return image, mask
    size = (int(round(width * factor)), int(round(height * factor)))
    image = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BICUBIC),
                       dtype=np.uint8)
    mask_rgb = encode_mask(mask).resize(size, Image.Resampling.NEAREST)
    return image, decode_mask(mask_rgb)


class HighResolutionPatchDataset(Dataset):
    def __init__(self, samples, patch_size: int, patches_per_image: int,
                 scale: float, seed: int, global_representation: bool = False,
                 metallography_augmentation: bool = False,
                 sampling_mode: str = "class_focused"):
        self.samples = list(samples)
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.scale = float(scale)
        self.seed = int(seed)
        self.global_representation = bool(global_representation)
        self.metallography_augmentation = bool(metallography_augmentation)
        if sampling_mode not in ("class_focused", "uniform_area"):
            raise ValueError(f"unsupported sampling mode: {sampling_mode}")
        self.sampling_mode = sampling_mode

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index % len(self.samples)]
        image, mask = load_pair(image_path, mask_path)
        image, mask = resize_pair(image, mask, self.scale, self.patch_size)
        height, width = mask.shape
        rng = random.Random(self.seed + index * 1009 + random.randrange(1_000_000))
        if self.sampling_mode == "uniform_area":
            # Uniform image coordinates make crop centers proportional to area,
            # matching the area-sampled crop policy of the B4 models.
            center_y, center_x = rng.randrange(height), rng.randrange(width)
        else:
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
        features = None
        if self.global_representation:
            # Inference computes z-score/CDF over the complete resized image.
            # Match that domain before cropping training features.
            if self.metallography_augmentation:
                image_float = image.astype(np.float32) / 255.0
                if rng.random() < 0.80:
                    contrast = rng.uniform(0.75, 1.25)
                    brightness = rng.uniform(-0.06, 0.06)
                    image_float = np.clip(
                        (image_float - 0.5) * contrast + 0.5 + brightness,
                        0.0, 1.0)
                if rng.random() < 0.65:
                    image_float = np.power(image_float, rng.uniform(0.80, 1.20))
                image = np.clip(image_float * 255.0, 0.0, 255.0).astype(np.uint8)
                if rng.random() < 0.15:
                    sigma = rng.uniform(0.25, 0.70)
                    image = np.clip(gaussian_filter(
                        image.astype(np.float32), sigma=(sigma, sigma, 0.0)),
                        0.0, 255.0).astype(np.uint8)
                if rng.random() < 0.20:
                    noise_std = rng.uniform(1.0, 4.0)
                    noise_rng = np.random.default_rng(rng.getrandbits(64))
                    noise = noise_rng.normal(0.0, noise_std, image.shape[:2])[..., None]
                    image = np.clip(image.astype(np.float32) + noise,
                                    0.0, 255.0).astype(np.uint8)
            elif rng.random() < 0.45:
                contrast = rng.uniform(0.94, 1.06)
                brightness = rng.uniform(-0.025, 0.025) * 255.0
                image = np.clip((image.astype(np.float32) - 127.5) * contrast
                                + 127.5 + brightness, 0.0, 255.0).astype(np.uint8)
            features = representation(image)
            features = features[:, top:top + self.patch_size,
                                left:left + self.patch_size]
        else:
            image = image[top:top + self.patch_size, left:left + self.patch_size]
        mask = mask[top:top + self.patch_size, left:left + self.patch_size]

        if rng.random() < 0.5:
            if features is not None:
                features = np.flip(features, 2).copy()
            else:
                image = np.flip(image, 1).copy()
            mask = np.flip(mask, 1).copy()
        if rng.random() < 0.5:
            if features is not None:
                features = np.flip(features, 1).copy()
            else:
                image = np.flip(image, 0).copy()
            mask = np.flip(mask, 0).copy()
        turns = rng.randrange(4)
        if turns:
            if features is not None:
                features = np.rot90(features, turns, axes=(1, 2)).copy()
            else:
                image = np.rot90(image, turns).copy()
            mask = np.rot90(mask, turns).copy()
        if self.metallography_augmentation:
            angle = rng.uniform(-np.pi, np.pi)
            zoom = rng.uniform(0.70, 1.40)
            cosine, sine = np.cos(angle), np.sin(angle)
            matrix = np.asarray([[cosine, sine], [-sine, cosine]],
                                dtype=np.float64) / zoom
            center = np.asarray([(self.patch_size - 1) / 2.0,
                                 (self.patch_size - 1) / 2.0])
            offset = center - matrix @ center
            features = np.stack([
                affine_transform(channel, matrix, offset,
                                 output_shape=(self.patch_size, self.patch_size),
                                 order=1, mode="reflect", prefilter=False)
                for channel in features
            ]).astype(np.float32)
            mask = affine_transform(
                mask, matrix, offset,
                output_shape=(self.patch_size, self.patch_size),
                order=0, mode="reflect", prefilter=False).astype(np.uint8)
        if features is None:
            if rng.random() < 0.45:
                contrast = rng.uniform(0.94, 1.06)
                brightness = rng.uniform(-0.025, 0.025) * 255.0
                image = np.clip((image.astype(np.float32) - 127.5) * contrast
                                + 127.5 + brightness, 0.0, 255.0).astype(np.uint8)
            features = representation(image)
        return torch.from_numpy(np.ascontiguousarray(features)), torch.from_numpy(mask).long()


def freeze_batchnorm_stats(model: nn.Module) -> None:
    """Batch-1 crops cannot update pooled 1x1 BN statistics reliably."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


@torch.no_grad()
def predict_scaled(model, image_path: Path, patch_size: int, stride: int,
                   scale: float, device: torch.device, use_amp: bool,
                   tta: str) -> np.ndarray:
    source = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    scaled, _ = resize_pair(source, np.zeros(source.shape[:2], dtype=np.int64),
                            scale, patch_size)
    height, width = scaled.shape[:2]
    features = representation(scaled)
    ys = [0] if height <= patch_size else list(range(0, height - patch_size + 1, stride))
    xs = [0] if width <= patch_size else list(range(0, width - patch_size + 1, stride))
    last_y, last_x = max(height - patch_size, 0), max(width - patch_size, 0)
    if ys[-1] != last_y:
        ys.append(last_y)
    if xs[-1] != last_x:
        xs.append(last_x)
    padded_height, padded_width = max(height, ys[-1] + patch_size), max(width, xs[-1] + patch_size)
    padded = np.pad(features, ((0, 0), (0, padded_height - height),
                               (0, padded_width - width)), mode="reflect")
    probability_sum = np.zeros((NUM_CLASSES, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    window = blend_window(patch_size, "gaussian")
    locations = [(top, left) for top in ys for left in xs]
    transforms = [(0, False)] if tta == "none" else views(tta)
    for top, left in locations:
        inputs = torch.from_numpy(padded[:, top:top + patch_size, left:left + patch_size])
        inputs = inputs.unsqueeze(0).to(device, non_blocking=True)
        probabilities = None
        for rotation, flip in transforms:
            transformed = transform_view(inputs, rotation, flip)
            with autocast(enabled=use_amp):
                value = torch.softmax(model(transformed), dim=1)
            value = inverse_view(value, rotation, flip)
            probabilities = value if probabilities is None else probabilities + value
        value = (probabilities / len(transforms)).float().cpu().numpy()[0]
        probability_sum[:, top:top + patch_size, left:left + patch_size] += value * window[None]
        weight_sum[top:top + patch_size, left:left + patch_size] += window
    probability_sum = probability_sum[:, :height, :width] / np.maximum(
        weight_sum[None, :height, :width], 1e-7)
    original_size = (source.shape[1], source.shape[0])
    restored = np.stack([
        np.asarray(Image.fromarray(probability_sum[c]).resize(
            original_size, Image.Resampling.BILINEAR), dtype=np.float32)
        for c in range(NUM_CLASSES)
    ])
    return restored / np.maximum(restored.sum(0, keepdims=True), 1e-7)


def evaluate(model, samples, args, device, use_amp, output_probs=False):
    model.eval()
    probabilities, targets = [], []
    for image_path, mask_path in samples:
        probabilities.append(predict_scaled(
            model, image_path, args.patch_size, args.val_stride, args.scale,
            device, use_amp, args.tta))
        targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
        print(f"validated {image_path.name}", flush=True)
    predictions = [value.argmax(0).astype(np.uint8) for value in probabilities]
    result = score(predictions, targets)
    if output_probs:
        result["probabilities"] = probabilities
    return result


def lovasz_gradient(sorted_foreground: torch.Tensor) -> torch.Tensor:
    count = sorted_foreground.numel()
    total = sorted_foreground.sum()
    intersection = total - sorted_foreground.cumsum(0)
    union = total + (1.0 - sorted_foreground).cumsum(0)
    gradient = 1.0 - intersection / union.clamp_min(1e-7)
    if count > 1:
        gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
    return gradient


def imagewise_cross_entropy_loss(logits: torch.Tensor,
                                 target: torch.Tensor) -> torch.Tensor:
    """Average pixel CE per present class, then per image, matching macro evaluation."""
    pixel_loss = F.cross_entropy(logits.float(), target, reduction="none")
    image_losses = []
    for batch_index in range(target.shape[0]):
        labels = target[batch_index].reshape(-1)
        losses = pixel_loss[batch_index].reshape(-1)
        counts = torch.bincount(labels, minlength=NUM_CLASSES).to(losses.dtype)
        class_losses = torch.stack([
            (losses * (labels == class_id)).sum() / counts[class_id].clamp_min(1.0)
            for class_id in range(NUM_CLASSES)
        ])
        present_count = (counts > 0).sum().clamp_min(1)
        image_losses.append(class_losses.sum() / present_count)
    return torch.stack(image_losses).mean()


def imagewise_soft_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth per-image, present-class IoU loss aligned with the evaluation macro."""
    probabilities = torch.softmax(logits.float(), dim=1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).to(probabilities.dtype)
    intersection = (probabilities * one_hot).sum(dim=(2, 3))
    union = probabilities.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3)) - intersection
    class_iou = (intersection + 1.0) / (union + 1.0)
    present = one_hot.sum(dim=(2, 3)) > 0
    per_image = (1.0 - class_iou).masked_fill(~present, 0.0).sum(dim=1)
    per_image = per_image / present.sum(dim=1).clamp_min(1)
    return per_image.mean()


def binary_foreground_boundary_loss(logits: torch.Tensor, target: torch.Tensor,
                                    boundary_radius: int = 3,
                                    boundary_weight: float = 2.0) -> torch.Tensor:
    """Balanced secondary-vs-Al loss with extra weight near the GT phase boundary."""
    logits = logits.float()
    secondary_logit = torch.logsumexp(logits[:, (0, 1, 3)], dim=1) - logits[:, 2]
    foreground = (target != 2).to(logits.dtype)
    kernel = 2 * int(boundary_radius) + 1
    foreground_4d = foreground.unsqueeze(1)
    padding = (boundary_radius,) * 4
    dilated = F.max_pool2d(F.pad(foreground_4d, padding, value=0.0),
                           kernel_size=kernel, stride=1)
    eroded = -F.max_pool2d(F.pad(-foreground_4d, padding, value=0.0),
                           kernel_size=kernel, stride=1)
    boundary_band = (dilated > eroded).squeeze(1).to(logits.dtype)
    pixel_bce = F.binary_cross_entropy_with_logits(
        secondary_logit, foreground, reduction="none")

    spatial_dims = (1, 2)
    positive = foreground.sum(dim=spatial_dims, keepdim=True)
    negative = (1.0 - foreground).sum(dim=spatial_dims, keepdim=True)
    positive_weight = negative / positive.clamp_min(1.0)
    balance = torch.where(foreground > 0.5, positive_weight, torch.ones_like(foreground))
    edge_weight = 1.0 + max(float(boundary_weight) - 1.0, 0.0) * boundary_band
    combined_weight = balance * edge_weight
    bce_per_image = (pixel_bce * combined_weight).sum(dim=spatial_dims) / \
        combined_weight.sum(dim=spatial_dims).clamp_min(1.0)

    probability = torch.sigmoid(secondary_logit)
    intersection = (probability * foreground).sum(dim=spatial_dims)
    union = probability.sum(dim=spatial_dims) + foreground.sum(dim=spatial_dims) - intersection
    iou_per_image = 1.0 - (intersection + 1.0) / (union + 1.0)
    return (bce_per_image + iou_per_image).mean()


def imagewise_lovasz_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Present-class Lovasz-Softmax, averaged per image to match the metric."""
    probabilities = torch.softmax(logits.float(), dim=1)
    image_losses = []
    for batch_index in range(target.shape[0]):
        labels = target[batch_index].reshape(-1)
        class_probabilities = probabilities[batch_index].permute(1, 2, 0).reshape(
            -1, NUM_CLASSES)
        class_losses = []
        present_classes = torch.nonzero(
            torch.bincount(labels, minlength=NUM_CLASSES), as_tuple=False).flatten().tolist()
        for class_id in present_classes:
            foreground = (labels == class_id).to(probabilities.dtype)
            errors = (foreground - class_probabilities[:, class_id]).abs()
            sorted_errors, permutation = torch.sort(errors, descending=True)
            class_losses.append(torch.dot(
                sorted_errors, lovasz_gradient(foreground[permutation])))
        if class_losses:
            image_losses.append(torch.stack(class_losses).mean())
    if not image_losses:
        return probabilities.sum() * 0.0
    return torch.stack(image_losses).mean()


def memory_probe(args):
    if not torch.cuda.is_available():
        raise RuntimeError("--memory-probe requires CUDA")
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
    model = Stride1NativeStemASPP().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.train()
    freeze_batchnorm_stats(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.decoder_lr,
                                  weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=not args.no_amp)
    images = torch.randn(1, 3, args.patch_size, args.patch_size, device=device)
    masks = torch.randint(NUM_CLASSES, (1, args.patch_size, args.patch_size), device=device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    try:
        with autocast(enabled=not args.no_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, masks, weight=HARD_CLASS_WEIGHTS.to(device)) \
                + dice_loss(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        print(json.dumps({
            "probe": "one synthetic train step",
            "patch_size": args.patch_size,
            "batch_size": 1,
            "memory_fraction": args.memory_fraction,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
            "loss": float(loss.detach()),
        }), flush=True)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(json.dumps({
            "probe": "OOM under capped allocation",
            "patch_size": args.patch_size,
            "memory_fraction": args.memory_fraction,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        }), flush=True)
        raise
    finally:
        del images, masks, model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def export_valid(args):
    """Export official-resolution validation probabilities from a saved model."""
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
    valid_samples = collect_split(args.data_dir, "valid")
    model = Stride1NativeStemASPP().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    result = evaluate(model, valid_samples, args, device, use_amp, output_probs=True)
    probabilities = result.pop("probabilities")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "valid_probs.npz", **{
        image_path.stem: value.astype(np.float16)
        for (image_path, _), value in zip(valid_samples, probabilities)
    })
    summary = {"checkpoint": str(args.checkpoint), "evaluation": result,
               "scale_factor": args.scale, "patch_size": args.patch_size,
               "tta": args.tta, "val_stride": args.val_stride}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "global_miou", "global_iou", "image_mean_miou",
        "image_mean_miou_exclude")}, ensure_ascii=False), flush=True)


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
        torch.backends.cudnn.benchmark = True
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    model = Stride1NativeStemASPP().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    encoder, decoder, native = [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith(("native_stem.", "native_to_head.")):
            native.append(parameter)
        elif name.startswith(("stem.", "enc1.", "enc2.", "enc3.", "enc4.")):
            encoder.append(parameter)
        else:
            decoder.append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": args.encoder_lr},
        {"params": decoder, "lr": args.decoder_lr},
        {"params": native, "lr": args.native_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.scheduler_epochs or args.epochs)
    scaler = GradScaler(enabled=use_amp)
    if args.metallography_augmentation and not args.global_representation:
        raise ValueError("--metallography-augmentation requires --global-representation")
    dataset = HighResolutionPatchDataset(
        train_samples, args.patch_size, args.patches_per_image, args.scale, args.seed,
        args.global_representation, args.metallography_augmentation,
        args.sampling_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=(args.num_workers > 0))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_score, best_state, best_record = -1.0, None, None
    history, start_time = [], time.time()
    print(f"device={device} scale={args.scale} patch={args.patch_size} "
          f"patches={len(dataset)} batch={args.batch_size} tta={args.tta}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        freeze_batchnorm_stats(model)
        total, count = 0.0, 0
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(images)
                if args.imagewise_cross_entropy:
                    cross_entropy = imagewise_cross_entropy_loss(logits, masks)
                else:
                    cross_entropy = F.cross_entropy(
                        logits, masks, weight=HARD_CLASS_WEIGHTS.to(device))
                hard_loss = cross_entropy + dice_loss(logits, masks)
                if args.imagewise_lovasz_weight > 0.0:
                    hard_loss = hard_loss + args.imagewise_lovasz_weight * imagewise_lovasz_loss(
                        logits, masks)
                if args.imagewise_soft_iou_weight > 0.0:
                    hard_loss = hard_loss + args.imagewise_soft_iou_weight * imagewise_soft_iou_loss(
                        logits, masks)
                if args.binary_foreground_weight > 0.0:
                    hard_loss = hard_loss + args.binary_foreground_weight * \
                        binary_foreground_boundary_loss(
                            logits, masks, args.boundary_radius, args.boundary_weight)
            if use_amp:
                scaler.scale(hard_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                hard_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(hard_loss.detach()) * images.shape[0]
            count += images.shape[0]
        valid = evaluate(model, valid_samples, args, device, use_amp)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": total / max(count, 1),
                  "global_miou": valid["global_miou"],
                  "image_mean_miou": valid["image_mean_miou"],
                  "image_mean_miou_exclude": valid["image_mean_miou_exclude"],
                  "global_iou": valid["global_iou"], "seconds": time.time() - start_time}
        history.append(record)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if record[args.selection_metric] > best_score:
            best_score, best_state, best_record = (
                float(record[args.selection_metric]), copy.deepcopy(model.state_dict()), record)
            torch.save({"model": best_state, "epoch": epoch, "metrics": valid,
                        "args": vars(args), "parent_checkpoint": str(args.checkpoint),
                        "scale_factor": args.scale}, output / "best.pt")
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    final = evaluate(model, valid_samples, args, device, use_amp, output_probs=True)
    probs = final.pop("probabilities")
    np.savez_compressed(output / "valid_probs.npz", **{
        path.stem: value.astype(np.float16)
        for (path, _), value in zip(valid_samples, probs)
    })
    base_loss_name = ("image-wise present-class CE + Dice" if args.imagewise_cross_entropy
                      else "weighted CE + Dice")
    summary = {"method": "supervised fine-tuning",
               "loss": base_loss_name + (
                   f" + {args.imagewise_lovasz_weight} x per-image present-class Lovasz-Softmax"
                   if args.imagewise_lovasz_weight > 0.0 else "") + (
                   f" + {args.imagewise_soft_iou_weight} x per-image present-class Soft-IoU"
                   if args.imagewise_soft_iou_weight > 0.0 else "") + (
                   f" + {args.binary_foreground_weight} x binary secondary-vs-Al boundary loss "
                   f"(radius={args.boundary_radius}, band_weight={args.boundary_weight})"
                   if args.binary_foreground_weight > 0.0 else ""),
               "selection_metric": args.selection_metric, "best_selection": best_score,
               "best_record": best_record, "final": final,
               "elapsed_seconds": time.time() - start_time,
               "parent_checkpoint": str(args.checkpoint),
               "scale_factor": args.scale, "patch_size": args.patch_size,
               "training_representation": (
                   "full-image statistics before crop" if args.global_representation
                   else "crop-local statistics")}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    del loader, dataset, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--patch-size", type=int, default=1536)
    parser.add_argument("--patches-per-image", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--scheduler-epochs", type=int, default=None,
                        help="cosine schedule length; useful for matched partial runs")
    parser.add_argument("--encoder-lr", type=float, default=2e-6)
    parser.add_argument("--decoder-lr", type=float, default=6e-6)
    parser.add_argument("--native-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--imagewise-lovasz-weight", type=float, default=0.0,
                        help="add per-image present-class Lovasz-Softmax to the supervised loss")
    parser.add_argument("--imagewise-cross-entropy", action="store_true",
                        help="replace weighted CE with per-image present-class macro CE")
    parser.add_argument("--imagewise-soft-iou-weight", type=float, default=0.0,
                        help="add per-image present-class soft-IoU loss")
    parser.add_argument("--binary-foreground-weight", type=float, default=0.0,
                        help="add balanced secondary-vs-Al boundary BCE + image-wise Soft-IoU")
    parser.add_argument("--boundary-radius", type=int, default=3,
                        help="half-width in pixels of the boundary emphasis band")
    parser.add_argument("--boundary-weight", type=float, default=2.0,
                        help="relative pixel weight inside the ground-truth boundary band")
    parser.add_argument("--val-stride", type=int, default=864)
    parser.add_argument("--tta", choices=("none", "d4"), default="d4")
    parser.add_argument("--selection-metric", choices=("image_mean_miou",
                                                         "image_mean_miou_exclude"),
                        default="image_mean_miou")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=253)
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    parser.add_argument("--memory-probe", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--global-representation", action="store_true",
                        help="compute z-score/CDF over the whole resized image before cropping")
    parser.add_argument("--metallography-augmentation", action="store_true",
                        help="add random 0.7–1.4x affine, arbitrary rotations, and stronger intensity augmentation")
    parser.add_argument("--sampling-mode", choices=("class_focused", "uniform_area"),
                        default="class_focused",
                        help="choose class-focused crops or sample crop centers uniformly by image area")
    parser.add_argument("--no-amp", action="store_true")
    arguments = parser.parse_args()
    if arguments.memory_probe:
        memory_probe(arguments)
    elif arguments.eval_only:
        export_valid(arguments)
    else:
        main(arguments)
