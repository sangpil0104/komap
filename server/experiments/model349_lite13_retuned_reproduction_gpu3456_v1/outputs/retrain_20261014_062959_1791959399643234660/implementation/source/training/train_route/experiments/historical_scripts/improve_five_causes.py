#!/usr/bin/env python3
"""High-resolution, aspect-preserving improvement for the KoMaP baseline.

This experiment addresses the five diagnosed bottlenecks together:
  1. preserve the native aspect ratio instead of resizing every image to a square;
  2. train on native-resolution patches and infer with overlapping tiles;
  3. oversample Primary Si/Eutectic Si patches and use corrected mild class weights;
  4. report both global-pixel and image-wise mIoU;
  5. optionally create grouped folds by specimen A-ID for leakage-aware CV.

The fixed-split training path starts from model 6's best checkpoint so that the
change can be compared directly with the existing 75.38% TTA result.
"""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import affine_transform, gaussian_filter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from model1_unet_resnet34 import (
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    decode_mask,
    encode_mask,
    seed_everything,
)
from train_models_2_to_6 import UNetPlusPlusResNet34


# Internal channel order is [Al3Ni, Primary Si, Al, Eutectic Si].  These
# weights are written in that order; the earlier experiments mistakenly used
# the canonical-name order and consequently emphasized the wrong channels.
PATCH_CLASS_WEIGHTS = torch.tensor([1.15, 1.45, 0.70, 2.00], dtype=torch.float32)
FOCUS_PROBS = ((3, 0.45), (1, 0.25), (0, 0.20), (2, 0.10))


def normalize_image(image_np: np.ndarray) -> torch.Tensor:
    image_float = image_np.astype(np.float32) / 255.0
    image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image_float.transpose(2, 0, 1)).float()


def load_pair(image_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = decode_mask(Image.open(mask_path).convert("RGB"))
    return image, mask


def scaled_pair(image: np.ndarray, mask: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Scale only when the short side is smaller than a patch; never distort ratio."""
    height, width = image.shape[:2]
    scale = max(1.0, float(patch_size) / float(min(height, width)))
    if abs(scale - 1.0) < 1e-6:
        return image, mask
    new_size = (max(patch_size, int(round(width * scale))), max(patch_size, int(round(height * scale))))
    image_out = np.asarray(Image.fromarray(image).resize(new_size, Image.Resampling.BICUBIC), dtype=np.uint8)
    mask_out = decode_mask(encode_mask(mask).resize(new_size, Image.Resampling.NEAREST))
    return image_out, mask_out


def crop_bounds(height: int, width: int, patch_size: int, center_y: int, center_x: int) -> Tuple[int, int]:
    top = int(np.clip(center_y - patch_size // 2, 0, max(height - patch_size, 0)))
    left = int(np.clip(center_x - patch_size // 2, 0, max(width - patch_size, 0)))
    return top, left


class RareAwarePatchDataset(Dataset):
    """Native-resolution crops with explicit rare-class/foreground sampling."""

    def __init__(
        self,
        samples: Sequence[Tuple[Path, Path]],
        patch_size: int,
        patches_per_image: int,
        augment: bool,
        seed: int,
        scale: float = 1.0,
        sampling_mode: str = "class_focused",
        metallography_augmentation: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.augment = augment
        self.seed = int(seed)
        self.scale = float(scale)
        self.sampling_mode = sampling_mode
        self.metallography_augmentation = bool(metallography_augmentation)
        if not self.samples:
            raise ValueError("No image/mask samples supplied")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.sampling_mode not in ("class_focused", "uniform_area", "area_weighted"):
            raise ValueError(f"unsupported sampling mode: {self.sampling_mode}")
        self.sample_area_cdf = None
        if self.sampling_mode == "area_weighted":
            areas = []
            for image_path, _ in self.samples:
                with Image.open(image_path) as image:
                    width, height = image.size
                areas.append(float(width * height))
            cumulative = np.cumsum(areas, dtype=np.float64)
            self.sample_area_cdf = cumulative / cumulative[-1]

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index % len(self.samples)]
        # Make workers deterministic enough for reproducible comparisons while
        # still drawing different crops on repeated visits.
        rng = random.Random(self.seed + index * 1009 + random.randrange(1_000_000))
        if self.sampling_mode == "area_weighted":
            sample_index = min(int(np.searchsorted(self.sample_area_cdf, rng.random(), side="right")),
                               len(self.samples) - 1)
            image_path, mask_path = self.samples[sample_index]
        image, mask = load_pair(image_path, mask_path)
        height, width = mask.shape
        # Cropping in source coordinates then magnifying only the crop gives
        # the same field of view without allocating a full 4x image per worker.
        effective_scale = max(1.0, self.scale,
                             float(self.patch_size) / float(min(height, width)))
        source_crop = max(1, int(round(self.patch_size / effective_scale)))
        source_crop = min(source_crop, height, width)

        if self.sampling_mode in ("uniform_area", "area_weighted"):
            center_y, center_x = rng.randrange(height), rng.randrange(width)
        else:
            focus_class = None
            if rng.random() < 0.90:
                draw = rng.random()
                cumulative = 0.0
                for class_id, probability in FOCUS_PROBS:
                    cumulative += probability
                    if draw <= cumulative:
                        focus_class = class_id
                        break
            if focus_class is not None:
                ys, xs = np.where(mask == focus_class)
            else:
                ys, xs = np.where(mask != 2)
            if len(xs) == 0:
                center_y, center_x = height // 2, width // 2
            else:
                selected = rng.randrange(len(xs))
                center_y, center_x = int(ys[selected]), int(xs[selected])
        top, left = crop_bounds(height, width, source_crop, center_y, center_x)
        image = image[top:top + source_crop, left:left + source_crop]
        mask = mask[top:top + source_crop, left:left + source_crop]
        if image.shape[:2] != (self.patch_size, self.patch_size):
            image = np.asarray(Image.fromarray(image).resize(
                (self.patch_size, self.patch_size), Image.Resampling.BICUBIC), dtype=np.uint8)
            mask = decode_mask(encode_mask(mask).resize(
                (self.patch_size, self.patch_size), Image.Resampling.NEAREST))

        if self.augment:
            if rng.random() < 0.5:
                image, mask = np.flip(image, axis=1).copy(), np.flip(mask, axis=1).copy()
            if rng.random() < 0.5:
                image, mask = np.flip(image, axis=0).copy(), np.flip(mask, axis=0).copy()
            quarter_turns = rng.randrange(4)
            if quarter_turns:
                image = np.rot90(image, quarter_turns).copy()
                mask = np.rot90(mask, quarter_turns).copy()
            if self.metallography_augmentation:
                if rng.random() < 0.80:
                    contrast = rng.uniform(0.75, 1.25)
                    brightness = rng.uniform(-0.06, 0.06) * 255.0
                    image = np.clip((image.astype(np.float32) - 127.5) * contrast
                                    + 127.5 + brightness, 0.0, 255.0).astype(np.uint8)
                if rng.random() < 0.65:
                    gamma = rng.uniform(0.80, 1.20)
                    image = np.clip(255.0 * (image.astype(np.float32) / 255.0) ** gamma,
                                    0.0, 255.0).astype(np.uint8)
                if rng.random() < 0.15:
                    sigma = rng.uniform(0.25, 0.70)
                    image = np.clip(gaussian_filter(
                        image.astype(np.float32), sigma=(sigma, sigma, 0.0)),
                        0.0, 255.0).astype(np.uint8)
                if rng.random() < 0.20:
                    noise_std = rng.uniform(1.0, 4.0)
                    noise = np.random.default_rng(rng.getrandbits(64)).normal(
                        0.0, noise_std, image.shape[:2])[..., None]
                    image = np.clip(image.astype(np.float32) + noise,
                                    0.0, 255.0).astype(np.uint8)
                angle = rng.uniform(-np.pi, np.pi)
                zoom = rng.uniform(0.70, 1.40)
                cosine, sine = np.cos(angle), np.sin(angle)
                matrix = np.asarray([[cosine, sine], [-sine, cosine]],
                                    dtype=np.float64) / zoom
                center = np.asarray([(self.patch_size - 1) / 2.0,
                                     (self.patch_size - 1) / 2.0])
                offset = center - matrix @ center
                image = np.stack([
                    affine_transform(image[..., channel].astype(np.float32), matrix, offset,
                                     output_shape=(self.patch_size, self.patch_size),
                                     order=1, mode="reflect", prefilter=False)
                    for channel in range(image.shape[2])
                ], axis=-1).clip(0.0, 255.0).astype(np.uint8)
                mask = affine_transform(
                    mask, matrix, offset,
                    output_shape=(self.patch_size, self.patch_size),
                    order=0, mode="reflect", prefilter=False).astype(np.uint8)
            elif rng.random() < 0.45:
                contrast = rng.uniform(0.94, 1.06)
                brightness = rng.uniform(-0.025, 0.025) * 255.0
                image = np.clip((image.astype(np.float32) - 127.5) * contrast + 127.5 + brightness,
                                0.0, 255.0).astype(np.uint8)

        return normalize_image(image), torch.from_numpy(mask).long()


def collect_split(data_dir: str | Path, split: str) -> List[Tuple[Path, Path]]:
    root = Path(data_dir) / split
    samples = []
    for image_path in sorted((root / "images").glob("*_image.png")):
        mask_path = root / "masks" / image_path.name.replace("_image.png", "_mask.png")
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        samples.append((image_path, mask_path))
    return samples


def specimen_id(path: Path) -> str:
    match = re.match(r"(A\d+)_", path.name)
    return match.group(1) if match else path.stem.split("_")[0]


def grouped_folds(samples: Sequence[Tuple[Path, Path]], n_folds: int, seed: int) -> List[List[int]]:
    groups: Dict[str, List[int]] = {}
    for index, (image_path, _) in enumerate(samples):
        groups.setdefault(specimen_id(image_path), []).append(index)
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for _, indices in sorted(group_items, key=lambda item: len(item[1]), reverse=True):
        fold = min(range(n_folds), key=lambda i: fold_sizes[i])
        folds[fold].extend(indices)
        fold_sizes[fold] += len(indices)
    return folds


def tile_starts(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tensor_batch(patches: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.stack([normalize_image(patch) for patch in patches]).to(device, non_blocking=True)


@torch.no_grad()
def predict_full_image(
    model: torch.nn.Module,
    image_path: Path,
    patch_size: int,
    stride: int,
    device: torch.device,
    use_amp: bool,
    tile_batch_size: int,
    tta: bool = False,
) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), patch_size)
    height, width = scaled.shape[:2]
    ys, xs = tile_starts(height, patch_size, stride), tile_starts(width, patch_size, stride)
    padded_height = max(height, ys[-1] + patch_size)
    padded_width = max(width, xs[-1] + patch_size)
    padded = np.pad(scaled, ((0, padded_height - height), (0, padded_width - width), (0, 0)), mode="reflect")
    probability_sum = np.zeros((NUM_CLASSES, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)

    locations: List[Tuple[int, int]] = [(top, left) for top in ys for left in xs]
    for start in range(0, len(locations), tile_batch_size):
        batch_locations = locations[start:start + tile_batch_size]
        patches = [padded[top:top + patch_size, left:left + patch_size] for top, left in batch_locations]
        inputs = tensor_batch(patches, device)
        with autocast(enabled=use_amp):
            probabilities = torch.softmax(model(inputs), dim=1)
            if tta:
                views = [torch.flip(inputs, dims=(-1,)), torch.flip(inputs, dims=(-2,)),
                         torch.rot90(inputs, 1, dims=(-2, -1))]
                undo = [lambda x: torch.flip(x, dims=(-1,)), lambda x: torch.flip(x, dims=(-2,)),
                        lambda x: torch.rot90(x, -1, dims=(-2, -1))]
                for view, undo_fn in zip(views, undo):
                    probabilities = probabilities + undo_fn(torch.softmax(model(view), dim=1))
                probabilities = probabilities / 4.0
        probabilities = probabilities.float().cpu().numpy()
        for tile_index, (top, left) in enumerate(batch_locations):
            probability_sum[:, top:top + patch_size, left:left + patch_size] += probabilities[tile_index]
            weight_sum[top:top + patch_size, left:left + patch_size] += 1.0

    scaled_prediction = (probability_sum / np.maximum(weight_sum[None], 1e-6)).argmax(axis=0).astype(np.uint8)
    scaled_prediction = scaled_prediction[:height, :width]
    return np.asarray(Image.fromarray(scaled_prediction).resize(image.shape[1::-1], Image.Resampling.NEAREST), dtype=np.uint8)


def confusion_for(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    bins = target.reshape(-1) * NUM_CLASSES + prediction.reshape(-1)
    return np.bincount(bins, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_predictions(predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> Dict[str, object]:
    from local_image_metrics import summary
    return summary(predictions, targets, CLASS_NAMES)


@torch.no_grad()
def evaluate_full(
    model: torch.nn.Module,
    samples: Sequence[Tuple[Path, Path]],
    patch_size: int,
    stride: int,
    device: torch.device,
    use_amp: bool,
    tile_batch_size: int,
    tta: bool,
) -> Dict[str, object]:
    model.eval()
    predictions, targets, names = [], [], []
    for image_path, mask_path in samples:
        prediction = predict_full_image(model, image_path, patch_size, stride, device, use_amp, tile_batch_size, tta)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        predictions.append(prediction)
        targets.append(target)
        names.append(image_path.name)
    result = metrics_from_predictions(predictions, targets)
    result["names"] = names
    result["tta"] = bool(tta)
    result["stride"] = int(stride)
    return result


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    denominator = probabilities.sum(dims) + one_hot.sum(dims)
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def patch_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = PATCH_CLASS_WEIGHTS.to(logits.device)
    return F.cross_entropy(logits, target, weight=weights) + dice_loss(logits, target)


def split_parameters(model: torch.nn.Module):
    prefixes = ("stem.", "enc1.", "enc2.", "enc3.", "enc4.")
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith(prefixes) else decoder).append(parameter)
    return encoder, decoder


def load_checkpoint(model: torch.nn.Module, checkpoint: str | Path, device: torch.device) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])


def train_fixed(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    train_ds = RareAwarePatchDataset(train_samples, args.patch_size, args.patches_per_image, True, args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    model = UNetPlusPlusResNet34(pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device)
    scaler = GradScaler(enabled=use_amp)
    encoder, decoder = split_parameters(model)
    for parameter in encoder:
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW([{"params": decoder, "lr": args.decoder_lr}], weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    stale = 0
    history = []
    total_epochs = args.freeze_epochs + args.finetune_epochs
    start_time = time.time()
    print(f"device={device} train_patches={len(train_ds)} valid_images={len(valid_samples)} "
          f"patch={args.patch_size} stride={args.val_stride} weights={PATCH_CLASS_WEIGHTS.tolist()}", flush=True)

    for epoch in range(1, total_epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for parameter in encoder:
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(
                [{"params": encoder, "lr": args.encoder_lr}, {"params": decoder, "lr": args.decoder_lr}],
                weight_decay=args.weight_decay)
        model.train()
        total_loss, count = 0.0, 0
        epoch_start = time.time()
        for images, masks in _local_progress(loader, 'improve_five_causes'):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                loss = patch_loss(model(images), masks)
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
        validation = evaluate_full(model, valid_samples, args.patch_size, args.val_stride, device, use_amp,
                                   args.tile_batch_size, False)
        record = {"epoch": epoch, "stage": "freeze" if epoch <= args.freeze_epochs else "finetune",
                  "train_loss": total_loss / max(count, 1), "valid": validation,
                  "epoch_seconds": time.time() - epoch_start}
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": validation, "args": vars(args)},
                   output_dir / "last.pt")
        # The competition metric is the mean of per-image, per-class mIoU.
        # Keep checkpoint selection aligned with model1 and the official rule;
        # global confusion-matrix mIoU is reported only as a diagnostic.
        score = float(validation["image_mean_miou"])
        print(f"epoch={epoch:02d} stage={record['stage']} loss={record['train_loss']:.4f} "
              f"global={score:.4f} image={validation['image_mean_miou']:.4f} "
              f"iou=" + ",".join(f"{v:.3f}" for v in validation["image_class_iou"].values()) +
              f" time={record['epoch_seconds']:.1f}s", flush=True)
        if score > best:
            best, stale = score, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": validation, "args": vars(args)},
                       output_dir / "best.pt")
        else:
            stale += 1
            if epoch > args.freeze_epochs and stale >= args.patience:
                print(f"early stopping after {args.patience} stale epochs", flush=True)
                break

    model.load_state_dict(torch.load(output_dir / "best.pt", map_location="cpu")["model"])
    final_normal = evaluate_full(model, valid_samples, args.patch_size, args.final_stride, device, use_amp,
                                 args.tile_batch_size, False)
    final_tta = evaluate_full(model, valid_samples, args.patch_size, args.final_stride, device, use_amp,
                               args.tile_batch_size, True)
    summary = {"best_training_image_mean_miou": best, "epochs_completed": len(history),
               "final_normal": final_normal, "final_tta": final_tta,
               "elapsed_seconds": time.time() - start_time, "class_weights": PATCH_CLASS_WEIGHTS.tolist(),
               "patch_size": args.patch_size, "val_stride": args.val_stride, "final_stride": args.final_stride}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def report_groups(args: argparse.Namespace) -> None:
    samples = collect_split(args.data_dir, "train") + collect_split(args.data_dir, "valid")
    folds = grouped_folds(samples, args.folds, args.seed)
    result = []
    for fold_id, valid_indices in enumerate(folds, 1):
        valid_groups = sorted({specimen_id(samples[i][0]) for i in valid_indices})
        train_indices = [i for i in range(len(samples)) if i not in set(valid_indices)]
        train_groups = sorted({specimen_id(samples[i][0]) for i in train_indices})
        result.append({"fold": fold_id, "train_images": len(train_indices), "valid_images": len(valid_indices),
                      "train_groups": train_groups, "valid_groups": valid_groups})
    print(json.dumps({"folds": result}, ensure_ascii=False, indent=2))


def predict_test(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    model = UNetPlusPlusResNet34(pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    output_dir = Path(args.output_dir) / "test_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        prediction = predict_full_image(model, image_path, args.patch_size, args.final_stride, device, use_amp,
                                        args.tile_batch_size, args.tta)
        encode_mask(prediction).save(output_dir / image_path.name.replace("_image.png", "_mask.png"))
    print(f"wrote {len(list(output_dir.glob('*.png')))} test masks to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["train", "groups", "predict-test"], default="train")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--patches-per-image", type=int, default=6)
    parser.add_argument("--val-stride", type=int, default=448)
    parser.add_argument("--final-stride", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--encoder-lr", type=float, default=5e-6)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "groups":
        report_groups(args)
    elif args.mode == "predict-test":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --mode predict-test")
        predict_test(args)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --mode train")
        train_fixed(args)
