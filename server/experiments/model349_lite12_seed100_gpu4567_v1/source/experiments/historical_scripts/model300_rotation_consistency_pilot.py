#!/usr/bin/env python3
"""One-fold, label-isolated rotation-consistency Mean-Teacher pilot.

The supervised initialization is Model297 fold 1. Its held-out specimen inputs
are available only as unlabeled images during adaptation; their masks are read
only after training for the paired OOF evaluation.
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
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from improve_five_causes import collect_split, decode_mask, grouped_folds, specimen_id
from model181_raw_zcdf_aspp import HARD_CLASS_WEIGHTS, dice_loss, representation
from model239_native_stem_finetune import Stride1NativeStemASPP
from model253_highres_scale_finetune import HighResolutionPatchDataset, predict_scaled
from model288_region_classifier_cv import image_score


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class UnlabeledPatchDataset(Dataset):
    """Crop image-only patches; this class never accepts or opens mask paths."""

    def __init__(self, image_paths: list[Path], patch_size: int,
                 patches_per_image: int, seed: int):
        self.image_paths = list(image_paths)
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.seed = int(seed)
        self.features = []
        for path in self.image_paths:
            image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            self.features.append(representation(image))

    def __len__(self) -> int:
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, index: int) -> torch.Tensor:
        image_index = index % len(self.image_paths)
        features = self.features[image_index]
        _, height, width = features.shape
        patch = self.patch_size
        pad_h, pad_w = max(0, patch - height), max(0, patch - width)
        if pad_h or pad_w:
            features = np.pad(features, ((0, 0), (0, pad_h), (0, pad_w)),
                              mode="reflect")
            height, width = features.shape[1:]
        rng = random.Random(self.seed + index * 1009 + random.randrange(1_000_000))
        top = rng.randrange(height - patch + 1)
        left = rng.randrange(width - patch + 1)
        crop = features[:, top:top + patch, left:left + patch]
        return torch.from_numpy(np.ascontiguousarray(crop))


def component_metric(probabilities: np.ndarray, mask_path: Path) -> float:
    target = decode_mask(Image.open(mask_path).convert("RGB"))
    prediction = probabilities.argmax(axis=0).astype(np.uint8)
    return image_score(prediction, target)


def rotation_consistency_loss(teacher: torch.nn.Module, student: torch.nn.Module,
                              images: torch.Tensor, use_amp: bool,
                              rng: random.Random) -> torch.Tensor:
    """Confidence-weighted MSE to an inverse-rotated four-view teacher mean."""
    teacher_views = []
    with torch.no_grad():
        for turns in range(4):
            rotated = torch.rot90(images, turns, dims=(-2, -1))
            with autocast(enabled=use_amp):
                probabilities = torch.softmax(teacher(rotated), dim=1).float()
            teacher_views.append(torch.rot90(probabilities, -turns, dims=(-2, -1)))
    stacked = torch.stack(teacher_views, dim=1)
    mean_probability = stacked.mean(dim=1).detach()
    log_mean = mean_probability.clamp_min(1e-6).log()
    per_view_kl = (stacked * (stacked.clamp_min(1e-6).log()
                              - log_mean[:, None])).sum(dim=2)
    uncertainty = per_view_kl.mean(dim=1)
    confidence = torch.exp(-4.0 * uncertainty).detach()

    turns = rng.randrange(4)
    rotated = torch.rot90(images, turns, dims=(-2, -1))
    with autocast(enabled=use_amp):
        student_probability = torch.softmax(student(rotated), dim=1).float()
    student_probability = torch.rot90(student_probability, -turns, dims=(-2, -1))
    pixel_error = (student_probability - mean_probability).square().mean(dim=1)
    return (pixel_error * confidence).mean()


def update_ema(teacher: torch.nn.Module, student: torch.nn.Module,
               decay: float) -> None:
    with torch.no_grad():
        student_state = student.state_dict()
        for name, teacher_value in teacher.state_dict().items():
            student_value = student_state[name]
            if teacher_value.is_floating_point():
                teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
            else:
                teacher_value.copy_(student_value)


def image_means(probabilities: dict[str, np.ndarray], samples,
                data_dir: Path) -> tuple[float, list[float]]:
    scores = []
    for image_path, _ in _local_progress(samples, 'model300_rotation_consistency_pilot'):
        mask_path = data_dir / "valid" / "masks" / image_path.name.replace(
            "_image.png", "_mask.png")
        scores.append(component_metric(probabilities[image_path.stem], mask_path))
    return float(np.mean(scores)), scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--base-dir", required=True,
                        help="Model297 output directory containing fold1/last.pt and train OOF probs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-id", type=int, default=1)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--unlabeled-patches-per-image", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--unsup-weight", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    args = parser.parse_args()

    started = time.time()
    seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
        torch.backends.cudnn.benchmark = True

    data_dir = Path(args.data_dir)
    base_dir = Path(args.base_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_samples = collect_split(data_dir, "train")
    valid_samples = collect_split(data_dir, "valid")
    folds = grouped_folds(train_samples, args.folds, 297)
    if not 1 <= args.fold_id <= len(folds):
        raise ValueError("fold-id is outside the grouped-fold range")
    heldout_indices = set(folds[args.fold_id - 1])
    fold_train = [sample for index, sample in enumerate(train_samples)
                  if index not in heldout_indices]
    fold_heldout = [train_samples[index] for index in sorted(heldout_indices)]
    train_groups = {specimen_id(path) for path, _ in fold_train}
    heldout_groups = {specimen_id(path) for path, _ in fold_heldout}
    if train_groups & heldout_groups:
        raise RuntimeError("specimen leakage between supervised and unlabeled fold")
    valid_image_paths = sorted((data_dir / "valid" / "images").glob("*_image.png"))
    if len(valid_image_paths) != len(valid_samples):
        raise RuntimeError("valid image/mask split mismatch")

    checkpoint = torch.load(base_dir / f"fold{args.fold_id}" / "last.pt",
                            map_location="cpu", weights_only=False)
    model = Stride1NativeStemASPP().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    labeled_dataset = HighResolutionPatchDataset(
        fold_train, args.patch_size, args.patches_per_image, 1.0,
        args.seed + args.fold_id, global_representation=True,
        metallography_augmentation=False, sampling_mode="class_focused")
    labeled_loader = DataLoader(
        labeled_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0))
    unlabeled_dataset = UnlabeledPatchDataset(
        [path for path, _ in fold_heldout], args.patch_size,
        args.unlabeled_patches_per_image, args.seed + 10_000 + args.fold_id)
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False)

    baseline_oof_archive = np.load(base_dir / "train_oof_probs.npz")
    baseline_oof = {image_path.stem: baseline_oof_archive[image_path.stem].astype(np.float32)
                    for image_path, _ in fold_heldout}
    baseline_valid: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for image_path in valid_image_paths:
            baseline_valid[image_path.stem] = predict_scaled(
                model, image_path, args.patch_size, 384, 1.0, device, use_amp, "d4")

    encoder, decoder, native = [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith(("native_stem.", "native_to_head.")):
            native.append(parameter)
        elif name.startswith(("stem.", "enc1.", "enc2.", "enc3.", "enc4.")):
            encoder.append(parameter)
        else:
            decoder.append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": 3e-6},
        {"params": decoder, "lr": 3e-5},
        {"params": native, "lr": 3e-5},
    ], weight_decay=1e-4)
    scaler = GradScaler(enabled=use_amp)
    unlabeled_iterator = iter(unlabeled_loader)
    rng = random.Random(args.seed + 30_000 + args.fold_id)
    epoch_history = []
    for epoch in _local_progress(range(1, args.epochs + 1), 'model300_rotation_consistency_pilot'):
        model.train()
        supervised_sum = consistency_sum = total_sum = 0.0
        step_count = 0
        for step, (images, masks) in enumerate(labeled_loader):
            try:
                unlabeled = next(unlabeled_iterator)
            except StopIteration:
                unlabeled_iterator = iter(unlabeled_loader)
                unlabeled = next(unlabeled_iterator)
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            unlabeled = unlabeled.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(images)
                supervised_loss = F.cross_entropy(
                    logits, masks, weight=HARD_CLASS_WEIGHTS.to(device)) + dice_loss(logits, masks)
            consistency_loss = rotation_consistency_loss(
                teacher, model, unlabeled, use_amp, rng)
            ramp = min(1.0, (step + 1) / max(1.0, 0.25 * len(labeled_loader)))
            loss = supervised_loss + args.unsup_weight * ramp * consistency_loss
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
            update_ema(teacher, model, args.ema_decay)
            supervised_sum += float(supervised_loss.detach())
            consistency_sum += float(consistency_loss.detach())
            total_sum += float(loss.detach())
            step_count += 1
        record = {
            "epoch": epoch,
            "supervised_loss": supervised_sum / max(step_count, 1),
            "rotation_consistency_loss": consistency_sum / max(step_count, 1),
            "total_loss": total_sum / max(step_count, 1),
            "steps": step_count,
        }
        epoch_history.append(record)
        print(json.dumps(record), flush=True)

    teacher.eval()
    adapted_oof: dict[str, np.ndarray] = {}
    heldout_scores = []
    with torch.no_grad():
        for image_path, mask_path in fold_heldout:
            probabilities = predict_scaled(
                teacher, image_path, args.patch_size, 384, 1.0, device, use_amp, "d4")
            adapted_oof[image_path.stem] = probabilities.astype(np.float16)
            heldout_scores.append(component_metric(probabilities, mask_path))
    adapted_valid: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for image_path in valid_image_paths:
            adapted_valid[image_path.stem] = predict_scaled(
                teacher, image_path, args.patch_size, 384, 1.0, device, use_amp, "d4")
    valid_base_score, valid_base_scores = image_means(baseline_valid, valid_samples, data_dir)
    valid_adapted_score, valid_adapted_scores = image_means(adapted_valid, valid_samples, data_dir)
    baseline_heldout_scores = [component_metric(
        baseline_oof[path.stem], mask_path) for path, mask_path in fold_heldout]
    result = {
        "model": 300,
        "method": "Model297 fold checkpoint + four-rotation uncertainty-weighted Mean Teacher consistency",
        "fold_id": args.fold_id,
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "train_image_count": len(fold_train),
        "unlabeled_heldout_image_count": len(fold_heldout),
        "heldout_specimen_groups": sorted(heldout_groups),
        "heldout_masks_used_for_training": False,
        "valid_masks_used_for_training": False,
        "teacher_ema_decay": args.ema_decay,
        "unsup_weight": args.unsup_weight,
        "rotation_confidence": "exp(-4 * mean-view KL to inverse-rotated teacher probability mean)",
        "epoch_history": epoch_history,
        "heldout_oof_baseline_image_mean_miou": float(np.mean(baseline_heldout_scores)),
        "heldout_oof_adapted_image_mean_miou": float(np.mean(heldout_scores)),
        "heldout_oof_delta_pp": float((np.mean(heldout_scores)
                                        - np.mean(baseline_heldout_scores)) * 100.0),
        "valid_baseline_image_mean_miou": valid_base_score,
        "valid_adapted_image_mean_miou": valid_adapted_score,
        "valid_delta_pp": (valid_adapted_score - valid_base_score) * 100.0,
        "valid_positive_image_count": int(sum(
            adapted > base for adapted, base in zip(valid_adapted_scores, valid_base_scores))),
        "valid_image_scores_baseline": valid_base_scores,
        "valid_image_scores_adapted": valid_adapted_scores,
        "elapsed_seconds": time.time() - started,
    }
    np.savez_compressed(output / "heldout_oof_adapted_probs.npz", **adapted_oof)
    np.savez_compressed(output / "valid_adapted_probs.npz", **{
        key: value.astype(np.float16) for key, value in adapted_valid.items()})
    torch.save({"model": teacher.state_dict(), "fold": args.fold_id,
                "heldout_groups": sorted(heldout_groups),
                "heldout_masks_used_for_training": False}, output / "teacher_ema.pt")
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "heldout_oof_baseline_image_mean_miou", "heldout_oof_adapted_image_mean_miou",
        "heldout_oof_delta_pp", "valid_baseline_image_mean_miou",
        "valid_adapted_image_mean_miou", "valid_delta_pp", "elapsed_seconds")},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
