#!/usr/bin/env python3
"""Leakage-aware boundary-enhancement continuation of the Model297 OOF folds.

Each continuation starts from a Model297 checkpoint that never saw its held-out
train specimen groups. Validation masks are withheld until all training and
checkpoint decisions are finished; this script uses the fixed final epoch.
"""

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
from scipy.ndimage import label as connected_components
from scipy.ndimage import maximum_filter, minimum_filter
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

from improve_five_causes import collect_split, decode_mask, specimen_id
from model181_raw_zcdf_aspp import HARD_CLASS_WEIGHTS, dice_loss
from model288_region_classifier_cv import image_score
from model253_highres_scale_finetune import HighResolutionPatchDataset, predict_scaled
from model297_train_only_oof_branch import seed_all
from model301_boundary_enhanced_finetune import BoundaryEnhancedASPP, smoke_test


def rotate_batch(images: torch.Tensor, masks: torch.Tensor,
                 max_degrees: float,
                 auxiliary_mask: torch.Tensor | None = None):
    """Apply one bilinear image / nearest-label rotation per patch."""
    if max_degrees <= 0:
        return ((images, masks) if auxiliary_mask is None
                else (images, masks, auxiliary_mask))
    batch = images.shape[0]
    angle = (torch.rand(batch, device=images.device) * 2.0 - 1.0)
    angle = angle * (max_degrees * np.pi / 180.0)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros((batch, 2, 3), dtype=images.dtype, device=images.device)
    theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
    theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection",
                           align_corners=False)
    masks = F.grid_sample(masks[:, None].float(), grid, mode="nearest",
                          padding_mode="reflection", align_corners=False)
    masks = masks[:, 0].long()
    if auxiliary_mask is None:
        return images, masks
    auxiliary_mask = F.grid_sample(auxiliary_mask[:, None].float(), grid, mode="nearest",
                                   padding_mode="zeros", align_corners=False)
    return images, masks, auxiliary_mask[:, 0]


def gap_region_from_mask(mask: np.ndarray, radius: int = 5, *,
                        exclude_crop_border: bool = True) -> np.ndarray:
    """Mark background pixels within radius of two distinct Primary-Si components.

    This approximates the pairwise dilation-intersection gap map used by GRAF,
    but uses local min/max component IDs to compute all overlaps in linear time.
    """
    values = np.asarray(mask)
    primary = values == 1  # main order: [Al3Ni, Primary Si, Al, Eutectic Si]
    component_ids, count = connected_components(
        primary, structure=np.ones((3, 3), dtype=np.uint8))
    if count < 2:
        return np.zeros(values.shape, dtype=np.uint8)
    radius = int(radius)
    if radius < 1:
        raise ValueError("gap radius must be >= 1")
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    footprint = (yy * yy + xx * xx) <= radius * radius
    labels = component_ids.astype(np.int32, copy=False)
    maximum_id = maximum_filter(labels, footprint=footprint, mode="nearest")
    nonzero_ids = np.where(labels > 0, labels, np.iinfo(np.int32).max).astype(np.int32)
    minimum_id = minimum_filter(nonzero_ids, footprint=footprint, mode="nearest")
    gap = (~primary) & (maximum_id > minimum_id)
    if exclude_crop_border:
        # A crop may cut a particle; do not treat artificial crop-edge fragments as gaps.
        gap[:radius] = False
        gap[-radius:] = False
        gap[:, :radius] = False
        gap[:, -radius:] = False
    return gap.astype(np.uint8)


class GapWeightedPatchDataset(Dataset):
    """Add an augmented-patch GRACE map without opening any held-out masks."""

    def __init__(self, dataset: Dataset, radius: int):
        self.dataset = dataset
        self.radius = int(radius)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        features, mask = self.dataset[index]
        gap = gap_region_from_mask(mask.numpy(), self.radius)
        return features, mask, torch.from_numpy(gap.astype(np.float32))


def train_epoch(model, loader, optimizer, scaler, device, use_amp,
                rotation_degrees: float = 0.0,
                gap_region_weight: float = 0.0) -> float:
    model.train()
    total, count = 0.0, 0
    weights = HARD_CLASS_WEIGHTS.to(device)
    for batch in _local_progress(loader, 'model302_boundary_oof_finetune'):
        if len(batch) == 3:
            images, masks, gap_map = batch
            gap_map = gap_map.to(device, non_blocking=True)
        else:
            images, masks = batch
            gap_map = None
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        rotated = rotate_batch(images, masks, rotation_degrees, gap_map)
        if gap_map is None:
            images, masks = rotated
        else:
            images, masks, gap_map = rotated
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            if gap_map is None or gap_region_weight <= 0:
                ce_loss = F.cross_entropy(logits, masks, weight=weights)
            else:
                per_pixel_ce = F.cross_entropy(
                    logits, masks, weight=weights, reduction="none")
                pixel_weights = 1.0 + gap_region_weight * gap_map
                normalizer = weights[masks].sum().clamp_min(1e-6)
                ce_loss = (per_pixel_ce * pixel_weights).sum() / normalizer
            loss = ce_loss + dice_loss(logits, masks)
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
        total += float(loss.detach()) * images.shape[0]
        count += images.shape[0]
    return total / max(count, 1)


def score_probabilities(probabilities: dict[str, np.ndarray], samples) -> list[float]:
    scores = []
    for image_path, mask_path in _local_progress(samples, 'model302_boundary_oof_finetune'):
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        prediction = probabilities[image_path.stem].astype(np.float32).argmax(0).astype(np.uint8)
        scores.append(image_score(prediction, target))
    return scores


def grouped_bootstrap_delta(samples, baseline_scores, new_scores, seed=302) -> list[float]:
    grouped_indices: dict[str, list[int]] = {}
    for index, (image_path, _) in enumerate(_local_progress(samples, 'model302_boundary_oof_finetune')):
        grouped_indices.setdefault(specimen_id(image_path), []).append(index)
    names = list(grouped_indices)
    deltas = np.asarray(new_scores, dtype=np.float64) - np.asarray(baseline_scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(10000, dtype=np.float64)
    for draw in range(len(draws)):
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = [idx for name in sampled for idx in grouped_indices[str(name)]]
        draws[draw] = deltas[indices].mean() * 100.0
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--parent-dir", default="outputs/model297_train_only_oof")
    parser.add_argument("--baseline-oof", default="outputs/model297_train_only_oof/train_oof_probs.npz")
    parser.add_argument("--baseline-valid", default="outputs/model297_train_only_oof/valid_fold_ensemble_probs.npz")
    parser.add_argument("--output-dir", default="outputs/model302_boundary_oof")
    parser.add_argument("--model-id", type=int, default=302)
    parser.add_argument("--baseline-model-id", type=int, default=297)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--stride", type=int, default=864)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=302)
    parser.add_argument("--lr-multiplier", type=float, default=1.0)
    parser.add_argument("--continuous-rotation-degrees", type=float, default=0.0,
                        help="sample a continuous angle uniformly from [-degrees, degrees]")
    parser.add_argument("--gap-region-weight", type=float, default=0.0,
                        help="extra CE weight for gaps between distinct Primary-Si components")
    parser.add_argument("--gap-radius", type=int, default=5,
                        help="dilation radius in pixels used to identify inter-particle gaps")
    parser.add_argument("--resume", action="store_true",
                        help="continue from complete folds recorded in progress.json")
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.continuous_rotation_degrees < 0 or args.continuous_rotation_degrees > 180:
        raise ValueError("continuous rotation range must be between 0 and 180 degrees")
    if args.gap_region_weight < 0 or args.gap_radius < 1:
        raise ValueError("gap-region weight must be >= 0 and radius must be >= 1")
    if args.smoke_test:
        smoke_test()
        sample = np.zeros((64, 64), dtype=np.uint8)
        sample[20:28, 10:18] = 1
        sample[20:28, 24:32] = 1
        gap = gap_region_from_mask(sample, radius=4)
        assert gap[23, 21] == 1 and gap[sample == 1].sum() == 0
        single = gap_region_from_mask(sample[:, :20], radius=4)
        assert single.sum() == 0
        print("gap-region smoke test: distinct-particle gap found", flush=True)
        return

    started = time.time()
    seed_all(args.seed)
    torch.set_num_threads(4)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was explicitly requested but is unavailable")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
        torch.backends.cudnn.benchmark = True
    elif device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.memory_fraction)

    output = Path(args.output_dir)
    if args.resume:
        if not output.is_dir() or not (output / "progress.json").is_file():
            raise FileNotFoundError(f"cannot resume without existing progress.json: {output}")
    else:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
        output.mkdir(parents=True)
    parent_dir = Path(args.parent_dir)
    train_samples = collect_split(args.data_dir, "train")
    valid_image_paths = sorted((Path(args.data_dir) / "valid" / "images").glob("*_image.png"))
    valid_samples = [(path, Path(args.data_dir) / "valid" / "masks" /
                      path.name.replace("_image.png", "_mask.png"))
                     for path in valid_image_paths]
    if not train_samples or not valid_samples:
        raise FileNotFoundError("expected non-empty train and valid splits")

    baseline_oof_archive = np.load(args.baseline_oof)
    baseline_oof = {path.stem: baseline_oof_archive[path.stem]
                    for path, _ in train_samples}
    baseline_valid_archive = np.load(args.baseline_valid)
    baseline_valid = {path.stem: baseline_valid_archive[path.stem]
                      for path in valid_image_paths}
    expected_train_names = {path.stem for path, _ in train_samples}
    if set(baseline_oof) != expected_train_names:
        raise RuntimeError("Model297 OOF archive does not cover current train split")

    oof_predictions: dict[str, np.ndarray] = {}
    valid_probability_sum: dict[str, np.ndarray] = {}
    if args.resume:
        progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
        fold_records = list(progress.get("completed_folds", []))
        completed_ids = [int(row["fold"]) for row in fold_records]
        if completed_ids != list(range(1, len(completed_ids) + 1)):
            raise RuntimeError(f"resume requires a complete prefix of folds, got {completed_ids}")
        for fold_id in completed_ids:
            fold_dir = output / f"fold{fold_id}"
            oof_path, valid_path = fold_dir / "oof_probs.npz", fold_dir / "valid_probs.npz"
            if not oof_path.is_file() or not valid_path.is_file():
                raise FileNotFoundError(f"fold {fold_id} is marked complete but artifacts are missing")
            with np.load(oof_path) as archive:
                for name in archive.files:
                    if name in oof_predictions:
                        raise RuntimeError(f"duplicate OOF prediction while resuming: {name}")
                    oof_predictions[name] = archive[name]
            with np.load(valid_path) as archive:
                if set(archive.files) != {path.stem for path in valid_image_paths}:
                    raise RuntimeError(f"fold {fold_id} valid archive has incomplete image coverage")
                for name in archive.files:
                    values = archive[name].astype(np.float32)
                    valid_probability_sum[name] = (valid_probability_sum.get(name, 0.0) + values)
        print(f"resume: restored {len(completed_ids)} complete folds and "
              f"{len(oof_predictions)} OOF images", flush=True)
    else:
        fold_records = []
    heldout_seen: set[str] = set(oof_predictions)

    for fold_id in range(1, 5):
        if fold_id <= len(fold_records):
            continue
        fold_started = time.time()
        fold_dir = output / f"fold{fold_id}"
        if fold_dir.exists() and any(fold_dir.iterdir()):
            raise RuntimeError(f"incomplete fold directory is non-empty; refusing overwrite: {fold_dir}")
        fold_dir.mkdir(exist_ok=True)
        checkpoint_path = parent_dir / f"fold{fold_id}" / "last.pt"
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train_groups = set(payload["train_groups"])
        heldout_groups = set(payload["heldout_groups"])
        if train_groups & heldout_groups:
            raise RuntimeError(f"specimen leakage in parent fold {fold_id}")
        fold_train = [(image, mask) for image, mask in train_samples
                      if specimen_id(image) in train_groups]
        fold_heldout = [(image, mask) for image, mask in train_samples
                        if specimen_id(image) in heldout_groups]
        heldout_names = {image.stem for image, _ in fold_heldout}
        if not fold_train or not fold_heldout or heldout_seen & heldout_names:
            raise RuntimeError(f"invalid or overlapping specimen holdout in fold {fold_id}")
        heldout_seen |= heldout_names

        dataset = HighResolutionPatchDataset(
            fold_train, args.patch_size, args.patches_per_image, 1.0,
            args.seed + fold_id, global_representation=True,
            metallography_augmentation=False, sampling_mode="class_focused")
        if args.gap_region_weight > 0:
            dataset = GapWeightedPatchDataset(dataset, args.gap_radius)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0))

        model = BoundaryEnhancedASPP().to(device)
        incompatible = model.load_state_dict(payload["model"], strict=False)
        missing = [key for key in incompatible.missing_keys
                   if not key.startswith("boundary_enhancer.")]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(f"fold {fold_id} checkpoint mismatch: "
                               f"missing={missing}, unexpected={incompatible.unexpected_keys}")

        encoder, decoder, native, boundary = [], [], [], []
        for name, parameter in model.named_parameters():
            if name.startswith("boundary_enhancer."):
                boundary.append(parameter)
            elif name.startswith(("native_stem.", "native_to_head.")):
                native.append(parameter)
            elif name.startswith(("stem.", "enc1.", "enc2.", "enc3.", "enc4.")):
                encoder.append(parameter)
            else:
                decoder.append(parameter)
        optimizer = torch.optim.AdamW([
            {"params": encoder, "lr": 5e-6 * args.lr_multiplier},
            {"params": decoder, "lr": 1.5e-5 * args.lr_multiplier},
            {"params": native, "lr": 1e-4 * args.lr_multiplier},
            {"params": boundary, "lr": 1e-4 * args.lr_multiplier},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = GradScaler(enabled=use_amp)
        epoch_history = []
        for epoch in _local_progress(range(1, args.epochs + 1), 'model302_boundary_oof_finetune'):
            loss = train_epoch(model, loader, optimizer, scaler, device, use_amp,
                               args.continuous_rotation_degrees,
                               args.gap_region_weight)
            scheduler.step()
            record = {"epoch": epoch, "train_loss": loss}
            epoch_history.append(record)
            print(f"fold={fold_id}/4 epoch={epoch}/{args.epochs} "
                  f"train_loss={loss:.5f}", flush=True)

        # Fixed final epoch: no valid mask is read to select a checkpoint.
        torch.save({"model": model.state_dict(), "fold": fold_id,
                    "parent_checkpoint": str(checkpoint_path),
                    "train_groups": sorted(train_groups),
                    "heldout_groups": sorted(heldout_groups),
                    "epochs": args.epochs,
                    "parent_model_id": args.baseline_model_id,
                    "architecture": ("BoundaryEnhancedASPP continuation"
                                     if args.baseline_model_id == 302 else
                                     f"Model{args.baseline_model_id} + shared "
                                     "multiscale boundary enhancement"),
                    "continuous_rotation_degrees": args.continuous_rotation_degrees,
                    "gap_region_weight": args.gap_region_weight,
                    "gap_radius": args.gap_radius,
                    "validation_masks_used_for_selection": False}, fold_dir / "last.pt")
        model.eval()
        with torch.no_grad():
            for image_path, _ in fold_heldout:
                probabilities = predict_scaled(
                    model, image_path, args.patch_size, args.stride, 1.0,
                    device, use_amp, "d4")
                oof_predictions[image_path.stem] = probabilities.astype(np.float16)
            fold_valid: dict[str, np.ndarray] = {}
            for image_path in valid_image_paths:
                probabilities = predict_scaled(
                    model, image_path, args.patch_size, args.stride, 1.0,
                    device, use_amp, "d4")
                fold_valid[image_path.stem] = probabilities.astype(np.float16)
                if image_path.stem not in valid_probability_sum:
                    valid_probability_sum[image_path.stem] = probabilities.astype(np.float32)
                else:
                    valid_probability_sum[image_path.stem] += probabilities.astype(np.float32)

        np.savez_compressed(fold_dir / "oof_probs.npz",
                            **{name: oof_predictions[name] for name in heldout_names})
        np.savez_compressed(fold_dir / "valid_probs.npz", **fold_valid)
        fold_record = {
            "fold": fold_id, "train_image_count": len(fold_train),
            "heldout_image_count": len(fold_heldout),
            "train_groups": sorted(train_groups),
            "heldout_groups": sorted(heldout_groups),
            "epoch_history": epoch_history,
            "elapsed_seconds": time.time() - fold_started,
        }
        fold_records.append(fold_record)
        (output / "progress.json").write_text(
            json.dumps({"completed_folds": fold_records,
                        "completed_oof_images": len(oof_predictions)},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"completed fold={fold_id}/4 heldout_images={len(heldout_names)} "
              f"elapsed={fold_record['elapsed_seconds']:.1f}s", flush=True)
        del model, loader, dataset, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    if heldout_seen != expected_train_names or set(oof_predictions) != expected_train_names:
        raise RuntimeError("four held-out groups do not cover every train image exactly once")
    valid_probabilities = {name: value / 4.0 for name, value in valid_probability_sum.items()}
    np.savez_compressed(output / "train_oof_probs.npz", **oof_predictions)
    np.savez_compressed(output / "valid_fold_ensemble_probs.npz", **valid_probabilities)

    # Ground-truth masks are read only after all training and fold predictions.
    baseline_oof_scores = score_probabilities(baseline_oof, train_samples)
    new_oof_scores = score_probabilities(oof_predictions, train_samples)
    baseline_valid_scores = score_probabilities(baseline_valid, valid_samples)
    new_valid_scores = score_probabilities(valid_probabilities, valid_samples)
    oof_delta = np.asarray(new_oof_scores) - np.asarray(baseline_oof_scores)
    method = f"Model{args.baseline_model_id} specimen-group OOF checkpoints fine-tuned"
    if args.gap_region_weight > 0:
        method += (f" with Primary-Si gap-region-weighted CE (weight="
                   f"{args.gap_region_weight:g}, radius={args.gap_radius}px)")
    elif args.baseline_model_id == 297:
        method += " with boundary enhancement"
    if args.continuous_rotation_degrees:
        method += (f" and random continuous rotation up to "
                   f"{args.continuous_rotation_degrees:g} degrees")
    result = {
        "model": args.model_id,
        "parent_model_id": args.baseline_model_id,
        "method": method,
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "train_image_count": len(train_samples), "valid_image_count": len(valid_samples),
        "validation_masks_used_for_training_selection_or_fold_decisions": False,
        "checkpoint_selection": "fixed final epoch in every fold",
        "continuous_rotation_degrees": args.continuous_rotation_degrees,
        "gap_region_weight": args.gap_region_weight,
        "gap_radius": args.gap_radius,
        f"parent_model{args.baseline_model_id}_oof_image_mean_miou":
            float(np.mean(baseline_oof_scores)),
        f"model{args.model_id}_oof_image_mean_miou": float(np.mean(new_oof_scores)),
        "oof_delta_pp": float(oof_delta.mean() * 100.0),
        "oof_positive_images": int((oof_delta > 0).sum()),
        "oof_group_bootstrap_95ci_delta_pp": grouped_bootstrap_delta(
            train_samples, baseline_oof_scores, new_oof_scores),
        f"valid_model{args.baseline_model_id}_fold_ensemble_image_mean_miou":
            float(np.mean(baseline_valid_scores)),
        f"valid_model{args.model_id}_fold_ensemble_image_mean_miou":
            float(np.mean(new_valid_scores)),
        "valid_delta_pp": float((np.asarray(new_valid_scores)
                                  - np.asarray(baseline_valid_scores)).mean() * 100.0),
        "valid_image_scores": new_valid_scores,
        "folds": fold_records,
        "elapsed_seconds": time.time() - started,
        "artifacts": {"train_oof_probs": str(output / "train_oof_probs.npz"),
                      "valid_fold_ensemble_probs": str(output / "valid_fold_ensemble_probs.npz")},
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_keys = [
        f"parent_model{args.baseline_model_id}_oof_image_mean_miou",
        f"model{args.model_id}_oof_image_mean_miou",
        "oof_delta_pp", "oof_positive_images", "oof_group_bootstrap_95ci_delta_pp",
        f"valid_model{args.baseline_model_id}_fold_ensemble_image_mean_miou",
        f"valid_model{args.model_id}_fold_ensemble_image_mean_miou",
        "valid_delta_pp", "elapsed_seconds",
    ]
    print(json.dumps({key: result[key] for key in summary_keys},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
