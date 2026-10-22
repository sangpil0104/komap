#!/usr/bin/env python3
"""OOF-derived Eutectic Si hard-example crop mining, specimen-group safe.

For every training image, the mining map comes from a Model303 grouped-OOF
predictor that did not train on that specimen. Each fold continues only its
recorded Model303 checkpoint using fold-train masks. Two extra crops per image
are centered on OOF Eutectic false negatives/false positives (fallback to GT
Eutectic or random centers when a type is absent). Validation masks are not
opened until all four folds and predictions are complete.
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
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

from improve_five_causes import collect_split, decode_mask, specimen_id
from model181_raw_zcdf_aspp import HARD_CLASS_WEIGHTS, dice_loss, representation
from model253_highres_scale_finetune import (
    HighResolutionPatchDataset, crop_bounds, predict_scaled, resize_pair,
)
from model288_region_classifier_cv import image_score
from model301_boundary_enhanced_finetune import BoundaryEnhancedASPP


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OOFHardPatchDataset(Dataset):
    def __init__(self, samples, oof_probs: dict[str, np.ndarray], patch_size: int,
                 random_patches_per_image: int, seed: int):
        self.samples = list(samples)
        self.base = HighResolutionPatchDataset(
            self.samples, patch_size, random_patches_per_image, 1.0, seed,
            global_representation=True, metallography_augmentation=False,
            sampling_mode="class_focused")
        self.patch_size = int(patch_size)
        self.seed = int(seed)
        self.hard_centers: list[tuple[Path, Path, float, float, str]] = []
        rng = np.random.default_rng(seed)
        for image_path, mask_path in self.samples:
            target = decode_mask(Image.open(mask_path).convert("RGB"))
            prob = oof_probs[image_path.stem].astype(np.float32)
            if prob.shape[-2:] != target.shape:
                raise ValueError(f"OOF/GT shape mismatch for {image_path.name}")
            pred = prob.argmax(0)
            eut_gt = target == 3
            false_negative = eut_gt & (pred != 3)
            false_positive = (pred == 3) & ~eut_gt
            eut_yx = np.argwhere(eut_gt)
            fn_yx = np.argwhere(false_negative)
            fp_yx = np.argwhere(false_positive)
            h, w = target.shape

            def choose(candidates: np.ndarray, fallback: np.ndarray, kind: str):
                source = candidates if len(candidates) else fallback
                if len(source):
                    y, x = source[rng.integers(0, len(source))]
                else:
                    y, x = rng.integers(0, h), rng.integers(0, w)
                self.hard_centers.append((image_path, mask_path,
                                          float(y / max(h - 1, 1)),
                                          float(x / max(w - 1, 1)), kind))

            choose(fn_yx, eut_yx, "eutectic_false_negative")
            choose(fp_yx, eut_yx, "eutectic_false_positive")

    def __len__(self) -> int:
        return len(self.base) + len(self.hard_centers)

    def __getitem__(self, index: int):
        if index < len(self.base):
            return self.base[index]
        hard_index = index - len(self.base)
        image_path, mask_path, y_fraction, x_fraction, _kind = self.hard_centers[hard_index]
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = decode_mask(Image.open(mask_path).convert("RGB"))
        image, mask = resize_pair(image, mask, 1.0, self.patch_size)
        height, width = mask.shape
        cy = int(round(y_fraction * max(height - 1, 1)))
        cx = int(round(x_fraction * max(width - 1, 1)))
        top, left = crop_bounds(height, width, self.patch_size, cy, cx)

        rng = random.Random(self.seed + hard_index * 1009 + random.randrange(1_000_000))
        if rng.random() < 0.35:
            contrast = rng.uniform(0.94, 1.06)
            brightness = rng.uniform(-0.025, 0.025) * 255.0
            image = np.clip((image.astype(np.float32) - 127.5) * contrast
                            + 127.5 + brightness, 0, 255).astype(np.uint8)
        features = representation(image)[:, top:top + self.patch_size, left:left + self.patch_size]
        mask = mask[top:top + self.patch_size, left:left + self.patch_size]
        if rng.random() < 0.5:
            features, mask = np.flip(features, axis=2).copy(), np.flip(mask, axis=1).copy()
        if rng.random() < 0.5:
            features, mask = np.flip(features, axis=1).copy(), np.flip(mask, axis=0).copy()
        turns = rng.randrange(4)
        if turns:
            features = np.rot90(features, turns, axes=(1, 2)).copy()
            mask = np.rot90(mask, turns).copy()
        return torch.from_numpy(np.ascontiguousarray(features)).float(), torch.from_numpy(mask).long()


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp) -> float:
    model.train()
    total, count = 0.0, 0
    weights = HARD_CLASS_WEIGHTS.to(device)
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
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
        total += float(loss.detach()) * images.shape[0]
        count += images.shape[0]
    return total / max(count, 1)


def score_maps(maps: dict[str, np.ndarray], samples):
    scores, classes = [], []
    for image_path, mask_path in samples:
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        pred = maps[image_path.stem].argmax(0).astype(np.uint8)
        scores.append(image_score(pred, target))
        per_class = []
        for c in range(4):
            p, t = pred == c, target == c
            union = np.logical_or(p, t).sum()
            per_class.append(float(np.logical_and(p, t).sum() / union) if union else 0.0)
        classes.append(per_class)
    return np.asarray(scores), np.asarray(classes)


def grouped_ci(samples, deltas, seed=311):
    by_group: dict[str, list[int]] = {}
    for i, (image, _) in enumerate(samples):
        by_group.setdefault(specimen_id(image), []).append(i)
    groups = np.asarray(sorted(by_group))
    rng = np.random.default_rng(seed)
    draws = np.empty(10000)
    for j in range(len(draws)):
        sampled = rng.choice(groups, len(groups), replace=True)
        ix = [i for group in sampled for i in by_group[str(group)]]
        draws[j] = deltas[ix].mean() * 100.0
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--parent-dir", default="outputs/model303_boundary_oof_8ep")
    ap.add_argument("--baseline-oof", default="outputs/model303_boundary_oof_8ep/train_oof_probs.npz")
    ap.add_argument("--baseline-valid", default="outputs/model303_boundary_oof_8ep/valid_fold_ensemble_probs.npz")
    ap.add_argument("--output-dir", default="outputs/model311_eutectic_oof_hard_mining")
    ap.add_argument("--patch-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=864)
    ap.add_argument("--random-patches-per-image", type=int, default=6,
                    help="six ordinary plus two OOF-hard crops keeps eight total crops/image")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=311)
    ap.add_argument("--lr-multiplier", type=float, default=1.0)
    ap.add_argument("--memory-fraction", type=float, default=0.90)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
        torch.backends.cudnn.benchmark = True

    train_samples = collect_split(args.data_dir, "train")
    valid_images = sorted((Path(args.data_dir) / "valid/images").glob("*_image.png"))
    valid_samples = [(p, Path(args.data_dir) / "valid/masks" /
                      p.name.replace("_image.png", "_mask.png")) for p in valid_images]
    oof_archive = np.load(args.baseline_oof)
    oof_probs = {p.stem: oof_archive[p.stem] for p, _ in train_samples}
    if set(oof_probs) != {p.stem for p, _ in train_samples}:
        raise RuntimeError("Model303 train OOF archive is incomplete")

    if args.smoke_test:
        probe_samples = train_samples[:2]
        dataset = OOFHardPatchDataset(probe_samples, oof_probs, 128, 1, args.seed)
        x, y = dataset[0]
        hx, hy = dataset[len(dataset.base)]
        assert x.shape == hx.shape == (3, 128, 128)
        assert y.shape == hy.shape == (128, 128)
        model = BoundaryEnhancedASPP().to(device).eval()
        sample = torch.randn(1, 3, 128, 128, device=device)
        with torch.no_grad():
            out = model(sample)
        print(json.dumps({"device": str(device), "random_sample": list(x.shape),
                          "hard_sample": list(hx.shape), "logits": list(out.shape),
                          "hard_crops": len(dataset.hard_centers)}))
        return

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)
    parent_dir = Path(args.parent_dir)
    baseline_valid_archive = np.load(args.baseline_valid)
    baseline_valid = {p.stem: baseline_valid_archive[p.stem] for p in valid_images}
    oof_candidate: dict[str, np.ndarray] = {}
    valid_probability_sum: dict[str, np.ndarray] = {}
    fold_records, heldout_seen = [], set()
    started = time.time()

    for fold in range(1, 5):
        fold_start = time.time()
        fold_dir = out_dir / f"fold{fold}"
        fold_dir.mkdir()
        parent_path = parent_dir / f"fold{fold}/last.pt"
        payload = torch.load(parent_path, map_location="cpu", weights_only=False)
        train_groups, heldout_groups = set(payload["train_groups"]), set(payload["heldout_groups"])
        if train_groups & heldout_groups:
            raise RuntimeError(f"specimen leakage in fold {fold}")
        fold_train = [(im, m) for im, m in train_samples if specimen_id(im) in train_groups]
        fold_held = [(im, m) for im, m in train_samples if specimen_id(im) in heldout_groups]
        held_names = {im.stem for im, _ in fold_held}
        if heldout_seen & held_names:
            raise RuntimeError(f"overlapping held-out images in fold {fold}")
        heldout_seen |= held_names

        dataset = OOFHardPatchDataset(fold_train, oof_probs, args.patch_size,
                                      args.random_patches_per_image, args.seed + fold)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=use_amp,
                            persistent_workers=args.num_workers > 0)
        model = BoundaryEnhancedASPP().to(device)
        model.load_state_dict(payload["model"], strict=True)
        enc, dec, native, boundary = [], [], [], []
        for name, parameter in model.named_parameters():
            if name.startswith("boundary_enhancer."):
                boundary.append(parameter)
            elif name.startswith(("native_stem.", "native_to_head.")):
                native.append(parameter)
            elif name.startswith(("stem.", "enc1.", "enc2.", "enc3.", "enc4.")):
                enc.append(parameter)
            else:
                dec.append(parameter)
        optimizer = torch.optim.AdamW([
            {"params": enc, "lr": 5e-6 * args.lr_multiplier},
            {"params": dec, "lr": 1.5e-5 * args.lr_multiplier},
            {"params": native, "lr": 1e-4 * args.lr_multiplier},
            {"params": boundary, "lr": 1e-4 * args.lr_multiplier},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = GradScaler(enabled=use_amp)
        history = []
        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, loader, optimizer, scaler, device, use_amp)
            scheduler.step()
            history.append({"epoch": epoch, "train_loss": loss})
            print(f"fold={fold}/4 epoch={epoch}/{args.epochs} loss={loss:.5f} "
                  f"train_images={len(fold_train)} random_crops={len(dataset.base)} "
                  f"OOF_hard_crops={len(dataset.hard_centers)}", flush=True)

        torch.save({"model": model.state_dict(), "fold": fold,
                    "parent_checkpoint": str(parent_path),
                    "train_groups": sorted(train_groups),
                    "heldout_groups": sorted(heldout_groups),
                    "epoch_history": history,
                    "sampling": "cross-fitted Model303 OOF Eutectic false-negative/false-positive centers",
                    "validation_masks_used_for_training_or_selection": False}, fold_dir / "last.pt")
        model.eval()
        with torch.no_grad():
            for image_path, _ in fold_held:
                prob = predict_scaled(model, image_path, args.patch_size, args.stride, 1.0,
                                      device, use_amp, "d4")
                oof_candidate[image_path.stem] = prob.astype(np.float16)
            fold_valid = {}
            for image_path in valid_images:
                prob = predict_scaled(model, image_path, args.patch_size, args.stride, 1.0,
                                      device, use_amp, "d4")
                fold_valid[image_path.stem] = prob.astype(np.float16)
                valid_probability_sum[image_path.stem] = (
                    valid_probability_sum.get(image_path.stem, 0.0) + prob.astype(np.float32))
        np.savez_compressed(fold_dir / "oof_probs.npz",
                            **{name: oof_candidate[name] for name in held_names})
        np.savez_compressed(fold_dir / "valid_probs.npz", **fold_valid)
        fold_record = {"fold": fold, "train_image_count": len(fold_train),
                       "heldout_image_count": len(fold_held),
                       "random_crop_count_per_epoch": len(dataset.base),
                       "oof_hard_crop_count_per_epoch": len(dataset.hard_centers),
                       "epoch_history": history, "elapsed_seconds": time.time()-fold_start}
        fold_records.append(fold_record)
        (out_dir / "progress.json").write_text(json.dumps({"folds": fold_records,
            "completed_oof_images": len(oof_candidate)}, indent=2))
        print(f"completed fold={fold}/4 elapsed={fold_record['elapsed_seconds']:.1f}s", flush=True)
        del model, loader, dataset, optimizer, scheduler, scaler
        if use_amp:
            torch.cuda.empty_cache()

    expected = {p.stem for p, _ in train_samples}
    if heldout_seen != expected or set(oof_candidate) != expected:
        raise RuntimeError("OOF coverage check failed")
    valid_candidate = {k: v / 4.0 for k, v in valid_probability_sum.items()}
    np.savez_compressed(out_dir / "train_oof_probs.npz", **oof_candidate)
    np.savez_compressed(out_dir / "valid_fold_ensemble_probs.npz", **valid_candidate)

    # Validation and training masks are first scored here, after all folds finish.
    base_oof_scores, base_oof_classes = score_maps(oof_probs, train_samples)
    new_oof_scores, new_oof_classes = score_maps(oof_candidate, train_samples)
    base_val_scores, _ = score_maps(baseline_valid, valid_samples)
    new_val_scores, _ = score_maps(valid_candidate, valid_samples)
    oof_delta = new_oof_scores - base_oof_scores
    result = {
        "model": 311,
        "parent_model_id": 303,
        "method": "Model303 fold checkpoint fine-tuning with cross-fitted train-OOF Eutectic FN/FP crop mining",
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "train_images": len(train_samples), "valid_images": len(valid_samples),
        "random_crops_per_image_per_epoch": args.random_patches_per_image,
        "hard_crops_per_image_per_epoch": 2,
        "validation_masks_used_for_training_selection_or_fold_decisions": False,
        "hard_mining_maps": "Model303 train-only grouped-OOF probabilities; per-image fold model excluded that specimen",
        "oof": {
            "baseline_image_mean_miou": float(base_oof_scores.mean()),
            "candidate_image_mean_miou": float(new_oof_scores.mean()),
            "delta_pp": float(oof_delta.mean()*100),
            "positive_images": int((oof_delta>0).sum()),
            "group_bootstrap_95ci_delta_pp": grouped_ci(train_samples, oof_delta),
            "per_class_mean_iou_baseline": np.mean(base_oof_classes, axis=0).tolist(),
            "per_class_mean_iou_candidate": np.mean(new_oof_classes, axis=0).tolist(),
            "per_class_delta_pp": (100*(np.mean(new_oof_classes,axis=0)-np.mean(base_oof_classes,axis=0))).tolist(),
        },
        "valid_secondary": {
            "baseline_image_mean_miou": float(base_val_scores.mean()),
            "candidate_image_mean_miou": float(new_val_scores.mean()),
            "delta_pp": float((new_val_scores-base_val_scores).mean()*100),
            "positive_images": int((new_val_scores>base_val_scores).sum()),
        },
        "folds": fold_records,
        "elapsed_seconds": time.time()-started,
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"model":311,"oof":result["oof"],"valid_secondary":result["valid_secondary"],
                      "elapsed_seconds":result["elapsed_seconds"]},indent=2), flush=True)


if __name__ == "__main__":
    main()
