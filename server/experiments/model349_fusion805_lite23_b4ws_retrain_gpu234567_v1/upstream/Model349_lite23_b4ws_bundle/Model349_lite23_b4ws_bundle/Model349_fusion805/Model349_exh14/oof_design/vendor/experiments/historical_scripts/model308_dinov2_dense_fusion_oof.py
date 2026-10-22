#!/usr/bin/env python3
"""Frozen DINOv2 multi-layer dense feature residual on Model303 grouped OOF.

The Model303 CNN fold checkpoints stay frozen.  A small, zero-initialized
decoder learns a per-pixel logit residual from four frozen DINOv2-Small
intermediate patch-token layers.  Train labels are restricted to each
checkpoint's recorded train groups; valid masks are withheld until the final
paired measurement.  All candidate/baseline inference uses identical 512px
tiles, overlap, Gaussian blending, and D4 TTA.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import Dinov2Model

from improve_five_causes import collect_split, decode_mask, specimen_id
from model181_raw_zcdf_aspp import HARD_CLASS_WEIGHTS, dice_loss
from model253_highres_scale_finetune import HighResolutionPatchDataset, predict_scaled
from model288_region_classifier_cv import image_score
from model301_boundary_enhanced_finetune import BoundaryEnhancedASPP


class DINOResidual(nn.Module):
    def __init__(self, channels: int = 384, classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 4, 128, 1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, classes, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        delta = self.net(features)
        return F.interpolate(delta, size=output_size, mode="bilinear", align_corners=False)


class FrozenDINOFusion(nn.Module):
    def __init__(self, base: nn.Module, dino: Dinov2Model):
        super().__init__()
        self.base = base
        self.dino = dino
        self.adapter = DINOResidual()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.base.parameters():
            p.requires_grad_(False)
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.base.eval()
        self.dino.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        self.dino.eval()
        self.adapter.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_logits = self.base(x)
            # Model303's channel 0 is an affine encoding of source grayscale:
            # (gray - .5) / .25.  Recover it and replicate to RGB for DINO.
            gray = (x[:, :1] * 0.25 + 0.5).clamp(0.0, 1.0)
            rgb = gray.expand(-1, 3, -1, -1)
            rgb = F.interpolate(rgb, size=(518, 518), mode="bilinear", align_corners=False)
            dino_input = (rgb - self.mean) / self.std
            output = self.dino(pixel_values=dino_input, output_hidden_states=True)
            grid_h, grid_w = 518 // 14, 518 // 14
            layer_features = []
            for layer_id in (3, 6, 9, 12):
                tokens = output.hidden_states[layer_id][:, 1:, :]
                if tokens.shape[1] != grid_h * grid_w:
                    raise RuntimeError(f"DINO patch grid mismatch: {tokens.shape}")
                tokens = self.dino.layernorm(tokens)
                feat = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_h, grid_w)
                layer_features.append(feat)
            dense = torch.cat(layer_features, dim=1)
        return base_logits + self.adapter(dense, x.shape[-2:])


def score_probability_map(prob: np.ndarray, mask_path: Path) -> tuple[float, list[float]]:
    target = decode_mask(Image.open(mask_path).convert("RGB"))
    pred = prob.argmax(axis=0).astype(np.uint8)
    score = image_score(pred, target)
    class_scores = []
    for c in range(4):
        p, t = pred == c, target == c
        union = np.logical_or(p, t).sum()
        inter = np.logical_and(p, t).sum()
        class_scores.append(float(inter / union) if union else 0.0)
    return score, class_scores


def bootstrap_group_ci(deltas: np.ndarray, groups: list[str], seed: int = 308) -> list[float]:
    grouped: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        grouped.setdefault(group, []).append(i)
    names = np.asarray(sorted(grouped))
    rng = np.random.default_rng(seed)
    draws = np.empty(10000, dtype=np.float64)
    for j in range(len(draws)):
        sampled = rng.choice(names, size=len(names), replace=True)
        ix = [i for name in sampled for i in grouped[str(name)]]
        draws[j] = deltas[ix].mean() * 100.0
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def evaluate_pair(candidate: dict[str, np.ndarray], baseline: dict[str, np.ndarray], samples):
    cand_scores, base_scores, cand_classes, base_classes = [], [], [], []
    names = []
    for image_path, mask_path in samples:
        name = image_path.stem
        cand_score, cand_iou = score_probability_map(candidate[name], mask_path)
        base_score, base_iou = score_probability_map(baseline[name], mask_path)
        names.append(name)
        cand_scores.append(cand_score)
        base_scores.append(base_score)
        cand_classes.append(cand_iou)
        base_classes.append(base_iou)
    cand_scores = np.asarray(cand_scores)
    base_scores = np.asarray(base_scores)
    delta = cand_scores - base_scores
    classes_delta = np.mean(cand_classes, axis=0) - np.mean(base_classes, axis=0)
    return {
        "image_count": len(samples),
        "baseline_image_mean_miou": float(base_scores.mean()),
        "candidate_image_mean_miou": float(cand_scores.mean()),
        "delta_pp": float(delta.mean() * 100),
        "positive_images": int((delta > 0).sum()),
        "group_bootstrap_95ci_delta_pp": bootstrap_group_ci(
            delta, [specimen_id(path) for path, _ in samples]),
        "per_class_mean_iou_delta_pp": (100 * classes_delta).tolist(),
        "image_scores_baseline": base_scores.tolist(),
        "image_scores_candidate": cand_scores.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--parent-dir", default="outputs/model303_boundary_oof_8ep")
    ap.add_argument("--dino-path", default="hf_cache/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056")
    ap.add_argument("--output-dir", default="outputs/model308_dinov2_dense_fusion")
    ap.add_argument("--patch-size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=384)
    ap.add_argument("--patches-per-image", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--memory-fraction", type=float, default=0.90)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    dino = Dinov2Model.from_pretrained(args.dino_path, local_files_only=True).to(device).eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    if args.smoke_test:
        base = BoundaryEnhancedASPP().to(device).eval()
        model = FrozenDINOFusion(base, dino).to(device).eval()
        x = torch.randn(2, 3, 128, 128, device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            reference = base(x)
        difference = float((logits - reference).abs().max())
        if not torch.isfinite(logits).all() or difference > 1e-6:
            raise RuntimeError(f"DINO identity smoke failed: maxdiff={difference}")
        print(json.dumps({"device": str(device), "dino_layers": len(dino.config.__dict__),
                          "initial_logit_maxdiff": difference, "shape": list(logits.shape)}))
        return

    started = time.time()
    data = Path(args.data_dir)
    train_samples = collect_split(data, "train")
    valid_samples = collect_split(data, "valid")
    valid_images = sorted((data / "valid/images").glob("*_image.png"))
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)

    oof_candidate: dict[str, np.ndarray] = {}
    oof_baseline: dict[str, np.ndarray] = {}
    valid_candidate_sum: dict[str, np.ndarray] = {}
    valid_baseline_sum: dict[str, np.ndarray] = {}
    folds = []
    heldout_all: set[str] = set()

    for fold in range(1, 5):
        fold_started = time.time()
        fold_dir = output / f"fold{fold}"
        fold_dir.mkdir()
        parent_path = Path(args.parent_dir) / f"fold{fold}/last.pt"
        payload = torch.load(parent_path, map_location="cpu", weights_only=False)
        train_groups, heldout_groups = set(payload["train_groups"]), set(payload["heldout_groups"])
        if train_groups & heldout_groups:
            raise RuntimeError(f"group leakage in fold {fold}")
        fold_train = [(im, mask) for im, mask in train_samples if specimen_id(im) in train_groups]
        fold_heldout = [(im, mask) for im, mask in train_samples if specimen_id(im) in heldout_groups]
        heldout_names = {im.stem for im, _ in fold_heldout}
        if heldout_all & heldout_names:
            raise RuntimeError(f"overlapping heldout images in fold {fold}")
        heldout_all |= heldout_names

        base = BoundaryEnhancedASPP().to(device)
        incompatible = base.load_state_dict(payload["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"checkpoint mismatch fold={fold}: {incompatible}")
        base.eval()
        model = FrozenDINOFusion(base, dino).to(device)
        dataset = HighResolutionPatchDataset(
            fold_train, args.patch_size, args.patches_per_image, 1.0,
            seed=308 + fold, global_representation=True,
            metallography_augmentation=False, sampling_mode="class_focused")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                            persistent_workers=(args.num_workers > 0))
        optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=8e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = GradScaler(enabled=use_amp)
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total, count = 0.0, 0
            for images, masks in loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    loss = F.cross_entropy(logits, masks, weight=HARD_CLASS_WEIGHTS.to(device)) + dice_loss(logits, masks)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1.0)
                    optimizer.step()
                total += float(loss.detach()) * images.shape[0]
                count += images.shape[0]
            scheduler.step()
            record = {"epoch": epoch, "train_loss": total / max(count, 1)}
            history.append(record)
            print(f"fold={fold}/4 epoch={epoch}/{args.epochs} loss={record['train_loss']:.5f}", flush=True)

        torch.save({"base_model": base.state_dict(), "adapter": model.adapter.state_dict(),
                    "fold": fold, "train_groups": sorted(train_groups),
                    "heldout_groups": sorted(heldout_groups), "history": history,
                    "parent_checkpoint": str(parent_path), "validation_masks_used_for_selection": False},
                   fold_dir / "last.pt")

        model.eval()
        with torch.no_grad():
            for image_path, _ in fold_heldout:
                cand = predict_scaled(model, image_path, args.patch_size, args.stride, 1.0,
                                      device, use_amp, "d4")
                base_prob = predict_scaled(base, image_path, args.patch_size, args.stride, 1.0,
                                           device, use_amp, "d4")
                oof_candidate[image_path.stem] = cand.astype(np.float16)
                oof_baseline[image_path.stem] = base_prob.astype(np.float16)
            fold_cand_valid, fold_base_valid = {}, {}
            for image_path in valid_images:
                cand = predict_scaled(model, image_path, args.patch_size, args.stride, 1.0,
                                      device, use_amp, "d4")
                base_prob = predict_scaled(base, image_path, args.patch_size, args.stride, 1.0,
                                           device, use_amp, "d4")
                name = image_path.stem
                fold_cand_valid[name] = cand.astype(np.float16)
                fold_base_valid[name] = base_prob.astype(np.float16)
                valid_candidate_sum[name] = valid_candidate_sum.get(name, 0.0) + cand.astype(np.float32)
                valid_baseline_sum[name] = valid_baseline_sum.get(name, 0.0) + base_prob.astype(np.float32)
        np.savez_compressed(fold_dir / "train_oof_probs.npz",
                            **{name: oof_candidate[name] for name in heldout_names})
        np.savez_compressed(fold_dir / "valid_probs.npz", **fold_cand_valid)
        folds.append({"fold": fold, "train_image_count": len(fold_train),
                      "heldout_image_count": len(fold_heldout), "history": history,
                      "elapsed_seconds": time.time() - fold_started})
        del base, model, optimizer, scheduler, scaler, loader, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    expected_names = {p.stem for p, _ in train_samples}
    if set(oof_candidate) != expected_names or heldout_all != expected_names:
        raise RuntimeError("OOF predictions do not cover each train image exactly once")
    valid_candidate = {k: v / 4 for k, v in valid_candidate_sum.items()}
    valid_baseline = {k: v / 4 for k, v in valid_baseline_sum.items()}
    np.savez_compressed(output / "train_oof_probs.npz", **oof_candidate)
    np.savez_compressed(output / "train_oof_baseline_probs.npz", **oof_baseline)
    np.savez_compressed(output / "valid_probs.npz", **valid_candidate)
    np.savez_compressed(output / "valid_baseline_probs.npz", **valid_baseline)

    # Ground-truth masks are first opened here, after all training and fold
    # checkpoint decisions are finished.
    oof = evaluate_pair(oof_candidate, oof_baseline, train_samples)
    valid = evaluate_pair(valid_candidate, valid_baseline, valid_samples)
    summary = {
        "model": 308,
        "method": "frozen DINOv2-Small layers 3/6/9/12 dense patch features; zero-init residual decoder; frozen Model303 CNN",
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "patch_size": args.patch_size, "stride": args.stride,
        "tta": "D4", "epochs_per_fold": args.epochs,
        "train_masks_used_only_in_recorded_fold_train_groups": True,
        "validation_masks_used_for_training_selection_or_fold_decisions": False,
        "dino_license": "Meta DINOv2 repository states code and weights are Apache-2.0",
        "oof": oof, "valid_secondary": valid, "folds": folds,
        "elapsed_seconds": time.time() - started,
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"model": 308, "oof": {k:v for k,v in oof.items() if not k.startswith("image_scores")},
                      "valid_secondary": {k:v for k,v in valid.items() if not k.startswith("image_scores")},
                      "elapsed_seconds": summary["elapsed_seconds"], "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
