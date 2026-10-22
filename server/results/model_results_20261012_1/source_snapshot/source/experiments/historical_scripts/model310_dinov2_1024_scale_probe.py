#!/usr/bin/env python3
"""Test Model308 frozen-DINO residual at the original 1024px inference scale.

No training occurs. Each fold uses its own Model308 adapter and Model303 base
checkpoint; OOF and official-valid probabilities are paired against the same
base CNN at exactly 1024px/stride864/D4. Masks are opened only after all
predictions have been generated.
"""
from __future__ import annotations
from local_image_metrics import progress as _local_progress

import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Dinov2Model

from improve_five_causes import collect_split, decode_mask, specimen_id
from model253_highres_scale_finetune import predict_scaled
from model288_region_classifier_cv import image_score
from model301_boundary_enhanced_finetune import BoundaryEnhancedASPP
from model308_dinov2_dense_fusion_oof import FrozenDINOFusion


def paired_ci(deltas: np.ndarray, groups: list[str], seed: int = 310) -> list[float]:
    by_group: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        by_group.setdefault(group, []).append(i)
    names = np.asarray(sorted(by_group))
    rng = np.random.default_rng(seed)
    draws = np.empty(10000)
    for j in range(len(draws)):
        sampled = rng.choice(names, len(names), replace=True)
        ix = [i for name in sampled for i in by_group[str(name)]]
        draws[j] = deltas[ix].mean() * 100
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def paired_scores(candidate: dict[str, np.ndarray], baseline: dict[str, np.ndarray], samples):
    cand_scores, base_scores, cand_cls, base_cls = [], [], [], []
    for image_path, mask_path in _local_progress(samples, 'model310_dinov2_1024_scale_probe'):
        name = image_path.stem
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        cp = candidate[name].argmax(0).astype(np.uint8)
        bp = baseline[name].argmax(0).astype(np.uint8)
        cand_scores.append(image_score(cp, target))
        base_scores.append(image_score(bp, target))
        ci, bi = [], []
        for c in range(4):
            cm, bm, tm = cp == c, bp == c, target == c
            cu = np.logical_or(cm, tm).sum()
            bu = np.logical_or(bm, tm).sum()
            ci.append(float(np.logical_and(cm, tm).sum() / cu) if cu else 0.0)
            bi.append(float(np.logical_and(bm, tm).sum() / bu) if bu else 0.0)
        cand_cls.append(ci); base_cls.append(bi)
    cand_scores, base_scores = np.asarray(cand_scores), np.asarray(base_scores)
    delta = cand_scores - base_scores
    return {
        "image_count": len(samples),
        "baseline_image_mean_miou": float(base_scores.mean()),
        "candidate_image_mean_miou": float(cand_scores.mean()),
        "delta_pp": float(delta.mean() * 100),
        "positive_images": int((delta > 0).sum()),
        "group_bootstrap_95ci_delta_pp": paired_ci(
            delta, [specimen_id(image) for image, _ in samples]),
        "per_class_mean_iou_delta_pp": (100 * (np.mean(cand_cls, axis=0)-np.mean(base_cls, axis=0))).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-existing", action="store_true",
                        help="score complete saved probability archives without running inference")
    args = parser.parse_args()
    data = Path("data")
    parent_dir = Path("outputs/model303_boundary_oof_8ep")
    fusion_dir = Path("outputs/model308_dinov2_dense_fusion")
    out = Path("outputs/model310_dinov2_1024_scale_probe")
    if out.exists() and not args.score_existing:
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    train_samples, valid_samples = collect_split(data, "train"), collect_split(data, "valid")
    valid_images = sorted((data / "valid/images").glob("*_image.png"))
    if args.score_existing:
        oof_cand = {k: v for k, v in np.load(out / "train_oof_candidate.npz").items()}
        oof_base = {k: v for k, v in np.load(out / "train_oof_baseline.npz").items()}
        valid_cand = {k: v for k, v in np.load(out / "valid_candidate.npz").items()}
        valid_base = {k: v for k, v in np.load(out / "valid_baseline.npz").items()}
        folds = []
        for fold in range(1, 5):
            payload = torch.load(parent_dir / f"fold{fold}/last.pt", map_location="cpu", weights_only=False)
            folds.append({"fold": fold, "heldout_groups": sorted(payload["heldout_groups"])})
        result = {
            "model": 310,
            "method": "Model308 DINO residual trained at 512px, evaluated without retraining at 1024px",
            "inference": {"patch_size": 1024, "stride": 864, "tta": "D4", "blend": "Gaussian"},
            "validation_masks_used_for_training_or_selection": False,
            "oof": paired_scores(oof_cand, oof_base, train_samples),
            "valid_secondary": paired_scores(valid_cand, valid_base, valid_samples),
            "folds": folds,
            "inference_seconds": 1343.0,
            "scoring_only_rerun": True,
        }
        # Check that the independently regenerated baseline matches Model303's
        # canonical 1024px probability archives before interpreting the delta.
        reference_train = np.load(parent_dir / "train_oof_probs.npz")
        reference_valid = np.load(parent_dir / "valid_fold_ensemble_probs.npz")
        def archive_mae(left, right):
            keys = sorted(set(left.files) & set(right.files))
            means, maxima = [], []
            for key in keys:
                diff = np.abs(left[key].astype(np.float32) - right[key].astype(np.float32))
                means.append(float(diff.mean()))
                maxima.append(float(diff.max()))
            return {"image_count": len(keys), "mean_absolute_probability_delta": float(np.mean(means)),
                    "max_absolute_probability_delta": float(np.max(maxima))}
        result["baseline_probability_parity_vs_model303"] = {
            "oof": archive_mae(np.load(out / "train_oof_baseline.npz"), reference_train),
            "valid": archive_mae(np.load(out / "valid_baseline.npz"), reference_valid),
        }
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    torch.set_num_threads(4)
    if use_amp:
        torch.cuda.set_per_process_memory_fraction(0.90, device.index or 0)
        torch.backends.cudnn.benchmark = True
    dino = Dinov2Model.from_pretrained(
        "hf_cache/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056",
        local_files_only=True).to(device).eval()
    for p in dino.parameters():
        p.requires_grad_(False)

    oof_cand, oof_base = {}, {}
    valid_cand_sum, valid_base_sum = {}, {}
    fold_rows, heldout_names_all = [], set()
    started = time.time()
    for fold in range(1, 5):
        parent = torch.load(parent_dir / f"fold{fold}/last.pt", map_location="cpu", weights_only=False)
        saved = torch.load(fusion_dir / f"fold{fold}/last.pt", map_location="cpu", weights_only=False)
        heldout_groups = set(parent["heldout_groups"])
        if heldout_groups != set(saved["heldout_groups"]):
            raise RuntimeError(f"Model303/308 heldout group mismatch fold {fold}")
        heldout = [(im, mask) for im, mask in train_samples if specimen_id(im) in heldout_groups]
        names = {im.stem for im, _ in heldout}
        if heldout_names_all & names:
            raise RuntimeError(f"duplicate OOF predictions fold {fold}")
        heldout_names_all |= names

        base = BoundaryEnhancedASPP().to(device)
        base.load_state_dict(parent["model"], strict=True)
        candidate = FrozenDINOFusion(base, dino).to(device)
        candidate.adapter.load_state_dict(saved["adapter"], strict=True)
        candidate.eval(); base.eval()
        with torch.no_grad():
            for image_path, _ in heldout:
                cp = predict_scaled(candidate, image_path, 1024, 864, 1.0, device, use_amp, "d4")
                bp = predict_scaled(base, image_path, 1024, 864, 1.0, device, use_amp, "d4")
                oof_cand[image_path.stem] = cp.astype(np.float16)
                oof_base[image_path.stem] = bp.astype(np.float16)
            fold_cand_valid, fold_base_valid = {}, {}
            for image_path in valid_images:
                cp = predict_scaled(candidate, image_path, 1024, 864, 1.0, device, use_amp, "d4")
                bp = predict_scaled(base, image_path, 1024, 864, 1.0, device, use_amp, "d4")
                name = image_path.stem
                fold_cand_valid[name] = cp.astype(np.float16)
                fold_base_valid[name] = bp.astype(np.float16)
                valid_cand_sum[name] = valid_cand_sum.get(name, 0.0) + cp.astype(np.float32)
                valid_base_sum[name] = valid_base_sum.get(name, 0.0) + bp.astype(np.float32)
        np.savez_compressed(out / f"fold{fold}_valid_candidate.npz", **fold_cand_valid)
        np.savez_compressed(out / f"fold{fold}_valid_baseline.npz", **fold_base_valid)
        fold_rows.append({"fold": fold, "heldout_images": len(heldout), "heldout_groups": sorted(heldout_groups)})
        print(f"fold={fold}/4 inference complete heldout={len(heldout)}", flush=True)
        del candidate, base
        if use_amp:
            torch.cuda.empty_cache()

    expected = {im.stem for im, _ in train_samples}
    if set(oof_cand) != expected or heldout_names_all != expected:
        raise RuntimeError("OOF image coverage check failed")
    valid_cand = {k: v / 4 for k, v in valid_cand_sum.items()}
    valid_base = {k: v / 4 for k, v in valid_base_sum.items()}
    np.savez_compressed(out / "train_oof_candidate.npz", **oof_cand)
    np.savez_compressed(out / "train_oof_baseline.npz", **oof_base)
    np.savez_compressed(out / "valid_candidate.npz", **valid_cand)
    np.savez_compressed(out / "valid_baseline.npz", **valid_base)

    # Only now open train/valid masks for the final paired metric.
    result = {
        "model": 310,
        "method": "Model308 DINO residual trained at 512px, evaluated without retraining at 1024px",
        "inference": {"patch_size": 1024, "stride": 864, "tta": "D4", "blend": "Gaussian"},
        "validation_masks_used_for_training_or_selection": False,
        "oof": paired_scores(oof_cand, oof_base, train_samples),
        "valid_secondary": paired_scores(valid_cand, valid_base, valid_samples),
        "folds": fold_rows,
        "elapsed_seconds": time.time() - started,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
