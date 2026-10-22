#!/usr/bin/env python3
"""Train-OOF-only expected-IoU per-image decoder probe for Model303.

The per-image biases are selected from probabilities alone using a
ratio-of-expectations mIoU surrogate. Ground-truth masks are opened only after
all per-image biases have been fixed, for paired OOF scoring. Official valid
masks are used only for a final, explicitly secondary measurement.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


COLORS = np.asarray(
    [(25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204)],
    dtype=np.uint8,
)
BIAS_GRID = np.asarray([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
TUNED_CLASSES = (0, 1, 3)  # Keep Al (class 2) fixed as the stable anchor.


def decode_mask(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    labels = np.full(rgb.shape[:2], -1, dtype=np.int8)
    for k, color in enumerate(COLORS):
        labels[np.all(rgb == color, axis=-1)] = k
    if np.any(labels < 0):
        raise ValueError(f"Unrecognized palette in {path}")
    return labels


def predict(prob: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.argmax(np.log(np.maximum(prob.astype(np.float32), 1e-7)) + bias[:, None, None], axis=0)


def expected_miou(prob: np.ndarray, bias: np.ndarray) -> float:
    pred = predict(prob, bias)
    scores = []
    for c in range(4):
        selected = pred == c
        area = int(selected.sum())
        expected_tp = float(prob[c][selected].sum(dtype=np.float64))
        expected_fn = float(prob[c][~selected].sum(dtype=np.float64))
        denom = area + expected_fn
        scores.append(expected_tp / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def actual_scores(pred: np.ndarray, target: np.ndarray) -> tuple[float, list[float]]:
    per_class = []
    for c in range(4):
        p, t = pred == c, target == c
        union = int(np.logical_or(p, t).sum())
        inter = int(np.logical_and(p, t).sum())
        per_class.append(inter / union if union else 0.0)
    return float(np.mean(per_class)), per_class


def group_name(image_id: str) -> str:
    match = re.match(r"(A\d+)", image_id)
    return match.group(1) if match else image_id


def paired_group_bootstrap(deltas: np.ndarray, groups: list[str], seed: int = 307,
                           repetitions: int = 10000) -> list[float]:
    unique = np.asarray(sorted(set(groups)))
    indices = {g: np.flatnonzero(np.asarray(groups) == g) for g in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for i in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([indices[g] for g in sampled])
        draws[i] = deltas[rows].mean()
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/workspace/komap_model1/data"))
    ap.add_argument("--oof", type=Path, default=Path("/workspace/komap_model1/outputs/model303_boundary_oof_8ep/train_oof_probs.npz"))
    ap.add_argument("--valid-probs", type=Path, default=Path("/workspace/komap_model1/outputs/model303_boundary_oof_8ep/valid_fold_ensemble_probs.npz"))
    ap.add_argument("--output", type=Path, default=Path("/workspace/komap_model1/outputs/model307_expected_iou_decoder/results.json"))
    ap.add_argument("--pool-size", type=int, default=128)
    ap.add_argument("--sweeps", type=int, default=2)
    args = ap.parse_args()

    oof = np.load(args.oof)
    image_ids = list(oof.files)
    all_biases, base_scores, decoded_scores, base_classes, classes = [], [], [], [], []
    for image_id in image_ids:
        full_prob = oof[image_id].astype(np.float32)
        h, w = full_prob.shape[-2:]
        small = F.adaptive_avg_pool2d(torch.from_numpy(full_prob)[None], (args.pool_size, args.pool_size))[0].numpy()
        logp = np.log(np.maximum(small, 1e-7))
        bias = np.zeros(4, dtype=np.float32)
        for _ in range(args.sweeps):
            for c in TUNED_CLASSES:
                best_score, best_bias = -1.0, float(bias[c])
                for candidate in BIAS_GRID:
                    trial = bias.copy()
                    trial[c] = candidate
                    score = expected_miou(np.exp(logp), trial)
                    if score > best_score:
                        best_score, best_bias = score, float(candidate)
                bias[c] = best_bias

        mask_path = args.data / "train/masks" / f"{image_id.replace('_image', '_mask')}.png"
        target = decode_mask(mask_path)
        if target.shape != (h, w):
            raise ValueError(f"OOF probability/mask shape mismatch for {image_id}: {(h,w)} vs {target.shape}")
        base, base_per_class = actual_scores(np.argmax(full_prob, axis=0), target)
        adjusted, per_class = actual_scores(predict(full_prob, bias), target)
        all_biases.append(bias.tolist())
        base_scores.append(base)
        decoded_scores.append(adjusted)
        base_classes.append(base_per_class)
        classes.append(per_class)

    base_scores = np.asarray(base_scores)
    decoded_scores = np.asarray(decoded_scores)
    deltas = decoded_scores - base_scores
    groups = [group_name(x) for x in image_ids]
    oof_result = {
        "image_count": len(image_ids),
        "baseline_image_mean_miou": float(base_scores.mean()),
        "expected_iou_decoder_image_mean_miou": float(decoded_scores.mean()),
        "delta_pp": float(deltas.mean() * 100),
        "positive_images": int((deltas > 0).sum()),
        "group_bootstrap_95ci_delta_pp": [100*x for x in paired_group_bootstrap(deltas, groups)],
        "mean_class_iou_baseline": np.mean(np.asarray(base_classes), axis=0).tolist(),
        "mean_class_iou_decoder": np.mean(np.asarray(classes), axis=0).tolist(),
        "per_class_delta_pp": (100*(np.mean(np.asarray(classes), axis=0) - np.mean(np.asarray(base_classes), axis=0))).tolist(),
        "bias_mean": np.mean(np.asarray(all_biases), axis=0).tolist(),
        "bias_values_by_class": {str(c): {str(float(v)): int(np.sum(np.isclose(np.asarray(all_biases)[:,c],v))) for v in BIAS_GRID} for c in TUNED_CLASSES},
        "selection_used_ground_truth": False,
        "selection_data": "per-image Model303 train grouped-OOF probabilities only",
        "scoring_masks": "train masks opened only after all per-image biases were selected",
        "classes": ["Al3Ni", "Primary Si", "Al", "Eutectic Si"],
    }

    result = {"model": 307, "method": "per-image probability-only coordinate search maximizing a ratio-of-expectations mIoU surrogate; class Al fixed", "pool_size": args.pool_size, "sweeps": args.sweeps, "bias_grid": BIAS_GRID.tolist(), "oof": oof_result}

    # Secondary diagnostic only: apply the same unlabeled per-image rule to
    # official valid predictions, then open valid masks solely for scoring.
    if args.valid_probs.exists():
        valid = np.load(args.valid_probs)
        rows = []
        for image_id in valid.files:
            prob = valid[image_id].astype(np.float32)
            small = F.adaptive_avg_pool2d(torch.from_numpy(prob)[None], (args.pool_size,args.pool_size))[0].numpy()
            bias = np.zeros(4, dtype=np.float32)
            for _ in range(args.sweeps):
                for c in TUNED_CLASSES:
                    best_score, best_bias = -1.0, float(bias[c])
                    for candidate in BIAS_GRID:
                        trial = bias.copy(); trial[c] = candidate
                        score = expected_miou(small, trial)
                        if score > best_score:
                            best_score, best_bias = score, float(candidate)
                    bias[c] = best_bias
            mask_id = image_id.replace("_image", "_mask")
            target = decode_mask(args.data / "valid/masks" / f"{mask_id}.png")
            if target.shape != prob.shape[-2:]:
                raise ValueError(f"valid probability/mask shape mismatch for {image_id}")
            raw, _ = actual_scores(np.argmax(prob,axis=0), target)
            dec, _ = actual_scores(predict(prob,bias), target)
            rows.append((raw,dec))
        pair = np.asarray(rows)
        result["official_valid_secondary_only"] = {
            "image_count": len(rows), "baseline_image_mean_miou": float(pair[:,0].mean()),
            "decoder_image_mean_miou": float(pair[:,1].mean()), "delta_pp": float((pair[:,1]-pair[:,0]).mean()*100),
            "used_for_bias_selection": False,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"model": 307, "oof": oof_result, "valid_secondary": result.get("official_valid_secondary_only"), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
