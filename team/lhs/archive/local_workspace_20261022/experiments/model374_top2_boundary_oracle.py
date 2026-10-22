#!/usr/bin/env python3
"""GT-assisted feasibility check for a predicted-boundary top-2 refiner.

Uses only grouped training OOF predictions. Results are oracle ceilings, not
deployable scores or evidence that a learned refiner will attain them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from model374_train_oof_error_diagnostic import image_ious, mask_to_ids


def edge_map(labels: np.ndarray) -> np.ndarray:
    edge = np.zeros(labels.shape, np.uint8)
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return edge


def expand(edge: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return edge.astype(bool)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    return cv2.dilate(edge, kernel).astype(bool)


def score(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(image_ious(pred, truth)))


def analyze(cv_data: Path, oof_root: Path, radius: int) -> dict:
    rows = []
    seen = set()
    total_wrong = total_correctable = total_band_pixels = total_pixels = 0
    gt_error_buckets = np.zeros(4, np.int64)
    for fold in range(1, 5):
        masks = {p.name.removesuffix("_mask.png"): p
                 for p in (cv_data / f"oof_fold{fold}" / "masks").glob("*_mask.png")}
        archive = oof_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        with np.load(archive, allow_pickle=False) as values:
            if set(values.files) != set(masks):
                raise ValueError(f"OOF image IDs and mask IDs differ in fold {fold}")
            for name in sorted(masks):
                if name in seen:
                    raise ValueError(f"duplicate OOF image: {name}")
                seen.add(name)
                truth = mask_to_ids(masks[name])
                prob = np.asarray(values[name], np.float32)
                if prob.shape != (4, *truth.shape) or not np.isfinite(prob).all():
                    raise ValueError(f"invalid probabilities for {name}")
                pred = prob.argmax(axis=0).astype(np.uint8)
                alternative_prob = prob.copy()
                np.put_along_axis(alternative_prob, pred[None], -np.inf, axis=0)
                second = alternative_prob.argmax(axis=0).astype(np.uint8)
                wrong = pred != truth
                pred_band = expand(edge_map(pred), radius)
                correctable = wrong & pred_band & (second == truth)
                oracle = pred.copy()
                oracle[correctable] = truth[correctable]

                gt_boundary = edge_map(truth)
                gt_0 = gt_boundary.astype(bool)
                gt_1 = expand(gt_boundary, 1)
                gt_3 = expand(gt_boundary, 3)
                buckets = [int((wrong & gt_0).sum()),
                           int((wrong & gt_1 & ~gt_0).sum()),
                           int((wrong & gt_3 & ~gt_1).sum()),
                           int((wrong & ~gt_3).sum())]
                gt_error_buckets += buckets
                wrong_count = int(wrong.sum())
                correctable_count = int(correctable.sum())
                total_wrong += wrong_count
                total_correctable += correctable_count
                total_band_pixels += int(pred_band.sum())
                total_pixels += pred.size
                rows.append({
                    "image_id": name, "fold": fold,
                    "base_image_miou_empty0": score(pred, truth),
                    "top2_pred_boundary_oracle_image_miou_empty0": score(oracle, truth),
                    "wrong_pixels": wrong_count,
                    "oracle_correctable_wrong_pixels": correctable_count,
                    "predicted_boundary_band_fraction": float(pred_band.mean()),
                    "wrong_pixel_gt_boundary_distance_buckets_on_edge_1px_2to3px_gt3px": buckets,
                })
    if len(rows) != 70:
        raise ValueError(f"expected 70 unique OOF images, got {len(rows)}")
    base = np.asarray([r["base_image_miou_empty0"] for r in rows])
    oracle = np.asarray([r["top2_pred_boundary_oracle_image_miou_empty0"] for r in rows])
    return {
        "scope": "70 train-only grouped OOF images, 4 folds; GT-assisted diagnostic",
        "metric": "image mean of four class IoUs; empty union=0",
        "predicted_boundary_dilation_radius_pixels": radius,
        "base_image_mean_miou_empty0": float(base.mean()),
        "top2_pred_boundary_oracle_image_mean_miou_empty0": float(oracle.mean()),
        "oracle_gain_percentage_points": float(100 * (oracle - base).mean()),
        "correctable_fraction_of_wrong_pixels": total_correctable / total_wrong,
        "predicted_boundary_band_area_fraction": total_band_pixels / total_pixels,
        "wrong_pixel_gt_boundary_distance_buckets_on_edge_1px_2to3px_gt3px": gt_error_buckets.tolist(),
        "wrong_pixel_gt_boundary_distance_bucket_fractions":
            (gt_error_buckets / total_wrong).tolist(),
        "per_fold": {str(fold): {
            "images": sum(r["fold"] == fold for r in rows),
            "base_mean": float(np.mean([r["base_image_miou_empty0"] for r in rows if r["fold"] == fold])),
            "oracle_mean": float(np.mean([r["top2_pred_boundary_oracle_image_miou_empty0"]
                                         for r in rows if r["fold"] == fold])),
        } for fold in range(1, 5)},
        "per_image": rows,
        "interpretation_guard": "Oracle reads GT per pixel. Its gain is only a feasibility ceiling for the constrained correction space.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-data", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.radius < 0:
        raise SystemExit("--radius must be nonnegative")
    result = analyze(args.cv_data, args.oof_root, args.radius)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
