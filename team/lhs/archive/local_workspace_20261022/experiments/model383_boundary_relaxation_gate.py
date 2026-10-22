#!/usr/bin/env python3
"""CPU-only feasibility gate for a fixed 3x3 Boundary Label Relaxation trial.

The oracle replaces predictions in the exact relaxed-label band with GT, so it
is only a feasibility ceiling. The distance/error gate asks whether errors are
mostly boundary-local. Partial folds are exploratory and cannot return GO.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from model374_train_oof_error_diagnostic import image_ious, mask_to_ids


GROUP_RE = re.compile(r"^(A\d+)")
NUM_CLASSES = 4
EXPECTED_FOLD_IMAGES = {1: 18, 2: 18, 3: 17, 4: 17}


def relaxed_label_sets(truth: np.ndarray) -> np.ndarray:
    """Return (4,H,W) allowed classes from the pixel's 3x3 GT neighborhood."""
    if truth.ndim != 2 or truth.size == 0 or truth.min() < 0 or truth.max() >= NUM_CLASSES:
        raise ValueError("truth must be a nonempty 2-D mask with class IDs 0..3")
    height, width = truth.shape
    allowed = np.zeros((NUM_CLASSES, height, width), dtype=bool)
    for class_id in range(NUM_CLASSES):
        padded = np.pad(truth == class_id, 1, mode="edge")
        present = np.zeros((height, width), dtype=bool)
        for dy in range(3):
            for dx in range(3):
                present |= padded[dy:dy + height, dx:dx + width]
        allowed[class_id] = present
    if not np.all(allowed.any(axis=0)):
        raise AssertionError("every pixel must have at least its own GT class allowed")
    return allowed


def relaxed_band(truth: np.ndarray) -> np.ndarray:
    """Pixels where a 3x3 GT neighborhood contains more than one class."""
    return relaxed_label_sets(truth).sum(axis=0) > 1


def euclidean_distance_to_band(band: np.ndarray) -> np.ndarray:
    if band.ndim != 2 or not band.any():
        raise ValueError("boundary band must be a nonempty 2-D mask")
    return cv2.distanceTransform(
        (~band).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )


def analyze(
    cv_root: Path,
    model374_root: Path,
    folds: tuple[int, ...] = (1, 2, 3, 4),
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold in folds:
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        archive_path = model374_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        if not mask_dir.is_dir() or not archive_path.is_file():
            raise FileNotFoundError(f"missing fold {fold} masks or Model374 OOF probabilities")
        masks = {
            path.name.removesuffix("_mask.png"): path
            for path in mask_dir.glob("*_mask.png")
        }
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(masks):
                raise ValueError(f"fold {fold}: OOF image IDs do not match GT mask IDs")
            if len(masks) != EXPECTED_FOLD_IMAGES[fold]:
                raise ValueError(f"fold {fold}: expected {EXPECTED_FOLD_IMAGES[fold]} images, found {len(masks)}")
            for image_id in sorted(masks):
                if image_id in seen:
                    raise ValueError(f"duplicate OOF image: {image_id}")
                seen.add(image_id)
                if GROUP_RE.match(image_id) is None:
                    raise ValueError(f"cannot parse specimen group from {image_id}")
                truth = mask_to_ids(masks[image_id])
                probabilities = np.asarray(archive[image_id], dtype=np.float32)
                if probabilities.shape != (NUM_CLASSES, *truth.shape) or not np.isfinite(probabilities).all():
                    raise ValueError(f"invalid probability shape/values for {image_id}")
                pred = probabilities.argmax(axis=0).astype(np.uint8)
                band = relaxed_band(truth)
                distance = euclidean_distance_to_band(band)
                wrong = pred != truth
                oracle = pred.copy()
                oracle[band] = truth[band]
                rows.append({
                    "image_id": image_id,
                    "group": GROUP_RE.match(image_id).group(1),
                    "fold": fold,
                    "base_miou": float(np.mean(image_ious(pred, truth))),
                    "boundary_oracle_miou": float(np.mean(image_ious(oracle, truth))),
                    "wrong_pixels": int(wrong.sum()),
                    "wrong_pixels_gt_distance_gt3": int((wrong & (distance > 3.0)).sum()),
                    "relaxed_band_fraction": float(band.mean()),
                })

    complete = set(folds) == {1, 2, 3, 4} and len(rows) == 70 and len({r["group"] for r in rows}) == 59
    base = np.asarray([row["base_miou"] for row in rows], dtype=np.float64)
    oracle = np.asarray([row["boundary_oracle_miou"] for row in rows], dtype=np.float64)
    total_wrong = sum(row["wrong_pixels"] for row in rows)
    total_far = sum(row["wrong_pixels_gt_distance_gt3"] for row in rows)
    oracle_gain_pp = float(100 * (oracle - base).mean())
    far_error_fraction = float(total_far / total_wrong) if total_wrong else 0.0
    gates = {
        "boundary_oracle_gain_at_least_1pp": oracle_gain_pp >= 1.0,
        "wrong_pixels_more_than_3px_from_relaxation_band_below_50pct": far_error_fraction < 0.50,
    }
    return {
        "model": 383,
        "method": "Model374 OOF GT oracle over the fixed 3x3 neighborhood label-relaxation band",
        "metric": "image-wise four-class mIoU; empty-union=0",
        "scope": "complete train OOF" if complete else "partial train OOF; exploratory only",
        "images": len(rows),
        "groups": len({row["group"] for row in rows}),
        "folds": sorted(set(folds)),
        "base_image_mean_miou_empty0_pct": float(base.mean() * 100),
        "boundary_oracle_image_mean_miou_empty0_pct": float(oracle.mean() * 100),
        "boundary_oracle_gain_pp": oracle_gain_pp,
        "fraction_wrong_pixels_gt3px_from_relaxation_band": far_error_fraction,
        "per_fold": {
            str(fold): {
                "images": sum(row["fold"] == fold for row in rows),
                "base_miou_pct": float(np.mean([row["base_miou"] for row in rows if row["fold"] == fold]) * 100),
                "oracle_miou_pct": float(np.mean([row["boundary_oracle_miou"] for row in rows if row["fold"] == fold]) * 100),
            }
            for fold in sorted(set(folds))
        },
        "preregistered_cpu_gate": gates if complete else None,
        "go": all(gates.values()) if complete else None,
        "per_image": rows,
        "interpretation_guard": "The GT-assisted oracle is an upper bound, never a model score. Only all 70 exactly-once grouped OOF images can decide the gate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--model374-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.folds or len(set(args.folds)) != len(args.folds) or any(f not in (1, 2, 3, 4) for f in args.folds):
        raise SystemExit("--folds must be a unique subset of 1,2,3,4")
    result = analyze(args.cv_root, args.model374_root, tuple(sorted(args.folds)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
