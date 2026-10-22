#!/usr/bin/env python3
"""Cross-fold check of a fixed Al3Ni/Eutectic component cleanup rule.

Select thresholds on one grouped OOF fold and score the other fold. This is a
small baseline for component correction, not a search on official validation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from model374_train_oof_error_diagnostic import image_ious, mask_to_ids


CLASS_IDS = (1, 3)  # Al3Ni, Eutectic_Si in Model336/374 channel order
THRESHOLDS = (0, 8, 32, 128)
CANDIDATES = tuple(itertools.product(THRESHOLDS, repeat=2))


def image_scores(probabilities: np.ndarray, target: np.ndarray) -> dict[tuple[int, int], float]:
    base = probabilities.argmax(axis=0).astype(np.uint8)
    if base.shape != target.shape:
        raise ValueError("probability/mask spatial shapes differ")
    alternatives = {}
    component_sizes = {}
    for class_id in CLASS_IDS:
        other_probabilities = probabilities.copy()
        other_probabilities[class_id] = -np.inf
        alternatives[class_id] = other_probabilities.argmax(axis=0).astype(np.uint8)
        foreground = (base == class_id).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        component_sizes[class_id] = stats[:, cv2.CC_STAT_AREA][labels]
        del count
    out = {}
    for threshold_1, threshold_3 in CANDIDATES:
        modified = base.copy()
        for class_id, threshold in ((1, threshold_1), (3, threshold_3)):
            if threshold > 0:
                small = (base == class_id) & (component_sizes[class_id] < threshold)
                modified[small] = alternatives[class_id][small]
        out[(threshold_1, threshold_3)] = float(np.mean(image_ious(modified, target)))
    return out


def evaluate(cv_data: Path, oof_root: Path) -> dict:
    rows = []
    seen = set()
    for fold in (1, 2):
        masks = {p.name.removesuffix("_mask.png"): p
                 for p in (cv_data / f"oof_fold{fold}" / "masks").glob("*_mask.png")}
        archive_path = oof_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(masks):
                raise ValueError(f"fold {fold} mask/probability IDs differ")
            for image_id in sorted(masks):
                if image_id in seen:
                    raise ValueError(f"duplicate image ID: {image_id}")
                seen.add(image_id)
                target = mask_to_ids(masks[image_id])
                probabilities = np.asarray(archive[image_id], dtype=np.float32)
                if probabilities.shape != (4, *target.shape) or not np.isfinite(probabilities).all():
                    raise ValueError(f"invalid probabilities for {image_id}")
                scores = image_scores(probabilities, target)
                rows.append({"image_id": image_id, "fold": fold,
                             "scores": {f"{a},{b}": score for (a, b), score in scores.items()}})
    if len(rows) != 36:
        raise ValueError(f"expected 36 images in OOF folds 1 and 2; found {len(rows)}")
    heldout = []
    for train_fold, eval_fold in ((1, 2), (2, 1)):
        train = [r for r in rows if r["fold"] == train_fold]
        test = [r for r in rows if r["fold"] == eval_fold]
        selected = max(CANDIDATES, key=lambda c: (
            np.mean([r["scores"][f"{c[0]},{c[1]}"] for r in train]),
            -(c[0] + c[1]),
        ))
        key = f"{selected[0]},{selected[1]}"
        gains = [r["scores"][key] - r["scores"]["0,0"] for r in test]
        heldout.append({
            "threshold_selection_fold": train_fold,
            "evaluation_fold": eval_fold,
            "selected_al3ni_eutectic_thresholds": list(selected),
            "train_image_mean_miou_empty0": float(np.mean([r["scores"][key] for r in train])),
            "eval_images": len(test),
            "eval_base_image_mean_miou_empty0": float(np.mean([r["scores"]["0,0"] for r in test])),
            "eval_cleanup_image_mean_miou_empty0": float(np.mean([r["scores"][key] for r in test])),
            "eval_gain_percentage_points": float(100 * np.mean(gains)),
            "eval_images_improved": int(np.count_nonzero(np.asarray(gains) > 1e-9)),
            "eval_images_worsened": int(np.count_nonzero(np.asarray(gains) < -1e-9)),
        })
    weighted_gain = sum(row["eval_gain_percentage_points"] * row["eval_images"] for row in heldout) / 36
    return {
        "scope": "training-only grouped OOF folds 1 and 2; 36 images",
        "metric": "image mean of 4 class IoUs, empty union=0",
        "thresholds_are_exclusive_pixel_area_cutoffs": list(THRESHOLDS),
        "class_order": ["Al", "Al3Ni", "Primary_Si", "Eutectic_Si"],
        "two_way_heldout_gain_percentage_points": weighted_gain,
        "heldout": heldout,
        "interpretation_guard": "Preliminary two-fold check; no official valid/test score and no full four-fold estimate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-data", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.cv_data, args.oof_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
