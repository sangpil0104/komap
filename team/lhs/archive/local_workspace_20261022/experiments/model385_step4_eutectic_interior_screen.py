#!/usr/bin/env python3
"""Train-OOF-only Stage-0 screen for a texture/phase-prototype experiment.

This is a descriptive feasibility gate, not a model evaluation or training
script. It uses Model374 grouped-OOF folds 1-2 and GT masks from the matching
training folds only. It measures what fraction of Step4 Eutectic false-negative
pixels are at least K native pixels from any phase boundary.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


PALETTE = np.asarray(
    [(153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)],
    dtype=np.uint8,
)
NUM_CLASSES = 4
EUTECTIC_SI = 3
DECISION_FOLDS = (1, 2)
BOUNDARY_DISTANCE_PX = 8.0
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 385
GROUP_RE = re.compile(r"^(A\d+)")


def mask_to_ids(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        ids[np.all(rgb == color, axis=2)] = class_id
    if np.any(ids == 255):
        unknown = np.unique(rgb[ids == 255], axis=0)
        raise ValueError(f"unknown mask colors in {path}: {unknown[:8].tolist()}")
    return ids


def phase_boundary(mask: np.ndarray) -> np.ndarray:
    """Mark both pixels on every 4-neighbor GT class transition."""
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def eutectic_false_negative_counts(
    target: np.ndarray, prediction: np.ndarray, k_px: float = BOUNDARY_DISTANCE_PX,
) -> dict[str, Any]:
    if target.shape != prediction.shape:
        raise ValueError(f"target/prediction shape mismatch: {target.shape} vs {prediction.shape}")
    boundary = phase_boundary(target)
    distance = distance_transform_edt(~boundary)
    eutectic_fn = (target == EUTECTIC_SI) & (prediction != EUTECTIC_SI)
    interior = eutectic_fn & (distance >= k_px)
    near = eutectic_fn & (distance < k_px)
    confusion_interior = np.bincount(
        prediction[interior].astype(np.int64), minlength=NUM_CLASSES
    )
    confusion_near = np.bincount(
        prediction[near].astype(np.int64), minlength=NUM_CLASSES
    )
    fn_count = int(eutectic_fn.sum())
    interior_count = int(interior.sum())
    return {
        "eutectic_gt_pixels": int((target == EUTECTIC_SI).sum()),
        "eutectic_false_negative_pixels": fn_count,
        "interior_fn_pixels_distance_ge_k": interior_count,
        "near_boundary_fn_pixels_distance_lt_k": int(near.sum()),
        "interior_fn_fraction": interior_count / fn_count if fn_count else None,
        "interior_fn_prediction_counts_by_class": confusion_interior.tolist(),
        "near_fn_prediction_counts_by_class": confusion_near.tolist(),
    }


def _group_bootstrap_ci(rows: list[dict[str, Any]], draws: int) -> list[float] | None:
    by_fold_group: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_fold_group[row["fold"]][row["specimen_group"]].append(row)
    if set(by_fold_group) != set(DECISION_FOLDS):
        raise ValueError("bootstrap requires both decision folds 1 and 2")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        numerator = denominator = 0
        for fold in DECISION_FOLDS:
            groups = sorted(by_fold_group[fold])
            selected = rng.integers(0, len(groups), size=len(groups))
            for position in selected:
                for row in by_fold_group[fold][groups[int(position)]]:
                    numerator += row["counts"]["interior_fn_pixels_distance_ge_k"]
                    denominator += row["counts"]["eutectic_false_negative_pixels"]
        samples[draw] = numerator / denominator if denominator else np.nan
    finite = samples[np.isfinite(samples)]
    if finite.size < int(0.95 * draws):
        return None
    return np.percentile(finite, [2.5, 97.5]).astype(float).tolist()


def analyze(cv_root: Path, oof_root: Path, draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    seen_group_fold: dict[str, int] = {}
    fold_counts: dict[str, dict[str, int]] = {}

    for fold in DECISION_FOLDS:
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        archive_path = oof_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        if not mask_dir.is_dir() or not archive_path.is_file():
            raise FileNotFoundError(f"fold {fold}: missing train OOF masks or probabilities")
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(masks):
                raise ValueError(f"fold {fold}: probability keys do not exactly match train masks")
            for image_id in sorted(masks):
                if image_id in seen_images:
                    raise ValueError(f"duplicate OOF image across folds: {image_id}")
                seen_images.add(image_id)
                group_match = GROUP_RE.match(image_id)
                if not group_match:
                    raise ValueError(f"cannot derive specimen group from {image_id}")
                group = group_match.group(1)
                if group in seen_group_fold and seen_group_fold[group] != fold:
                    raise ValueError(f"specimen group split across folds: {group}")
                seen_group_fold[group] = fold

                if "_step4_" not in image_id.lower():
                    continue
                probabilities = np.asarray(archive[image_id], dtype=np.float32)
                target = mask_to_ids(masks[image_id])
                if probabilities.shape != (NUM_CLASSES, *target.shape):
                    raise ValueError(f"invalid probability shape for {image_id}: {probabilities.shape}")
                if not np.isfinite(probabilities).all():
                    raise ValueError(f"non-finite probabilities for {image_id}")
                if np.max(np.abs(probabilities.sum(axis=0) - 1.0)) > 0.03:
                    raise ValueError(f"unnormalized probabilities for {image_id}")
                prediction = probabilities.argmax(axis=0).astype(np.uint8)
                counts = eutectic_false_negative_counts(target, prediction)
                rows.append({
                    "image_id": image_id,
                    "specimen_group": group,
                    "fold": fold,
                    "counts": counts,
                })
        fold_counts[str(fold)] = {
            "all_oof_images": len(masks),
            "step4_images": sum("_step4_" in name.lower() for name in masks),
            "step4_groups": len({r["specimen_group"] for r in rows if r["fold"] == fold}),
        }

    if not rows:
        raise ValueError("no Step4 images found in decision folds")
    fn_total = sum(r["counts"]["eutectic_false_negative_pixels"] for r in rows)
    interior_total = sum(r["counts"]["interior_fn_pixels_distance_ge_k"] for r in rows)
    if fn_total == 0:
        raise ValueError("no Step4 Eutectic false-negative pixels; Stage-0 fraction undefined")
    interior_fraction = interior_total / fn_total
    by_fold = {}
    for fold in DECISION_FOLDS:
        fold_rows = [r for r in rows if r["fold"] == fold]
        d = sum(r["counts"]["eutectic_false_negative_pixels"] for r in fold_rows)
        n = sum(r["counts"]["interior_fn_pixels_distance_ge_k"] for r in fold_rows)
        by_fold[str(fold)] = {
            "step4_images": len(fold_rows),
            "step4_groups": len({r["specimen_group"] for r in fold_rows}),
            "eutectic_fn_pixels": d,
            "interior_fn_fraction": n / d if d else None,
        }

    prediction_counts = {"interior": [0] * NUM_CLASSES, "near_boundary": [0] * NUM_CLASSES}
    for row in rows:
        for class_id, count in enumerate(row["counts"]["interior_fn_prediction_counts_by_class"]):
            prediction_counts["interior"][class_id] += count
        for class_id, count in enumerate(row["counts"]["near_fn_prediction_counts_by_class"]):
            prediction_counts["near_boundary"][class_id] += count

    return {
        "model": 385,
        "stage": "Stage 0; training-free feasibility screen for a separate prototype-loss experiment",
        "scope": "Model374 train-only grouped OOF folds 1-2; official valid/test excluded",
        "metric": "Step4 GT Eutectic false-negative pixel fraction at least K native pixels from any 4-neighbor GT phase boundary",
        "boundary_distance_native_pixels_k": BOUNDARY_DISTANCE_PX,
        "images_total_oof": len(seen_images),
        "specimen_groups_total_oof": len(seen_group_fold),
        "step4_images": len(rows),
        "step4_groups": len({r["specimen_group"] for r in rows}),
        "eutectic_fn_pixels": fn_total,
        "interior_fn_pixels": interior_total,
        "interior_fn_fraction": interior_fraction,
        "group_bootstrap_95ci": _group_bootstrap_ci(rows, draws),
        "bootstrap_draws": draws,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "per_fold": by_fold,
        "fn_destination_counts_by_zone_and_class_id": prediction_counts,
        "precommitted_screen_rule": "prototype feasibility screen passes iff pooled interior_fn_fraction >= 0.40; this is not authorization to train",
        "prototype_feasibility_screen_passed": bool(interior_fraction >= 0.40),
        "fold_image_counts": fold_counts,
        "interpretation_guard": "Two folds only; pixel-level error share is a feasibility diagnostic, not a generalization estimate or causal attribution. A pass only justifies drafting/running a separately preregistered grouped-OOF experiment when compute and model checkpoints are available.",
        "per_image": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True,
                        help="train-only grouped CV root containing oof_fold1..4/masks")
    parser.add_argument("--oof-root", type=Path, required=True,
                        help="Model374 output root containing foldN_oof/valid_probabilities.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        raise SystemExit("--bootstrap-draws must be >= 100")
    result = analyze(args.cv_root, args.oof_root, args.bootstrap_draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {k: v for k, v in result.items() if k != "per_image"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
