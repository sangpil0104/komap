#!/usr/bin/env python3
"""Train-OOF-only distance-stratified Step4 Eutectic error-rate diagnostic.

This is a descriptive CPU analysis, not a training or model-selection script.
It corrects the denominator issue in Model385 by reporting false-negative
rates relative to GT Eutectic pixels in each distance-to-phase-boundary band.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


PALETTE = np.asarray(
    [(153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)],
    dtype=np.uint8,
)
NUM_CLASSES = 4
EUTECTIC_SI = 3
GROUP_RE = re.compile(r"^(A\d+)")
BIN_NAMES = ("0_to_lt2px", "2_to_lt4px", "4_to_lt8px", "8px_or_more")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 386


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
    """Mark both sides of every 4-neighbor transition between any GT classes."""
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def eutectic_distance_bin_counts(
    target: np.ndarray, prediction: np.ndarray,
) -> dict[str, Any]:
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must be same-size 2-D arrays")
    boundary = phase_boundary(target)
    if not boundary.any():
        raise ValueError("GT mask has no phase boundary")
    distance = cv2.distanceTransform(
        (~boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    eutectic = target == EUTECTIC_SI
    fn = eutectic & (prediction != EUTECTIC_SI)
    masks = (
        eutectic & (distance < 2.0),
        eutectic & (distance >= 2.0) & (distance < 4.0),
        eutectic & (distance >= 4.0) & (distance < 8.0),
        eutectic & (distance >= 8.0),
    )
    bins: dict[str, dict[str, Any]] = {}
    for name, support in zip(BIN_NAMES, masks):
        fn_mask = support & fn
        destinations = np.bincount(
            prediction[fn_mask].astype(np.int64), minlength=NUM_CLASSES
        )
        support_count = int(support.sum())
        fn_count = int(fn_mask.sum())
        bins[name] = {
            "gt_eutectic_pixels": support_count,
            "eutectic_false_negative_pixels": fn_count,
            "false_negative_rate": fn_count / support_count if support_count else None,
            "fn_destination_counts_by_class_id": destinations.tolist(),
        }
    return {
        "gt_eutectic_pixels": int(eutectic.sum()),
        "eutectic_false_negative_pixels": int(fn.sum()),
        "bins": bins,
    }


def _sum_bin_counts(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    support = np.zeros(len(BIN_NAMES), dtype=np.int64)
    misses = np.zeros(len(BIN_NAMES), dtype=np.int64)
    for row in rows:
        for index, name in enumerate(BIN_NAMES):
            support[index] += row["counts"]["bins"][name]["gt_eutectic_pixels"]
            misses[index] += row["counts"]["bins"][name]["eutectic_false_negative_pixels"]
    return support, misses


def _derived_rates(support: np.ndarray, misses: np.ndarray) -> dict[str, float | None]:
    near_support = int(support[0] + support[1])
    near_misses = int(misses[0] + misses[1])
    interior_support = int(support[3])
    interior_misses = int(misses[3])
    near_rate = near_misses / near_support if near_support else None
    interior_rate = interior_misses / interior_support if interior_support else None
    return {
        "near_0_to_lt4px_fn_rate": near_rate,
        "interior_ge8px_fn_rate": interior_rate,
        "near_minus_interior_rate_difference": (
            near_rate - interior_rate
            if near_rate is not None and interior_rate is not None else None
        ),
        "near_to_interior_rate_ratio": (
            near_rate / interior_rate
            if near_rate is not None and interior_rate is not None and interior_rate > 0
            else None
        ),
    }


def _group_bootstrap(rows: list[dict[str, Any]], draws: int) -> dict[str, Any]:
    by_fold_group: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_fold_group[row["fold"]][row["specimen_group"]].append(row)
    folds = sorted(by_fold_group)
    if len(folds) < 2:
        raise ValueError("group bootstrap requires at least two folds")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences: list[float] = []
    ratios: list[float] = []
    for _ in range(draws):
        selected_rows: list[dict[str, Any]] = []
        for fold in folds:
            groups = sorted(by_fold_group[fold])
            sampled = rng.integers(0, len(groups), size=len(groups))
            for index in sampled:
                selected_rows.extend(by_fold_group[fold][groups[int(index)]])
        support, misses = _sum_bin_counts(selected_rows)
        rates = _derived_rates(support, misses)
        delta = rates["near_minus_interior_rate_difference"]
        ratio = rates["near_to_interior_rate_ratio"]
        if delta is not None and np.isfinite(delta):
            differences.append(float(delta))
        if ratio is not None and np.isfinite(ratio):
            ratios.append(float(ratio))

    def interval(values: list[float]) -> list[float] | None:
        if len(values) < int(0.95 * draws):
            return None
        return np.percentile(values, [2.5, 97.5]).astype(float).tolist()

    return {
        "unit": "specimen group resampled within each included fold; pooled pixel rates recomputed per draw",
        "draws": draws,
        "seed": BOOTSTRAP_SEED,
        "near_minus_interior_rate_difference_95ci": interval(differences),
        "near_to_interior_rate_ratio_95ci": interval(ratios),
        "finite_difference_draws": len(differences),
        "finite_ratio_draws": len(ratios),
    }


def analyze(cv_root: Path, oof_root: Path, folds: tuple[int, ...], draws: int) -> dict[str, Any]:
    if folds not in ((1, 2), (3, 4)):
        raise ValueError("use folds (1,2) for exploratory pilot or (3,4) for held-out diagnostic")
    rows: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    group_fold: dict[str, int] = {}
    fold_counts: dict[str, dict[str, int]] = {}
    prediction_input_type_by_fold: dict[str, str] = {}

    for fold in folds:
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        fold_oof = oof_root / f"fold{fold}_oof"
        probability_path = fold_oof / "valid_probabilities.npz"
        prediction_path = fold_oof / "valid_predictions.npz"
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"fold {fold}: missing train OOF masks")
        if probability_path.is_file() and prediction_path.is_file():
            raise ValueError(f"fold {fold}: ambiguous inputs; provide probabilities or argmax maps, not both")
        if probability_path.is_file():
            archive_path = probability_path
            input_type = "OOF probabilities; prediction computed by argmax"
        elif prediction_path.is_file():
            archive_path = prediction_path
            input_type = "uint8 argmax predictions deterministically exported from OOF probabilities"
        else:
            raise FileNotFoundError(f"fold {fold}: missing train OOF probabilities or predictions")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"fold {fold}: missing train OOF masks")
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(masks):
                raise ValueError(f"fold {fold}: OOF prediction keys do not match train masks")
            for image_id in sorted(masks):
                if image_id in seen_images:
                    raise ValueError(f"duplicate OOF image across folds: {image_id}")
                seen_images.add(image_id)
                match = GROUP_RE.match(image_id)
                if not match:
                    raise ValueError(f"cannot derive specimen group from {image_id}")
                group = match.group(1)
                if group in group_fold and group_fold[group] != fold:
                    raise ValueError(f"specimen group {group} crosses folds")
                group_fold[group] = fold
                if "_step4_" not in image_id.lower():
                    continue

                target = mask_to_ids(masks[image_id])
                values = np.asarray(archive[image_id])
                if values.shape == (NUM_CLASSES, *target.shape):
                    probabilities = values.astype(np.float32, copy=False)
                    if not np.isfinite(probabilities).all():
                        raise ValueError(f"non-finite probabilities for {image_id}")
                    if np.max(np.abs(probabilities.sum(axis=0) - 1.0)) > 0.03:
                        raise ValueError(f"unnormalized probabilities for {image_id}")
                    prediction = probabilities.argmax(axis=0).astype(np.uint8)
                elif values.shape == target.shape and np.issubdtype(values.dtype, np.integer):
                    if values.size and (int(values.min()) < 0 or int(values.max()) >= NUM_CLASSES):
                        raise ValueError(f"invalid class id in argmax map for {image_id}")
                    prediction = values.astype(np.uint8, copy=False)
                else:
                    raise ValueError(f"bad probability/prediction shape for {image_id}: {values.shape}")
                counts = eutectic_distance_bin_counts(
                    target, prediction
                )
                rows.append({
                    "image_id": image_id,
                    "specimen_group": group,
                    "fold": fold,
                    "counts": counts,
                })
        fold_counts[str(fold)] = {
            "all_oof_images": len(masks),
            "step4_images": sum("_step4_" in name.lower() for name in masks),
            "step4_groups": len({row["specimen_group"] for row in rows if row["fold"] == fold}),
        }
        prediction_input_type_by_fold[str(fold)] = input_type

    if not rows:
        raise ValueError("no Step4 images found in selected OOF folds")
    support, misses = _sum_bin_counts(rows)
    bins: dict[str, Any] = {}
    for index, name in enumerate(BIN_NAMES):
        bins[name] = {
            "gt_eutectic_pixels": int(support[index]),
            "eutectic_false_negative_pixels": int(misses[index]),
            "false_negative_rate": float(misses[index] / support[index]) if support[index] else None,
            "share_of_gt_eutectic_pixels": float(support[index] / support.sum()) if support.sum() else None,
            "share_of_eutectic_fn_pixels": float(misses[index] / misses.sum()) if misses.sum() else None,
        }

    per_fold: dict[str, Any] = {}
    for fold in folds:
        fold_rows = [row for row in rows if row["fold"] == fold]
        fold_support, fold_misses = _sum_bin_counts(fold_rows)
        per_fold[str(fold)] = {
            "step4_images": len(fold_rows),
            "step4_groups": len({row["specimen_group"] for row in fold_rows}),
            "bins": {
                name: {
                    "gt_eutectic_pixels": int(fold_support[i]),
                    "false_negative_pixels": int(fold_misses[i]),
                    "false_negative_rate": float(fold_misses[i] / fold_support[i]) if fold_support[i] else None,
                }
                for i, name in enumerate(BIN_NAMES)
            },
            **_derived_rates(fold_support, fold_misses),
        }

    selected = tuple(sorted(folds))
    return {
        "model": 386,
        "scope": (
            "exploratory train-only grouped OOF folds1-2; not confirmatory and no model GO"
            if selected == (1, 2)
            else "folds3-4 train-only grouped OOF diagnostic; selection-conditional and no automatic model GO"
        ),
        "metric": "Step4 GT Eutectic false-negative rate within each distance-to-any-GT-phase-boundary band",
        "boundary_definition": "both pixels at every 4-neighbor GT class transition; exact L2 distance on native mask grid",
        "distance_bins_native_pixels": ["[0,2)", "[2,4)", "[4,8)", "[8,infinity)"],
        "official_valid_or_test_read": False,
        "folds": list(selected),
        "prediction_input_type_by_fold": prediction_input_type_by_fold,
        "oof_images_total": len(seen_images),
        "specimen_groups_total": len(group_fold),
        "step4_images": len(rows),
        "step4_groups": len({row["specimen_group"] for row in rows}),
        "gt_eutectic_pixels": int(support.sum()),
        "eutectic_false_negative_pixels": int(misses.sum()),
        "distance_bins": bins,
        "near_vs_interior": _derived_rates(support, misses),
        "group_bootstrap": _group_bootstrap(rows, draws),
        "per_fold": per_fold,
        "fold_image_counts": fold_counts,
        "interpretation_guard": (
            "Distance-bin rates correct the GT-support denominator but are still descriptive. "
            "Folds1-2 are exploratory because prior candidate decisions used them. "
            "No training GO follows from this script; any confirmatory-style report must use untouched folds3-4 "
            "and preserve the existing Model383 preregistered gate."
        ),
        "per_image": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs=2, choices=(1, 2, 3, 4), default=(1, 2))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        raise SystemExit("--bootstrap-draws must be >=100")
    result = analyze(args.cv_root, args.oof_root, tuple(sorted(args.folds)), args.bootstrap_draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
