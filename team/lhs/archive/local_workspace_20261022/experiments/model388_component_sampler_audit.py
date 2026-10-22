#!/usr/bin/env python3
"""CPU-only sampler anchor-dose and train-OOF component-headroom audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


PALETTE = np.asarray(
    ((153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)),
    dtype=np.uint8,
)
MINORITY_CLASSES = (1, 3)
CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
AREA_BINS = (
    (1, 64, "1_to_lt64"),
    (64, 256, "64_to_lt256"),
    (256, 1024, "256_to_lt1024"),
    (1024, 4096, "1024_to_lt4096"),
    (4096, None, "4096_or_more"),
)
ELIGIBLE_MIN_AREA = 64
EPOCHS = 70
SAMPLES_PER_IMAGE = 12
POSITIVE_PROBABILITY = 0.5
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 388
GROUP_RE = re.compile(r"^(A\d+)_")
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_to_ids(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        ids[np.all(rgb == color, axis=2)] = class_id
    if np.any(ids == 255):
        raise ValueError(f"unknown RGB labels in {path}: {np.unique(rgb[ids == 255], axis=0)[:8].tolist()}")
    return ids


def component_areas(mask: np.ndarray, class_id: int) -> np.ndarray:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask == class_id).astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
    )
    if count <= 1:
        return np.zeros(0, dtype=np.int64)
    return stats[1:, cv2.CC_STAT_AREA].astype(np.int64)


def area_bin(area: int) -> str:
    for low, high, name in AREA_BINS:
        if area >= low and (high is None or area < high):
            return name
    raise ValueError(f"invalid component area: {area}")


def load_fold_masks(cv_root: Path) -> tuple[dict[int, dict[str, Path]], dict[str, int]]:
    by_fold: dict[int, dict[str, Path]] = {}
    group_fold: dict[str, int] = {}
    seen: set[str] = set()
    for fold in range(1, 5):
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"missing fixed-fold train masks: {mask_dir}")
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        if not masks:
            raise FileNotFoundError(f"no train masks in {mask_dir}")
        by_fold[fold] = masks
        for image_id in masks:
            if image_id in seen:
                raise ValueError(f"duplicate train image in fixed OOF folds: {image_id}")
            seen.add(image_id)
            match = GROUP_RE.match(image_id)
            if not match:
                raise ValueError(f"cannot parse specimen group: {image_id}")
            group = match.group(1)
            if group in group_fold and group_fold[group] != fold:
                raise ValueError(f"specimen group crosses folds: {group}")
            group_fold[group] = fold
    return by_fold, group_fold


def exposure_audit(cv_root: Path, folds: tuple[int, int]) -> dict[str, Any]:
    if folds not in ((1, 2), (3, 4)):
        raise ValueError("run folds (1,2) exploratory or (3,4) primary")
    by_fold, group_fold = load_fold_masks(cv_root)
    per_outer_fold: dict[str, Any] = {}
    all_hashes: dict[str, str] = {}

    for heldout_fold in folds:
        components_by_cell: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        train_ids: set[str] = set()
        class_anchor_totals = {"current": defaultdict(float), "candidate": defaultdict(float)}
        for source_fold in range(1, 5):
            if source_fold == heldout_fold:
                continue
            for image_id, mask_path in sorted(by_fold[source_fold].items()):
                train_ids.add(image_id)
                all_hashes[str(mask_path)] = sha256(mask_path)
                mask = mask_to_ids(mask_path)
                areas_by_class = {c: component_areas(mask, c) for c in MINORITY_CLASSES}
                present = [c for c in MINORITY_CLASSES if areas_by_class[c].size]
                if not present:
                    continue
                positive_anchors_per_class = (
                    EPOCHS * SAMPLES_PER_IMAGE * POSITIVE_PROBABILITY / len(present)
                )
                group = GROUP_RE.match(image_id).group(1)
                for class_id in present:
                    areas = areas_by_class[class_id]
                    total_pixels = int(areas.sum())
                    eligible = areas >= ELIGIBLE_MIN_AREA
                    eligible_count = int(eligible.sum())
                    class_anchor_totals["current"][class_id] += positive_anchors_per_class
                    class_anchor_totals["candidate"][class_id] += positive_anchors_per_class
                    for area in areas.tolist():
                        is_eligible = area >= ELIGIBLE_MIN_AREA
                        current = positive_anchors_per_class * area / total_pixels
                        if eligible_count:
                            candidate = positive_anchors_per_class / eligible_count if is_eligible else 0.0
                        else:
                            candidate = current  # fallback to existing pixel sampler
                        components_by_cell[(class_id, area_bin(int(area)))].append({
                            "image_id": image_id,
                            "specimen_group": group,
                            "area": int(area),
                            "current_expected_anchors": float(current),
                            "candidate_expected_anchors": float(candidate),
                        })

        class_results: dict[str, Any] = {}
        for class_id in MINORITY_CLASSES:
            class_bins: dict[str, Any] = {}
            for _, _, bin_name in AREA_BINS:
                records = components_by_cell[(class_id, bin_name)]
                current = sum(row["current_expected_anchors"] for row in records)
                candidate = sum(row["candidate_expected_anchors"] for row in records)
                class_bins[bin_name] = {
                    "components": len(records),
                    "specimen_groups": len({row["specimen_group"] for row in records}),
                    "current_expected_anchor_count": float(current),
                    "candidate_expected_anchor_count": float(candidate),
                    "candidate_to_current_dose_ratio": float(candidate / current) if current else None,
                    "eligible_for_candidate": bin_name != "1_to_lt64",
                }
            current_total = class_anchor_totals["current"][class_id]
            candidate_total = class_anchor_totals["candidate"][class_id]
            class_results[CLASS_NAMES[class_id]] = {
                "positive_anchors_current": float(current_total),
                "positive_anchors_candidate": float(candidate_total),
                "class_total_anchor_delta": float(candidate_total - current_total),
                "component_size_bins": class_bins,
            }
        per_outer_fold[str(heldout_fold)] = {
            "training_images": len(train_ids),
            "training_groups": len({GROUP_RE.match(name).group(1) for name in train_ids}),
            "heldout_fold_excluded_from_exposure_calculation": heldout_fold,
            "classes": class_results,
        }

    return {
        "model": 388,
        "mode": "E1 expected positive-anchor dose manipulation check",
        "scope": "primary train-only folds3-4" if folds == (3, 4) else "exploratory train-only folds1-2",
        "folds": list(folds),
        "train_image_count": sum(len(m) for m in by_fold.values()),
        "train_group_count": len(group_fold),
        "official_valid_or_test_read": False,
        "oof_predictions_read": False,
        "sampler_parameters": {
            "epochs": EPOCHS,
            "samples_per_image_per_epoch": SAMPLES_PER_IMAGE,
            "positive_probability": POSITIVE_PROBABILITY,
            "minority_classes": [CLASS_NAMES[c] for c in MINORITY_CLASSES],
            "native_crop_size": 256,
            "jitter_each_axis_px": 64,
            "current_coordinate_cap_per_image_class": 4000,
            "eligible_component_min_area": ELIGIBLE_MIN_AREA,
        },
        "candidate_definition": (
            "preserve positive-crop probability and equal class selection; when class has any 8-connected "
            "component >=64 px, choose uniformly among those components then uniformly among its pixels; "
            "otherwise fallback to current pixel-uniform sampler"
        ),
        "exposure_definition": (
            "expected count of selected positive anchor pixels inside a GT component over 70x12 samples/image; "
            "the anchor is guaranteed inside the native 256px crop under +/-64px jitter; not a >=50%-area crop count"
        ),
        "per_outer_fold": per_outer_fold,
        "source_sha256": all_hashes,
        "interpretation_guard": (
            "Training-fold GT only. This is a dose/manipulation check; it does not link an excluded held-out "
            "component's exposure to its OOF error and does not estimate a sampler causal effect."
        ),
    }


def official_image_miou(target: np.ndarray, prediction: np.ndarray) -> float:
    confusion = np.bincount(
        target.astype(np.int64).ravel() * 4 + prediction.astype(np.int64).ravel(), minlength=16
    ).reshape(4, 4)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.zeros(4), where=union > 0)
    return float(iou.mean())


def component_bin_oracle_gains(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    baseline = official_image_miou(target, prediction)
    result: dict[str, Any] = {}
    for class_id in MINORITY_CLASSES:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (target == class_id).astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
        )
        areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.zeros(0, dtype=np.int64)
        for _, _, bin_name in AREA_BINS:
            selected_labels = [idx + 1 for idx, area in enumerate(areas) if area_bin(int(area)) == bin_name]
            selected = np.isin(labels, selected_labels) if selected_labels else np.zeros(target.shape, dtype=bool)
            oracle = prediction.copy()
            oracle[selected] = target[selected]
            result[f"{class_id}:{bin_name}"] = {
                "component_count": len(selected_labels),
                "fn_pixels": int(np.count_nonzero(selected & (prediction != target))),
                "oracle_miou_gain": float(official_image_miou(target, oracle) - baseline),
            }
    return {"baseline_miou": baseline, "cells": result}


def _verify_model388_lock(lock_path: Path) -> dict[str, Any]:
    receipt = json.loads(lock_path.read_text(encoding="utf-8"))
    for relative, expected in receipt["artifacts_sha256"].items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Model388 preregistration hash mismatch: {relative}")
    return receipt


def _group_bootstrap_cell(rows: list[dict[str, Any]], folds: tuple[int, int], cell: str,
                          draws: int) -> dict[str, Any]:
    by_fold_group: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_fold_group[row["fold"]][row["specimen_group"]].append(row)
    rng = np.random.default_rng(BOOTSTRAP_SEED + sum(ord(ch) for ch in cell))
    values: list[float] = []
    for _ in range(draws):
        sample: list[dict[str, Any]] = []
        for fold in folds:
            groups = sorted(by_fold_group[fold])
            if not groups:
                raise ValueError(f"no groups in fold {fold}")
            for index in rng.integers(0, len(groups), size=len(groups)):
                sample.extend(by_fold_group[fold][groups[int(index)]])
        values.append(float(np.mean([row["cells"][cell]["oracle_miou_gain"] for row in sample])))
    return {
        "ci_95_two_sided": np.percentile(values, [2.5, 97.5]).astype(float).tolist(),
        "ci_99_5_two_sided": np.percentile(values, [0.25, 99.75]).astype(float).tolist(),
        "draws": draws,
        "seed": BOOTSTRAP_SEED + sum(ord(ch) for ch in cell),
        "unit": "specimen group resampled within each held-out fold; all images retained",
    }


def headroom_audit(cv_root: Path, model382_root: Path, folds: tuple[int, int],
                   model382_lock: Path, model388_lock: Path, draws: int) -> dict[str, Any]:
    import model382_oof_ensemble_compare as model382

    if folds not in ((1, 2), (3, 4)):
        raise ValueError("run folds (1,2) exploratory or (3,4) primary")
    fold_masks, group_fold = load_fold_masks(cv_root)
    lock382 = json.loads(model382_lock.read_text(encoding="utf-8"))
    lock388 = _verify_model388_lock(model388_lock)
    rows_by_seed: dict[str, list[dict[str, Any]]] = {"1": [], "2": []}
    sources: dict[str, str] = {}
    for seed in (1, 2):
        rows = rows_by_seed[str(seed)]
        seen: set[str] = set()
        for fold in folds:
            masks = fold_masks[fold]
            oof_dir = model382_root / f"seed{seed}" / f"fold{fold}_oof"
            archive_path = model382._verify_model382_manifest(
                oof_dir, seed, fold, float(lock382["locked_at_unix"])
            )
            manifest_path = oof_dir / "model382_manifest.json"
            if fold in (3, 4):
                lock_time = float(lock388["locked_at_unix"])
                if archive_path.stat().st_mtime <= lock_time or manifest_path.stat().st_mtime <= lock_time:
                    raise ValueError(f"primary OOF artifact predates Model388 lock: {oof_dir}")
            with np.load(archive_path, allow_pickle=False) as archive:
                if set(archive.files) != set(masks):
                    raise ValueError(f"OOF image IDs mismatch: {archive_path}")
                for image_id, mask_path in sorted(masks.items()):
                    if image_id in seen:
                        raise ValueError(f"duplicate OOF ID across folds: {image_id}")
                    seen.add(image_id)
                    group = GROUP_RE.match(image_id).group(1)
                    target = mask_to_ids(mask_path)
                    probabilities = np.asarray(archive[image_id], dtype=np.float32)
                    if probabilities.shape != (4, *target.shape) or not np.isfinite(probabilities).all():
                        raise ValueError(f"invalid OOF probabilities for {image_id}")
                    if np.max(np.abs(probabilities.sum(axis=0) - 1.0)) > 0.03:
                        raise ValueError(f"unnormalized OOF probabilities for {image_id}")
                    prediction = probabilities.argmax(axis=0).astype(np.uint8)
                    diagnostic = component_bin_oracle_gains(target, prediction)
                    rows.append({
                        "image_id": image_id,
                        "specimen_group": group,
                        "fold": fold,
                        **diagnostic,
                    })
                    sources[str(mask_path)] = sha256(mask_path)
            sources[str(archive_path)] = sha256(archive_path)
            sources[str(manifest_path)] = sha256(manifest_path)
            checkpoint = oof_dir.parent / f"fold{fold}" / "last.pt"
            sources[str(checkpoint)] = sha256(checkpoint)
    results: dict[str, Any] = {}
    for seed, rows in rows_by_seed.items():
        seed_summary: dict[str, Any] = {
            "images": len(rows),
            "groups": len({row["specimen_group"] for row in rows}),
            "baseline_imagewise_miou": float(np.mean([row["baseline_miou"] for row in rows])),
            "cells": {},
        }
        for class_id in MINORITY_CLASSES:
            for _, _, bin_name in AREA_BINS:
                cell = f"{class_id}:{bin_name}"
                cell_rows = [row["cells"][cell] for row in rows]
                gain = float(np.mean([row["oracle_miou_gain"] for row in cell_rows]))
                by_fold: dict[str, Any] = {}
                for fold in folds:
                    fold_rows = [row for row in rows if row["fold"] == fold]
                    by_fold[str(fold)] = {
                        "images": len(fold_rows),
                        "groups": len({row["specimen_group"] for row in fold_rows}),
                        "mean_imagewise_oracle_gain": float(np.mean([
                            row["cells"][cell]["oracle_miou_gain"] for row in fold_rows
                        ])),
                    }
                group_gain: dict[str, float] = defaultdict(float)
                for row in rows:
                    group_gain[row["specimen_group"]] += row["cells"][cell]["oracle_miou_gain"]
                total_group_gain = sum(group_gain.values())
                seed_summary["cells"][cell] = {
                    "class": CLASS_NAMES[class_id],
                    "area_bin": bin_name,
                    "component_count": sum(row["cells"][cell]["component_count"] for row in cell_rows),
                    "groups_with_components": len({
                        row["specimen_group"] for row in rows if row["cells"][cell]["component_count"]
                    }),
                    "fn_pixels": sum(row["cells"][cell]["fn_pixels"] for row in cell_rows),
                    "mean_imagewise_oracle_miou_gain": gain,
                    "max_group_share_of_total_gain": (
                        max(group_gain.values()) / total_group_gain if total_group_gain > 0 else None
                    ),
                    "per_fold": by_fold,
                    "group_bootstrap": _group_bootstrap_cell(rows, folds, cell, draws),
                }
        results[seed] = seed_summary
    return {
        "model": 388,
        "mode": "E2 held-out train-OOF component-size oracle headroom; not a sampler effect estimate",
        "scope": "primary train-only folds3-4" if folds == (3, 4) else "exploratory train-only folds1-2",
        "folds": list(folds),
        "seed_results": results,
        "metric": "official image-wise mean of four class IoUs; empty union=0",
        "official_valid_or_test_read": False,
        "training_started": False,
        "source_sha256": sources,
        "model382_lock_sha256": sha256(model382_lock),
        "model388_lock_sha256": sha256(model388_lock),
        "interpretation_guard": (
            "Oracle correction is an unattainable upper bound. OOF components were excluded from their model's "
            "training; no per-heldout-component crop exposure or causal association is estimated."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("exposure", "headroom"), required=True)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs=2, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--model382-root", type=Path)
    parser.add_argument("--model382-lock", type=Path,
                        default=Path("experiments/results/model382_e5_preregistration_lock.json"))
    parser.add_argument("--model388-lock", type=Path,
                        default=Path("experiments/results/model388_preregistration_lock.json"))
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folds = tuple(sorted(args.folds))
    if args.mode == "exposure":
        result = exposure_audit(args.cv_root, folds)
    else:
        if args.model382_root is None:
            parser.error("--model382-root is required for headroom mode")
        if args.bootstrap_draws < 100:
            parser.error("--bootstrap-draws must be >=100")
        result = headroom_audit(args.cv_root, args.model382_root, folds,
                                args.model382_lock, args.model388_lock, args.bootstrap_draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("source_sha256", "seed_results")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
