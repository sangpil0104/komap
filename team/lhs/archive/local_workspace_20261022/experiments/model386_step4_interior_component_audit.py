#!/usr/bin/env python3
"""Exploratory component audit for Model386 Step4 interior Eutectic errors."""

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

from model386_step4_eutectic_distance_rate_pilot import (
    EUTECTIC_SI,
    NUM_CLASSES,
    mask_to_ids,
    phase_boundary,
)


GROUP_RE = re.compile(r"^(A\d+)")
INTERIOR_DISTANCE_PX = 8.0


def components(target: np.ndarray, prediction: np.ndarray) -> list[dict[str, Any]]:
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must be same-size 2-D arrays")
    boundary = phase_boundary(target)
    if not boundary.any():
        raise ValueError("GT mask has no phase boundary")
    distance = cv2.distanceTransform(
        (~boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    interior = (target == EUTECTIC_SI) & (distance >= INTERIOR_DISTANCE_PX)
    count, labels = cv2.connectedComponents(interior.astype(np.uint8), connectivity=8)
    rows: list[dict[str, Any]] = []
    for component_id in range(1, count):
        pixels = labels == component_id
        ys, xs = np.where(pixels)
        miss = pixels & (prediction != EUTECTIC_SI)
        destination_counts = np.bincount(
            prediction[miss].astype(np.int64), minlength=NUM_CLASSES
        )
        support = int(pixels.sum())
        fn = int(miss.sum())
        rows.append({
            "component_id": component_id,
            "gt_eutectic_interior_pixels": support,
            "false_negative_pixels": fn,
            "false_negative_rate": fn / support,
            "prediction_destination_counts_by_class_id": destination_counts.tolist(),
            "bbox_xyxy_inclusive": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        })
    return rows


def analyze(cv_root: Path, oof_root: Path, folds: tuple[int, ...] = (1, 2)) -> dict[str, Any]:
    if tuple(sorted(folds)) != (1, 2):
        raise ValueError("this post-hoc component audit is fixed to exploratory folds1-2")
    image_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    group_fold: dict[str, int] = {}
    seen_images: set[str] = set()

    for fold in (1, 2):
        mask_dir = cv_root / f"oof_fold{fold}" / "masks"
        prediction_path = oof_root / f"fold{fold}_oof" / "valid_predictions.npz"
        if not mask_dir.is_dir() or not prediction_path.is_file():
            raise FileNotFoundError(f"fold{fold}: missing train masks or argmax OOF predictions")
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        with np.load(prediction_path, allow_pickle=False) as archive:
            if set(archive.files) != set(masks):
                raise ValueError(f"fold{fold}: prediction keys do not match train masks")
            for image_id in sorted(masks):
                if image_id in seen_images:
                    raise ValueError(f"duplicate image across folds: {image_id}")
                seen_images.add(image_id)
                match = GROUP_RE.match(image_id)
                if not match:
                    raise ValueError(f"cannot parse specimen group from {image_id}")
                group = match.group(1)
                if group in group_fold and group_fold[group] != fold:
                    raise ValueError(f"specimen group {group} crosses folds")
                group_fold[group] = fold
                if "_step4_" not in image_id.lower():
                    continue
                target = mask_to_ids(masks[image_id])
                prediction = np.asarray(archive[image_id])
                if prediction.shape != target.shape or not np.issubdtype(prediction.dtype, np.integer):
                    raise ValueError(f"invalid argmax map for {image_id}: {prediction.shape}")
                if prediction.size and (int(prediction.min()) < 0 or int(prediction.max()) >= NUM_CLASSES):
                    raise ValueError(f"invalid class id in prediction map: {image_id}")
                prediction = prediction.astype(np.uint8, copy=False)
                found = components(target, prediction)
                support = sum(row["gt_eutectic_interior_pixels"] for row in found)
                fn = sum(row["false_negative_pixels"] for row in found)
                destinations = [0] * NUM_CLASSES
                for row in found:
                    destinations = [
                        a + b for a, b in zip(
                            destinations,
                            row["prediction_destination_counts_by_class_id"],
                        )
                    ]
                image_rows.append({
                    "image_id": image_id,
                    "specimen_group": group,
                    "fold": fold,
                    "component_count": len(found),
                    "gt_eutectic_interior_pixels": support,
                    "false_negative_pixels": fn,
                    "false_negative_rate": fn / support if support else None,
                    "prediction_destination_counts_by_class_id": destinations,
                })
                for row in found:
                    component_rows.append({
                        "image_id": image_id,
                        "specimen_group": group,
                        "fold": fold,
                        **row,
                    })

    if len(seen_images) != 36 or len(group_fold) != 30:
        raise ValueError(f"expected 36 OOF images/30 groups, found {len(seen_images)}/{len(group_fold)}")

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        by_group[row["specimen_group"]].append(row)
    group_rows = []
    for group, rows in by_group.items():
        support = sum(row["gt_eutectic_interior_pixels"] for row in rows)
        fn = sum(row["false_negative_pixels"] for row in rows)
        destinations = [0] * NUM_CLASSES
        for row in rows:
            destinations = [a + b for a, b in zip(destinations, row["prediction_destination_counts_by_class_id"])]
        group_rows.append({
            "specimen_group": group,
            "fold": rows[0]["fold"],
            "images": len(rows),
            "component_count": sum(row["component_count"] for row in rows),
            "gt_eutectic_interior_pixels": support,
            "false_negative_pixels": fn,
            "false_negative_rate": fn / support if support else None,
            "prediction_destination_counts_by_class_id": destinations,
        })
    group_rows.sort(key=lambda row: (-row["gt_eutectic_interior_pixels"], row["specimen_group"]))

    support_total = sum(row["gt_eutectic_interior_pixels"] for row in component_rows)
    fn_total = sum(row["false_negative_pixels"] for row in component_rows)
    components_90 = [row for row in component_rows if row["false_negative_rate"] >= 0.90]
    groups_with_support = [row for row in group_rows if row["gt_eutectic_interior_pixels"] > 0]
    return {
        "model": 386,
        "scope": "post-hoc descriptive audit of exploratory train-OOF folds1-2; no confirmatory claim or model GO",
        "input": "Model374 argmax OOF maps deterministically exported from probability archives; same-fold train GT only",
        "official_valid_or_test_read": False,
        "interior_definition": "GT Eutectic pixels at least 8 native pixels from any 4-neighbor GT phase boundary",
        "component_connectivity": 8,
        "folds": [1, 2],
        "oof_images": len(seen_images),
        "specimen_groups": len(group_fold),
        "step4_images": len(image_rows),
        "step4_groups": len(group_rows),
        "groups_with_interior_support": len(groups_with_support),
        "groups_without_interior_support": len(group_rows) - len(groups_with_support),
        "interior_support_pixels": support_total,
        "interior_false_negative_pixels": fn_total,
        "components_total": len(component_rows),
        "components_with_fn_rate_at_least_90pct": len(components_90),
        "share_of_interior_support_in_components_with_fn_rate_at_least_90pct": (
            sum(row["gt_eutectic_interior_pixels"] for row in components_90) / support_total
            if support_total else None
        ),
        "group_concentration": {
            "top1_support_share": sum(row["gt_eutectic_interior_pixels"] for row in group_rows[:1]) / support_total if support_total else None,
            "top3_support_share": sum(row["gt_eutectic_interior_pixels"] for row in group_rows[:3]) / support_total if support_total else None,
            "top1_fn_share": sum(row["false_negative_pixels"] for row in group_rows[:1]) / fn_total if fn_total else None,
            "top3_fn_share": sum(row["false_negative_pixels"] for row in group_rows[:3]) / fn_total if fn_total else None,
        },
        "groups": group_rows,
        "per_image": image_rows,
        "per_component": sorted(
            component_rows,
            key=lambda row: (-row["gt_eutectic_interior_pixels"], row["specimen_group"], row["component_id"]),
        ),
        "interpretation_guard": "Connected pixels within a component are not independent samples. This post-hoc 2-fold audit cannot justify training or change Model385/383 gates.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.cv_root, args.oof_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("per_image", "per_component")}, indent=2))


if __name__ == "__main__":
    main()
