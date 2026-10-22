#!/usr/bin/env python3
"""Train-OOF-only class and boundary error audit for Eutectic Si.

No official validation data is read. The archive is expected to contain one
held-out prediction per train image, with team class order [Al, Al3Ni,
Primary Si, Eutectic Si].
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion


PALETTE = np.asarray(
    [[153, 127, 76], [25, 76, 153], [76, 178, 76], [204, 204, 204]],
    dtype=np.uint8,
)
CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")
STRUCTURE_8 = np.ones((3, 3), dtype=bool)


def load_target(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    target = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        target[np.all(rgb == color, axis=-1)] = class_id
    if np.any(target == 255):
        raise ValueError(f"Unknown mask color: {path}")
    return target


def group_for(image_id: str) -> str:
    match = re.match(r"(A\d+)", image_id)
    if not match:
        raise ValueError(f"Cannot extract specimen group from {image_id}")
    return match.group(1)


def group_bootstrap_ci(values: np.ndarray, groups: list[str], seed: int = 332,
                       repetitions: int = 10000) -> list[float]:
    unique = np.asarray(sorted(set(groups)))
    rows = {group: np.flatnonzero(np.asarray(groups) == group) for group in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for i in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([rows[group] for group in sampled])
        draws[i] = values[indices].mean()
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/workspace/komap_model1/data"))
    parser.add_argument(
        "--oof", type=Path,
        default=Path("/workspace/komap_model1/outputs/model322_team_b4_group_oof/train_oof_probs.npz"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("/workspace/komap_model1/outputs/model332_eutectic_error_geometry/audit.json"),
    )
    parser.add_argument("--expected-images", type=int, default=70)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {args.out}")

    masks = args.data / "train" / "masks"
    per_image_ious: list[list[float]] = []
    per_image_mious: list[float] = []
    image_ids: list[str] = []
    fn_confusion = np.zeros(4, dtype=np.int64)
    fp_confusion = np.zeros(4, dtype=np.int64)
    counts = {key: 0 for key in (
        "gt_eutectic_pixels", "gt_eutectic_pixels_in_band", "eutectic_fn",
        "eutectic_fn_in_band", "eutectic_fp", "eutectic_fp_in_band",
        "near_intersection", "near_union", "far_intersection", "far_union",
    )}

    with np.load(args.oof) as archive:
        image_ids = sorted(archive.files)
        if len(image_ids) != args.expected_images:
            raise ValueError(f"Expected {args.expected_images} images, found {len(image_ids)}")
        for image_id in image_ids:
            probability = np.asarray(archive[image_id])
            if probability.ndim != 3 or probability.shape[0] != 4:
                raise ValueError(f"Expected (4,H,W) probabilities for {image_id}, got {probability.shape}")
            prediction = probability.argmax(axis=0)
            mask_stem = image_id.replace("_image", "")
            target = load_target(masks / f"{mask_stem}_mask.png")
            if target.shape != prediction.shape:
                raise ValueError(f"Shape mismatch for {image_id}: {target.shape} != {prediction.shape}")

            class_ious = []
            for class_id in range(4):
                gt = target == class_id
                pred = prediction == class_id
                union = int(np.count_nonzero(gt | pred))
                intersection = int(np.count_nonzero(gt & pred))
                class_ious.append(intersection / union if union else 0.0)
            per_image_ious.append(class_ious)
            per_image_mious.append(float(np.mean(class_ious)))

            gt_eutectic = target == 3
            pred_eutectic = prediction == 3
            edge = gt_eutectic & ~binary_erosion(
                gt_eutectic, structure=STRUCTURE_8, border_value=0
            )
            band = binary_dilation(edge, structure=STRUCTURE_8, iterations=2)
            false_negative = gt_eutectic & ~pred_eutectic
            false_positive = ~gt_eutectic & pred_eutectic

            counts["gt_eutectic_pixels"] += int(gt_eutectic.sum())
            counts["gt_eutectic_pixels_in_band"] += int((gt_eutectic & band).sum())
            counts["eutectic_fn"] += int(false_negative.sum())
            counts["eutectic_fn_in_band"] += int((false_negative & band).sum())
            counts["eutectic_fp"] += int(false_positive.sum())
            counts["eutectic_fp_in_band"] += int((false_positive & band).sum())
            fn_confusion += np.bincount(prediction[false_negative], minlength=4)
            fp_confusion += np.bincount(target[false_positive], minlength=4)

            for region, prefix in ((band, "near"), (~band, "far")):
                gt_region = gt_eutectic & region
                pred_region = pred_eutectic & region
                counts[f"{prefix}_intersection"] += int((gt_region & pred_region).sum())
                counts[f"{prefix}_union"] += int((gt_region | pred_region).sum())

    class_ious = np.asarray(per_image_ious, dtype=np.float64)
    groups = [group_for(name) for name in image_ids]
    group_count = len(set(groups))
    def pct(numerator: int, denominator: int) -> float | None:
        return 100.0 * numerator / denominator if denominator else None

    output = {
        "audit": "Model322 grouped train OOF only; no valid/test images or masks read",
        "image_count": len(image_ids),
        "specimen_group_count": group_count,
        "class_order": list(CLASS_NAMES),
        "empty_union_policy": "IoU=0",
        "image_mean_miou_pct": 100.0 * float(np.mean(per_image_mious)),
        "per_image_class_iou_mean_pct": {
            CLASS_NAMES[i]: 100.0 * float(class_ious[:, i].mean()) for i in range(4)
        },
        "eutectic_class_iou_group_bootstrap_95ci_pct": group_bootstrap_ci(
            class_ious[:, 3], groups
        ),
        "boundary_band": "GT Eutectic inner boundary, dilated by 2 iterations with 8-connectivity",
        "gt_eutectic_pixels_in_band_pct": pct(
            counts["gt_eutectic_pixels_in_band"], counts["gt_eutectic_pixels"]
        ),
        "eutectic_false_negative_pixels_in_band_pct": pct(
            counts["eutectic_fn_in_band"], counts["eutectic_fn"]
        ),
        "eutectic_false_positive_pixels_in_band_pct": pct(
            counts["eutectic_fp_in_band"], counts["eutectic_fp"]
        ),
        "eutectic_pooled_iou_near_boundary_pct": pct(
            counts["near_intersection"], counts["near_union"]
        ),
        "eutectic_pooled_iou_far_from_boundary_pct": pct(
            counts["far_intersection"], counts["far_union"]
        ),
        "eutectic_false_negative_prediction_counts": {
            CLASS_NAMES[i]: int(fn_confusion[i]) for i in range(4)
        },
        "eutectic_false_positive_gt_source_counts": {
            CLASS_NAMES[i]: int(fp_confusion[i]) for i in range(4)
        },
        "per_image": [
            {"image": image_ids[i], "specimen_group": groups[i],
             "image_miou_pct": 100.0 * per_image_mious[i],
             "class_iou_pct": {
                 CLASS_NAMES[c]: 100.0 * per_image_ious[i][c] for c in range(4)
             }}
            for i in range(len(image_ids))
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in output.items() if key != "per_image"}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
