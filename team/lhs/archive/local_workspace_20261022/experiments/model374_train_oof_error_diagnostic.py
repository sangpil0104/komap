#!/usr/bin/env python3
"""Train-only diagnostics for complete Model374 grouped OOF probabilities.

This is descriptive, not a tuning/search script. It reports strict official
image-wise mIoU, the fraction of errors near GT phase boundaries, mIoU after
excluding a narrow GT-boundary band (diagnostic only), and specimen-level
score concentration. It never reads valid.zip or test.zip.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CLASS_NAMES = ["Al", "Al3Ni", "Primary_Si", "Eutectic_Si"]
PALETTE = np.asarray(
    [(153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)],
    dtype=np.uint8,
)
GROUP_RE = re.compile(r"^(A\d+)")
NUM_CLASSES = 4


def mask_to_ids(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    result = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(PALETTE):
        result[np.all(rgb == color, axis=2)] = class_id
    if np.any(result == 255):
        colors = np.unique(rgb[result == 255], axis=0)
        raise ValueError(f"unknown RGB mask colors in {path}: {colors[:8].tolist()}")
    return result


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    if radius <= 0:
        return boundary
    height, width = mask.shape
    padded = np.pad(boundary, ((radius, radius), (radius, radius)))
    dilated = np.zeros(mask.shape, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            dilated |= padded[dy:dy + height, dx:dx + width]
    return dilated


def image_ious(pred: np.ndarray, target: np.ndarray, keep: np.ndarray | None = None) -> list[float]:
    if keep is None:
        truth = target.reshape(-1).astype(np.int64)
        guess = pred.reshape(-1).astype(np.int64)
    else:
        truth = target[keep].astype(np.int64)
        guess = pred[keep].astype(np.int64)
    confusion = np.bincount(
        truth * NUM_CLASSES + guess, minlength=NUM_CLASSES**2
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    return np.divide(
        intersection, union, out=np.zeros(NUM_CLASSES, dtype=np.float64), where=union > 0
    ).tolist()


def analyze(cv_data: Path, oof_root: Path, radius: int, folds: tuple[int, ...] = (1, 2, 3, 4)) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    fold_counts: dict[str, int] = {}

    for fold in folds:
        split_dir = cv_data / f"oof_fold{fold}"
        mask_dir = split_dir / "masks"
        archive_path = oof_root / f"fold{fold}_oof" / "valid_probabilities.npz"
        if not mask_dir.is_dir() or not archive_path.is_file():
            raise FileNotFoundError(f"required grouped OOF input missing: {mask_dir} or {archive_path}")
        masks = {p.name.removesuffix("_mask.png"): p for p in mask_dir.glob("*_mask.png")}
        with np.load(archive_path, allow_pickle=False) as archive:
            names = set(archive.files)
            if names != set(masks):
                raise ValueError(
                    f"fold {fold} OOF/mask IDs differ: missing predictions={sorted(set(masks)-names)[:5]}, "
                    f"missing masks={sorted(names-set(masks))[:5]}"
                )
            fold_count = 0
            for name in sorted(names):
                if name in seen_images:
                    raise ValueError(f"image occurs in more than one OOF fold: {name}")
                seen_images.add(name)
                group_match = GROUP_RE.match(name)
                if not group_match:
                    raise ValueError(f"cannot derive specimen group from image ID: {name}")
                probabilities = np.asarray(archive[name], dtype=np.float32)
                target = mask_to_ids(masks[name])
                if probabilities.ndim != 3 or probabilities.shape[0] != NUM_CLASSES:
                    raise ValueError(f"invalid probability shape for {name}: {probabilities.shape}")
                if probabilities.shape[1:] != target.shape:
                    raise ValueError(f"probability/mask shape mismatch for {name}: "
                                     f"{probabilities.shape[1:]} vs {target.shape}")
                if not np.isfinite(probabilities).all():
                    raise ValueError(f"non-finite probabilities for {name}")
                sums = probabilities.sum(axis=0)
                if np.max(np.abs(sums - 1.0)) > 0.03:
                    raise ValueError(f"probabilities not normalized for {name}: max error="
                                     f"{float(np.max(np.abs(sums - 1.0)))}")

                pred = probabilities.argmax(axis=0).astype(np.uint8)
                strict_class_ious = image_ious(pred, target)
                band = boundary_band(target, radius)
                ignored_boundary_ious = image_ious(pred, target, keep=~band)
                wrong = pred != target
                n_wrong = int(wrong.sum())
                n_boundary_wrong = int((wrong & band).sum())
                per_class_boundary_error = []
                for class_id in range(NUM_CLASSES):
                    class_error = (pred == class_id) != (target == class_id)
                    denominator = int(class_error.sum())
                    numerator = int((class_error & band).sum())
                    per_class_boundary_error.append(
                        numerator / denominator if denominator else 0.0
                    )
                records.append({
                    "image_id": name,
                    "specimen_group": group_match.group(1),
                    "fold": fold,
                    "strict_image_miou_empty0": float(np.mean(strict_class_ious)),
                    "strict_class_iou_empty0": strict_class_ious,
                    "boundary_ignored_image_miou_empty0_diagnostic_only":
                        float(np.mean(ignored_boundary_ious)),
                    "boundary_error_pixel_fraction":
                        n_boundary_wrong / n_wrong if n_wrong else 0.0,
                    "wrong_pixels": n_wrong,
                    "wrong_pixels_in_gt_boundary_band": n_boundary_wrong,
                    "pixels_in_gt_boundary_band": int(band.sum()),
                    "total_pixels": int(band.size),
                    "boundary_error_fraction_by_class_fp_plus_fn": per_class_boundary_error,
                    "gt_boundary_band_pixel_fraction": float(band.mean()),
                })
                fold_count += 1
            fold_counts[str(fold)] = fold_count

    if not records:
        raise ValueError("no grouped OOF predictions were analyzed")
    scores = np.asarray([r["strict_image_miou_empty0"] for r in records], dtype=np.float64)
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        groups[record["specimen_group"]].append(record["strict_image_miou_empty0"])
    group_rows = [
        {"specimen_group": group, "images": len(values), "mean_image_miou_empty0": float(np.mean(values))}
        for group, values in groups.items()
    ]
    group_rows.sort(key=lambda item: (item["mean_image_miou_empty0"], item["specimen_group"]))
    q_count = max(1, math.ceil(0.25 * len(group_rows)))
    group_deficits = [max(0.0, float(scores.mean()) - row["mean_image_miou_empty0"])
                      for row in group_rows]
    total_deficit = sum(group_deficits)
    bottom_quartile_deficit_share = (
        sum(group_deficits[:q_count]) / total_deficit if total_deficit else 0.0
    )

    total_wrong = sum(r["wrong_pixels"] for r in records)
    total_boundary_wrong = sum(r["wrong_pixels_in_gt_boundary_band"] for r in records)
    total_band_pixels = sum(r["pixels_in_gt_boundary_band"] for r in records)
    total_pixels = sum(r["total_pixels"] for r in records)
    boundary_error_density = total_boundary_wrong / total_band_pixels if total_band_pixels else 0.0
    interior_error_density = ((total_wrong - total_boundary_wrong) / (total_pixels - total_band_pixels)
                              if total_pixels > total_band_pixels else 0.0)
    class_boundary_numerators = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_boundary_denominators = np.zeros(NUM_CLASSES, dtype=np.float64)
    for record in records:
        for class_id, fraction in enumerate(record["boundary_error_fraction_by_class_fp_plus_fn"]):
            # Weight image-level class error shares by their FP+FN counts, which
            # can be recovered from the pixel-confusion asymmetry only with the
            # prediction archive. Preserve the per-image fractions separately;
            # use an unweighted image mean here to avoid claiming pixel weighting.
            class_boundary_numerators[class_id] += fraction
            class_boundary_denominators[class_id] += 1

    return {
        "model": 374,
        "scope": "training images only; grouped out-of-fold predictions; no valid/test data",
        "folds": list(folds),
        "metric": "mean over images of four class IoUs, empty union=0",
        "boundary_radius_pixels": radius,
        "images": len(records),
        "specimen_groups": len(groups),
        "fold_image_counts": fold_counts,
        "image_mean_miou_empty0": float(scores.mean()),
        "mean_boundary_ignored_image_miou_empty0_diagnostic_only": float(np.mean([
            r["boundary_ignored_image_miou_empty0_diagnostic_only"] for r in records
        ])),
        "all_wrong_pixels_in_gt_boundary_band_fraction":
            total_boundary_wrong / total_wrong if total_wrong else 0.0,
        "gt_boundary_band_area_fraction": total_band_pixels / total_pixels,
        "boundary_wrong_pixel_density": boundary_error_density,
        "interior_wrong_pixel_density": interior_error_density,
        "boundary_vs_interior_error_density_ratio": (
            boundary_error_density / interior_error_density if interior_error_density else None
        ),
        "mean_image_boundary_error_fraction_by_class_fp_plus_fn": {
            name: float(class_boundary_numerators[i] / class_boundary_denominators[i])
            for i, name in enumerate(CLASS_NAMES)
        },
        "bottom_quartile_specimen_groups": group_rows[:q_count],
        "bottom_quartile_share_of_positive_group_miou_deficits": bottom_quartile_deficit_share,
        "specimen_group_scores_ascending": group_rows,
        "per_image": records,
        "interpretation_guard": (
            "Boundary-ignored mIoU is diagnostic, not the competition score. "
            "These results do not justify changing the model by themselves."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-data", type=Path, required=True,
                        help="data_model322_group_cv containing oof_fold1..4")
    parser.add_argument("--oof-root", type=Path, required=True,
                        help="Model374 output root containing foldN_oof/valid_probabilities.npz")
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--expected-images", type=int, default=70)
    parser.add_argument("--expected-groups", type=int, default=59)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.boundary_radius < 0:
        raise SystemExit("--boundary-radius must be nonnegative")
    folds = tuple(args.folds)
    if not folds or len(set(folds)) != len(folds) or any(fold not in (1, 2, 3, 4) for fold in folds):
        raise SystemExit("--folds must contain unique fold numbers from 1 to 4")
    result = analyze(args.cv_data, args.oof_root, args.boundary_radius, folds)
    if args.expected_images and result["images"] != args.expected_images:
        raise SystemExit(f"expected {args.expected_images} unique OOF images, got {result['images']}")
    if args.expected_groups and result["specimen_groups"] != args.expected_groups:
        raise SystemExit(f"expected {args.expected_groups} specimen groups, got {result['specimen_groups']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_image", "specimen_group_scores_ascending")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
