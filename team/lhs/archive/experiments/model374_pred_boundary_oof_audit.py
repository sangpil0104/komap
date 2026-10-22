#!/usr/bin/env python3
"""Prediction-boundary oracle audit on exactly-once grouped train OOF maps.

Unlike a GT-boundary audit, the candidate correction region here is derived
only from the model prediction. GT is used only to measure oracle ceilings.
This is diagnostic and does not fit or deploy a correction.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

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
        raise ValueError(f"Unknown mask color in {path}")
    return target


def group_for(image_id: str) -> str:
    match = re.match(r"(A\d+)", image_id)
    if not match:
        raise ValueError(f"Cannot extract specimen group from {image_id}")
    return match.group(1)


def score(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    matrix = np.bincount(target.ravel().astype(np.int64) * 4
                         + pred.ravel().astype(np.int64), minlength=16).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.zeros(4), where=union > 0)
    return float(iou.mean()), iou


def prediction_boundary(pred: np.ndarray, radius: int) -> np.ndarray:
    edge = np.zeros(pred.shape, dtype=bool)
    for axis in (0, 1):
        left = [slice(None), slice(None)]
        right = [slice(None), slice(None)]
        left[axis] = slice(1, None)
        right[axis] = slice(None, -1)
        changed = pred[tuple(left)] != pred[tuple(right)]
        edge[tuple(left)] |= changed
        edge[tuple(right)] |= changed
    return binary_dilation(edge, structure=STRUCTURE_8, iterations=radius)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/workspace/komap_model1/data"))
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--fold-oof", type=Path, action="append", default=[],
                        help="Optional per-fold OOF archives; repeat once per fold to verify group ownership")
    parser.add_argument("--reference-results", type=Path,
                        help="Optional Model329 results.json used to assert official OOF score parity")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=70)
    parser.add_argument("--radii", type=int, nargs="+", default=(1, 3))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if any(radius < 1 for radius in args.radii):
        raise ValueError("All radii must be positive")

    per_image = []
    with np.load(args.oof) as archive:
        names = sorted(archive.files)
        if len(names) != args.expected_images or len(set(names)) != len(names):
            raise ValueError(f"Expected {args.expected_images} unique OOF images, found {len(names)}")
        for index, name in enumerate(names):
            probabilities = np.asarray(archive[name], dtype=np.float32)
            if probabilities.ndim != 3 or probabilities.shape[0] != 4:
                raise ValueError(f"Expected (4,H,W) probabilities for {name}; got {probabilities.shape}")
            target = load_target(args.data / "train" / "masks" / f"{name}_mask.png")
            if target.shape != probabilities.shape[1:]:
                raise ValueError(f"Shape mismatch for {name}: {target.shape} vs {probabilities.shape[1:]}")
            pred = probabilities.argmax(axis=0).astype(np.uint8)
            baseline, _ = score(pred, target)
            perfect, _ = score(target, target)
            ranking = np.argsort(probabilities, axis=0)
            top2 = ranking[-2]
            row = {
                "image": name,
                "specimen_group": group_for(name),
                "image_miou_empty0": baseline,
                "perfect_label_image_miou_empty0": perfect,
                "pixels": int(target.size),
                "misclassified_pixels": int((pred != target).sum()),
                "top2_contains_gt_fraction_all_errors": float(
                    ((pred != target) & (top2 == target)).sum()
                    / max(int((pred != target).sum()), 1)),
                "radii": {},
            }
            for radius in args.radii:
                band = prediction_boundary(pred, radius)
                errors = pred != target
                in_errors = errors & band
                corrected = pred.copy()
                corrected[in_errors] = target[in_errors]
                top2_correctable = in_errors & (top2 == target)
                corrected_top2 = pred.copy()
                corrected_top2[top2_correctable] = target[top2_correctable]
                score_all, _ = score(corrected, target)
                score_top2, _ = score(corrected_top2, target)
                corrected_far = pred.copy()
                far_errors = errors & ~band
                corrected_far[far_errors] = target[far_errors]
                score_far, _ = score(corrected_far, target)
                row["radii"][str(radius)] = {
                    "prediction_boundary_band_pixels": int(band.sum()),
                    "misclassified_pixels_in_band": int(in_errors.sum()),
                    "in_band_errors_with_gt_in_top2": int(top2_correctable.sum()),
                    "prediction_boundary_band_pixel_fraction": float(band.mean()),
                    "fraction_of_all_errors_inside_prediction_band": float(
                        in_errors.sum() / max(int(errors.sum()), 1)),
                    "fraction_of_in_band_errors_with_gt_in_top2": float(
                        top2_correctable.sum() / max(int(in_errors.sum()), 1)),
                    "oracle_miou_correct_all_in_band_errors": score_all,
                    "oracle_miou_correct_only_top2_in_band_errors": score_top2,
                    "oracle_miou_correct_only_outside_band_errors": score_far,
                }
            per_image.append(row)
            if (index + 1) % 10 == 0 or index + 1 == len(names):
                print(f"audited OOF images {index + 1}/{len(names)}", flush=True)

    fold_ownership = {}
    fold_names_by_index = {}
    if args.fold_oof:
        combined_names = set(names)
        for fold_index, fold_path in enumerate(args.fold_oof, start=1):
            with np.load(fold_path) as fold_archive:
                fold_names = set(fold_archive.files)
            if not fold_names:
                raise ValueError(f"Empty fold archive: {fold_path}")
            fold_names_by_index[fold_index] = fold_names
            for image_name in fold_names:
                if image_name not in combined_names:
                    raise ValueError(f"Unexpected fold OOF image {image_name} in {fold_path}")
                group = group_for(image_name)
                previous = fold_ownership.setdefault(group, fold_index)
                if previous != fold_index:
                    raise ValueError(f"Specimen group {group} appears in multiple OOF folds")
        if set(fold_ownership.values()) != set(range(1, len(args.fold_oof) + 1)):
            raise ValueError("At least one fold archive has no specimen-group owners")
        fold_union = set().union(*fold_names_by_index.values())
        if fold_union != combined_names:
            raise ValueError("Per-fold OOF archive union does not match combined archive")

    groups = {row["specimen_group"] for row in per_image}
    result = {
        "analysis": "prediction_boundary_oracle_oof",
        "source_model": "Model329 exactly-once specimen-group train OOF",
        "audit": "diagnostic only; prediction-derived boundary band; no validation/test data",
        "metric": "per-image mean of four class IoUs; empty union=0",
        "image_count": len(per_image),
        "specimen_group_count": len(groups),
        "fold_count_verified": len(args.fold_oof) if args.fold_oof else None,
        "groups_assigned_to_exactly_one_fold": bool(args.fold_oof),
        "group_count_by_fold": ({str(fold): len({group_for(name) for name in fold_names})
                                 for fold, fold_names in fold_names_by_index.items()}
                                if args.fold_oof else None),
        "class_order": list(CLASS_NAMES),
        "boundary_definition": (
            "both sides of 4-neighbor class transitions, then 8-connected binary dilation; "
            "radius is dilation iterations, not total band width"),
        "baseline_image_mean_miou_empty0": float(np.mean(
            [row["image_miou_empty0"] for row in per_image])),
        "all_labels_correct_image_mean_miou_empty0": float(np.mean(
            [row["perfect_label_image_miou_empty0"] for row in per_image])),
        "official_reference_score": None,
        "aggregate_by_radius": {},
        "pooled_pixel_fractions_by_radius": {},
        "per_image": per_image,
    }
    if args.reference_results:
        reference = json.loads(args.reference_results.read_text(encoding="utf-8"))
        expected = float(reference["b4_oof_standalone"]["image_mean_miou_empty_zero"])
        actual = result["baseline_image_mean_miou_empty0"]
        if abs(expected - actual) > 1e-7:
            raise AssertionError(f"Model329 OOF score mismatch: {actual} vs {expected}")
        result["official_reference_score"] = expected
    for radius in args.radii:
        fields = [row["radii"][str(radius)] for row in per_image]
        result["aggregate_by_radius"][str(radius)] = {
            key: float(np.mean([row[key] for row in fields]))
            for key in (
                "prediction_boundary_band_pixel_fraction",
                "fraction_of_all_errors_inside_prediction_band",
                "fraction_of_in_band_errors_with_gt_in_top2",
                "oracle_miou_correct_all_in_band_errors",
                "oracle_miou_correct_only_top2_in_band_errors",
                "oracle_miou_correct_only_outside_band_errors",
            )
        }
        total_pixels = sum(row["pixels"] for row in per_image)
        total_errors = sum(row["misclassified_pixels"] for row in per_image)
        total_band_pixels = sum(row["radii"][str(radius)]["prediction_boundary_band_pixels"]
                                for row in per_image)
        total_band_errors = sum(row["radii"][str(radius)]["misclassified_pixels_in_band"]
                                for row in per_image)
        total_top2_errors = sum(row["radii"][str(radius)]["in_band_errors_with_gt_in_top2"]
                                for row in per_image)
        result["pooled_pixel_fractions_by_radius"][str(radius)] = {
            "boundary_band_pixels_over_all_pixels": total_band_pixels / total_pixels,
            "errors_inside_band_over_all_errors": total_band_errors / max(total_errors, 1),
            "in_band_errors_with_gt_in_top2": total_top2_errors / max(total_band_errors, 1),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "baseline_image_mean_miou_empty0", "all_labels_correct_image_mean_miou_empty0",
        "official_reference_score", "aggregate_by_radius", "pooled_pixel_fractions_by_radius")},
        indent=2), flush=True)


if __name__ == "__main__":
    main()
