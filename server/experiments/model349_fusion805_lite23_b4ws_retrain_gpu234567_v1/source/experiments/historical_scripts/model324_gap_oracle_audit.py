#!/usr/bin/env python3
"""Train-OOF upper-bound audit for GRAF-style Primary-Si gap weighting.

The gap map is built at the exact minimum-side resolution used by the Model302
patch dataset, then mapped back to the saved OOF prediction grid. GT is used
only to define the oracle correction; this script does not train or tune a model.
"""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from improve_five_causes import collect_split, decode_mask
from model302_boundary_oof_finetune import gap_region_from_mask


def score(prediction: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    confusion = np.bincount(target.astype(np.int64).ravel() * 4
                            + prediction.astype(np.int64).ravel(),
                            minlength=16).reshape(4, 4)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.zeros(4, dtype=np.float64),
                    where=union > 0)
    return float(iou.mean()), iou


def specimen_group(path: Path) -> str:
    match = re.match(r"(A\d+)_", path.name)
    if not match:
        raise ValueError(f"cannot parse specimen group from {path.name}")
    return match.group(1)


def probabilities_in_main_order(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 3 and value.shape[0] == 4:
        return value
    if value.ndim == 3 and value.shape[-1] == 4:
        return np.moveaxis(value, -1, 0)
    raise ValueError(f"expected four probability channels, got {value.shape}")


def gap_at_training_resolution(target: np.ndarray, min_side: int,
                               radius: int) -> np.ndarray:
    target = np.asarray(target, dtype=np.uint8)
    height, width = target.shape
    factor = max(1.0, float(min_side) / min(height, width))
    scaled_size = (int(round(width * factor)), int(round(height * factor)))
    if scaled_size == (width, height):
        scaled_target = target
    else:
        scaled_target = np.asarray(Image.fromarray(target).resize(
            scaled_size, Image.Resampling.NEAREST), dtype=np.uint8)
    gap_scaled = gap_region_from_mask(
        scaled_target, radius, exclude_crop_border=False)
    if scaled_size == (width, height):
        return gap_scaled.astype(bool)
    gap_original = Image.fromarray(gap_scaled).resize(
        (width, height), Image.Resampling.NEAREST)
    return np.asarray(gap_original, dtype=np.uint8).astype(bool)


def group_bootstrap_ci(groups: list[str], deltas: np.ndarray,
                       seed: int = 324, draws: int = 10000) -> list[float]:
    grouped: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        grouped.setdefault(group, []).append(index)
    names = list(grouped)
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(names, size=len(names), replace=True)
        indices = [index for name in sampled for index in grouped[str(name)]]
        values[draw] = deltas[indices].mean() * 100.0
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--min-side", type=int, default=1024)
    parser.add_argument("--gap-radius", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.min_side < 1 or args.gap_radius < 1:
        raise ValueError("min-side and gap-radius must be >= 1")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.out}")

    samples = collect_split(str(args.data_dir), args.split)
    with np.load(args.probabilities) as archive:
        missing = [image.stem for image, _ in samples if image.stem not in archive]
        if missing:
            raise KeyError(f"probability archive missing {len(missing)} images: {missing[:3]}")
        rows = []
        for image_path, mask_path in _local_progress(samples, 'model324_gap_oracle_audit'):
            target = decode_mask(Image.open(mask_path).convert("RGB"))
            probabilities = probabilities_in_main_order(archive[image_path.stem])
            if probabilities.shape[1:] != target.shape:
                raise ValueError(f"shape mismatch for {image_path.name}: "
                                 f"{probabilities.shape[1:]} vs {target.shape}")
            prediction = probabilities.argmax(0).astype(np.uint8)
            gap = gap_at_training_resolution(target, args.min_side, args.gap_radius)
            oracle = prediction.copy()
            oracle[gap] = target[gap]
            baseline_score, baseline_class = score(prediction, target)
            oracle_score, oracle_class = score(oracle, target)
            rows.append({
                "image": image_path.name,
                "specimen_group": specimen_group(image_path),
                "baseline_miou": baseline_score,
                "oracle_miou": oracle_score,
                "delta_pp": (oracle_score - baseline_score) * 100.0,
                "baseline_class_iou": baseline_class.tolist(),
                "oracle_class_iou": oracle_class.tolist(),
                "gap_pixel_fraction": float(gap.mean()),
                "gap_error_fraction": (float((prediction[gap] != target[gap]).mean())
                                       if gap.any() else 0.0),
                "gap_pixels": int(gap.sum()),
            })

    deltas = np.asarray([row["delta_pp"] / 100.0 for row in rows], dtype=np.float64)
    result = {
        "audit": "oracle upper bound: replace predictions inside GT-defined gap pixels by GT",
        "split": args.split,
        "metric": "image-wise mean of 4 class IoUs; absent union=0",
        "gap_definition": {
            "class": "Primary Si (main channel id 1)",
            "connectivity": 8,
            "radius_at_training_resolution_px": args.gap_radius,
            "minimum_side_used_by_training": args.min_side,
            "crop_border_excluded": False,
        },
        "image_count": len(rows),
        "specimen_group_count": len({row["specimen_group"] for row in rows}),
        "baseline_image_mean_miou": float(np.mean([row["baseline_miou"] for row in rows])),
        "gap_oracle_image_mean_miou": float(np.mean([row["oracle_miou"] for row in rows])),
        "gap_oracle_delta_pp": float(np.mean([row["delta_pp"] for row in rows])),
        "gap_oracle_group_bootstrap_95ci_delta_pp": group_bootstrap_ci(
            [row["specimen_group"] for row in rows], deltas),
        "mean_gap_pixel_fraction": float(np.mean([row["gap_pixel_fraction"] for row in rows])),
        "mean_gap_error_fraction": float(np.mean([row["gap_error_fraction"] for row in rows])),
        "per_class_baseline_image_mean_iou": np.mean(
            [row["baseline_class_iou"] for row in rows], axis=0).tolist(),
        "per_class_gap_oracle_image_mean_iou": np.mean(
            [row["oracle_class_iou"] for row in rows], axis=0).tolist(),
        "images": rows,
        "warning": "This is a GT-informed upper bound, not an achievable model score.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "split", "image_count", "specimen_group_count", "baseline_image_mean_miou",
        "gap_oracle_image_mean_miou", "gap_oracle_delta_pp",
        "gap_oracle_group_bootstrap_95ci_delta_pp", "mean_gap_pixel_fraction",
        "mean_gap_error_fraction", "per_class_baseline_image_mean_iou",
        "per_class_gap_oracle_image_mean_iou")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
