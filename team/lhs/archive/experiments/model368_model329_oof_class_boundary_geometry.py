#!/usr/bin/env python3
"""Train-OOF-only classwise boundary/error geometry audit for Model329.

Reports each class's signed area error per GT boundary pixel, plus near/far
false-negative and false-positive fractions. No validation or test data is read.
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


def group_bootstrap_ci(values: np.ndarray, groups: list[str], seed: int,
                       repetitions: int = 10000) -> list[float]:
    unique = np.asarray(sorted(set(groups)))
    rows = {group: np.flatnonzero(np.asarray(groups) == group) for group in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for i in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([rows[group] for group in sampled])
        draws[i] = values[indices].mean()
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/workspace/komap_model1/data"))
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=70)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if args.radius < 1:
        raise ValueError("--radius must be positive")

    names = []
    per_image = []
    with np.load(args.oof) as archive:
        names = sorted(archive.files)
        if len(names) != args.expected_images:
            raise ValueError(f"Expected {args.expected_images} OOF images, found {len(names)}")
        for index, name in enumerate(names):
            probabilities = np.asarray(archive[name])
            if probabilities.ndim != 3 or probabilities.shape[0] != 4:
                raise ValueError(f"Expected (4,H,W) probabilities for {name}, got {probabilities.shape}")
            prediction = probabilities.argmax(axis=0).astype(np.uint8)
            target = load_target(args.data / "train" / "masks" / f"{name}_mask.png")
            if target.shape != prediction.shape:
                raise ValueError(f"Shape mismatch for {name}: {target.shape} vs {prediction.shape}")

            row = {"image": name, "specimen_group": group_for(name), "classes": {}}
            for class_id, class_name in enumerate(CLASS_NAMES):
                gt = target == class_id
                pred = prediction == class_id
                edge = gt & ~binary_erosion(gt, structure=STRUCTURE_8, border_value=0)
                band = binary_dilation(edge, structure=STRUCTURE_8,
                                       iterations=args.radius)
                fn = gt & ~pred
                fp = pred & ~gt
                row["classes"][class_name] = {
                    "gt_pixels": int(gt.sum()),
                    "pred_pixels": int(pred.sum()),
                    "gt_boundary_pixels": int(edge.sum()),
                    "signed_area_error_per_boundary_pixel": float(
                        (pred.sum() - gt.sum()) / max(int(edge.sum()), 1)),
                    "gt_fraction_in_band": float((gt & band).sum() / max(int(gt.sum()), 1)),
                    "fn_pixels": int(fn.sum()),
                    "fn_fraction_in_band": float((fn & band).sum() / max(int(fn.sum()), 1)),
                    "fp_pixels": int(fp.sum()),
                    "fp_fraction_in_band": float((fp & band).sum() / max(int(fp.sum()), 1)),
                }
            per_image.append(row)
            if (index + 1) % 10 == 0 or index + 1 == len(names):
                print(f"audited OOF images {index + 1}/{len(names)}", flush=True)

    groups = [row["specimen_group"] for row in per_image]
    summary = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        rows = [row["classes"][class_name] for row in per_image]
        values = np.asarray([r["signed_area_error_per_boundary_pixel"] for r in rows])
        totals = {
            key: sum(r[key] for r in rows)
            for key in ("gt_pixels", "pred_pixels", "gt_boundary_pixels", "fn_pixels", "fp_pixels")
        }
        summary[class_name] = {
            "signed_area_error_per_boundary_pixel_mean": float(values.mean()),
            "signed_area_error_per_boundary_pixel_group_bootstrap_95ci":
                group_bootstrap_ci(values, groups, seed=368 + class_id,
                                   repetitions=args.bootstrap),
            "signed_area_error_per_boundary_pixel_pooled": float(
                (totals["pred_pixels"] - totals["gt_pixels"])
                / max(totals["gt_boundary_pixels"], 1)),
            "gt_fraction_in_band_pooled": float(sum(
                r["gt_pixels"] * r["gt_fraction_in_band"] for r in rows)
                / max(totals["gt_pixels"], 1)),
            "fn_fraction_in_band_pooled": float(sum(
                r["fn_pixels"] * r["fn_fraction_in_band"] for r in rows)
                / max(totals["fn_pixels"], 1)),
            "fp_fraction_in_band_pooled": float(sum(
                r["fp_pixels"] * r["fp_fraction_in_band"] for r in rows)
                / max(totals["fp_pixels"], 1)),
            "pixel_totals": totals,
        }

    result = {
        "model": 368,
        "audit": "Model329 grouped train OOF only; no validation/test files read",
        "image_count": len(names),
        "specimen_group_count": len(set(groups)),
        "class_order": list(CLASS_NAMES),
        "boundary_band": f"per-class GT inner boundary dilated {args.radius}px; 8-connectivity",
        "empty_union_policy": "IoU=0 in the source Model329 OOF audit",
        "classwise": summary,
        "per_image": per_image,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({"model": 368, "image_count": len(names),
                      "specimen_group_count": len(set(groups)),
                      "classwise": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
