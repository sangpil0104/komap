#!/usr/bin/env python3
"""Re-score every saved validation probability archive with both empty-class policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Existing KoMaP model archives use this fixed channel order.
MAIN_PALETTE = np.asarray([
    [25, 76, 153],   # Al3Ni
    [76, 178, 76],   # Primary Si
    [153, 127, 76],  # Al
    [204, 204, 204], # Eutectic Si
], dtype=np.uint8)
# Teammate B4 code uses [Al, Al3Ni, Primary Si, Eutectic Si].
TEAM_TO_MAIN = (1, 2, 0, 3)


def decode_mask(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    target = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(MAIN_PALETTE):
        target[np.all(rgb == color, axis=-1)] = class_id
    if np.any(target == 255):
        raise ValueError(f"unknown mask color in {path}")
    return target


def normalize_prob(value: np.ndarray, team_order: bool) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 2:
        return value.astype(np.uint8)
    if value.ndim != 3:
        raise ValueError(f"expected HxW label or 3D probabilities, got {value.shape}")
    if value.shape[0] == 4:
        probabilities = value
    elif value.shape[-1] == 4:
        probabilities = np.moveaxis(value, -1, 0)
    else:
        raise ValueError(f"cannot identify four class channels: {value.shape}")
    if team_order:
        probabilities = probabilities[list(TEAM_TO_MAIN)]
    return probabilities


def score_image(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, list[float], int]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/mask shape mismatch: {prediction.shape} vs {target.shape}")
    confusion = np.bincount(target.astype(np.int64).ravel() * 4 + prediction.ravel().astype(np.int64),
                            minlength=16).reshape(4, 4)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.full(4, np.nan), where=union > 0)
    fixed_zero = float(np.nan_to_num(iou, nan=0.0).mean())
    skip_empty = float(np.nanmean(iou))
    empty_count = int((union == 0).sum())
    return fixed_zero, skip_empty, np.nan_to_num(iou, nan=0.0).tolist(), empty_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--team-order-prefix", default="model322_team_b4_group_oof")
    args = parser.parse_args()

    valid_images = sorted((args.data / "valid" / "images").glob("*_image.png"))
    valid = {}
    for image in valid_images:
        name = image.name.replace("_image.png", "")
        mask = args.data / "valid" / "masks" / image.name.replace("_image.png", "_mask.png")
        if not mask.is_file():
            raise FileNotFoundError(mask)
        valid[name] = (image, mask, decode_mask(mask))
    if len(valid) != 20:
        raise ValueError(f"expected 20 validation images, found {len(valid)}")

    records, skipped = [], []
    for path in sorted(args.outputs.rglob("*.npz")):
        try:
            with np.load(path) as archive:
                key_map = {key.removesuffix("_image"): key for key in archive.files}
                if not set(valid).issubset(key_map):
                    continue
                team_order = args.team_order_prefix in path.relative_to(args.outputs).parts
                fixed, skipped_empty, per_class, empty_pairs = [], [], [], 0
                for name, (_, _, target) in valid.items():
                    probabilities = normalize_prob(archive[key_map[name]], team_order)
                    if probabilities.ndim == 2:
                        prediction = probabilities
                    else:
                        prediction = probabilities.argmax(0).astype(np.uint8)
                    a, b, class_iou, empty_count = score_image(prediction, target)
                    fixed.append(a)
                    skipped_empty.append(b)
                    per_class.append(class_iou)
                    empty_pairs += empty_count
            record = {
                "archive": str(path),
                "channel_order": "team [Al, Al3Ni, Primary Si, Eutectic Si]" if team_order
                                else "main [Al3Ni, Primary Si, Al, Eutectic Si]",
                "image_count": len(valid),
                "image_mean_miou_empty_zero": float(np.mean(fixed)),
                "image_mean_miou_empty_skipped": float(np.mean(skipped_empty)),
                "mean_delta_skip_minus_zero_pp": float((np.mean(skipped_empty)
                                                         - np.mean(fixed)) * 100),
                "per_class_image_mean_iou_empty_zero": np.mean(per_class, axis=0).tolist(),
                "empty_image_class_pairs": empty_pairs,
            }
            records.append(record)
        except Exception as exc:
            skipped.append({"archive": str(path), "error": f"{type(exc).__name__}: {exc}"})

    records.sort(key=lambda row: row["image_mean_miou_empty_zero"], reverse=True)
    by_skip = sorted(records, key=lambda row: row["image_mean_miou_empty_skipped"], reverse=True)
    result = {
        "metric": "mean over 20 images of per-image mean of four class IoUs",
        "empty_policies": {"empty_zero": "union=0 class IoU is 0; always average four classes",
                           "empty_skipped": "union=0 class is NaN; nanmean over present unions"},
        "valid_image_count": len(valid),
        "scanned_npz_count": len(list(args.outputs.rglob("*.npz"))),
        "scored_archive_count": len(records),
        "top_empty_zero": records[:20],
        "top_empty_skipped": by_skip[:20],
        "all_scored": records,
        "errors": skipped,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "valid_image_count", "scanned_npz_count", "scored_archive_count",
        "top_empty_zero", "top_empty_skipped")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
