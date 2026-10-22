#!/usr/bin/env python3
"""Reconstruct Model270 once and audit image/class/boundary failure headroom."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from improve_five_causes import collect_split, decode_mask, specimen_id
from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import bilateral_weights, make_mixed, predict


CLASS_NAMES = ("Al3Ni", "Primary Si", "Al", "Eutectic Si")


def confusion(prediction: np.ndarray, target: np.ndarray,
              valid: np.ndarray | None = None) -> np.ndarray:
    if valid is not None:
        prediction, target = prediction[valid], target[valid]
    return np.bincount(target.astype(np.int64).ravel() * 4
                       + prediction.astype(np.int64).ravel(),
                       minlength=16).reshape(4, 4)


def class_iou(matrix: np.ndarray) -> np.ndarray:
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    return np.divide(intersection, union, out=np.zeros(4), where=union > 0)


def boundary_band(target: np.ndarray, radius: int) -> np.ndarray:
    boundary = np.zeros(target.shape, dtype=bool)
    for axis in (0, 1):
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis] = slice(1, None)
        second[axis] = slice(None, -1)
        change = target[tuple(first)] != target[tuple(second)]
        boundary[tuple(first)] |= change
        boundary[tuple(second)] |= change
    return ndimage.binary_dilation(boundary, iterations=radius)


def score_from_confusions(matrices: list[np.ndarray]) -> tuple[float, np.ndarray]:
    per_image = np.asarray([class_iou(matrix) for matrix in matrices])
    return float(per_image.mean(axis=1).mean()), per_image.mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--entries", type=Path, required=True)
    parser.add_argument("--model195", type=Path, required=True)
    parser.add_argument("--tent20", type=Path, required=True)
    parser.add_argument("--model269", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--boundary-radii", type=int, nargs="+", default=(1, 2, 3))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.out}")

    with args.entries.open("rb") as handle:
        entries = pickle.load(handle)
    archive_paths = {"195": args.model195, "217": args.tent20, "269": args.model269}
    archives = {key: np.load(path) for key, path in archive_paths.items()}
    key_orders = [tuple(archive.files) for archive in archives.values()]
    if len(set(key_orders)) != 1:
        raise ValueError("Model270 component archives differ in image keys/order")
    names = key_orders[0]
    samples = collect_split(str(args.data_dir), "valid")
    sample_names = tuple(image.stem for image, _ in samples)
    if sample_names != names or len(entries) != len(names):
        raise ValueError("valid manifest, cache entries, and component archive order differ")
    source_values = {
        source: [archive[name].astype(np.float32) for name in names]
        for source, archive in archives.items()
    }

    setting = {
        "stage": "model270_reference",
        "primary_source": "195", "primary_gamma": 0.75,
        "al3ni_source": "217", "al3ni_gamma": 0.40,
        "eutectic_source": "269", "eutectic_gamma": 0.50,
        "prior_scale": 1.25, "beta_scale": 0.80,
        "al_delta": 0.0, "eut_delta": 0.15,
        "gaussian_blend": 0.50, "eut_edge_blend": 0.25,
        "cleanup_threshold": 128,
    }
    confusions: list[np.ndarray] = []
    oracle_confusions = {name: [] for name in CLASS_NAMES}
    boundary_confusions = {radius: [] for radius in args.boundary_radii}
    records = []
    image_scores = []

    for index, ((image_path, mask_path), item) in enumerate(zip(samples, entries)):
        target = decode_mask(Image.open(mask_path).convert("RGB")).astype(np.uint8)
        cached_target = item["target"].astype(np.uint8)
        if target.shape != cached_target.shape or not np.array_equal(target, cached_target):
            raise ValueError(f"cached GT differs from mask for {image_path.name}")
        row = {
            "item": item,
            "base": undo_model181_primary_blend(item),
            "sources": {"base": None, "181": item["extra"].astype(np.float32),
                        **{key: values[index] for key, values in source_values.items()}},
        }
        mixed = make_mixed(row, setting)
        prediction = predict(item, mixed, setting, bilateral_weights(item["z"]), exact=True)
        matrix = confusion(prediction, target)
        class_scores = class_iou(matrix)
        confusions.append(matrix)
        image_score = float(class_scores.mean())
        image_scores.append(image_score)

        # GT-informed single-class oracle: make one class perfect and correct
        # false predictions of that class, leaving all other pixels unchanged.
        oracle_deltas = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            corrected = prediction.copy()
            edit = (target == class_id) | (prediction == class_id)
            corrected[edit] = target[edit]
            oracle_matrix = confusion(corrected, target)
            oracle_confusions[class_name].append(oracle_matrix)
            oracle_deltas[class_name] = float(
                (class_iou(oracle_matrix).mean() - image_score) * 100.0)

        for radius in args.boundary_radii:
            valid_pixels = ~boundary_band(target, radius)
            boundary_confusions[radius].append(confusion(prediction, target, valid_pixels))

        gt_counts = matrix.sum(1)
        pred_counts = matrix.sum(0)
        absent_predictions = [CLASS_NAMES[c] for c in range(4)
                             if gt_counts[c] > 0 and pred_counts[c] == 0]
        records.append({
            "image": image_path.name,
            "specimen_group": specimen_id(image_path),
            "image_mean_miou": image_score,
            "class_iou": dict(zip(CLASS_NAMES, class_scores.tolist())),
            "gt_pixels": dict(zip(CLASS_NAMES, gt_counts.astype(int).tolist())),
            "pred_pixels": dict(zip(CLASS_NAMES, pred_counts.astype(int).tolist())),
            "gt_present_pred_absent": absent_predictions,
            "single_class_oracle_delta_pp": oracle_deltas,
            "confusion": matrix.tolist(),
        })

    mean_miou, class_means = score_from_confusions(confusions)
    if abs(mean_miou - 0.787209255349963) > 1e-6:
        raise RuntimeError(f"Model270 baseline did not reproduce: {mean_miou:.12f}")
    total_confusion = np.sum(confusions, axis=0)
    global_class_iou = class_iou(total_confusion)
    single_class_oracles = {}
    for class_name, matrices in oracle_confusions.items():
        oracle_score, oracle_class_means = score_from_confusions(matrices)
        single_class_oracles[class_name] = {
            "oracle_image_mean_miou": oracle_score,
            "delta_pp": (oracle_score - mean_miou) * 100.0,
            "oracle_class_image_mean_iou": oracle_class_means.tolist(),
        }
    boundary_results = {}
    for radius, matrices in boundary_confusions.items():
        boundary_score, per_class = score_from_confusions(matrices)
        boundary_results[str(radius)] = {
            "ignored_boundary_pixels_px": radius,
            "image_mean_miou_on_remaining_pixels": boundary_score,
            "delta_pp_vs_full_image_metric": (boundary_score - mean_miou) * 100.0,
            "per_class_image_mean_iou": per_class.tolist(),
        }

    records_sorted = sorted(records, key=lambda row: row["image_mean_miou"])
    output = {
        "audit": "diagnostic only; validation GT was not used to tune a model or route",
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "route": setting,
        "image_count": len(records),
        "specimen_group_count": len({row["specimen_group"] for row in records}),
        "image_mean_miou": mean_miou,
        "class_image_mean_iou": dict(zip(CLASS_NAMES, class_means.tolist())),
        "class_global_iou": dict(zip(CLASS_NAMES, global_class_iou.tolist())),
        "gt_present_pred_absent_count_by_class": {
            name: int(sum(name in row["gt_present_pred_absent"] for row in records))
            for name in CLASS_NAMES
        },
        "single_class_oracle_upper_bounds": single_class_oracles,
        "boundary_tolerance_diagnostics": boundary_results,
        "lowest_scoring_images": records_sorted[:10],
        "all_images": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in (
        "image_mean_miou", "class_image_mean_iou", "class_global_iou",
        "gt_present_pred_absent_count_by_class", "single_class_oracle_upper_bounds",
        "boundary_tolerance_diagnostics", "lowest_scoring_images")},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
