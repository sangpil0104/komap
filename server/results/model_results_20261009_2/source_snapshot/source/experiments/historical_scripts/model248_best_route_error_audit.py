#!/usr/bin/env python3
"""Per-image/per-class audit of the current best exact validation route."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from improve_five_causes import collect_split
from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import bilateral_weights, make_mixed, predict


CLASS_NAMES = ("Al3Ni", "Primary Si", "Al", "Eutectic Si")


def confusion(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    bins = target.reshape(-1) * 4 + prediction.reshape(-1)
    return np.bincount(bins, minlength=16).reshape(4, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--entries", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--model195", required=True)
    parser.add_argument("--model217", required=True)
    parser.add_argument("--model244", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    with Path(args.settings).open(encoding="utf-8") as handle:
        setting = json.load(handle)["best_image"]
    archives = {key: np.load(path) for key, path in (
        ("195", args.model195), ("217", args.model217), ("244", args.model244))}
    key_orders = [tuple(archive.files) for archive in archives.values()]
    if len(set(key_orders)) != 1:
        raise ValueError("probability archives differ in image keys/order")
    source_values = {
        name: [archive[key].astype(np.float32) for key in archive.files]
        for name, archive in archives.items()
    }
    samples = collect_split(args.data_dir, "valid")
    if len(samples) != len(entries):
        raise ValueError("validation samples and cached entries differ in count")

    records = []
    for index, ((image_path, _), item) in enumerate(zip(samples, entries)):
        row = {
            "item": item,
            "base": undo_model181_primary_blend(item),
            "sources": {"base": None, "181": item["extra"].astype(np.float32),
                        **{name: values[index] for name, values in source_values.items()}},
        }
        mixed = make_mixed(row, setting)
        prediction = predict(item, mixed, setting, bilateral_weights(item["z"]), exact=True)
        target = item["target"].astype(np.uint8)
        matrix = confusion(prediction, target)
        intersection = np.diag(matrix).astype(np.float64)
        union = matrix.sum(0) + matrix.sum(1) - intersection
        iou = intersection / np.maximum(union, 1.0)
        records.append({
            "image": image_path.name,
            "specimen": image_path.name.split("_", 1)[0],
            "image_mean_miou": float(iou.mean()),
            "class_iou": {name: float(value) for name, value in zip(CLASS_NAMES, iou)},
            "gt_pixels": {name: int(value) for name, value in zip(CLASS_NAMES, matrix.sum(1))},
            "pred_pixels": {name: int(value) for name, value in zip(CLASS_NAMES, matrix.sum(0))},
            "union_pixels": {name: int(value) for name, value in zip(CLASS_NAMES, union)},
            "empty_union_classes": [CLASS_NAMES[c] for c in range(4) if union[c] == 0],
            "confusion": matrix.tolist(),
        })

    class_image_means = {
        name: float(np.mean([record["class_iou"][name] for record in records]))
        for name in CLASS_NAMES
    }
    class_global = {}
    total = np.sum([np.asarray(record["confusion"], dtype=np.int64)
                    for record in records], axis=0)
    intersection = np.diag(total).astype(np.float64)
    union = total.sum(0) + total.sum(1) - intersection
    class_global = {name: float(value) for name, value in zip(
        CLASS_NAMES, intersection / np.maximum(union, 1.0))}
    records.sort(key=lambda row: row["image_mean_miou"])
    output = {
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "route": {key: setting.get(key) for key in (
            "primary_source", "primary_gamma", "al3ni_source", "al3ni_gamma",
            "eutectic_source", "eutectic_gamma", "prior_scale", "beta_scale",
            "al_delta", "eut_delta", "gaussian_blend", "eut_edge_blend",
            "cleanup_threshold")},
        "image_mean_miou": float(np.mean([record["image_mean_miou"] for record in records])),
        "class_image_mean_iou": class_image_means,
        "class_image_class_iou": class_global,
        "empty_union_count_by_class": {
            name: int(sum(name in record["empty_union_classes"] for record in records))
            for name in CLASS_NAMES
        },
        "lowest_scoring_images": records[:10],
        "all_images": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in (
        "image_mean_miou", "class_image_mean_iou", "class_image_class_iou",
        "empty_union_count_by_class", "lowest_scoring_images")},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
