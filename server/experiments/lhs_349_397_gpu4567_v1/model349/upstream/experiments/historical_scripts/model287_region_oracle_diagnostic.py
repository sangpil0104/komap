#!/usr/bin/env python3
"""Oracle diagnostics separating predicted region-label and phase-boundary errors."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import ndimage

from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import bilateral_weights, make_mixed, predict


def image_score(prediction: np.ndarray, target: np.ndarray) -> float:
    matrix = np.bincount(
        target.reshape(-1).astype(np.int64) * 4 + prediction.reshape(-1).astype(np.int64),
        minlength=16,
    ).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    return float((intersection / np.maximum(union, 1.0)).mean())


def component_label_oracle(prediction: np.ndarray, target: np.ndarray,
                           connectivity: int) -> tuple[np.ndarray, int]:
    """Keep predicted secondary-phase shape; relabel each connected region by GT majority."""
    binary = prediction != 2
    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, count = ndimage.label(binary, structure=structure)
    encoded = labels[binary].astype(np.int64) * 4 + target[binary].astype(np.int64)
    counts = np.bincount(encoded, minlength=(count + 1) * 4).reshape(count + 1, 4)
    majority = counts.argmax(axis=1).astype(np.uint8)
    output = prediction.copy()
    output[binary] = majority[labels[binary]]
    return output, int(count)


def class_component_label_oracle(prediction: np.ndarray, target: np.ndarray,
                                 connectivity: int) -> tuple[np.ndarray, int]:
    """Relabel components separately within each predicted secondary phase."""
    structure = ndimage.generate_binary_structure(2, connectivity)
    output = prediction.copy()
    total_count = 0
    for class_id in (0, 1, 3):
        binary = prediction == class_id
        labels, count = ndimage.label(binary, structure=structure)
        if count == 0:
            continue
        encoded = labels[binary].astype(np.int64) * 4 + target[binary].astype(np.int64)
        counts = np.bincount(encoded, minlength=(count + 1) * 4).reshape(count + 1, 4)
        majority = counts.argmax(axis=1).astype(np.uint8)
        output[binary] = majority[labels[binary]]
        total_count += int(count)
    return output, total_count


def foreground_shape_oracle(prediction: np.ndarray, mixed: np.ndarray,
                            target: np.ndarray) -> np.ndarray:
    """Use GT secondary-vs-Al shape, retaining predicted secondary labels where possible."""
    secondary_ids = np.asarray((0, 1, 3), dtype=np.uint8)
    best_secondary = secondary_ids[mixed[secondary_ids].argmax(axis=0)]
    output = np.full_like(target, 2, dtype=np.uint8)
    gt_secondary = target != 2
    kept = gt_secondary & (prediction != 2)
    output[kept] = prediction[kept]
    fill = gt_secondary & (prediction == 2)
    output[fill] = best_secondary[fill]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--model195", required=True)
    parser.add_argument("--tent20", required=True)
    parser.add_argument("--model244", required=True)
    parser.add_argument("--model249", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.entries).open("rb") as stream:
        entries = pickle.load(stream)
    with Path(args.settings).open(encoding="utf-8") as stream:
        setting = json.load(stream)["best_image"]
    archives = {
        "195": np.load(args.model195),
        "217": np.load(args.tent20),
        "244": np.load(args.model244),
        "249": np.load(args.model249),
    }
    orders = {tuple(archive.files) for archive in archives.values()}
    if len(orders) != 1 or len(next(iter(orders))) != len(entries):
        raise ValueError("validation probability archives do not align with cache entries")
    keys = tuple(next(iter(orders)))
    values = {
        source: [archive[key].astype(np.float32) for key in keys]
        for source, archive in archives.items()
    }

    names, baseline, comp4, comp8, class_comp4, class_comp8, shape = (
        [], [], [], [], [], [], [])
    region_counts4, region_counts8, class_region_counts4, class_region_counts8 = (
        [], [], [], [])
    for index, (name, item) in enumerate(zip(keys, entries)):
        row = {
            "item": item,
            "base": undo_model181_primary_blend(item),
            "sources": {
                "base": None,
                "181": item["extra"].astype(np.float32),
                **{source: probabilities[index] for source, probabilities in values.items()},
            },
        }
        mixed = make_mixed(row, setting)
        target = item["target"].astype(np.uint8)
        prediction = predict(item, mixed, setting, bilateral_weights(item["z"]), exact=True)
        oracle4, count4 = component_label_oracle(prediction, target, connectivity=1)
        oracle8, count8 = component_label_oracle(prediction, target, connectivity=2)
        class_oracle4, class_count4 = class_component_label_oracle(
            prediction, target, connectivity=1)
        class_oracle8, class_count8 = class_component_label_oracle(
            prediction, target, connectivity=2)
        shape_oracle = foreground_shape_oracle(prediction, mixed, target)
        names.append(name)
        baseline.append(image_score(prediction, target))
        comp4.append(image_score(oracle4, target))
        comp8.append(image_score(oracle8, target))
        class_comp4.append(image_score(class_oracle4, target))
        class_comp8.append(image_score(class_oracle8, target))
        shape.append(image_score(shape_oracle, target))
        region_counts4.append(count4)
        region_counts8.append(count8)
        class_region_counts4.append(class_count4)
        class_region_counts8.append(class_count8)
        print(f"{name}: base={baseline[-1]*100:.4f}% cc4={comp4[-1]*100:.4f}% "
              f"cc8={comp8[-1]*100:.4f}% classcc4={class_comp4[-1]*100:.4f}% "
              f"classcc8={class_comp8[-1]*100:.4f}% shape={shape[-1]*100:.4f}% "
              f"regions={count4}/{count8};class={class_count4}/{class_count8}", flush=True)

    arrays = {
        "baseline": np.asarray(baseline),
        "component_oracle_4": np.asarray(comp4),
        "component_oracle_8": np.asarray(comp8),
        "class_component_oracle_4": np.asarray(class_comp4),
        "class_component_oracle_8": np.asarray(class_comp8),
        "foreground_shape_oracle": np.asarray(shape),
    }
    means = {key: float(scores.mean()) for key, scores in arrays.items()}
    result = {
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "route": setting,
        "image_count": len(names),
        "specimen_group_count": len({name.split("_", 1)[0] for name in names}),
        "mean_scores": means,
        "mean_delta_pp_vs_baseline": {
            key: float((scores - arrays["baseline"]).mean() * 100.0)
            for key, scores in arrays.items() if key != "baseline"
        },
        "group_bootstrap_95ci_delta_pp": {},
        "mean_region_count": {
            "4_connected": float(np.mean(region_counts4)),
            "8_connected": float(np.mean(region_counts8)),
            "classwise_4_connected": float(np.mean(class_region_counts4)),
            "classwise_8_connected": float(np.mean(class_region_counts8)),
        },
        "images": [{
            "name": name,
            "specimen": name.split("_", 1)[0],
            "baseline": float(arrays["baseline"][index]),
            "component_oracle_4": float(arrays["component_oracle_4"][index]),
            "component_oracle_8": float(arrays["component_oracle_8"][index]),
            "class_component_oracle_4": float(arrays["class_component_oracle_4"][index]),
            "class_component_oracle_8": float(arrays["class_component_oracle_8"][index]),
            "foreground_shape_oracle": float(arrays["foreground_shape_oracle"][index]),
            "regions_4_connected": region_counts4[index],
            "regions_8_connected": region_counts8[index],
            "class_regions_4_connected": class_region_counts4[index],
            "class_regions_8_connected": class_region_counts8[index],
        } for index, name in enumerate(names)],
    }
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(name.split("_", 1)[0], []).append(index)
    rng = np.random.default_rng(287)
    group_names = list(groups)
    for key in ("component_oracle_4", "component_oracle_8", "class_component_oracle_4",
                "class_component_oracle_8", "foreground_shape_oracle"):
        delta = arrays[key] - arrays["baseline"]
        bootstrap = np.empty(10000, dtype=np.float64)
        for draw in range(len(bootstrap)):
            sampled = rng.choice(group_names, size=len(group_names), replace=True)
            indices = [index for group in sampled for index in groups[str(group)]]
            bootstrap[draw] = delta[indices].mean() * 100.0
        result["group_bootstrap_95ci_delta_pp"][key] = [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "mean_scores", "mean_delta_pp_vs_baseline", "group_bootstrap_95ci_delta_pp",
        "mean_region_count")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
