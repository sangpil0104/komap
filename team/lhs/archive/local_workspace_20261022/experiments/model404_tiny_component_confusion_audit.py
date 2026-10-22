#!/usr/bin/env python3
"""Audit which classes absorb tiny Al3Ni/Eutectic GT components in fixed OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import label

import model396_fixed_architecture_oof as m396


ROOT = Path(__file__).resolve().parents[1]
CV_ROOT = "data_model322_group_cv"
B4_ROOT = "outputs/model329_fullstats_width4000_uniform_group_oof"
MODEL392_ROOT = "outputs/model392_widthnorm"
GROUP_RE = re.compile(r"^(A\d+)_")
TARGETS = {1: "Al3Ni", 3: "Eutectic Si"}
CLASSES = ("Al", "Al3Ni", "Primary Si", "Eutectic Si")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_component_confusion(truth: np.ndarray, prediction: np.ndarray,
                                  class_id: int) -> dict:
    labels, count = label(truth == class_id, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    confusion = np.zeros(4, dtype=np.int64)
    majority_predicted_as = np.zeros(4, dtype=np.int64)
    recalls: list[float] = []
    component_count = 0
    full_miss = 0
    majority_miss = 0
    for component_id in range(1, count + 1):
        area = int(areas[component_id])
        if not 1 <= area < 64:
            continue
        selected = labels == component_id
        predicted_counts = np.bincount(prediction[selected].astype(np.int64), minlength=4)
        confusion += predicted_counts
        majority_predicted_as[int(np.argmax(predicted_counts))] += 1
        recall = float(predicted_counts[class_id] / area)
        recalls.append(recall)
        component_count += 1
        full_miss += int(recall == 0.0)
        majority_miss += int(recall < 0.50)
    support = int(confusion.sum())
    return {
        "components": component_count,
        "support_pixels": support,
        "pixel_confusion_by_predicted_class": {
            name: int(confusion[index]) for index, name in enumerate(CLASSES)
        },
        "pixel_confusion_percent_by_predicted_class": {
            name: float(100.0 * confusion[index] / support) if support else None
            for index, name in enumerate(CLASSES)
        },
        "majority_predicted_class_by_component": {
            name: int(majority_predicted_as[index])
            for index, name in enumerate(CLASSES)
        },
        "component_recall_sum": float(np.sum(recalls)),
        "mean_component_recall_percent": float(100.0 * np.mean(recalls)) if recalls else None,
        "component_recall_lt_50_percent": float(100.0 * majority_miss / component_count)
        if component_count else None,
        "component_recall_eq_0_percent": float(100.0 * full_miss / component_count)
        if component_count else None,
    }


def run(root: Path) -> dict:
    masks_by_fold = m396.mask_files(root)
    arms = {
        "model392_seed1": {class_id: [] for class_id in TARGETS},
        "model397_fixed_blend": {class_id: [] for class_id in TARGETS},
    }
    input_hashes: dict[str, str] = {}
    image_count = 0
    groups: set[str] = set()
    for fold, masks in masks_by_fold.items():
        ids = set(masks)
        paths = {
            "b4": root / B4_ROOT / f"fold{fold}_oof/probabilities.npz",
            "hrnet_seed1": root / MODEL392_ROOT / f"seed1/fold{fold}_oof/valid_probabilities.npz",
            "hrnet_seed2": root / MODEL392_ROOT / f"seed2/fold{fold}_oof/valid_probabilities.npz",
        }
        archives = {name: np.load(path, allow_pickle=False) for name, path in paths.items()}
        try:
            if any(set(archive.files) != ids for archive in archives.values()):
                raise ValueError(f"OOF ID inventory mismatch for fold {fold}")
            for name, path in paths.items():
                input_hashes[str(path.relative_to(root))] = sha256(path)
            for image_id, mask_path in sorted(masks.items()):
                match = GROUP_RE.match(image_id)
                if match is None:
                    raise ValueError(f"cannot parse specimen group from {image_id}")
                groups.add(match.group(1))
                input_hashes[str(mask_path.relative_to(root))] = sha256(mask_path)
                truth = m396.mask_to_ids(mask_path)
                p_b4 = np.asarray(archives["b4"][image_id], dtype=np.float32)
                p_h1 = np.asarray(archives["hrnet_seed1"][image_id], dtype=np.float32)
                p_h2 = np.asarray(archives["hrnet_seed2"][image_id], dtype=np.float32)
                if any(p.shape != (4, *truth.shape) or not np.isfinite(p).all()
                       for p in (p_b4, p_h1, p_h2)):
                    raise ValueError(f"invalid probability tensor for {image_id}")
                predictions = {
                    "model392_seed1": p_h1.argmax(axis=0),
                    "model397_fixed_blend": (0.50 * p_b4 + 0.25 * p_h1 + 0.25 * p_h2).argmax(axis=0),
                }
                for arm, prediction in predictions.items():
                    for class_id in TARGETS:
                        arms[arm][class_id].append(
                            summarize_component_confusion(truth, prediction, class_id)
                        )
                image_count += 1
        finally:
            for archive in archives.values():
                archive.close()

    summary = {}
    for arm, classes in arms.items():
        summary[arm] = {}
        for class_id, image_rows in classes.items():
            pooled = {
                "components": sum(row["components"] for row in image_rows),
                "support_pixels": sum(row["support_pixels"] for row in image_rows),
            }
            confusion = {
                name: sum(row["pixel_confusion_by_predicted_class"][name] for row in image_rows)
                for name in CLASSES
            }
            majority = {
                name: sum(row["majority_predicted_class_by_component"][name] for row in image_rows)
                for name in CLASSES
            }
            n = pooled["components"]
            pooled.update({
                "pixel_confusion_by_predicted_class": confusion,
                "pixel_confusion_percent_by_predicted_class": {
                    name: 100.0 * confusion[name] / pooled["support_pixels"]
                    if pooled["support_pixels"] else None for name in CLASSES
                },
                "majority_predicted_class_by_component": majority,
                "mean_component_recall_percent": (
                    100.0 * sum(row["component_recall_sum"] for row in image_rows) / n if n else None
                ),
                "component_recall_lt_50_percent": 100.0 * sum(
                    row["components"] * (row["component_recall_lt_50_percent"] or 0.0) / 100.0
                    for row in image_rows
                ) / n if n else None,
                "component_recall_eq_0_percent": 100.0 * sum(
                    row["components"] * (row["component_recall_eq_0_percent"] or 0.0) / 100.0
                    for row in image_rows
                ) / n if n else None,
            })
            summary[arm][TARGETS[class_id]] = pooled

    return {
        "experiment": "Model404 tiny-component class-confusion audit",
        "prediction": "fixed OOF Model392 seed1 and Model397 0.50 B4 + 0.25 HRNet seed1 + 0.25 HRNet seed2; argmax",
        "scope": "native-grid 8-connected GT components with area 1-63 px, classes Al3Ni/Eutectic Si",
        "metric_scope": "component recall and predicted-class confusion only; no mIoU computation or tuning",
        "images": image_count,
        "groups": len(groups),
        "valid_read": False,
        "test_read": False,
        "input_sha256": input_hashes,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.out}")
    result = run(args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({
        "result": str(args.out),
        "images": result["images"],
        "groups": result["groups"],
        "summary": result["summary"],
        "valid_read": False,
        "test_read": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
