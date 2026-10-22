#!/usr/bin/env python3
"""Descriptive native-size confusion audit for completed Model406 grouped OOF.

This secondary diagnostic does not compute or modify the preregistered mIoU
adoption gate. It reports which predicted class absorbs each tiny GT phase,
paired between the locked Model406 candidate and control.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import label

import model396_fixed_architecture_oof as m396
import model406_component_hardnegative_v3_lock as m406lock


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {1: "Al3Ni", 3: "Eutectic_Si"}
AREA_BINS = {
    "1-3px": (1, 4),
    "4-15px": (4, 16),
    "16-63px": (16, 64),
}
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 40603
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def component_image_stats(
    truth: np.ndarray,
    prediction: np.ndarray,
    class_id: int,
    bins: dict[str, tuple[int, int]] = AREA_BINS,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Summarize native 8-connected GT components for one image and class."""
    labels, component_count = label(truth == class_id, structure=CONNECTIVITY)
    areas = np.bincount(labels.ravel(), minlength=component_count + 1)
    output = {
        name: {
            "gt_components": 0,
            "gt_pixels": 0,
            "confusion_pixels": np.zeros(4, dtype=np.int64),
            "component_recall_sum": 0.0,
            "zero_recall_components": 0,
            "lt50_recall_components": 0,
        }
        for name in bins
    }
    for component_id in range(1, component_count + 1):
        area = int(areas[component_id])
        bin_name = next(
            (name for name, (lower, upper) in bins.items()
             if lower <= area < upper),
            None,
        )
        if bin_name is None:
            continue
        selected = labels == component_id
        predicted_counts = np.bincount(
            prediction[selected].astype(np.int64), minlength=4
        )[:4]
        recall = float(predicted_counts[class_id] / area)
        row = output[bin_name]
        row["gt_components"] += 1
        row["gt_pixels"] += area
        row["confusion_pixels"] += predicted_counts
        row["component_recall_sum"] += recall
        row["zero_recall_components"] += int(recall == 0.0)
        row["lt50_recall_components"] += int(recall < 0.50)
    return output


def _metrics(
    confusion_pixels: np.ndarray,
    gt_pixels: np.ndarray,
    gt_components: np.ndarray,
    component_recall_sum: np.ndarray,
    zero_recall_components: np.ndarray,
    lt50_recall_components: np.ndarray,
) -> dict[str, np.ndarray]:
    confusion_total = confusion_pixels.sum(axis=0)
    pixel_total = int(gt_pixels.sum())
    component_total = int(gt_components.sum())
    return {
        "predicted_class_pixel_percent": (
            100.0 * confusion_total / pixel_total if pixel_total else np.zeros(4)
        ),
        "mean_component_recall_percent": np.asarray(
            100.0 * component_recall_sum.sum() / component_total
            if component_total else 0.0
        ),
        "zero_recall_component_percent": np.asarray(
            100.0 * zero_recall_components.sum() / component_total
            if component_total else 0.0
        ),
        "lt50_recall_component_percent": np.asarray(
            100.0 * lt50_recall_components.sum() / component_total
            if component_total else 0.0
        ),
    }


def _bootstrap_metrics(arrays: dict[str, np.ndarray], weights: np.ndarray) -> dict[str, np.ndarray]:
    confusion = weights @ arrays["confusion_pixels"]
    gt_pixels = weights @ arrays["gt_pixels"]
    components = weights @ arrays["gt_components"]
    recall_sum = weights @ arrays["component_recall_sum"]
    zero = weights @ arrays["zero_recall_components"]
    lt50 = weights @ arrays["lt50_recall_components"]
    pixel_denominator = np.maximum(gt_pixels, 1)
    component_denominator = np.maximum(components, 1)
    return {
        "predicted_class_pixel_percent": 100.0 * confusion / pixel_denominator[:, None],
        "mean_component_recall_percent": 100.0 * recall_sum / component_denominator,
        "zero_recall_component_percent": 100.0 * zero / component_denominator,
        "lt50_recall_component_percent": 100.0 * lt50 / component_denominator,
    }


def _plain(value: np.ndarray | float | int) -> list | float:
    def convert(item):
        if isinstance(item, list):
            return [convert(child) for child in item]
        return float(item)

    return convert(np.asarray(value).tolist())


def _summarize_cell(
    per_image: dict[str, dict[str, np.ndarray]],
    draw_weights: np.ndarray,
    class_names: list[str],
) -> dict:
    arm_metrics = {}
    boot_metrics = {}
    for arm in ("control", "candidate"):
        row = per_image[arm]
        arm_metrics[arm] = _metrics(
            row["confusion_pixels"], row["gt_pixels"], row["gt_components"],
            row["component_recall_sum"], row["zero_recall_components"],
            row["lt50_recall_components"],
        )
        boot_metrics[arm] = _bootstrap_metrics(row, draw_weights)

    deltas = {}
    for metric_name in arm_metrics["candidate"]:
        delta = boot_metrics["candidate"][metric_name] - boot_metrics["control"][metric_name]
        deltas[metric_name] = {
            "candidate_minus_control": _plain(
                arm_metrics["candidate"][metric_name] - arm_metrics["control"][metric_name]
            ),
            "paired_group_bootstrap_95ci": _plain(
                np.quantile(delta, [0.025, 0.975], axis=0)
            ),
        }

    return {
        "control": {
            **{
                name: _plain(value)
                for name, value in arm_metrics["control"].items()
            },
            "predicted_class_order": class_names,
        },
        "candidate": {
            **{
                name: _plain(value)
                for name, value in arm_metrics["candidate"].items()
            },
            "predicted_class_order": class_names,
        },
        "candidate_vs_control": deltas,
    }


def audit(root: Path, candidate_root: Path, control_root: Path) -> dict:
    m406lock.verify_oof(root)
    masks_by_fold = m396.mask_files(root)
    rows: list[dict] = []
    records = {
        arm: {
            target: {
                area_bin: {
                    "gt_pixels": [],
                    "gt_components": [],
                    "confusion_pixels": [],
                    "component_recall_sum": [],
                    "zero_recall_components": [],
                    "lt50_recall_components": [],
                }
                for area_bin in AREA_BINS
            }
            for target in TARGETS.values()
        }
        for arm in ("control", "candidate")
    }
    input_sha256: dict[str, str] = {}

    for fold, mask_map in masks_by_fold.items():
        prediction_paths = {
            "candidate": candidate_root / f"fold{fold}_oof/valid_probabilities.npz",
            "control": control_root / f"fold{fold}_oof/valid_probabilities.npz",
        }
        archives = {
            arm: np.load(path, allow_pickle=False)
            for arm, path in prediction_paths.items()
        }
        try:
            for arm, path in prediction_paths.items():
                input_sha256[path.resolve().relative_to(root.resolve()).as_posix()] = sha256(path)
                if set(archives[arm].files) != set(mask_map):
                    raise ValueError(f"{arm} OOF IDs differ from fixed fold {fold}")
            for image_id, mask_path in sorted(mask_map.items()):
                truth = m396.mask_to_ids(mask_path)
                group_match = m396.GROUP_RE.match(image_id)
                step_match = m396.STEP_RE.search(image_id)
                if group_match is None or step_match is None:
                    raise ValueError(f"cannot parse group/step from {image_id}")
                input_sha256[mask_path.resolve().relative_to(root.resolve()).as_posix()] = sha256(mask_path)
                rows.append({
                    "image": image_id,
                    "group": group_match.group(1),
                    "step": int(step_match.group(1)),
                    "fold": fold,
                })
                for arm in ("control", "candidate"):
                    probability = np.asarray(archives[arm][image_id], dtype=np.float32)
                    m396.class_ious(probability, truth)  # shape/range/normalization check
                    prediction = probability.argmax(axis=0)
                    for class_id, target_name in TARGETS.items():
                        image_stats = component_image_stats(truth, prediction, class_id)
                        for area_bin, stats in image_stats.items():
                            for key, value in stats.items():
                                records[arm][target_name][area_bin][key].append(value)
        finally:
            for archive in archives.values():
                archive.close()

    if (len(rows) != 70 or len({row["image"] for row in rows}) != 70
            or len({row["group"] for row in rows}) != 59):
        raise ValueError("secondary audit must cover exactly 70 images and 59 groups")
    bootstrap_indices = m396.group_bootstrap_indices(
        rows, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED
    )
    draw_weights = np.zeros((BOOTSTRAP_DRAWS, len(rows)), dtype=np.int16)
    for draw, indices in enumerate(bootstrap_indices):
        draw_weights[draw] = np.bincount(indices, minlength=len(rows))

    classes = list(m396.CLASS_NAMES)
    result_classes = {}
    for target_name in TARGETS.values():
        result_classes[target_name] = {}
        for area_bin in AREA_BINS:
            per_image = {}
            arm_summaries = {}
            for arm in ("control", "candidate"):
                per_image[arm] = {
                    key: np.asarray(value, dtype=np.float64)
                    for key, value in records[arm][target_name][area_bin].items()
                }
                arm_summaries[arm] = {
                    "images_with_components": int(np.count_nonzero(
                        per_image[arm]["gt_components"]
                    )),
                    "components": int(per_image[arm]["gt_components"].sum()),
                    "gt_pixels": int(per_image[arm]["gt_pixels"].sum()),
                }
            result_classes[target_name][area_bin] = {
                "control_support": arm_summaries["control"],
                "candidate_support": arm_summaries["candidate"],
                **_summarize_cell(per_image, draw_weights, classes),
            }

    return {
        "experiment": "Model406 native tiny-GT-component confusion secondary audit",
        "scope": "fixed grouped train OOF only: 70 images / 59 specimen groups",
        "metric": "native 8-connected GT components; per-pixel predicted-class confusion and component recall",
        "area_bins_native_pixels": {name: list(bounds) for name, bounds in AREA_BINS.items()},
        "target_classes": TARGETS,
        "paired_group_bootstrap_draws": BOOTSTRAP_DRAWS,
        "paired_group_bootstrap_seed": BOOTSTRAP_SEED,
        "descriptive_only_no_miou_gate_or_threshold_selection": True,
        "official_valid_read": False,
        "test_read": False,
        "source_lock_sha256": sha256(root / m406lock.SOURCE_LOCK),
        "oof_lock_sha256": sha256(root / m406lock.OOF_LOCK),
        "input_sha256": input_sha256,
        "classes": result_classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite secondary audit: {out}")
    result = audit(root, args.candidate_root.resolve(), args.control_root.resolve())
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "result": str(out),
        "images": 70,
        "specimen_groups": 59,
        "area_bins": list(AREA_BINS),
        "descriptive_only": True,
        "official_valid_read": False,
        "test_read": False,
    }, ensure_ascii=False, indent=2))


def self_test() -> None:
    truth = np.zeros((16, 16), dtype=np.uint8)
    truth[2:4, 2:4] = 1  # 4px component
    truth[8:9, 8:9] = 1  # 1px component
    truth[8, 8] = 1
    pred = truth.copy()
    pred[2:4, 2:4] = 3
    pred[8, 8] = 3
    stats = component_image_stats(truth, pred, class_id=1)
    assert stats["1-3px"]["gt_components"] == 1
    assert stats["1-3px"]["confusion_pixels"].tolist() == [0, 0, 0, 1]
    assert stats["1-3px"]["zero_recall_components"] == 1
    assert stats["4-15px"]["gt_components"] == 1
    assert stats["4-15px"]["confusion_pixels"].tolist() == [0, 0, 0, 4]
    empty_row = {
        "gt_pixels": np.asarray([4.0, 1.0]),
        "gt_components": np.asarray([1.0, 1.0]),
        "confusion_pixels": np.asarray([[0, 0, 0, 4], [0, 1, 0, 0]], dtype=np.float64),
        "component_recall_sum": np.asarray([0.0, 1.0]),
        "zero_recall_components": np.asarray([1.0, 0.0]),
        "lt50_recall_components": np.asarray([1.0, 0.0]),
    }
    candidate_row = {
        **empty_row,
        "confusion_pixels": np.asarray([[0, 2, 0, 2], [0, 1, 0, 0]], dtype=np.float64),
        "component_recall_sum": np.asarray([0.5, 1.0]),
        "zero_recall_components": np.asarray([0.0, 0.0]),
        "lt50_recall_components": np.asarray([0.0, 0.0]),
    }
    summary = _summarize_cell(
        {"control": empty_row, "candidate": candidate_row},
        np.asarray([[1, 1], [2, 0]], dtype=np.int16),
        ["Al", "Al3Ni", "Primary Si", "Eutectic Si"],
    )
    recall_delta = summary["candidate_vs_control"]["mean_component_recall_percent"]
    assert recall_delta["candidate_minus_control"] == 25.0
    assert len(recall_delta["paired_group_bootstrap_95ci"]) == 2
    print("Model406 secondary confusion-audit smoke passed.")


if __name__ == "__main__":
    if os.environ.get("MODEL406_AUDIT_SELF_TEST") == "1":
        self_test()
    else:
        main()
