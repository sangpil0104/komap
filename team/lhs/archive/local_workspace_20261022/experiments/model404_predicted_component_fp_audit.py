#!/usr/bin/env python3
"""Read-only FP-component diagnostic for fixed Model404 train-OOF maps.

This supplements (but never changes) the locked Model404 adoption scorer. It
uses only the 70 grouped train-OOF images and reports pixel FP burden plus
predicted connected components that have no same-class GT overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_SCRIPTS = ROOT / "model397_reproduction_bundle" / "scripts"
sys.path.insert(0, str(BUNDLE_SCRIPTS))
sys.path.insert(0, str(ROOT / "experiments"))

from score_model397_oof import FOLD_FILE, bootstrap_indices, group_and_step  # noqa: E402
import score_model404_component_recall_oof as shared  # noqa: E402


TARGET_CLASSES = {1: "Al3Ni", 3: "Eutectic Si"}
AREA_BINS = {"tiny_1_to_lt64": (1, 64), "small_64_to_lt256": (64, 256), "large_ge256": (256, None)}
ARMS = ("candidate", "control", "model392_parent")


def image_metrics(probability: np.ndarray, truth: np.ndarray) -> dict[str, dict[str, float]]:
    probability = np.asarray(probability, dtype=np.float32)
    if probability.shape != (4, *truth.shape) or not np.isfinite(probability).all():
        raise ValueError(f"invalid probability map {probability.shape}, GT={truth.shape}")
    if (probability.min() < -1e-4 or probability.max() > 1.0001
            or np.max(np.abs(probability.sum(axis=0) - 1.0)) > 0.03):
        raise ValueError("probability map is outside [0,1] or channels are not normalized")
    prediction = probability.argmax(axis=0)
    result: dict[str, dict[str, float]] = {}
    for class_id, class_name in TARGET_CLASSES.items():
        pred_class = prediction == class_id
        gt_class = truth == class_id
        item = {
            "predicted_pixels": float(pred_class.sum()),
            "false_positive_pixels": float((pred_class & ~gt_class).sum()),
            "false_negative_pixels": float((~pred_class & gt_class).sum()),
        }
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.ascontiguousarray(pred_class, dtype=np.uint8),
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        overlaps = np.bincount(
            labels.reshape(-1), weights=gt_class.reshape(-1), minlength=count
        ).astype(np.int64, copy=False)
        for bin_name in AREA_BINS:
            item[f"pred_components_{bin_name}"] = 0.0
            item[f"unmatched_components_{bin_name}"] = 0.0
            item[f"majority_fp_components_{bin_name}"] = 0.0
            item[f"unmatched_component_area_{bin_name}"] = 0.0
            item[f"component_fp_pixels_{bin_name}"] = 0.0
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            bin_name = next(
                (name for name, (lo, hi) in AREA_BINS.items()
                 if area >= lo and (hi is None or area < hi)),
                None,
            )
            if bin_name is None:
                continue
            overlap = int(overlaps[component_id])
            item[f"pred_components_{bin_name}"] += 1.0
            item[f"component_fp_pixels_{bin_name}"] += float(area - overlap)
            if overlap == 0:
                item[f"unmatched_components_{bin_name}"] += 1.0
                item[f"unmatched_component_area_{bin_name}"] += float(area)
            if 2 * overlap < area:
                item[f"majority_fp_components_{bin_name}"] += 1.0
        result[class_name] = item
    return result


def audit(data_root: Path, candidate_root: Path, control_root: Path,
          baseline_root: Path, draws: int = 20_000, seed: int = 4041) -> dict:
    data_root = data_root.resolve()
    image_paths = sorted((data_root / "train/images").glob("*_image.png"))
    if len(image_paths) != 70:
        raise ValueError(f"expected 70 train images, found {len(image_paths)}")
    ids = [path.name.removesuffix("_image.png") for path in image_paths]
    metadata = {image_id: group_and_step(image_id) for image_id in ids}
    if len({group for group, _ in metadata.values()}) != 59:
        raise ValueError("expected 59 specimen groups")

    fold_manifest = json.loads(FOLD_FILE.read_text(encoding="utf-8"))
    fold_for_group: dict[str, int] = {}
    for fold_data in fold_manifest["folds"]:
        fold = int(fold_data["fold"])
        for group in fold_data["heldout_groups"]:
            if group in fold_for_group:
                raise ValueError(f"specimen group assigned to multiple folds: {group}")
            fold_for_group[group] = fold
    if set(fold_for_group) != {group for group, _ in metadata.values()}:
        raise ValueError("fixed fold manifest does not cover all train groups")

    roots = {
        "candidate": candidate_root.resolve(),
        "control": control_root.resolve(),
        "model392_parent": baseline_root.resolve() / "seed1",
    }
    rows: list[dict] = []
    arrays: dict[str, dict[str, list[float]]] = {
        arm: {} for arm in ARMS
    }
    for fold in range(1, 5):
        expected = {
            image_id for image_id in ids
            if fold_for_group[metadata[image_id][0]] == fold
        }
        archives = {
            arm: shared.load_archive(
                root / f"fold{fold}_oof/valid_probabilities.npz", expected
            )
            for arm, root in roots.items()
        }
        for image_id in sorted(expected):
            truth = shared.read_mask(data_root / "train/masks" / f"{image_id}_mask.png")
            group, step = metadata[image_id]
            row = {"fold": fold, "group": group, "step": step, "image_id": image_id}
            for arm in ARMS:
                values = image_metrics(archives[arm][image_id], truth)
                row[arm] = values
                for class_name, metrics in values.items():
                    for metric_name, value in metrics.items():
                        key = f"{class_name}/{metric_name}"
                        arrays[arm].setdefault(key, []).append(float(value))
            rows.append(row)
    if len(rows) != 70 or len({row["image_id"] for row in rows}) != 70:
        raise ValueError("OOF maps must cover each of the 70 train images exactly once")

    samples = bootstrap_indices(rows, draws=draws, seed=seed)
    summaries: dict[str, dict[str, float]] = {}
    comparisons: dict[str, dict[str, dict[str, float]]] = {}
    keys = sorted(arrays["candidate"])
    for arm in ARMS:
        summaries[arm] = {
            key: float(np.mean(arrays[arm][key])) for key in keys
        }
    for reference in ("control", "model392_parent"):
        comparisons[f"candidate_vs_{reference}"] = {}
        for key in keys:
            delta = np.asarray(arrays["candidate"][key]) - np.asarray(arrays[reference][key])
            boot = np.asarray([delta[index].mean() for index in samples], dtype=np.float64)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            by_step: dict[str, dict[str, float | int | None]] = {}
            for step_name, row_mask in (
                ("step4", np.asarray([row["step"] == 4 for row in rows], dtype=bool)),
                ("non_step4", np.asarray([row["step"] != 4 for row in rows], dtype=bool)),
            ):
                selected = np.flatnonzero(row_mask)
                if selected.size == 0:
                    by_step[step_name] = {
                        "images": 0,
                        "groups": 0,
                        "per_image_mean_delta": None,
                        "paired_group_bootstrap_95ci": None,
                    }
                    continue
                subgroup_boot = []
                for sample in samples:
                    selected_sample = sample[row_mask[sample]]
                    if selected_sample.size:
                        subgroup_boot.append(float(delta[selected_sample].mean()))
                if len(subgroup_boot) < 100:
                    raise ValueError(f"too few nonempty bootstrap samples for {step_name}")
                step_lo, step_hi = np.quantile(np.asarray(subgroup_boot), [0.025, 0.975])
                by_step[step_name] = {
                    "images": int(selected.size),
                    "groups": len({rows[int(index)]["group"] for index in selected}),
                    "per_image_mean_delta": float(delta[selected].mean()),
                    "paired_group_bootstrap_95ci": [float(step_lo), float(step_hi)],
                }
            comparisons[f"candidate_vs_{reference}"][key] = {
                "per_image_mean_delta": float(delta.mean()),
                "paired_group_bootstrap_95ci": [float(lo), float(hi)],
                "by_step": by_step,
            }
    return {
        "experiment": "Model404 predicted-component FP diagnostic",
        "scope": "fixed grouped train-OOF only: 70 images, 59 specimen groups",
        "metric": "argmax semantic prediction; 8-connected predicted components",
        "unmatched_definition": "predicted class component with zero overlap with same-class GT",
        "majority_fp_definition": "less than 50% of predicted component pixels overlap same-class GT",
        "area_bins_native_pixels": AREA_BINS,
        "arms_per_image_mean": summaries,
        "paired_candidate_deltas": comparisons,
        "draws": draws,
        "valid_read": False,
        "test_read": False,
        "adoption_gate": False,
        "interpretation": (
            "Descriptive diagnostic only; not an adoption gate and not a validation/test estimate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--control-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=4041)
    args = parser.parse_args()
    if args.draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite prior audit: {args.out}")
    result = audit(
        args.data_root, args.candidate_root, args.control_root,
        args.baseline_root, draws=args.draws, seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"out": str(args.out), "scope": result["scope"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
