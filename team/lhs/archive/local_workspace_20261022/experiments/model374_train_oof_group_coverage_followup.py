#!/usr/bin/env python3
"""Post-hoc descriptive leverage, class, and hypothetical headroom checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rho(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def analyze(coverage: dict, oof: dict) -> dict:
    class_scores: dict[tuple[int, str], list[float]] = {}
    for row in oof["per_image"]:
        key = (int(row["fold"]), row["specimen_group"])
        class_scores.setdefault(key, []).append(row["strict_class_iou_empty0"])
    class_means = {
        key: np.mean(np.asarray(values, dtype=np.float64), axis=0)
        for key, values in class_scores.items()
    }

    folds = {}
    overall_current = []
    total_weight = 0
    total_top_quartile_delta = 0.0
    for fold_key, fold in coverage["folds"].items():
        rows = fold["rows"]
        distance = np.asarray([row["appearance_distance"] for row in rows], dtype=np.float64)
        score = np.asarray([row["image_mean_miou_empty0"] for row in rows], dtype=np.float64)
        order = np.argsort(distance, kind="stable")
        top_two = set(order[-2:].tolist())
        keep = np.asarray([i for i in range(len(rows)) if i not in top_two], dtype=np.int64)
        fold_classes = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            values = np.asarray([
                class_means[(int(fold_key), row["group"])][class_id]
                for row in rows
            ], dtype=np.float64)
            fold_classes[class_name] = rho(distance, values)

        q75 = float(np.quantile(distance, 0.75))
        far = distance >= q75
        near_median = float(np.median(score[~far])) if np.any(~far) else float(np.median(score))
        image_weights = np.asarray([row["images"] for row in rows], dtype=np.float64)
        current_image_mean = float(np.average(score, weights=image_weights))
        hypothetical = score.copy()
        hypothetical[far] = np.maximum(hypothetical[far], near_median)
        hypothetical_image_mean = float(np.average(hypothetical, weights=image_weights))
        total_top_quartile_delta += (hypothetical_image_mean - current_image_mean) * int(image_weights.sum())
        total_weight += int(image_weights.sum())
        overall_current.extend([row["image_mean_miou_empty0"] for row in rows for _ in range(row["images"])])

        folds[fold_key] = {
            "heldout_groups": len(rows),
            "appearance_distance_quantiles": {
                key: float(np.quantile(distance, quantile))
                for key, quantile in (("min", 0.0), ("p25", 0.25), ("median", 0.5),
                                      ("p75", 0.75), ("p90", 0.9), ("max", 1.0))
            },
            "full_fold_spearman_distance_vs_group_miou": rho(distance, score),
            "spearman_after_dropping_two_farthest_groups_descriptive": rho(distance[keep], score[keep]),
            "farthest_groups": [rows[i]["group"] for i in order[-2:][::-1]],
            "class_iou_spearman_distance_vs_group_class_iou": fold_classes,
            "top_distance_quartile_scenario": {
                "groups": int(far.sum()),
                "images": int(image_weights[far].sum()),
                "median_score_of_nonfar_groups": near_median,
                "current_fold_image_mean": current_image_mean,
                "hypothetical_if_far_groups_reached_nonfar_median": hypothetical_image_mean,
                "hypothetical_gain_percentage_points": (hypothetical_image_mean - current_image_mean) * 100,
            },
        }

    actual = float(np.mean(overall_current))
    scenario_miou = actual + total_top_quartile_delta / total_weight
    return {
        "scope": "post-hoc exploratory analysis of Model374 train-only grouped OOF; no valid/test inputs or scores",
        "metric": "per-image four-class mIoU, empty union=0",
        "base_image_mean_miou_empty0": actual,
        "folds": folds,
        "all_fold_top_distance_quartile_scenario": {
            "hypothetical_image_mean_miou": scenario_miou,
            "gain_percentage_points": (scenario_miou - actual) * 100,
            "assumption": "within each fold, every group in the top appearance-distance quartile is raised to at least the non-far groups' median group score; descriptive scenario, not an achievable model estimate or causal upper bound",
        },
        "interpretation_guard": (
            "All checks are descriptive and selection-conditional. Fold training sets overlap, groups per fold are small, "
            "and no association or hypothetical headroom is an automatic GPU training gate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-json", type=Path, required=True)
    parser.add_argument("--oof-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coverage = json.loads(args.coverage_json.read_text(encoding="utf-8"))
    oof = json.loads(args.oof_json.read_text(encoding="utf-8"))
    result = analyze(coverage, oof)
    result["source_sha256"] = {
        "coverage_audit": sha256(args.coverage_json),
        "model374_oof_summary": sha256(args.oof_json),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"base_image_mean_miou_empty0": result["base_image_mean_miou_empty0"],
                      "all_fold_top_distance_quartile_scenario": result["all_fold_top_distance_quartile_scenario"],
                      "folds": result["folds"], "interpretation_guard": result["interpretation_guard"]},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
