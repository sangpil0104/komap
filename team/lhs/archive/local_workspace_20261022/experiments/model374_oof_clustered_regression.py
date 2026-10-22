#!/usr/bin/env python3
"""Descriptive OOF regression with specimen-clustered uncertainty.

Fits image-wise mIoU and Eutectic IoU separately against GT boundary density,
step, native resolution, and held-out fold. Uses only Model374 train OOF.
This is association analysis, not causal inference or test-score prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def clustered_ols(y: np.ndarray, x: np.ndarray, groups: np.ndarray,
                  names: list[str]) -> dict:
    n, k = x.shape
    unique_groups = np.unique(groups)
    g = len(unique_groups)
    if np.linalg.matrix_rank(x) != k or n <= k or g <= 1:
        raise ValueError(f"rank/size problem: n={n}, k={k}, groups={g}")
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((k, k), dtype=np.float64)
    for group in unique_groups:
        selected = groups == group
        score = x[selected].T @ residual[selected]
        meat += np.outer(score, score)
    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    critical = float(t.ppf(0.975, df=g - 1))
    rows = {}
    for index, name in enumerate(names):
        estimate = float(beta[index])
        se = float(standard_error[index])
        stat = estimate / se if se > 0 else None
        p_value = float(2 * t.sf(abs(stat), df=g - 1)) if stat is not None else None
        rows[name] = {
            "estimate": estimate,
            "cluster_robust_se": se,
            "ci95": [estimate - critical * se, estimate + critical * se],
            "t": stat,
            "p_two_sided_descriptive": p_value,
        }
    return {
        "n_images": int(n),
        "n_specimen_groups": int(g),
        "r_squared": float(1 - (residual @ residual) / np.sum((y - y.mean()) ** 2)),
        "design_rank": int(np.linalg.matrix_rank(x)),
        "design_columns": names,
        "coefficients": rows,
    }


def analyze(path: Path) -> dict:
    source = json.loads(path.read_text(encoding="utf-8"))
    records = source["per_image"]
    if len(records) != 70 or len({row["specimen_group"] for row in records}) != 59:
        raise ValueError("expected exactly 70 train OOF images from 59 specimen groups")
    resolution_levels = ["1230x875", "1278x909", "1440x1024", "675x480"]
    step_levels = ["step1", "step2", "step3", "step4"]
    fold_levels = [1, 2, 3, 4]
    names = ["intercept", "gt_boundary_fraction"]
    columns = [np.ones(len(records)),
               np.asarray([row["gt_boundary_band_fraction"] for row in records], dtype=np.float64)]
    for step in step_levels[1:]:
        names.append(f"{step}_vs_step1")
        columns.append(np.asarray([row["step"] == step for row in records], dtype=np.float64))
    for resolution in resolution_levels[1:]:
        names.append(f"resolution_{resolution}_vs_1230x875")
        columns.append(np.asarray([row["resolution"] == resolution for row in records], dtype=np.float64))
    for fold in fold_levels[1:]:
        names.append(f"fold{fold}_vs_fold1")
        columns.append(np.asarray([row["fold"] == fold for row in records], dtype=np.float64))
    design = np.column_stack(columns)
    groups = np.asarray([row["specimen_group"] for row in records])
    outcomes = {
        "image_miou_empty0": np.asarray([row["image_miou_empty0"] for row in records]),
        "eutectic_iou_empty0": np.asarray([row["class_iou_empty0"][3] for row in records]),
    }
    return {
        "model": 374,
        "scope": "70 training images, exactly-once grouped OOF; valid/test excluded",
        "formula": "OLS with CR1 specimen-group-clustered covariance; categorical references step1, 1230x875, fold1",
        "interpretation_guard": (
            "Observational associations only. Boundary density, resolution, step, and specimen "
            "morphology are entangled; intervals do not establish causality or predict test mIoU."
        ),
        "outcomes": {
            name: clustered_ols(values, design, groups, names)
            for name, values in outcomes.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stratified-oof-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.stratified_oof_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
