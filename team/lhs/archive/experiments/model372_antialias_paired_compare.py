#!/usr/bin/env python3
"""Paired image/group comparison of scale-3 flip4 antialias inference."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

CLASS_NAMES = ("Al", "Al3Ni", "Primary_Si", "Eutectic_Si")


def load_rows(path: Path) -> dict[str, dict[str, object]]:
    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            name = row["image_id"]
            if name in rows:
                raise ValueError(f"duplicate image {name} in {path}")
            match = re.match(r"(A\d+)_", name)
            if not match:
                raise ValueError(f"cannot parse specimen group for {name}")
            rows[name] = {
                "group": match.group(1),
                "image_miou_empty_zero": float(row["image_miou_empty_zero"]),
                "class_iou_empty_zero": {
                    class_name: float(row[f"iou_{class_name}"])
                    if row[f"iou_{class_name}"] not in ("", "nan", "NaN") else 0.0
                    for class_name in CLASS_NAMES
                },
            }
    return rows


def bootstrap_group_delta(delta: np.ndarray, groups: list[str], repetitions: int,
                          seed: int) -> list[float]:
    unique = np.asarray(sorted(set(groups)))
    group_rows = {group: np.flatnonzero(np.asarray(groups) == group)
                  for group in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled])
        draws[index] = delta[rows].mean()
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=372)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    baseline = load_rows(args.baseline_csv)
    candidate = load_rows(args.candidate_csv)
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline/candidate image keys differ")
    names = sorted(baseline)
    if len(names) != 20:
        raise ValueError(f"expected 20 valid images, found {len(names)}")
    groups = [str(baseline[name]["group"]) for name in names]
    if len(set(groups)) != 19 or any(
            baseline[name]["group"] != candidate[name]["group"] for name in names):
        raise ValueError("expected matching 19-group validation metadata")

    baseline_scores = np.asarray([
        baseline[name]["image_miou_empty_zero"] for name in names], dtype=np.float64)
    candidate_scores = np.asarray([
        candidate[name]["image_miou_empty_zero"] for name in names], dtype=np.float64)
    score_delta = candidate_scores - baseline_scores
    class_results = {}
    for class_name in CLASS_NAMES:
        baseline_values = np.asarray([
            baseline[name]["class_iou_empty_zero"][class_name] for name in names],
            dtype=np.float64)
        candidate_values = np.asarray([
            candidate[name]["class_iou_empty_zero"][class_name] for name in names],
            dtype=np.float64)
        class_results[class_name] = {
            "baseline_mean_iou_empty0": float(baseline_values.mean()),
            "candidate_mean_iou_empty0": float(candidate_values.mean()),
            "delta_pp": float((candidate_values - baseline_values).mean() * 100.0),
        }

    result = {
        "audit": "paired scale-3 flip4 baseline vs probability-downsample antialias",
        "metric": "20-image mean of four class IoUs; empty union=0",
        "baseline_csv": str(args.baseline_csv),
        "candidate_csv": str(args.candidate_csv),
        "images": len(names),
        "specimen_groups": len(set(groups)),
        "baseline_image_mean_miou_empty0": float(baseline_scores.mean()),
        "candidate_image_mean_miou_empty0": float(candidate_scores.mean()),
        "paired_delta_pp": float(score_delta.mean() * 100.0),
        "group_bootstrap_95ci_delta_pp": bootstrap_group_delta(
            score_delta * 100.0, groups, args.bootstrap, args.seed),
        "bootstrap_repetitions": args.bootstrap,
        "seed": args.seed,
        "class_mean_iou_empty0": class_results,
        "per_image": [{"image": name, "group": groups[index],
                       "baseline_miou_empty0": float(baseline_scores[index]),
                       "candidate_miou_empty0": float(candidate_scores[index]),
                       "delta_pp": float(score_delta[index] * 100.0)}
                      for index, name in enumerate(names)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "baseline_image_mean_miou_empty0", "candidate_image_mean_miou_empty0",
        "paired_delta_pp", "group_bootstrap_95ci_delta_pp",
        "class_mean_iou_empty0")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
