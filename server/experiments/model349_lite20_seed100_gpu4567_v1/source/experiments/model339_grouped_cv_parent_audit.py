#!/usr/bin/env python3
"""Compare Model339's group-held-out route selection directly with Model338."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


CONFIG_FIELDS = (
    "primary_source", "primary_gamma", "al3ni_source", "al3ni_gamma",
    "al_source", "al_gamma", "eutectic_source", "eutectic_gamma",
    "prior_scale", "beta_scale", "al_delta", "eut_delta",
    "gaussian_blend", "eut_edge_blend", "cleanup_threshold",
)


def specimen_group(name: str) -> str:
    canonical = name[:-6] if name.endswith("_image") else name
    match = re.match(r"(A\d+)_", Path(canonical).name)
    if not match:
        raise ValueError(f"cannot parse specimen group from image key: {name}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    results_path = args.results or args.root / "outputs/model339_joint_gamma_route_search/results.json"
    output = args.output or results_path.with_name("cv_vs_model338.json")
    result = json.loads(results_path.read_text(encoding="utf-8"))
    records = result["records"]
    config = result["fixed_experts_from_model338_strict"]

    parent = next((record for record in records
                   if all(record.get(key) == config.get(key) for key in CONFIG_FIELDS)), None)
    if parent is None:
        raise ValueError("Model338 strict route is missing from the Model339 joint grid")
    # Model339 used Model195 archive key order when constructing its per-image vectors.
    with np.load(args.root / "outputs/model195_native_resume/valid_probs.npz") as archive:
        names = list(archive.files)
    if len(names) != len(parent["image_scores"]):
        raise ValueError("image-key count does not match saved route score vectors")

    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(specimen_group(name), []).append(index)

    score_field = "image_scores"
    reference_scores = np.asarray(parent[score_field], dtype=np.float64)
    delta = np.zeros(len(names), dtype=np.float64)
    folds = []
    for held_group, held_indices in groups.items():
        train_indices = [index for group, indices in groups.items() if group != held_group
                         for index in indices]
        selected = max(records, key=lambda row: float(
            np.asarray(row[score_field], dtype=np.float64)[train_indices].mean()))
        selected_scores = np.asarray(selected[score_field], dtype=np.float64)
        delta[held_indices] = selected_scores[held_indices] - reference_scores[held_indices]
        folds.append({
            "heldout_specimen": held_group,
            "selected_route": selected["candidate"],
            "delta_pp": float(delta[held_indices].mean() * 100.0),
        })

    rng = np.random.default_rng(341)
    group_names = list(groups)
    bootstrap = np.empty(10000, dtype=np.float64)
    for draw in range(len(bootstrap)):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [index for group in sampled for index in groups[str(group)]]
        bootstrap[draw] = delta[indices].mean() * 100.0

    raw_delta_pp = (result["best_image_empty_zero"]["image_mean_miou"]
                    - parent["image_mean_miou"]) * 100.0
    summary = {
        "comparison": "Model339 leave-one-specimen-group-out route selection vs fixed Model338 strict route",
        "metric": "per-image mean of four class IoUs; empty union=0",
        "parent_model338_route": parent["candidate"],
        "parent_model338_valid_miou": parent["image_mean_miou"],
        "model339_raw_best_route": result["best_image_empty_zero"]["candidate"],
        "model339_raw_best_valid_miou": result["best_image_empty_zero"]["image_mean_miou"],
        "raw_best_delta_pp": raw_delta_pp,
        "grouped_cv_delta_pp": float(delta.mean() * 100.0),
        "positive_image_count": int((delta > 0).sum()),
        "image_count": len(names),
        "specimen_group_count": len(groups),
        "group_bootstrap_95ci_delta_pp": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "route_selection_counts": dict(Counter(row["selected_route"] for row in folds)),
        "folds": folds,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
