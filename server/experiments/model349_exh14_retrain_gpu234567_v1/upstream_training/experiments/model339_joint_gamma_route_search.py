#!/usr/bin/env python3
"""Jointly refine all four Model338 class-wise blend gammas.

The selected experts stay fixed to Model338's strict empty-union=0 route.
This evaluates their gamma interactions together (rather than one class at a
time), then reports leave-one-specimen-group-out selection against Model270.
CPU-only: cached validation probabilities are used; no model inference/training.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import model335_exhaustive_b4_route_search as route


GAMMA_GRID = {
    "al3ni_gamma": (0.3, 0.4, 0.5, 0.6, 0.7),
    "primary_gamma": (0.7, 0.8, 0.9, 1.0),
    "al_gamma": (0.3, 0.4, 0.5, 0.6, 0.7),
    "eutectic_gamma": (0.7, 0.8, 0.9, 1.0),
}
FIELDS = (
    "primary_source", "primary_gamma", "al3ni_source", "al3ni_gamma",
    "al_source", "al_gamma", "eutectic_source", "eutectic_gamma",
    "prior_scale", "beta_scale", "al_delta", "eut_delta",
    "gaussian_blend", "eut_edge_blend", "cleanup_threshold",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/komap_model1"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-routes", type=int, default=0,
                        help="prefix limit for a smoke check; 0 evaluates the full grid")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root
    output = args.output or root / "outputs/model339_joint_gamma_route_search/results.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")

    model335 = json.loads((root / "outputs/model335_exhaustive_b4_routes/results.json")
                          .read_text(encoding="utf-8"))
    model338 = json.loads((root / "outputs/model338_fine_gamma_route_search/results.json")
                          .read_text(encoding="utf-8"))
    entries_path = root / "outputs/model183_metric_calibration/entries.pkl"
    with entries_path.open("rb") as stream:
        entries = pickle.load(stream)

    source_paths = {
        "195": (root / "outputs/model195_native_resume/valid_probs.npz", False),
        "217": (root / "outputs/model220_native_tent_extended/valid_probs_06.npz", False),
        "269": (root / "outputs/model269_fullstats_1x_extended/valid_probs.npz", False),
        "322": (root / "outputs/model322_team_b4_group_oof/valid_ensemble/probabilities.npz", True),
        "326": (root / "outputs/model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz", True),
    }
    archives = {}
    names = None
    for source, (path, team_order) in source_paths.items():
        archive_names, values = route.load_archive(path, team_order=team_order)
        if names is None:
            names = archive_names
        elif set(archive_names) != set(names):
            raise ValueError(f"validation image-key mismatch for Model{source}")
        archives[source] = values
    if names is None or len(entries) != len(names):
        raise ValueError("cached entries and probability image counts differ")

    archives["b4mean"] = {
        name: 0.5 * (archives["322"][name] + archives["326"][name])
        for name in names
    }
    rows = []
    for index, name in enumerate(names):
        item = entries[index]
        rows.append({
            "index": index,
            "item": item,
            "base": route.undo_model181_primary_blend(item),
            "sources": {
                "base": None,
                "181": item["extra"].astype("float32"),
                **{source: values[name] for source, values in archives.items()},
            },
        })

    baseline = model335["baseline"]
    strict_parent = model338["best_image_empty_zero"]
    baseline_setting = {key: baseline[key] for key in FIELDS if key in baseline}
    baseline_setting.update(stage="model270_reference", candidate="Model270 reference")
    settings = [baseline_setting]
    grid_keys = tuple(GAMMA_GRID)
    combinations = list(itertools.product(*(GAMMA_GRID[key] for key in grid_keys)))
    for combination in combinations:
        setting = {key: strict_parent[key] for key in FIELDS if key in strict_parent}
        for key, gamma in zip(grid_keys, combination):
            setting[key] = gamma
        setting["stage"] = "model339_joint_gamma"
        setting["candidate"] = "joint gamma " + ", ".join(
            f"{key}={value:.1f}" for key, value in zip(grid_keys, combination))
        settings.append(setting)

    if len(settings) != 401:
        raise AssertionError(f"expected baseline + 400 combinations, got {len(settings)}")
    if args.max_routes:
        if args.max_routes < 2 or args.max_routes > len(settings):
            raise ValueError("--max-routes must be between 2 and 401")
        settings = settings[:args.max_routes]

    records, _ = route.evaluate(rows, settings, names, args.workers)
    strict_best = max(records, key=lambda record: record["image_mean_miou"])
    skip_best = max(records, key=lambda record: record["image_mean_miou_exclude"])
    cv_strict = route.grouped_cv(records, names, "image_scores", seed=339)
    cv_skip = route.grouped_cv(records, names, "image_scores_exclude", seed=340)
    result = {
        "model": 339,
        "metric": model335["metric"],
        "candidate_count": len(records),
        "full_grid_count_including_model270_reference": 401,
        "gamma_grid": GAMMA_GRID,
        "fixed_experts_from_model338_strict": {
            key: strict_parent[key] for key in FIELDS if key in strict_parent
        },
        "best_image_empty_zero": strict_best,
        "best_image_empty_skipped": skip_best,
        "grouped_cv_image_empty_zero": cv_strict,
        "grouped_cv_image_empty_skipped": cv_skip,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidate_count": len(records),
        "strict_best": strict_best["image_mean_miou"],
        "skip_best": skip_best["image_mean_miou_exclude"],
        "grouped_cv_strict_delta_pp": cv_strict["image_mean_delta_pp"],
        "grouped_cv_strict_95ci_delta_pp": cv_strict["group_bootstrap_95ci_delta_pp"],
        "output": str(output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
