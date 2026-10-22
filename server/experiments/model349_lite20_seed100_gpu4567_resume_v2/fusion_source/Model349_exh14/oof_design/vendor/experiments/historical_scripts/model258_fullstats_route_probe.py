#!/usr/bin/env python3
"""Exact class-wise fusion and specimen-group CV for model257."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import default_setting, evaluate


def main(args: argparse.Namespace) -> None:
    branch_id = args.source_id
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    archives = {
        "195": np.load(args.model195),
        "217": np.load(args.tent20),
        "244": np.load(args.model244),
        "249": np.load(args.model249),
        branch_id: np.load(args.source_probs),
    }
    key_orders = {tuple(archive.files) for archive in archives.values()}
    if len(key_orders) != 1:
        raise ValueError("probability archives differ in image keys/order")
    image_names = tuple(archives[branch_id].files)
    values = {
        source: [archive[key].astype(np.float32) for key in archive.files]
        for source, archive in archives.items()
    }
    if any(len(source_values) != len(entries) for source_values in values.values()):
        raise ValueError("probability archive and route cache sizes differ")

    rows = [{
        "index": index,
        "item": item,
        "base": undo_model181_primary_blend(item),
        "sources": {
            "base": None,
            "181": item["extra"].astype(np.float32),
            **{source: source_values[index] for source, source_values in values.items()},
        },
    } for index, item in enumerate(entries)]

    def setting(**changes):
        config = {
            "stage": f"model{branch_id}_full_image_stats_grouped_cv",
            "primary_source": "195", "primary_gamma": 0.75,
            "al3ni_source": "217", "al3ni_gamma": 0.40,
            "eutectic_source": "217", "eutectic_gamma": 0.40,
            "prior_scale": 1.25, "beta_scale": 0.80,
            "al_delta": 0.0, "eut_delta": 0.15,
            "gaussian_blend": 0.50, "eut_edge_blend": 0.25,
            "cleanup_threshold": 128,
        }
        config.update(changes)
        return default_setting(**config)

    reference = setting(stage=f"model{branch_id}_model225_reference")
    settings = [reference]
    classes = ("al3ni", "eutectic", "primary")
    for subset in range(1, 1 << len(classes)):
        for gamma in args.gammas:
            changes = {}
            for bit, class_name in enumerate(classes):
                if subset & (1 << bit):
                    changes[f"{class_name}_source"] = branch_id
                    changes[f"{class_name}_gamma"] = gamma
            settings.append(setting(**changes))

    exact = evaluate(rows, settings, exact=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".exact.json").write_text(
        json.dumps({"candidate_count": len(settings), "records": exact},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    reference_scores = next(row["image_scores"] for row in exact
                            if row["stage"] == f"model{branch_id}_model225_reference")

    groups: dict[str, list[int]] = {}
    for index, name in enumerate(image_names):
        match = re.match(r"(A\d+)_", Path(name).name)
        group = match.group(1) if match else Path(name).stem.split("_")[0]
        groups.setdefault(group, []).append(index)

    cv_delta = np.zeros(len(entries), dtype=np.float64)
    fold_routes = []
    for heldout_group, heldout_indices in groups.items():
        train_indices = [index for group, indices in groups.items() if group != heldout_group
                         for index in indices]
        selected = max(exact, key=lambda row: float(np.mean(
            np.asarray(row["image_scores"])[train_indices])))
        selected_scores = selected["image_scores"]
        for index in heldout_indices:
            cv_delta[index] = selected_scores[index] - reference_scores[index]
        fold_routes.append({"heldout_specimen": heldout_group,
                            "selected": {key: selected[key] for key in (
                                "primary_source", "primary_gamma", "al3ni_source",
                                "al3ni_gamma", "eutectic_source", "eutectic_gamma")}})

    rng = np.random.default_rng(int(branch_id))
    group_names = list(groups)
    bootstrap = np.empty(10000, dtype=np.float64)
    for draw in range(len(bootstrap)):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [index for group in sampled for index in groups[str(group)]]
        bootstrap[draw] = float(cv_delta[indices].mean())
    cv_summary = {
        "image_mean_delta_pp": float(cv_delta.mean() * 100.0),
        "positive_image_count": int((cv_delta > 0).sum()),
        "heldout_image_count": len(entries),
        "specimen_group_count": len(groups),
        "group_bootstrap_95ci_delta_pp": [
            float(value * 100.0) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "fold_routes": fold_routes,
    }
    result = {
        "probe": f"model{branch_id} full-image-stat branch fusion; leave-one-specimen-out selection",
        "candidate_count": len(settings),
        "exact_count": len(exact),
        "grouped_cv_vs_model225": cv_summary,
        "best_image": max(exact, key=lambda row: row["image_mean_miou"]),
        "best_exclude": max(exact, key=lambda row: row["image_mean_miou_exclude"]),
        "best_global": max(exact, key=lambda row: row["global_miou"]),
        "records": exact,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("best_image", "best_exclude", "best_global"):
        print(key, json.dumps(result[key], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--model195", required=True)
    parser.add_argument("--tent20", required=True)
    parser.add_argument("--model244", required=True)
    parser.add_argument("--model249", required=True)
    parser.add_argument("--source-id", default="257")
    parser.add_argument("--source-probs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gammas", nargs="+", type=float, default=(0.30, 0.50),
                        help="source blend weights (default: 0.30 0.50)")
    main(parser.parse_args())
