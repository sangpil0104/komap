#!/usr/bin/env python3
"""Grouped-CV probe of high- versus native-resolution branch fusion."""

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
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    archives = {"195": np.load(args.model195), "217": np.load(args.tent20),
                "244": np.load(args.model244), "249": np.load(args.model249),
                args.id253: np.load(args.model253), args.id255: np.load(args.model255)}
    key_orders = {tuple(archive.files) for archive in archives.values()}
    if len(key_orders) != 1:
        raise ValueError("probability archives differ in image keys/order")
    image_names = tuple(archives[args.id253].files)
    source_values = {
        name: [archive[key].astype(np.float32) for key in archive.files]
        for name, archive in archives.items()
    }
    if any(len(values) != len(entries) for values in source_values.values()):
        raise ValueError("probability archive and calibration cache sizes differ")
    rows = [{
        "index": index,
        "item": item,
        "base": undo_model181_primary_blend(item),
        "sources": {
            "base": None,
            "181": item["extra"].astype(np.float32),
            **{name: values[index] for name, values in source_values.items()},
        },
    } for index, item in enumerate(entries)]

    def setting(**changes):
        values = {
            "stage": "model254_highres_route_grouped_cv",
            "primary_source": "195", "primary_gamma": 0.75,
            "al3ni_source": "217", "al3ni_gamma": 0.40,
            "eutectic_source": "217", "eutectic_gamma": 0.40,
            "prior_scale": 1.25, "beta_scale": 0.80,
            "al_delta": 0.0, "eut_delta": 0.15,
            "gaussian_blend": 0.50, "eut_edge_blend": 0.25,
            "cleanup_threshold": 128,
        }
        values.update(changes)
        return default_setting(**values)

    reference = setting(stage="model254_model225_reference")
    settings = [reference]
    # Preregister a small set: 30%/50% replacements for every class subset,
    # using either the named 3x branch or matched 1x control.
    class_names = ("al3ni", "eutectic", "primary")
    for source in (args.id253, args.id255):
        for subset in range(1, 1 << len(class_names)):
            for gamma in (0.30, 0.50):
                changes = {}
                for bit, class_name in enumerate(class_names):
                    if subset & (1 << bit):
                        changes[f"{class_name}_source"] = source
                        changes[f"{class_name}_gamma"] = gamma
                settings.append(setting(**changes))
        settings.append(setting(
            primary_source=source, primary_gamma=1.0,
            al3ni_source=source, al3ni_gamma=1.0,
            eutectic_source=source, eutectic_gamma=1.0))
    for gamma in (0.30, 0.50):
        settings.append(setting(al3ni_source=args.id253, al3ni_gamma=gamma,
                                eutectic_source=args.id255, eutectic_gamma=gamma))
        settings.append(setting(al3ni_source=args.id255, al3ni_gamma=gamma,
                                eutectic_source=args.id253, eutectic_gamma=gamma))

    exact = evaluate(rows, settings, exact=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".exact.json").write_text(
        json.dumps({"candidate_count": len(settings), "records": exact},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    reference_scores = next(row["image_scores"] for row in exact
                            if row["stage"] == "model254_model225_reference")
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(entries):
        # The route cache stores arrays in collect_split(valid)'s sorted order;
        # the probability archives use that same order and retain image stems.
        name = image_names[index]
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

    rng = np.random.default_rng(254)
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
        "probe": f"preregistered {args.id253} 3x vs {args.id255} matched 1x fusion; leave-one-specimen-out selection",
        "candidate_count": len(settings),
        "exact_count": len(exact),
        "grouped_cv_vs_model225": cv_summary,
        "best_image": max(exact, key=lambda row: row["image_mean_miou"]),
        "best_exclude": max(exact, key=lambda row: row["image_mean_miou_exclude"]),
        "best_image_compat": max(exact, key=lambda row: row["image_miou_compat"]),
        "records": exact,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("best_image", "best_exclude", "best_image_compat"):
        print(key, json.dumps(result[key], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--model195", required=True)
    parser.add_argument("--tent20", required=True)
    parser.add_argument("--model244", required=True)
    parser.add_argument("--model249", required=True)
    parser.add_argument("--model253", required=True)
    parser.add_argument("--model255", required=True)
    parser.add_argument("--id253", default="253")
    parser.add_argument("--id255", default="255")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
