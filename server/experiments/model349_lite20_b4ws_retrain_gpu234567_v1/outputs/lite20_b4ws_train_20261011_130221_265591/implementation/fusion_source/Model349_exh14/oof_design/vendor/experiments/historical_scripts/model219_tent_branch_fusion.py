#!/usr/bin/env python3
"""Cache-aware fusion search including the best native-TENT branch."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from model187_schedule_branch_blend import undo_model181_primary_blend
from model198_fast_branch_search import default_setting, evaluate, top


def main(args):
    entries = pickle.load(Path(args.entries).open("rb"))
    paths = {"195": args.model195, "217": args.model217}
    values = {}
    for name, path in paths.items():
        archive = np.load(path)
        values[name] = [archive[key].astype(np.float32) for key in archive.files]
    rows = []
    for index, item in enumerate(entries):
        rows.append({
            "index": index, "item": item,
            "base": undo_model181_primary_blend(item),
            "sources": {"base": None, "181": item["extra"].astype(np.float32),
                        **{name: values[name][index] for name in values}},
        })

    primary_settings = []
    for source in ("195", "217"):
        for gamma in (0.0, 0.25, 0.50, 0.65, 0.75, 0.85, 1.0):
            primary_settings.append(default_setting(
                stage="tent_primary", primary_source=source, primary_gamma=gamma))
    first = evaluate(rows, primary_settings, exact=False)

    rare_settings = []
    # Keep this pass deliberately narrow; the existing model-197 calibration
    # already provides the broad prior/bias search.  Here we only test whether
    # the new TENT branch itself transfers to a class route.
    for parent in top(first, "image_mean_miou", 4):
        for class_name in ("al3ni", "eutectic"):
            for source in ("195", "217"):
                for gamma in args.rare_gammas:
                    candidate = dict(parent)
                    candidate["stage"] = f"tent_{class_name}"
                    candidate[f"{class_name}_source"] = source
                    candidate[f"{class_name}_gamma"] = gamma
                    rare_settings.append(candidate)
        for al_source in ("195", "217"):
            for eut_source in ("195", "217"):
                for gamma in args.rare_gammas:
                    candidate = dict(parent)
                    candidate["stage"] = "tent_joint_rare"
                    candidate["al3ni_source"] = al_source
                    candidate["eutectic_source"] = eut_source
                    candidate["al3ni_gamma"] = gamma
                    candidate["eutectic_gamma"] = gamma
                    rare_settings.append(candidate)
    second = evaluate(rows, rare_settings, exact=False)

    calibration_settings = []
    for parent in top(second, "image_mean_miou", 4):
        for prior_scale in (1.20, 1.25, 1.30):
            for beta_scale in (0.70, 0.80, 0.90):
                candidate = dict(parent)
                candidate["stage"] = "tent_calibration"
                candidate["prior_scale"] = prior_scale
                candidate["beta_scale"] = beta_scale
                calibration_settings.append(candidate)
        for eut_delta in (-0.15, 0.0, 0.15):
            candidate = dict(parent); candidate["stage"] = "tent_eut_delta"
            candidate["eut_delta"] = eut_delta; calibration_settings.append(candidate)
        for al_delta in (0.0, 0.10):
            candidate = dict(parent); candidate["stage"] = "tent_al_delta"
            candidate["al_delta"] = al_delta; calibration_settings.append(candidate)
        for eut_edge in (0.15, 0.25, 0.40):
            candidate = dict(parent); candidate["stage"] = "tent_eut_edge"
            candidate["eut_edge_blend"] = eut_edge; calibration_settings.append(candidate)
    third = evaluate(rows, calibration_settings, exact=False)
    cheap = first + second + third
    exact_settings = [default_setting()]
    exact_settings.extend(top(cheap, "image_mean_miou", args.exact_count))
    exact = evaluate(rows, exact_settings, exact=True)
    records = cheap + exact
    result = {
        "record_count": len(records), "cheap_count": len(cheap),
        "exact_count": len(exact),
        "best_image": max(records, key=lambda row: row["image_mean_miou"]),
        "best_exclude": max(records, key=lambda row: row["image_mean_miou_exclude"]),
        "best_global": max(records, key=lambda row: row["global_miou"]),
        "records": records,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("best_image", "best_exclude", "best_global"):
        print(key, json.dumps(result[key], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--model195", required=True)
    parser.add_argument("--model217", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exact-count", type=int, default=40)
    parser.add_argument("--rare-gammas", nargs="+", type=float,
                        default=(0.0, 0.05, 0.10, 0.20),
                        help="rare-class source mixing weights to evaluate")
    main(parser.parse_args())
