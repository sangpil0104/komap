#!/usr/bin/env python3
"""Narrow prior-scale refinement around model-189."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from model183_metric_calibration import prior_terms
from model187_schedule_branch_blend import replace_branch, undo_model181_primary_blend
from model189_branch_calibration import evaluate


def main(args: argparse.Namespace) -> None:
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    probs = np.load(args.valid_probs)
    new_probs = [probs[key].astype(np.float32) for key in probs.files]
    rows = []
    for item, new in zip(entries, new_probs):
        base = undo_model181_primary_blend(item)
        mixed = replace_branch(base, new, "primary", 0.80)
        base_prior, delta = prior_terms(item, mixed)
        rows.append({"item": item, "mixed": mixed,
                     "base_prior": base_prior, "delta": delta})
    records = []
    for prior_scale in (1.20, 1.25, 1.30, 1.35, 1.40):
        for beta_scale in (0.60, 0.70, 0.80, 0.90):
            value = evaluate(rows, prior_scale, beta_scale)
            records.append({"prior_scale": prior_scale, "beta_scale": beta_scale,
                            "primary_bias": 0.0, **value})
    best_image = max(records, key=lambda row: row["image_mean_miou"])
    best_exclude = max(records, key=lambda row: row["image_mean_miou_exclude"])
    best_image_compat = max(records, key=lambda row: row["image_miou_compat"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best_image": best_image,
                                  "best_exclude": best_exclude,
                                  "best_image_compat": best_image_compat,
                                  "records": records},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_image": best_image, "best_exclude": best_exclude,
                      "best_image_compat": best_image_compat}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--valid-probs", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
