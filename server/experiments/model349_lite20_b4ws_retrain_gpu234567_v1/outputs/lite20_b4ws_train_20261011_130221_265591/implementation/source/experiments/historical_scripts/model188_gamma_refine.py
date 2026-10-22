#!/usr/bin/env python3
"""Narrow gamma refinement around model-187's Primary-only branch blend."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from model187_schedule_branch_blend import evaluate


def main(args: argparse.Namespace) -> None:
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    probabilities = np.load(args.valid_probs)
    keys = list(probabilities.files)
    new_probs = [probabilities[key].astype(np.float32) for key in keys]
    records = []
    for gamma in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        value = evaluate(entries, new_probs, "primary", gamma)
        records.append({"mode": "primary", "gamma": gamma, **value})
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
