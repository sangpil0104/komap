#!/usr/bin/env python3
"""Narrowly refine model-196 branch and rare-class blend weights."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from aspp768_addition import cleanup_primary
from model181_raw_zcdf_aspp import score
from model183_metric_calibration import prior_terms
from model184_edge_refine import bilateral
from model187_schedule_branch_blend import replace_branch, undo_model181_primary_blend


def evaluate(rows: list[dict], primary_gamma: float, mode: str,
             rare_gamma: float) -> dict:
    predictions, targets = [], []
    for row in rows:
        mixed = replace_branch(row["base"], row["new"], "primary", primary_gamma)
        if mode == "eutectic":
            mixed = replace_branch(mixed, row["new"], "eutectic", rare_gamma)
        elif mode == "al3ni":
            mixed = replace_branch(mixed, row["new"], "al3ni", rare_gamma)
        elif mode == "rare":
            mixed = replace_branch(mixed, row["new"], "rare", rare_gamma)
        base_prior, delta = prior_terms(row["item"], mixed)
        prior = 1.25 * base_prior
        prior[0] += 1.25 * 0.80 * row["item"]["params"]["beta"] * delta
        logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
        logits[2] += row["item"]["params"]["al_bias"]
        logits[3] += row["item"]["params"]["eut_bias"]
        refined = logits.copy()
        refined[1] = (0.50 * gaussian_filter(logits[1], sigma=0.50, mode="reflect")
                      + 0.50 * bilateral(logits[1], row["item"]["z"], 2, 1.00, 0.60))
        refined[3] = 0.75 * logits[3] + 0.25 * bilateral(
            logits[3], row["item"]["z"], 2, 1.00, 0.60)
        prediction = cleanup_primary(refined.argmax(0).astype(np.uint8), refined, 128)
        if float((prediction == 1).mean()) <= 0.01:
            mask = prediction == 1
            alternative = refined.copy()
            alternative[1] = -np.inf
            prediction[mask] = alternative[:, mask].argmax(0).astype(np.uint8)
        predictions.append(prediction)
        targets.append(row["item"]["target"])
    return score(predictions, targets)


def main(args: argparse.Namespace) -> None:
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    probabilities = np.load(args.valid_probs)
    new_probs = [probabilities[key].astype(np.float32) for key in probabilities.files]
    rows = [{"item": item, "base": undo_model181_primary_blend(item), "new": new}
            for item, new in zip(entries, new_probs)]
    records = []
    primary_values = (0.70, 0.75, 0.80, 0.85, 0.90)
    rare_values = (0.05, 0.075, 0.10, 0.125, 0.15)
    for primary_gamma in primary_values:
        records.append({"primary_gamma": primary_gamma, "mode": "none", "rare_gamma": 0.0,
                        **evaluate(rows, primary_gamma, "none", 0.0)})
        for mode in ("rare", "al3ni", "eutectic"):
            for rare_gamma in rare_values:
                records.append({"primary_gamma": primary_gamma, "mode": mode,
                                "rare_gamma": rare_gamma,
                                **evaluate(rows, primary_gamma, mode, rare_gamma)})
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
