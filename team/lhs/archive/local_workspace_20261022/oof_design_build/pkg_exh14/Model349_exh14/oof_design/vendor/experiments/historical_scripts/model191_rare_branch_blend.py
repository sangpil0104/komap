#!/usr/bin/env python3
"""Add small rare-class model-186 blends on top of model-190 Primary route."""

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


def evaluate(rows: list[dict], mode: str, gamma: float) -> dict:
    predictions, targets = [], []
    for row in rows:
        mixed = replace_branch(row["base"], row["new"], "primary", 0.80)
        if mode == "eutectic":
            mixed = replace_branch(mixed, row["new"], "eutectic", gamma)
        elif mode == "al3ni":
            mixed = replace_branch(mixed, row["new"], "al3ni", gamma)
        elif mode == "rare":
            mixed = replace_branch(mixed, row["new"], "rare", gamma)
        base_prior, delta = prior_terms(row["item"], mixed)
        prior = 1.25 * base_prior
        prior[0] += 1.25 * 0.80 * row["item"]["params"]["beta"] * delta
        logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
        logits[2] += row["item"]["params"]["al_bias"]
        logits[3] += row["item"]["params"]["eut_bias"]
        refined = logits.copy()
        refined[1] = (0.50 * gaussian_filter(logits[1], sigma=0.50, mode="reflect")
                      + 0.50 * bilateral(logits[1], row["item"]["z"], 2, 1.00, 0.60))
        refined[3] = 0.75 * logits[3] + 0.25 * bilateral(logits[3], row["item"]["z"], 2, 1.00, 0.60)
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
    probs = np.load(args.valid_probs)
    new_probs = [probs[key].astype(np.float32) for key in probs.files]
    rows = [{"item": item, "base": undo_model181_primary_blend(item), "new": new}
            for item, new in zip(entries, new_probs)]
    records = []
    for mode in ("eutectic", "al3ni", "rare"):
        for gamma in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
            records.append({"mode": mode, "gamma": gamma,
                            **evaluate(rows, mode, gamma)})
    best_image = max(records, key=lambda row: row["image_mean_miou"])
    best_exclude = max(records, key=lambda row: row["image_mean_miou_exclude"])
    best_global = max(records, key=lambda row: row["global_miou"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best_image": best_image,
                                  "best_exclude": best_exclude,
                                  "best_global": best_global,
                                  "records": records},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_image": best_image, "best_exclude": best_exclude,
                      "best_global": best_global}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--valid-probs", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
