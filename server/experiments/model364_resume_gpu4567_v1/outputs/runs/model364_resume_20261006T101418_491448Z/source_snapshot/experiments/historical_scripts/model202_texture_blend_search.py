#!/usr/bin/env python3
"""Small exact blend search for the model-201 texture branch."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

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
        if mode == "rare":
            mixed = replace_branch(mixed, row["new"], "rare", rare_gamma)
        elif mode in ("al3ni", "eutectic"):
            mixed = replace_branch(mixed, row["new"], mode, rare_gamma)
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
    entries = pickle.load(Path(args.entries).open("rb"))
    values = np.load(args.valid_probs)
    new = [values[key].astype(np.float32) for key in values.files]
    rows = [{"item": item, "base": undo_model181_primary_blend(item), "new": value}
            for item, value in zip(entries, new)]
    settings = []
    for gamma in (0.0, 0.25, 0.50, 0.75, 1.0):
        settings.append((gamma, "none", 0.0))
    for gamma in (0.0, 0.05, 0.075, 0.10, 0.15, 0.25):
        for mode in ("rare", "al3ni", "eutectic"):
            settings.append((0.75, mode, gamma))
    records = []
    for primary_gamma, mode, rare_gamma in _local_progress(settings, 'model202_texture_blend_search'):
        records.append({"primary_gamma": primary_gamma, "mode": mode,
                        "rare_gamma": rare_gamma,
                        **evaluate(rows, primary_gamma, mode, rare_gamma)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "best_image": max(records, key=lambda row: row["image_mean_miou"]),
        "best_exclude": max(records, key=lambda row: row["image_mean_miou_exclude"]),
        "best_image_compat": max(records, key=lambda row: row["image_miou_compat"]),
        "records": records,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--valid-probs", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
