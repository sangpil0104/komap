#!/usr/bin/env python3
"""Class-specific edge-aware refinement on the model-184 route."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from aspp768_addition import cleanup_primary
from model181_raw_zcdf_aspp import score
from model184_edge_refine import bilateral


def logits_for(item: dict) -> np.ndarray:
    prior = 1.10 * item["base_prior"].copy()
    prior[0] += 1.10 * 0.75 * item["params"]["beta"] * item["delta"]
    logits = item["log_mixed"] + 0.09 * prior
    logits[2] += item["params"]["al_bias"]
    logits[3] += item["params"]["eut_bias"]
    return logits


def predict(logits: np.ndarray, primary: np.ndarray,
            replacements: dict[int, np.ndarray]) -> np.ndarray:
    refined = logits.copy()
    refined[1] = primary
    for class_id, value in replacements.items():
        refined[class_id] = value
    output = cleanup_primary(refined.argmax(0).astype(np.uint8), refined, 128)
    if float((output == 1).mean()) <= 0.01:
        mask = output == 1
        alternative = refined.copy()
        alternative[1] = -np.inf
        output[mask] = alternative[:, mask].argmax(0).astype(np.uint8)
    return output


def evaluate(rows: list[dict], mode: str, blend: float,
             edge_values: dict[int, list[np.ndarray]]) -> dict:
    predictions, targets = [], []
    if mode == "none":
        classes = ()
    elif mode == "rare":
        classes = (0, 3)
    elif mode == "al3ni":
        classes = (0,)
    elif mode == "al":
        classes = (2,)
    elif mode == "eutectic":
        classes = (3,)
    elif mode == "al3ni_eutectic":
        classes = (0, 3)
    elif mode == "all_non_primary":
        classes = (0, 2, 3)
    else:
        raise ValueError(mode)
    for index, row in enumerate(rows):
        replacements = {
            class_id: (1.0 - blend) * row["logits"][class_id]
            + blend * edge_values[class_id][index]
            for class_id in classes
        }
        predictions.append(predict(row["logits"], row["primary"], replacements))
        targets.append(row["target"])
    return score(predictions, targets)


def main(args: argparse.Namespace) -> None:
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    rows = []
    for item in entries:
        logits = logits_for(item)
        primary_raw = bilateral(item["log_mixed"][1], item["z"], 2, 1.00, 0.60)
        primary = 0.50 * gaussian_filter(logits[1], sigma=0.50, mode="reflect") + 0.50 * primary_raw
        rows.append({"item": item, "logits": logits, "primary": primary,
                     "target": item["target"]})

    edge_values = {class_id: [bilateral(row["logits"][class_id], row["item"]["z"],
                                         2, 1.00, 0.60) for row in rows]
                   for class_id in (0, 2, 3)}
    records = []
    baseline = evaluate(rows, "none", 0.0, edge_values)
    records.append({"mode": "model184_baseline", "blend": 0.0, **baseline})
    for mode in ("al3ni", "al", "eutectic", "rare", "al3ni_eutectic", "all_non_primary"):
        for blend in (0.25, 0.50, 0.75, 1.00):
            records.append({"mode": mode, "blend": blend,
                            **evaluate(rows, mode, blend, edge_values)})
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
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
