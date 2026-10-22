#!/usr/bin/env python3
"""Test a label-free-looking Primary-Si suppression on the A71-like route."""

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


def route(item: dict, new: np.ndarray, primary_delta: float,
          force_no_primary: bool = False) -> np.ndarray:
    mixed = replace_branch(undo_model181_primary_blend(item), new, "primary", 0.75)
    mixed = replace_branch(mixed, new, "rare", 0.075)
    base_prior, delta = prior_terms(item, mixed)
    prior = 1.25 * base_prior
    prior[0] += 1.25 * 0.80 * item["params"]["beta"] * delta
    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
    logits[1] += primary_delta
    logits[2] += item["params"]["al_bias"]
    logits[3] += item["params"]["eut_bias"]
    refined = logits.copy()
    refined[1] = 0.50 * gaussian_filter(logits[1], 0.50, mode="reflect") + 0.50 * bilateral(
        logits[1], item["z"], 2, 1.00, 0.60)
    refined[3] = 0.75 * logits[3] + 0.25 * bilateral(
        logits[3], item["z"], 2, 1.00, 0.60)
    if force_no_primary:
        refined[1] = -np.inf
    return cleanup_primary(refined.argmax(0).astype(np.uint8), refined, 128)


def main(args: argparse.Namespace) -> None:
    entries = pickle.load(Path(args.entries).open("rb"))
    values = np.load(args.valid_probs)
    probabilities = [values[key].astype(np.float32) for key in values.files]
    records = []
    for index, (item, new) in enumerate(zip(entries, probabilities)):
        if args.index >= 0 and index != args.index:
            continue
        target = item["target"]
        for delta in (-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5):
            pred = route(item, new, delta)
            value = score([pred], [target])
            records.append({"index": index, "primary_delta": delta, **value})
        pred = route(item, new, 0.0, True)
        records.append({"index": index, "primary_delta": "force_no_primary",
                        **score([pred], [target])})
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--valid-probs", required=True)
    parser.add_argument("--index", type=int, default=16)
    main(parser.parse_args())
