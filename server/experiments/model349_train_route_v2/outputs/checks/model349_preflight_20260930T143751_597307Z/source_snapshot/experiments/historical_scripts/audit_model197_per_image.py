#!/usr/bin/env python3
"""Print per-image confusion/IoU diagnostics for the current model-197 route."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from aspp768_addition import cleanup_primary
from improve_five_causes import collect_split
from model181_raw_zcdf_aspp import score
from model183_metric_calibration import prior_terms
from model184_edge_refine import bilateral
from model187_schedule_branch_blend import replace_branch, undo_model181_primary_blend


def make_prediction(item: dict, new: np.ndarray, primary_gamma: float,
                    rare_gamma: float) -> np.ndarray:
    base = undo_model181_primary_blend(item)
    mixed = replace_branch(base, new, "primary", primary_gamma)
    mixed = replace_branch(mixed, new, "rare", rare_gamma)
    base_prior, delta = prior_terms(item, mixed)
    prior = 1.25 * base_prior
    prior[0] += 1.25 * 0.80 * item["params"]["beta"] * delta
    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
    logits[2] += item["params"]["al_bias"]
    logits[3] += item["params"]["eut_bias"]
    refined = logits.copy()
    refined[1] = (0.50 * gaussian_filter(logits[1], sigma=0.50, mode="reflect")
                  + 0.50 * bilateral(logits[1], item["z"], 2, 1.00, 0.60))
    refined[3] = 0.75 * logits[3] + 0.25 * bilateral(
        logits[3], item["z"], 2, 1.00, 0.60)
    prediction = cleanup_primary(refined.argmax(0).astype(np.uint8), refined, 128)
    if float((prediction == 1).mean()) <= 0.01:
        mask = prediction == 1
        alternative = refined.copy()
        alternative[1] = -np.inf
        prediction[mask] = alternative[:, mask].argmax(0).astype(np.uint8)
    return prediction


def main(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    values = np.load(args.valid_probs)
    probabilities = [values[key].astype(np.float32) for key in values.files]
    samples = collect_split(args.data_dir, "valid")
    output = []
    for index, (item, new, sample) in enumerate(zip(entries, probabilities, samples)):
        target = item["target"]
        prediction = make_prediction(item, new, args.primary_gamma, args.rare_gamma)
        matrix = np.bincount(
            target.reshape(-1) * 4 + prediction.reshape(-1),
            minlength=16,
        ).reshape(4, 4)
        intersection = np.diag(matrix).astype(np.float64)
        union = matrix.sum(0) + matrix.sum(1) - intersection
        iou = intersection / np.maximum(union, 1.0)
        target_area = np.bincount(target.reshape(-1), minlength=4) / target.size
        prediction_area = np.bincount(prediction.reshape(-1), minlength=4) / prediction.size
        output.append({
            "index": index,
            "name": sample[0].name,
            "image_mean_miou": float(iou.mean()),
            "iou": iou.tolist(),
            "target_area": target_area.tolist(),
            "prediction_area": prediction_area.tolist(),
            "confusion": matrix.tolist(),
        })
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--entries", required=True)
    parser.add_argument("--valid-probs", required=True)
    parser.add_argument("--primary-gamma", type=float, default=0.75)
    parser.add_argument("--rare-gamma", type=float, default=0.075)
    main(parser.parse_args())
