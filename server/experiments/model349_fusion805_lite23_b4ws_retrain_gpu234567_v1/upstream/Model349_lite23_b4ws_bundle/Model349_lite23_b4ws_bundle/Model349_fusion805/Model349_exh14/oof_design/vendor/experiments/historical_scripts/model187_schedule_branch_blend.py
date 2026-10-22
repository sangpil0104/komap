#!/usr/bin/env python3
"""Blend the longer-trained model-186 representation branch into model 185."""

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


def undo_model181_primary_blend(item: dict, gamma: float = 0.40) -> np.ndarray:
    """Recover the pre-auxiliary probability from model-182's mixed map."""
    mixed = item["mixed"]
    extra = item["extra"]
    primary = mixed[1]
    scale = (1.0 - gamma + gamma * extra[1]) / (1.0 - gamma + gamma * primary)
    base = mixed * scale[None]
    base[1] = 1.0 - base[0] - base[2] - base[3]
    return base / np.maximum(base.sum(0, keepdims=True), 1e-7)


def replace_branch(base: np.ndarray, new: np.ndarray, mode: str, gamma: float) -> np.ndarray:
    output = base.copy()
    if mode == "primary":
        classes = (1,)
    elif mode == "all":
        classes = (0, 1, 2, 3)
    elif mode == "rare":
        classes = (0, 3)
    elif mode == "eutectic":
        classes = (3,)
    elif mode == "al3ni":
        classes = (0,)
    else:
        raise ValueError(mode)
    for class_id in classes:
        output[class_id] = (1.0 - gamma) * base[class_id] + gamma * new[class_id]
    return output / np.maximum(output.sum(0, keepdims=True), 1e-7)


def route_logits(item: dict, mixed: np.ndarray) -> np.ndarray:
    base_prior, delta = prior_terms(item, mixed)
    prior = 1.10 * base_prior
    prior[0] += 1.10 * 0.75 * item["params"]["beta"] * delta
    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
    logits[2] += item["params"]["al_bias"]
    logits[3] += item["params"]["eut_bias"]
    return logits


def predict(logits: np.ndarray, guide: np.ndarray) -> np.ndarray:
    refined = logits.copy()
    primary_edge = bilateral(logits[1], guide, 2, 1.00, 0.60)
    refined[1] = (0.50 * gaussian_filter(logits[1], sigma=0.50, mode="reflect")
                  + 0.50 * primary_edge)
    eutectic_edge = bilateral(logits[3], guide, 2, 1.00, 0.60)
    refined[3] = 0.75 * logits[3] + 0.25 * eutectic_edge
    output = cleanup_primary(refined.argmax(0).astype(np.uint8), refined, 128)
    if float((output == 1).mean()) <= 0.01:
        mask = output == 1
        alternative = refined.copy()
        alternative[1] = -np.inf
        output[mask] = alternative[:, mask].argmax(0).astype(np.uint8)
    return output


def evaluate(entries: list[dict], new_probs: list[np.ndarray], mode: str,
              gamma: float) -> dict:
    predictions, targets = [], []
    for item, new in zip(entries, new_probs):
        base = undo_model181_primary_blend(item)
        mixed = replace_branch(base, new, mode, gamma)
        predictions.append(predict(route_logits(item, mixed), item["z"]))
        targets.append(item["target"])
    return score(predictions, targets)


def main(args: argparse.Namespace) -> None:
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    probabilities = np.load(args.valid_probs)
    keys = list(probabilities.files)
    if len(keys) != len(entries):
        raise ValueError(f"valid probability count {len(keys)} != entries {len(entries)}")
    new_probs = [probabilities[key].astype(np.float32) for key in keys]
    records = []
    for mode in ("primary", "all", "rare", "eutectic", "al3ni"):
        for gamma in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00):
            value = evaluate(entries, new_probs, mode, gamma)
            records.append({"mode": mode, "gamma": gamma, **value})
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
