#!/usr/bin/env python3
"""Edge-aware refinement on the model-183 route cache.

This experiment keeps every neural branch and calibration parameter fixed and
replaces part of the Primary-Si Gaussian logit smoothing with a small bilateral
filter guided by the per-image z-score intensity map.  It is intentionally
inference-only: no validation mask is used to choose an image-specific route.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from aspp768_addition import cleanup_primary
from model181_raw_zcdf_aspp import score


def bilateral(field: np.ndarray, guide: np.ndarray, radius: int,
              sigma_spatial: float, sigma_range: float) -> np.ndarray:
    """Small reflect-padded bilateral filter for a single logit channel."""
    if sigma_range <= 0:
        return field.copy()
    pad = radius
    field_pad = np.pad(field, pad, mode="reflect")
    guide_pad = np.pad(guide, pad, mode="reflect")
    center = guide_pad[pad:pad + guide.shape[0], pad:pad + guide.shape[1]]
    denominator = np.zeros_like(field, dtype=np.float32)
    numerator = np.zeros_like(field, dtype=np.float32)
    spatial_denominator = 2.0 * sigma_spatial * sigma_spatial
    range_denominator = 2.0 * sigma_range * sigma_range
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial_distance = float(dx * dx + dy * dy)
            spatial_weight = np.exp(-spatial_distance / spatial_denominator)
            neighbour_field = field_pad[
                pad + dy:pad + dy + guide.shape[0],
                pad + dx:pad + dx + guide.shape[1],
            ]
            neighbour_guide = guide_pad[
                pad + dy:pad + dy + guide.shape[0],
                pad + dx:pad + dx + guide.shape[1],
            ]
            range_weight = np.exp(-((neighbour_guide - center) ** 2) / range_denominator)
            weight = (spatial_weight * range_weight).astype(np.float32)
            numerator += weight * neighbour_field
            denominator += weight
    return numerator / np.maximum(denominator, 1e-7)


def logits_for(item: dict) -> np.ndarray:
    """Reproduce the model-183 image-wise-optimal calibrated logits."""
    prior = 1.10 * item["base_prior"].copy()
    prior[0] += 1.10 * 0.75 * item["params"]["beta"] * item["delta"]
    logits = item["log_mixed"] + 0.09 * prior
    logits[2] += item["params"]["al_bias"]
    logits[3] += item["params"]["eut_bias"]
    return logits


def prediction(logits: np.ndarray, primary_logit: np.ndarray) -> np.ndarray:
    refined = logits.copy()
    refined[1] = primary_logit
    output = cleanup_primary(refined.argmax(0).astype(np.uint8), refined, 128)
    if float((output == 1).mean()) <= 0.01:
        mask = output == 1
        alternative = refined.copy()
        alternative[1] = -np.inf
        output[mask] = alternative[:, mask].argmax(0).astype(np.uint8)
    return output


def prepare(entries: list[dict]) -> list[dict]:
    prepared = []
    for item in entries:
        logits = logits_for(item)
        prepared.append({
            "item": item,
            "logits": logits,
            "gaussian": gaussian_filter(logits[1], sigma=0.50, mode="reflect"),
        })
    return prepared


def evaluate_primary(prepared: list[dict], primary_values: list[np.ndarray]) -> dict:
    predictions, targets = [], []
    for row, primary in zip(prepared, primary_values):
        predictions.append(prediction(row["logits"], primary))
        targets.append(row["item"]["target"])
    return score(predictions, targets)


def main(args: argparse.Namespace) -> None:
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    prepared = prepare(entries)
    records = []
    baseline = evaluate_primary(prepared, [row["gaussian"] for row in prepared])
    records.append({"mode": "gaussian_baseline", "radius": 0,
                    "sigma_spatial": 0.50, "sigma_range": 0.0, "blend": 0.0,
                    **baseline})
    for radius in (1, 2):
        for sigma_spatial in (0.50, 0.75, 1.00):
            for sigma_range in (0.10, 0.20, 0.35, 0.60, 1.00):
                edge_values = [bilateral(item["log_mixed"][1], item["z"], radius,
                                         sigma_spatial, sigma_range)
                               for item in entries]
                for blend in (0.25, 0.50, 0.75, 1.00):
                    primary_values = [
                        (1.0 - blend) * row["gaussian"] + blend * edge
                        for row, edge in zip(prepared, edge_values)
                    ]
                    value = evaluate_primary(prepared, primary_values)
                    records.append({"mode": "bilateral_primary", "radius": radius,
                                    "sigma_spatial": sigma_spatial,
                                    "sigma_range": sigma_range, "blend": blend,
                                    **value})
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
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
