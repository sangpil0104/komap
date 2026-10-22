#!/usr/bin/env python3
"""Search class-specific fusion of the independent representation branches.

Model 197 uses the model-195 native branch for Primary and both rare classes
with one shared gamma.  This search keeps the same route and lets each class
choose a complementary branch (model 186, model 192, model 195, or the
pre-branch route) independently.  It also performs a small calibration
coordinate search around the strongest fusion candidates.
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
from model183_metric_calibration import prior_terms
from model184_edge_refine import bilateral
from model187_schedule_branch_blend import undo_model181_primary_blend


CLASS_IDS = (0, 1, 2, 3)


def replace_one(mixed: np.ndarray, source: np.ndarray | None,
                class_id: int, gamma: float) -> np.ndarray:
    if source is None or abs(gamma) < 1e-12:
        return mixed
    output = mixed.copy()
    output[class_id] = (1.0 - gamma) * mixed[class_id] + gamma * source[class_id]
    return output / np.maximum(output.sum(axis=0, keepdims=True), 1e-7)


def make_mixed(row: dict, setting: dict) -> np.ndarray:
    mixed = row["base"].copy()
    for class_id, name in ((1, "primary"), (0, "al3ni"), (3, "eutectic")):
        mixed = replace_one(mixed, row["sources"][setting[f"{name}_source"]],
                            class_id, setting[f"{name}_gamma"])
    return mixed


def predict(item: dict, mixed: np.ndarray, setting: dict) -> np.ndarray:
    base_prior, delta = prior_terms(item, mixed)
    prior_scale = setting["prior_scale"]
    beta_scale = setting["beta_scale"]
    prior = prior_scale * base_prior
    prior[0] += prior_scale * beta_scale * item["params"]["beta"] * delta
    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
    logits[2] += item["params"]["al_bias"] + setting["al_delta"]
    logits[3] += item["params"]["eut_bias"] + setting["eut_delta"]
    refined = logits.copy()
    refined[1] = (setting["gaussian_blend"] * gaussian_filter(
        logits[1], sigma=0.50, mode="reflect")
        + (1.0 - setting["gaussian_blend"]) * bilateral(
            logits[1], item["z"], 2, 1.00, 0.60))
    refined[3] = ((1.0 - setting["eut_edge_blend"]) * logits[3]
                  + setting["eut_edge_blend"] * bilateral(
                      logits[3], item["z"], 2, 1.00, 0.60))
    prediction = cleanup_primary(
        refined.argmax(axis=0).astype(np.uint8), refined,
        setting["cleanup_threshold"])
    if float((prediction == 1).mean()) <= 0.01:
        mask = prediction == 1
        alternative = refined.copy()
        alternative[1] = -np.inf
        prediction[mask] = alternative[:, mask].argmax(axis=0).astype(np.uint8)
    return prediction


def evaluate(rows: list[dict], setting: dict) -> dict:
    predictions, targets = [], []
    for row in rows:
        mixed = make_mixed(row, setting)
        predictions.append(predict(row["item"], mixed, setting))
        targets.append(row["item"]["target"])
    return score(predictions, targets)


def setting(primary_source: str = "195", primary_gamma: float = 0.75,
            al3ni_source: str = "195", al3ni_gamma: float = 0.075,
            eutectic_source: str = "195", eutectic_gamma: float = 0.075,
            prior_scale: float = 1.25, beta_scale: float = 0.80,
            al_delta: float = 0.0, eut_delta: float = 0.0,
            gaussian_blend: float = 0.50, eut_edge_blend: float = 0.25,
            cleanup_threshold: int = 128) -> dict:
    return {
        "primary_source": primary_source, "primary_gamma": primary_gamma,
        "al3ni_source": al3ni_source, "al3ni_gamma": al3ni_gamma,
        "eutectic_source": eutectic_source, "eutectic_gamma": eutectic_gamma,
        "prior_scale": prior_scale, "beta_scale": beta_scale,
        "al_delta": al_delta, "eut_delta": eut_delta,
        "gaussian_blend": gaussian_blend, "eut_edge_blend": eut_edge_blend,
        "cleanup_threshold": cleanup_threshold,
    }


def main(args: argparse.Namespace) -> None:
    with Path(args.entries).open("rb") as handle:
        entries = pickle.load(handle)
    paths = {
        "186": args.model186,
        "192": args.model192,
        "195": args.model195,
    }
    probabilities = {}
    for name, path in paths.items():
        archive = np.load(path)
        probabilities[name] = [archive[key].astype(np.float32) for key in archive.files]
    rows = []
    for index, item in enumerate(entries):
        rows.append({
            "item": item,
            "base": undo_model181_primary_blend(item),
            "sources": {
                "base": None,
                "186": probabilities["186"][index],
                "192": probabilities["192"][index],
                "195": probabilities["195"][index],
            },
        })

    records: list[dict] = []
    # Stage 0: compare independent branches as Primary replacements.
    primary_candidates = []
    for source in ("base", "186", "192", "195"):
        for gamma in (0.0, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00):
            candidate = setting(primary_source=source, primary_gamma=gamma)
            value = evaluate(rows, candidate)
            record = {"stage": "primary", **candidate, **value}
            records.append(record)
            primary_candidates.append(record)
    primary_candidates = sorted(primary_candidates,
                               key=lambda row: row["image_mean_miou"], reverse=True)[:8]

    # Stage 1: independently select the rare-class sources and strengths.
    rare_candidates = []
    for parent in primary_candidates:
        for class_name in ("al3ni", "eutectic"):
            for source in ("base", "186", "192", "195"):
                for gamma in (-0.10, 0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30):
                    candidate = dict(parent)
                    candidate["stage"] = f"{class_name}_coordinate"
                    candidate[f"{class_name}_source"] = source
                    candidate[f"{class_name}_gamma"] = gamma
                    value = evaluate(rows, candidate)
                    record = {**candidate, **value}
                    records.append(record)
                    rare_candidates.append(record)
    parents = sorted(rare_candidates,
                     key=lambda row: row["image_mean_miou"], reverse=True)[:12]

    # Stage 2: jointly re-evaluate the best class-specific configurations.
    joint = []
    for parent in parents:
        for al_source in ("base", "186", "192", "195"):
            for eut_source in ("base", "186", "192", "195"):
                candidate = dict(parent)
                candidate["stage"] = "joint_source"
                candidate["al3ni_source"] = al_source
                candidate["eutectic_source"] = eut_source
                value = evaluate(rows, candidate)
                record = {**candidate, **value}
                records.append(record)
                joint.append(record)
    parents = sorted(joint + parents,
                     key=lambda row: row["image_mean_miou"], reverse=True)[:12]

    # Stage 3: small calibration coordinate search around the best fusions.
    calibration = []
    for parent in parents:
        for field, values in (
            ("prior_scale", (1.00, 1.10, 1.20, 1.25, 1.30, 1.40)),
            ("beta_scale", (0.50, 0.70, 0.80, 1.00, 1.20)),
            ("al_delta", (-0.20, -0.10, 0.0, 0.10, 0.20)),
            ("eut_delta", (-0.30, -0.15, 0.0, 0.15, 0.30)),
            ("gaussian_blend", (0.0, 0.25, 0.50, 0.75, 1.0)),
            ("eut_edge_blend", (0.0, 0.15, 0.25, 0.40, 0.60)),
            ("cleanup_threshold", (0, 64, 128, 256)),
        ):
            for value_to_try in values:
                candidate = dict(parent)
                candidate["stage"] = f"calibration_{field}"
                candidate[field] = value_to_try
                value = evaluate(rows, candidate)
                record = {**candidate, **value}
                records.append(record)
                calibration.append(record)

    best_image = max(records, key=lambda row: row["image_mean_miou"])
    best_exclude = max(records, key=lambda row: row["image_mean_miou_exclude"])
    best_image_compat = max(records, key=lambda row: row["image_miou_compat"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "best_image": best_image,
        "best_exclude": best_exclude,
        "best_image_compat": best_image_compat,
        "records": records,
        "record_count": len(records),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "record_count": len(records),
        "best_image": best_image,
        "best_exclude": best_exclude,
        "best_image_compat": best_image_compat,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", required=True)
    parser.add_argument("--model186", required=True)
    parser.add_argument("--model192", required=True)
    parser.add_argument("--model195", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
