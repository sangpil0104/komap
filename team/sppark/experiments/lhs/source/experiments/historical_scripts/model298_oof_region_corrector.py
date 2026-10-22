#!/usr/bin/env python3
"""Fit a component corrector on train-only grouped OOF predictions."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from improve_five_causes import collect_split, decode_mask, specimen_id
from model181_raw_zcdf_aspp import representation
from model288_region_classifier_cv import extract_regions, fit_predict, image_score


THRESHOLDS = (0.60, 0.75, 0.85, 0.90, 0.95)
SOFT_BLEND = 0.50


def extract_record(image_path: Path, mask_path: Path, probabilities: np.ndarray) -> dict:
    target = decode_mask(Image.open(mask_path).convert("RGB"))
    prediction = probabilities.argmax(axis=0).astype(np.uint8)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    features = representation(image)
    region_x, region_y, weights, predicted, component_ids, component_map = extract_regions(
        prediction, target, probabilities.astype(np.float32),
        {"z": features[1], "q": features[2]}, connectivity=1)
    if region_x.shape[1] == 0:
        region_x = np.empty((0, 32), dtype=np.float32)
    return {
        "name": image_path.stem,
        "group": specimen_id(image_path),
        "prediction": prediction,
        "target": target,
        "base_probs": probabilities.astype(np.float32),
        "features": region_x,
        "labels": region_y,
        "weights": weights,
        "predicted": predicted,
        "component_ids": component_ids,
        "component_map": component_map,
        "base_score": image_score(prediction, target),
    }


def apply_region_probabilities(record: dict, region_probabilities: np.ndarray,
                               threshold: float) -> np.ndarray:
    """Blend region posteriors over accepted whole components, without pixel loops."""
    base = record["base_probs"]
    ids = record["component_ids"]
    if not len(ids):
        return base.copy()
    predicted = record["predicted"]
    proposed = region_probabilities.argmax(axis=1)
    confidence = region_probabilities.max(axis=1)
    accepted = (confidence >= threshold) & (proposed != predicted)

    component_count = int(ids.max())
    posterior_lookup = np.zeros((4, component_count + 1), dtype=np.float32)
    posterior_lookup[:, ids] = region_probabilities.T.astype(np.float32)
    alpha_lookup = np.zeros(component_count + 1, dtype=np.float32)
    alpha_lookup[ids[accepted]] = SOFT_BLEND
    component_map = record["component_map"]
    alpha = alpha_lookup[component_map]
    region_posterior = posterior_lookup[:, component_map]
    soft = (1.0 - alpha[None]) * base + alpha[None] * region_posterior
    soft /= np.maximum(soft.sum(axis=0, keepdims=True), 1e-7)
    return soft.astype(np.float32)


def group_bootstrap(delta: np.ndarray, groups: list[str], draws: int, seed: int) -> list[float]:
    group_members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        group_members.setdefault(group, []).append(index)
    names = list(group_members)
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sample = rng.choice(names, size=len(names), replace=True)
        indices = [index for name in sample for index in group_members[str(name)]]
        values[draw] = delta[indices].mean() * 100.0
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--oof-probs", required=True)
    parser.add_argument("--valid-probs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mlp-epochs", type=int, default=25)
    parser.add_argument("--meta-folds", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=298)
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(4)

    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    train_archive = np.load(args.oof_probs)
    valid_archive = np.load(args.valid_probs)
    train_names = {image_path.stem for image_path, _ in train_samples}
    valid_names = {image_path.stem for image_path, _ in valid_samples}
    if set(train_archive.files) != train_names:
        raise ValueError("OOF probability cache does not exactly cover train images")
    if set(valid_archive.files) != valid_names:
        raise ValueError("fold-ensemble cache does not exactly cover valid images")

    train_records = []
    for image_path, mask_path in train_samples:
        record = extract_record(image_path, mask_path,
                                train_archive[image_path.stem].astype(np.float32))
        train_records.append(record)
        print(f"train OOF regions {image_path.name}: {len(record['labels'])}", flush=True)
    valid_records = []
    for image_path, mask_path in valid_samples:
        record = extract_record(image_path, mask_path,
                                valid_archive[image_path.stem].astype(np.float32))
        valid_records.append(record)
        print(f"valid regions {image_path.name}: {len(record['labels'])}", flush=True)

    groups = sorted({record["group"] for record in train_records})
    fold_groups = [groups[index::args.meta_folds] for index in range(args.meta_folds)]
    if any(not values for values in fold_groups):
        raise ValueError("meta-fold count exceeds number of train specimen groups")
    cv_scores = {threshold: np.full(len(train_records), np.nan, dtype=np.float64)
                 for threshold in THRESHOLDS}
    for fold_id, heldout_groups in enumerate(fold_groups, start=1):
        heldout_set = set(heldout_groups)
        heldout_indices = [i for i, record in enumerate(train_records)
                           if record["group"] in heldout_set]
        fit_indices = [i for i, record in enumerate(train_records)
                       if record["group"] not in heldout_set]
        fit_x = np.concatenate([train_records[i]["features"] for i in fit_indices])
        fit_y = np.concatenate([train_records[i]["labels"] for i in fit_indices])
        fit_w = np.concatenate([train_records[i]["weights"] for i in fit_indices])
        test_x = np.concatenate([train_records[i]["features"] for i in heldout_indices])
        region_probs = fit_predict(fit_x, fit_y, fit_w, test_x,
                                   seed=args.seed + fold_id, epochs=args.mlp_epochs)
        cursor = 0
        for index in heldout_indices:
            record = train_records[index]
            count = len(record["labels"])
            local_probs = region_probs[cursor:cursor + count]
            cursor += count
            for threshold in THRESHOLDS:
                soft = apply_region_probabilities(record, local_probs, threshold)
                cv_scores[threshold][index] = image_score(
                    soft.argmax(axis=0).astype(np.uint8), record["target"])
        print(f"completed region meta-fold {fold_id}/{args.meta_folds} "
              f"heldout_groups={len(heldout_groups)}", flush=True)

    baselines = np.asarray([record["base_score"] for record in train_records])
    threshold_results = {}
    train_group_names = [record["group"] for record in train_records]
    for threshold in THRESHOLDS:
        scores = cv_scores[threshold]
        delta = scores - baselines
        threshold_results[str(threshold)] = {
            "image_mean_miou": float(scores.mean()),
            "delta_pp_vs_base": float(delta.mean() * 100.0),
            "positive_image_count": int((delta > 0).sum()),
            "group_bootstrap_95ci_delta_pp": group_bootstrap(
                delta, train_group_names, args.bootstrap_draws, args.seed + int(threshold * 100)),
        }
    selected_threshold = max(
        THRESHOLDS, key=lambda value: threshold_results[str(value)]["image_mean_miou"])

    all_train_x = np.concatenate([record["features"] for record in train_records])
    all_train_y = np.concatenate([record["labels"] for record in train_records])
    all_train_w = np.concatenate([record["weights"] for record in train_records])
    all_valid_x = np.concatenate([record["features"] for record in valid_records])
    valid_region_probs = fit_predict(
        all_train_x, all_train_y, all_train_w, all_valid_x,
        seed=args.seed, epochs=args.mlp_epochs)

    valid_base_scores = np.asarray([record["base_score"] for record in valid_records])
    valid_corrected_scores = []
    corrected_payload = {}
    cursor = 0
    for record in valid_records:
        count = len(record["labels"])
        local_probs = valid_region_probs[cursor:cursor + count]
        cursor += count
        soft = apply_region_probabilities(record, local_probs, selected_threshold)
        valid_corrected_scores.append(image_score(
            soft.argmax(axis=0).astype(np.uint8), record["target"]))
        corrected_payload[record["name"]] = soft.astype(np.float16)
    valid_corrected_scores = np.asarray(valid_corrected_scores, dtype=np.float64)
    valid_delta = valid_corrected_scores - valid_base_scores
    valid_groups = [record["group"] for record in valid_records]
    overlap_groups = {record["group"] for record in train_records}
    overlap_mask = np.asarray([group in overlap_groups for group in valid_groups])

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corrected_path = output / "valid_probs_region_corrected.npz"
    np.savez_compressed(corrected_path, **corrected_payload)
    result = {
        "model": 298,
        "method": "train-only grouped OOF region MLP; soft component posterior blend",
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "region_target": "ground-truth majority class in predicted 4-connected component",
        "ground_truth_use": "train OOF/meta-CV fit; valid used only for final evaluation",
        "train_image_count": len(train_records),
        "train_component_count": int(sum(len(record["labels"]) for record in train_records)),
        "valid_image_count": len(valid_records),
        "valid_component_count": int(sum(len(record["labels"]) for record in valid_records)),
        "meta_fold_count": args.meta_folds,
        "mlp_epochs": args.mlp_epochs,
        "soft_blend": SOFT_BLEND,
        "threshold_selected_on_train_oof_group_cv": selected_threshold,
        "train_meta_cv_thresholds": threshold_results,
        "valid_base_image_mean_miou": float(valid_base_scores.mean()),
        "valid_corrected_image_mean_miou": float(valid_corrected_scores.mean()),
        "valid_delta_pp": float(valid_delta.mean() * 100.0),
        "valid_positive_image_count": int((valid_delta > 0).sum()),
        "valid_group_bootstrap_95ci_delta_pp": group_bootstrap(
            valid_delta, valid_groups, args.bootstrap_draws, args.seed + 1000),
        "valid_train_specimen_overlap": {
            "image_count": int(overlap_mask.sum()),
            "delta_pp": float(valid_delta[overlap_mask].mean() * 100.0)
            if overlap_mask.any() else None,
        },
        "valid_train_specimen_independent": {
            "image_count": int((~overlap_mask).sum()),
            "delta_pp": float(valid_delta[~overlap_mask].mean() * 100.0)
            if (~overlap_mask).any() else None,
        },
        "valid_images": [{
            "name": record["name"], "specimen": record["group"],
            "baseline": float(valid_base_scores[index]),
            "corrected": float(valid_corrected_scores[index]),
            "delta_pp": float(valid_delta[index] * 100.0),
        } for index, record in enumerate(valid_records)],
        "corrected_probabilities": str(corrected_path),
        "elapsed_seconds": time.time() - started,
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "threshold_selected_on_train_oof_group_cv", "train_meta_cv_thresholds",
        "valid_base_image_mean_miou", "valid_corrected_image_mean_miou",
        "valid_delta_pp", "valid_positive_image_count",
        "valid_group_bootstrap_95ci_delta_pp", "elapsed_seconds")},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
