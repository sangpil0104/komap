#!/usr/bin/env python3
"""Fit component relabeling on train predictions; evaluate once on valid."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from improve_five_causes import collect_split, decode_mask, specimen_id
from model181_raw_zcdf_aspp import representation
from model239_native_stem_finetune import Stride1NativeStemASPP
from model253_highres_scale_finetune import predict_scaled
from model288_region_classifier_cv import extract_regions, fit_predict, image_score


def _features(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    represented = representation(image)
    return represented[1], represented[2]


def _extract_one(image_path: Path, mask_path: Path, probabilities: np.ndarray):
    target = decode_mask(Image.open(mask_path).convert("RGB"))
    prediction = probabilities.argmax(axis=0).astype(np.uint8)
    z, q = _features(image_path)
    features, labels, weights, predicted, component_ids, component_map = extract_regions(
        prediction, target, probabilities.astype(np.float32), {"z": z, "q": q})
    record = {
        "name": image_path.stem,
        "group": specimen_id(image_path),
        "prediction": prediction,
        "target": target,
        "base_probs": probabilities.astype(np.float32),
        "features": features,
        "labels": labels,
        "weights": weights,
        "predicted": predicted,
        "component_ids": component_ids,
        "component_map": component_map,
        "base_score": image_score(prediction, target),
    }
    return record


def _apply_region_predictions(record: dict, region_probabilities: np.ndarray,
                              threshold: float, soft_blend: float):
    output = record["prediction"].copy()
    soft = record["base_probs"].copy()
    if len(region_probabilities):
        labels = region_probabilities.argmax(axis=1)
        confidence = region_probabilities.max(axis=1)
        for index, new_label in enumerate(labels):
            component_mask = record["component_map"] == record["component_ids"][index]
            if confidence[index] < threshold:
                continue
            old_label = int(record["predicted"][index])
            if int(new_label) != old_label:
                output[component_mask] = int(new_label)
                soft[:, component_mask] = (
                    (1.0 - soft_blend) * soft[:, component_mask]
                    + soft_blend * region_probabilities[index, :, None]
                )
    soft /= np.maximum(soft.sum(axis=0, keepdims=True), 1e-7)
    return output, soft


def main(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Stride1NativeStemASPP().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    train_feature_parts, train_label_parts, train_weight_parts = [], [], []
    started = time.time()
    for index, (image_path, mask_path) in enumerate(train_samples, start=1):
        probabilities = predict_scaled(
            model, image_path, args.patch_size, args.val_stride, 1.0,
            device, amp, args.tta)
        record = _extract_one(image_path, mask_path, probabilities)
        if len(record["labels"]):
            train_feature_parts.append(record["features"])
            train_label_parts.append(record["labels"])
            train_weight_parts.append(record["weights"])
        if index % 10 == 0 or index == len(train_samples):
            print(f"train features {index}/{len(train_samples)} "
                  f"components={sum(len(value) for value in train_label_parts)} "
                  f"elapsed={time.time() - started:.1f}s", flush=True)

    if not train_feature_parts:
        raise RuntimeError("no secondary-phase components were found in train predictions")
    train_x = np.concatenate(train_feature_parts).astype(np.float32)
    train_y = np.concatenate(train_label_parts).astype(np.int64)
    train_w = np.concatenate(train_weight_parts).astype(np.float32)
    print(f"fit samples={len(train_y)} features={train_x.shape[1]} "
          f"train_groups={len({specimen_id(p) for p, _ in train_samples})}", flush=True)

    records = []
    valid_archive = np.load(args.valid_probs)
    expected_keys = {path.stem for path, _ in valid_samples}
    if set(valid_archive.files) != expected_keys:
        raise ValueError("validation archive keys do not match current valid split")
    for index, (image_path, mask_path) in enumerate(valid_samples, start=1):
        probabilities = valid_archive[image_path.stem].astype(np.float32)
        record = _extract_one(image_path, mask_path, probabilities)
        records.append(record)
        print(f"valid features {index}/{len(valid_samples)} {image_path.name} "
              f"components={len(record['labels'])}", flush=True)

    feature_count = train_x.shape[1]
    valid_feature_parts = [record["features"] if record["features"].shape[1] == feature_count
                           else np.empty((0, feature_count), dtype=np.float32)
                           for record in records]
    valid_feature_counts = [len(value) for value in valid_feature_parts]
    all_valid_x = np.concatenate(valid_feature_parts, axis=0)
    all_valid_region_probabilities = fit_predict(
        train_x, train_y, train_w, all_valid_x, seed=args.seed, epochs=args.epochs)
    cursor = 0
    for record, count in zip(records, valid_feature_counts):
        record["region_probabilities"] = all_valid_region_probabilities[cursor:cursor + count]
        cursor += count

    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record["group"], []).append(index)
    thresholds = tuple(args.thresholds)
    scores = {value: np.zeros(len(records), dtype=np.float64) for value in thresholds}
    fixed_threshold = min(thresholds, key=lambda value:
                          abs(float(value) - float(args.fixed_threshold)))
    fixed_soft = []
    baselines = np.asarray([record["base_score"] for record in records])
    for index, record in enumerate(records):
        for threshold in thresholds:
            prediction, soft = _apply_region_predictions(
                record, record["region_probabilities"], threshold, args.soft_blend)
            scores[threshold][index] = image_score(prediction, record["target"])
            if threshold == fixed_threshold:
                fixed_soft.append(soft)

    # Group-held-out threshold selection: the selected threshold for an image's
    # specimen is chosen using other valid specimen groups only.
    cv_scores = np.zeros(len(records), dtype=np.float64)
    cv_thresholds = {}
    for heldout, indices in groups.items():
        training_indices = [i for group, members in groups.items() if group != heldout
                            for i in members]
        selected = max(thresholds, key=lambda value:
                       float(scores[value][training_indices].mean()))
        for index in indices:
            cv_scores[index] = scores[selected][index]
        cv_thresholds[heldout] = float(selected)

    bootstrap_rng = np.random.default_rng(args.seed)
    group_names = list(groups)
    deltas = cv_scores - baselines
    bootstrap = np.empty(args.bootstrap_draws, dtype=np.float64)
    for draw in range(args.bootstrap_draws):
        sampled = bootstrap_rng.choice(group_names, size=len(group_names), replace=True)
        indices = [i for group in sampled for i in groups[str(group)]]
        bootstrap[draw] = deltas[indices].mean() * 100.0
    train_groups = {specimen_id(path) for path, _ in train_samples}
    overlap_mask = np.asarray([record["group"] in train_groups for record in records])
    independent_mask = ~overlap_mask

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "valid_probs_region_corrected.npz", **{
        record["name"]: probabilities.astype(np.float16)
        for record, probabilities in zip(records, fixed_soft)
    })

    result = {
        "method": "train-fitted class-component region MLP relabeling",
        "train_prediction_source": str(args.checkpoint),
        "train_prediction_note": "in-sample predictions from a model trained on train labels",
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "train_image_count": len(train_samples),
        "train_specimen_groups": len({specimen_id(path) for path, _ in train_samples}),
        "train_valid_specimen_overlap_groups": sorted(
            train_groups & {record["group"] for record in records}),
        "train_component_count": int(len(train_y)),
        "valid_image_count": len(records),
        "valid_specimen_groups": len(groups),
        "feature_count": int(train_x.shape[1]),
        "baseline_image_mean_miou": float(baselines.mean()),
        "thresholds": {},
        "group_heldout_threshold_selection": {
            "image_mean_miou": float(cv_scores.mean()),
            "delta_pp_vs_baseline": float(deltas.mean() * 100.0),
            "positive_image_count": int((deltas > 0).sum()),
            "group_bootstrap_95ci_delta_pp": [
                float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
            "threshold_by_heldout_group": cv_thresholds,
            "nonoverlap_group_image_count": int(independent_mask.sum()),
            "nonoverlap_group_delta_pp": (
                float(deltas[independent_mask].mean() * 100.0)
                if independent_mask.any() else None),
            "overlap_group_image_count": int(overlap_mask.sum()),
            "overlap_group_delta_pp": (
                float(deltas[overlap_mask].mean() * 100.0)
                if overlap_mask.any() else None),
        },
        "fixed_threshold_for_route_probe": fixed_threshold,
        "images": [],
        "elapsed_seconds": time.time() - started,
    }
    for threshold in thresholds:
        delta = scores[threshold] - baselines
        result["thresholds"][str(threshold)] = {
            "image_mean_miou": float(scores[threshold].mean()),
            "delta_pp_vs_baseline": float(delta.mean() * 100.0),
            "positive_image_count": int((delta > 0).sum()),
            "image_scores": scores[threshold].tolist(),
        }
    result["images"] = [{
        "name": record["name"], "group": record["group"],
        "baseline": float(baselines[index]),
        **{f"score_threshold_{threshold}": float(scores[threshold][index])
           for threshold in thresholds},
        "cv_selected_threshold": cv_thresholds[record["group"]],
        "cv_selected_score": float(cv_scores[index]),
    } for index, record in enumerate(records)]
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": result["baseline_image_mean_miou"],
                      "thresholds": result["thresholds"],
                      "group_cv": result["group_heldout_threshold_selection"],
                      "probabilities": str(output / "valid_probs_region_corrected.npz")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid-probs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--val-stride", type=int, default=576)
    parser.add_argument("--tta", choices=("none", "d4"), default="d4")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=(0.75, 0.85, 0.90, 0.95))
    parser.add_argument("--fixed-threshold", type=float, default=0.90)
    parser.add_argument("--soft-blend", type=float, default=0.50)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=293)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
