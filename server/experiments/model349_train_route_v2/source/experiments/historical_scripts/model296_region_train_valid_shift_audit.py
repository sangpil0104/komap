#!/usr/bin/env python3
"""Audit train in-sample vs valid component errors for region-corrector viability."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

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
from model288_region_classifier_cv import extract_regions


def summarize(parts: list[dict]) -> dict:
    if not parts:
        return {"image_count": 0, "component_count": 0}
    predicted = np.concatenate([part["predicted"] for part in parts])
    truth = np.concatenate([part["truth"] for part in parts])
    area = np.concatenate([part["area"] for part in parts])
    confidence = np.concatenate([part["confidence"] for part in parts])
    entropy = np.concatenate([part["entropy"] for part in parts])
    margin = np.concatenate([part["margin"] for part in parts])
    correct = predicted == truth
    bins = np.linspace(0.0, 1.0, 11)
    confidence_bins = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = (confidence >= lower) & (confidence < upper if upper < 1.0
                                              else confidence <= upper)
        if selected.any():
            confidence_bins.append({
                "range": [float(lower), float(upper)],
                "component_count": int(selected.sum()),
                "mean_confidence": float(confidence[selected].mean()),
                "component_accuracy": float(correct[selected].mean()),
                "area_weighted_accuracy": float(np.average(correct[selected],
                                                             weights=area[selected])),
            })
    class_summary = {}
    for class_id in (0, 1, 3):
        selected = predicted == class_id
        class_summary[str(class_id)] = {
            "component_count": int(selected.sum()),
            "component_accuracy": float(correct[selected].mean()) if selected.any() else None,
            "area_weighted_accuracy": float(np.average(correct[selected],
                                                         weights=area[selected]))
            if selected.any() else None,
            "mean_confidence": float(confidence[selected].mean()) if selected.any() else None,
            "mean_entropy": float(entropy[selected].mean()) if selected.any() else None,
        }
    return {
        "image_count": len(parts),
        "component_count": int(len(predicted)),
        "component_accuracy": float(correct.mean()),
        "component_error_rate": float((~correct).mean()),
        "area_weighted_accuracy": float(np.average(correct, weights=area)),
        "area_weighted_error_rate": float(1.0 - np.average(correct, weights=area)),
        "mean_confidence": float(confidence.mean()),
        "mean_entropy": float(entropy.mean()),
        "mean_margin": float(margin.mean()),
        "confidence_bins": confidence_bins,
        "predicted_class": class_summary,
    }


def extract_one(image_path: Path, mask_path: Path, probabilities: np.ndarray) -> dict:
    target = decode_mask(Image.open(mask_path).convert("RGB"))
    prediction = probabilities.argmax(axis=0).astype(np.uint8)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    represented = representation(image)
    features, truth, _, predicted, _, _ = extract_regions(
        prediction, target, probabilities.astype(np.float32),
        {"z": represented[1], "q": represented[2]}, connectivity=1)
    if not len(predicted):
        return {"predicted": np.empty(0, dtype=np.int64),
                "truth": np.empty(0, dtype=np.int64), "area": np.empty(0),
                "confidence": np.empty(0), "entropy": np.empty(0),
                "margin": np.empty(0)}
    rows = np.arange(len(predicted))
    return {
        "predicted": predicted,
        "truth": truth,
        "area": np.maximum(np.expm1(features[:, 13].astype(np.float64)), 0.0),
        "confidence": features[rows, 3 + predicted].astype(np.float64),
        "entropy": features[:, 11].astype(np.float64),
        "margin": features[:, 12].astype(np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid-probs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=576)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Stride1NativeStemASPP().to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    train_parts: list[dict] = []
    for index, (image_path, mask_path) in enumerate(train_samples, start=1):
        probs = predict_scaled(model, image_path, args.patch_size, args.stride,
                               1.0, device, use_amp, "d4")
        train_parts.append(extract_one(image_path, mask_path, probs))
        if index % 10 == 0 or index == len(train_samples):
            count = sum(len(part["predicted"]) for part in train_parts)
            print(f"train in-sample audit {index}/{len(train_samples)} components={count}",
                  flush=True)

    valid_archive = np.load(args.valid_probs)
    expected = {path.stem for path, _ in valid_samples}
    if set(valid_archive.files) != expected:
        raise ValueError("valid probability cache keys do not match valid split")
    train_groups = {specimen_id(path) for path, _ in train_samples}
    valid_parts: list[dict] = []
    overlap_parts: list[dict] = []
    independent_parts: list[dict] = []
    for image_path, mask_path in _local_progress(valid_samples, 'model296_region_train_valid_shift_audit'):
        part = extract_one(image_path, mask_path,
                           valid_archive[image_path.stem].astype(np.float32))
        part["name"] = image_path.name
        valid_parts.append(part)
        (overlap_parts if specimen_id(image_path) in train_groups else independent_parts).append(part)

    train_summary = summarize(train_parts)
    valid_summary = summarize(valid_parts)
    result = {
        "model": 296,
        "method": "model257 in-sample train vs saved valid component error/calibration audit",
        "metric": "4-connected predicted secondary components; correctness is component GT-majority class",
        "ground_truth_use": "diagnostic only; no threshold or model selection",
        "inference": {"patch_size": args.patch_size, "stride": args.stride,
                      "scale": 1.0, "tta": "d4"},
        "train_image_count": len(train_samples),
        "train_specimen_group_count": len(train_groups),
        "valid_image_count": len(valid_samples),
        "valid_overlap_image_count": len(overlap_parts),
        "valid_independent_image_count": len(independent_parts),
        "train_in_sample": train_summary,
        "valid_all": valid_summary,
        "valid_train_specimen_overlap": summarize(overlap_parts),
        "valid_train_specimen_independent": summarize(independent_parts),
        "valid_per_image": [{
            "name": part["name"],
            "component_count": int(len(part["predicted"])),
            "component_error_rate": float(np.mean(part["predicted"] != part["truth"])),
            "mean_confidence": float(part["confidence"].mean()),
            "mean_entropy": float(part["entropy"].mean()),
        } for part in valid_parts],
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "train_in_sample", "valid_all", "valid_train_specimen_overlap",
        "valid_train_specimen_independent", "elapsed_seconds")},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
