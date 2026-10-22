#!/usr/bin/env python3
"""Evaluate the corrected Eutectic fine-tune as an ensemble member."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import metrics, predict_probabilities
from ensemble_with_aspp import combine_class_conditional, load_aspp
from hard_negative_finetune import load_model
from improve_five_causes import CLASS_NAMES, collect_split, decode_mask, encode_mask
from resnet34_refinements import build_model as build_refined_model, load_compatible


def load_ema(checkpoint, base_checkpoint, device):
    model = build_refined_model("ema")
    load_compatible(model, base_checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def bias_internal(primary, al3ni, eutectic, al):
    return np.asarray([al3ni, primary, al, eutectic], dtype=np.float32)[:, None, None]


def per_image_metrics(prediction, target):
    ious = []
    for class_id in range(4):
        intersection = np.logical_and(prediction == class_id, target == class_id).sum()
        union = np.logical_or(prediction == class_id, target == class_id).sum()
        ious.append(float(intersection / max(union, 1)))
    return {"miou": float(np.mean(ious)), "iou": ious,
            "iou_by_class": {name: float(value) for name, value in zip(CLASS_NAMES, ious)},
            "error_rate": float(np.mean(prediction != target)),
            "error_pixels": int(np.sum(prediction != target)), "pixels": int(target.size)}


def evaluate(cached, new_weight, bias):
    predictions, targets = [], []
    for item in cached:
        base = combine_class_conditional(item["base"], 0.20)
        combined = (1.0 - new_weight) * base + new_weight * item["corrected"]
        combined = 0.75 * combined + 0.25 * item["ema"]
        combined = combined / np.maximum(combined.sum(axis=0, keepdims=True), 1e-7)
        prediction = (np.log(np.maximum(combined, 1e-7)) + bias).argmax(axis=0).astype(np.uint8)
        predictions.append(prediction)
        targets.append(item["target"])
    return metrics(predictions, targets), predictions, targets


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    samples = collect_split(args.data_dir, "valid")
    specs = [
        ("model98", lambda: load_model(args.checkpoint98, device), 512, 384, 4),
        ("model99", lambda: load_model(args.checkpoint99, device), 512, 384, 4),
        ("model101", lambda: load_model(args.checkpoint101, device), 640, 512, 2),
        ("model109", lambda: load_aspp(args.checkpoint109, args.checkpoint98, device), 512, 384, 4),
        ("model125_corrected", lambda: load_model(args.checkpoint125, device), 512, 384, 4),
        ("model123_ema", lambda: load_ema(args.checkpoint123, args.checkpoint109, device), 512, 384, 4),
    ]
    values = []
    for name, loader, patch_size, stride, batch_size in specs:
        print(f"loading {name}", flush=True)
        model = loader()
        current = []
        for image_path, _ in _local_progress(samples, 'evaluate_corrected_ensemble'):
            current.append(predict_probabilities(model, image_path, patch_size, stride, device, use_amp,
                                                 batch_size, "d4", "gaussian"))
            print(f"  predicted {image_path.name}", flush=True)
        values.append(current)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cached = []
    for index, (image_path, mask_path) in enumerate(_local_progress(samples, 'evaluate_corrected_ensemble')):
        cached.append({"name": image_path.name.replace("_image.png", ""),
                       "base": [item[index] for item in values[:4]],
                       "corrected": values[4][index], "ema": values[5][index],
                       "target": decode_mask(Image.open(mask_path).convert("RGB"))})

    bias_specs = [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.10, -0.30),
        (0.0, 0.0, 0.20, -0.20),
        (0.0, 0.0, 0.30, -0.10),
        (0.0, 0.0, 0.20, 0.00),
    ]
    records = []
    for new_weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        for official in bias_specs:
            result, _, _ = evaluate(cached, new_weight, bias_internal(*official))
            records.append({"corrected_weight": new_weight,
                            "bias_official": {"Primary Si": official[0], "Al3Ni": official[1],
                                              "Eutectic Si": official[2], "Al": official[3]}, **result})
    best = max(records, key=lambda item: item["image_mean_miou"])
    best_image_compat = max(records, key=lambda item: item["image_miou_compat"])
    best_bias = tuple(best["bias_official"][name] for name in ("Primary Si", "Al3Ni", "Eutectic Si", "Al"))
    result, predictions, targets = evaluate(cached, best["corrected_weight"], bias_internal(*best_bias))
    output = Path(args.output_dir)
    for name in ("prediction", "ground_truth", "error_map"):
        (output / name).mkdir(parents=True, exist_ok=True)
    records_by_name = []
    for item, prediction, target in zip(cached, predictions, targets):
        stem = item["name"]
        encode_mask(prediction).save(output / "prediction" / f"{stem}_mask.png")
        encode_mask(target).save(output / "ground_truth" / f"{stem}_mask.png")
        error = np.zeros((*target.shape, 3), dtype=np.uint8)
        error[:] = (25, 29, 36)
        error[prediction == target] = (38, 105, 76)
        error[prediction != target] = (232, 67, 67)
        Image.fromarray(error, mode="RGB").save(output / "error_map" / f"{stem}_error.png")
        records_by_name.append({"name": stem, "image": f"{stem}_image.png", **per_image_metrics(prediction, target)})
    manifest = {"model": "corrected Eutectic ensemble: model57 + model125", "best": best,
                "best_image_compat": best_image_compat, "records": records_by_name, "global": result}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_image_mean": best, "best_image_compat": best_image_compat, "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint98", required=True)
    parser.add_argument("--checkpoint99", required=True)
    parser.add_argument("--checkpoint101", required=True)
    parser.add_argument("--checkpoint109", required=True)
    parser.add_argument("--checkpoint123", required=True)
    parser.add_argument("--checkpoint125", required=True)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
