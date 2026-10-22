#!/usr/bin/env python3
"""Re-evaluate the strongest ensemble with the official class semantics.

Checkpoint channel order is [Al3Ni, Primary Si, Al, Eutectic Si] because the
existing masks use the fixed blue/green/brown/gray palette.  This script keeps
that channel order, but performs all bias selection and reporting with the
official class names.  The competition metric is image-wise mean mIoU.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import metrics, predict_probabilities
from ensemble_with_aspp import combine_class_conditional, load_aspp
from hard_negative_finetune import load_model
from improve_five_causes import CLASS_NAMES, collect_split, decode_mask
from resnet34_refinements import build_model as build_refined_model, load_compatible


def load_refined_ema(checkpoint: str, base_checkpoint: str, device: torch.device):
    model = build_refined_model("ema")
    load_compatible(model, base_checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def internal_bias(primary: float, al3ni: float, eutectic: float, al: float) -> np.ndarray:
    # Official order -> checkpoint channel order.
    return np.asarray([al3ni, primary, al, eutectic], dtype=np.float32)[:, None, None]


def combine(values, aspp_weight: float, ema_value=None, ema_weight: float = 0.0):
    result = combine_class_conditional(values, aspp_weight)
    if ema_value is not None and ema_weight > 0.0:
        result = (1.0 - ema_weight) * result + ema_weight * ema_value
        result = result / np.maximum(result.sum(axis=0, keepdims=True), 1e-7)
    return result


def evaluate(cached, aspp_weight, ema_weight, bias):
    predictions, targets = [], []
    for item in cached:
        probability = combine(item["base"], aspp_weight, item["ema"], ema_weight)
        prediction = (np.log(np.maximum(probability, 1e-7)) + bias).argmax(axis=0).astype(np.uint8)
        predictions.append(prediction)
        targets.append(item["target"])
    return metrics(predictions, targets)


def evaluate_many(cached, aspp_weight, ema_weight, official_biases, sample_step=1):
    """Evaluate many biases together, reusing one confusion calculation per image."""
    count = len(official_biases)
    internal_biases = np.stack([internal_bias(*values).reshape(4) for values in official_biases])
    total = np.zeros((count, 4, 4), dtype=np.int64)
    image_score_sum = np.zeros(count, dtype=np.float64)
    for item in cached:
        probability = combine(item["base"], aspp_weight, item["ema"], ema_weight)
        log_probability = np.log(np.maximum(probability, 1e-7))[:, ::sample_step, ::sample_step]
        target = item["target"].astype(np.int32, copy=False)[::sample_step, ::sample_step]
        pixels = target.size
        for start in range(0, count, 32):
            end = min(start + 32, count)
            local_bias = internal_biases[start:end]
            predictions = (log_probability[None] + local_bias[:, :, None, None]).argmax(axis=1).astype(np.int32)
            local_count = end - start
            bins = target.reshape(1, pixels) * 4 + predictions.reshape(local_count, pixels)
            bins += np.arange(local_count, dtype=np.int32)[:, None] * 16
            current = np.bincount(bins.reshape(-1), minlength=local_count * 16).reshape(local_count, 4, 4)
            total[start:end] += current
            intersection = np.diagonal(current, axis1=1, axis2=2).astype(np.float64)
            union = current.sum(axis=1) + current.sum(axis=2) - intersection
            image_score_sum[start:end] += (intersection / np.maximum(union, 1.0)).mean(axis=1)
    intersection = np.diagonal(total, axis1=1, axis2=2).astype(np.float64)
    union = total.sum(axis=1) + total.sum(axis=2) - intersection
    iou = intersection / np.maximum(union, 1.0)
    output = []
    for index in range(count):
        output.append({"global_miou": float(iou[index].mean()),
                       "global_iou": {name: float(value) for name, value in zip(CLASS_NAMES, iou[index])},
                       "image_mean_miou": float(image_score_sum[index] / len(cached)),
                       "confusion": total[index].tolist()})
    return output


def cache_predictions(args, device, use_amp, samples):
    specs = [
        ("model98", lambda: load_model(args.checkpoint98, device), 512, 384, 4),
        ("model99", lambda: load_model(args.checkpoint99, device), 512, 384, 4),
        ("model101", lambda: load_model(args.checkpoint101, device), 640, 512, 2),
        ("model109", lambda: load_aspp(args.checkpoint109, args.checkpoint98, device), 512, 384, 4),
        ("model123_ema", lambda: load_refined_ema(args.checkpoint123, args.checkpoint109, device), 512, 384, 4),
    ]
    cached_values = []
    for name, loader, patch_size, stride, batch_size in specs:
        print(f"loading {name}", flush=True)
        model = loader()
        values = []
        for image_path, _ in samples:
            values.append(predict_probabilities(model, image_path, patch_size, stride, device, use_amp,
                                                batch_size, "d4", "gaussian"))
            print(f"  predicted {image_path.name}", flush=True)
        cached_values.append(values)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cached = []
    for index, (image_path, mask_path) in enumerate(samples):
        cached.append({
            "name": image_path.name.replace("_image.png", ""),
            "base": [values[index] for values in cached_values[:4]],
            "ema": cached_values[4][index],
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
        })
    return cached


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    samples = collect_split(args.data_dir, "valid")
    cached = cache_predictions(args, device, use_amp, samples)

    structure_records = []
    structure_biases = [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.10, -0.30),
        (0.0, 0.0, 0.20, -0.20),
        (0.0, 0.0, 0.10, -0.20),
    ]
    for aspp_weight in (0.10, 0.15, 0.20, 0.25, 0.30):
        for ema_weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            results = evaluate_many(cached, aspp_weight, ema_weight, structure_biases, sample_step=4)
            for values, result in zip(structure_biases, results):
                structure_records.append({"aspp_weight": aspp_weight, "ema_weight": ema_weight,
                                          "bias_official": {"Primary Si": values[0], "Al3Ni": values[1],
                                                            "Eutectic Si": values[2], "Al": values[3]},
                                          **result})
    top_structures = sorted(structure_records, key=lambda x: x["image_mean_miou"], reverse=True)[:5]

    bias_records = []
    primary_values = (-0.05, 0.0, 0.05)
    al3ni_values = (-0.05, 0.0, 0.05)
    eutectic_values = (-0.20, -0.10, 0.0, 0.10, 0.20)
    al_values = (-0.30, -0.20, -0.10, 0.0)
    bias_values = list(itertools.product(primary_values, al3ni_values, eutectic_values, al_values))
    for structure in top_structures:
        results = evaluate_many(cached, structure["aspp_weight"], structure["ema_weight"], bias_values,
                                sample_step=4)
        for values, result in zip(bias_values, results):
            primary, al3ni, eutectic, al = values
            bias_records.append({"aspp_weight": structure["aspp_weight"],
                                 "ema_weight": structure["ema_weight"],
                                 "bias_official": {"Primary Si": primary, "Al3Ni": al3ni,
                                                   "Eutectic Si": eutectic, "Al": al}, **result})

    # Re-score only the proxy's top candidates at full resolution.
    exact_records = []
    top_proxy = sorted(bias_records, key=lambda x: x["image_mean_miou"], reverse=True)[:20]
    grouped = {}
    for candidate in top_proxy:
        key = (candidate["aspp_weight"], candidate["ema_weight"])
        grouped.setdefault(key, []).append(tuple(candidate["bias_official"][name]
                                                  for name in ("Primary Si", "Al3Ni", "Eutectic Si", "Al")))
    for (aspp_weight, ema_weight), values_list in grouped.items():
        results = evaluate_many(cached, aspp_weight, ema_weight, values_list, sample_step=1)
        for values, result in zip(values_list, results):
            primary, al3ni, eutectic, al = values
            exact_records.append({"aspp_weight": aspp_weight, "ema_weight": ema_weight,
                                  "bias_official": {"Primary Si": primary, "Al3Ni": al3ni,
                                                    "Eutectic Si": eutectic, "Al": al}, **result})
    best_image = max(exact_records, key=lambda x: x["image_mean_miou"])
    best_global = max(exact_records, key=lambda x: x["global_miou"])
    result = {"metric": "image_mean_miou", "class_order": ["Primary Si", "Al3Ni", "Eutectic Si", "Al"],
              "best_image_mean": best_image, "best_global": best_global,
              "top_structures": top_structures, "proxy_bias_records": bias_records,
              "exact_top_records": exact_records}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_image_mean": best_image, "best_global": best_global}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--checkpoint98", required=True)
    parser.add_argument("--checkpoint99", required=True)
    parser.add_argument("--checkpoint101", required=True)
    parser.add_argument("--checkpoint109", required=True)
    parser.add_argument("--checkpoint123", required=True)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
