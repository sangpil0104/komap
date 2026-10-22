#!/usr/bin/env python3
"""Small, controlled 3-model probability ensemble search."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from calibrate_al import predict_probabilities
from hard_negative_finetune import load_model
from improve_five_causes import CLASS_NAMES, NUM_CLASSES, collect_split, decode_mask, encode_mask


def confusion(prediction, target):
    bins = target.reshape(-1) * NUM_CLASSES + prediction.reshape(-1)
    return np.bincount(bins, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def score(prediction, target):
    current = confusion(prediction, target)
    intersection = np.diag(current).astype(np.float64)
    union = current.sum(0) + current.sum(1) - intersection
    return float(np.mean(intersection / np.maximum(union, 1.0)))


def metrics(predictions, targets):
    from local_image_metrics import summary
    return summary(predictions, targets, CLASS_NAMES)


def cache(args, device, use_amp):
    models = [load_model(args.checkpoint98, device), load_model(args.checkpoint99, device),
              load_model(args.checkpoint101, device)]
    samples = collect_split(args.data_dir, "valid")
    values = []
    for image_path, mask_path in _local_progress(samples, 'three_ensemble'):
        p98 = predict_probabilities(models[0], image_path, 512, 384, device, use_amp, 4, True)
        p99 = predict_probabilities(models[1], image_path, 512, 384, device, use_amp, 4, True)
        p101 = predict_probabilities(models[2], image_path, 640, 512, device, use_amp, 2, True)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        values.append((p98, p99, p101, target))
        print(f"processed {image_path.name}", flush=True)
    return values


def evaluate(values, weights, biases):
    predictions, targets = [], []
    bias = np.asarray(biases, dtype=np.float32)[:, None, None]
    for p98, p99, p101, target in values:
        probability = weights[0] * p98 + weights[1] * p99 + weights[2] * p101
        prediction = (np.log(np.maximum(probability, 1e-7)) + bias).argmax(axis=0).astype(np.uint8)
        predictions.append(prediction)
        targets.append(target)
    return metrics(predictions, targets)


def predict_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    models = [load_model(args.checkpoint98, device), load_model(args.checkpoint99, device),
              load_model(args.checkpoint101, device)]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights = (args.weight98, args.weight99, args.weight101)
    biases = np.asarray([0.0, args.primary_bias, args.al_bias, args.eutectic_bias],
                        dtype=np.float32)[:, None, None]
    images = sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png"))
    for image_path in images:
        p98 = predict_probabilities(models[0], image_path, 512, 384, device, use_amp, 4, True)
        p99 = predict_probabilities(models[1], image_path, 512, 384, device, use_amp, 4, True)
        p101 = predict_probabilities(models[2], image_path, 640, 512, device, use_amp, 2, True)
        probability = weights[0] * p98 + weights[1] * p99 + weights[2] * p101
        prediction = (np.log(np.maximum(probability, 1e-7)) + biases).argmax(axis=0).astype(np.uint8)
        encode_mask(prediction).save(output / image_path.name.replace("_image.png", "_mask.png"))
    print(f"wrote {len(images)} masks to {output}", flush=True)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    values = cache(args, device, use_amp)
    candidates = [
        ((0.85, 0.15, 0.00), (0.0, 0.0, 0.10, -0.20)),
        ((0.75, 0.15, 0.10), (0.0, 0.0, 0.10, -0.20)),
        ((0.70, 0.15, 0.15), (0.0, 0.0, 0.10, -0.20)),
        ((0.75, 0.10, 0.15), (0.0, 0.0, 0.10, -0.20)),
        ((0.80, 0.10, 0.10), (0.0, 0.0, 0.10, -0.20)),
        ((0.85, 0.10, 0.05), (0.0, 0.0, 0.10, -0.20)),
        ((0.75, 0.15, 0.10), (0.0, 0.0, 0.10, -0.30)),
        ((0.75, 0.15, 0.10), (0.0, 0.0, 0.10, -0.10)),
    ]
    records = []
    for weights, biases in candidates:
        result = evaluate(values, weights, biases)
        records.append({"weights": weights, "biases": biases, **result})
    best = max(records, key=lambda x: x["image_miou_compat"])
    best_image = max(records, key=lambda x: x["image_mean_miou"])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best_image_compat": best, "best_image_mean": best_image, "records": records},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_image_compat": best, "best_image_mean": best_image}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sweep", "predict-test"], default="sweep")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint98", required=True)
    parser.add_argument("--checkpoint99", required=True)
    parser.add_argument("--checkpoint101", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir", default="outputs/model102_three_ensemble/test_masks")
    parser.add_argument("--weight98", type=float, default=0.75)
    parser.add_argument("--weight99", type=float, default=0.10)
    parser.add_argument("--weight101", type=float, default=0.15)
    parser.add_argument("--primary-bias", type=float, default=0.0)
    parser.add_argument("--eutectic-bias", type=float, default=0.10)
    parser.add_argument("--al-bias", type=float, default=-0.20)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.mode == "predict-test":
        predict_test(args)
    else:
        main(args)
