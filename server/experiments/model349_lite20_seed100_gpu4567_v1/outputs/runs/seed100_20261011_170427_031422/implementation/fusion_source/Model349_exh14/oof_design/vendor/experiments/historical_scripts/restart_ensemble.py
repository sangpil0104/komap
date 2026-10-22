#!/usr/bin/env python3
"""Controlled native-resolution ensemble search for the restart track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from calibrate_al import predict_probabilities
from hard_negative_finetune import load_model
from improve_five_causes import CLASS_NAMES, NUM_CLASSES, collect_split, decode_mask, encode_mask


def confusion(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    bins = target.reshape(-1) * NUM_CLASSES + prediction.reshape(-1)
    return np.bincount(bins, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics(predictions, targets):
    total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    image_scores = []
    for prediction, target in zip(predictions, targets):
        current = confusion(prediction, target)
        total += current
        intersection = np.diag(current).astype(np.float64)
        union = current.sum(0) + current.sum(1) - intersection
        image_scores.append(float(np.mean(intersection / np.maximum(union, 1.0))))
    intersection = np.diag(total).astype(np.float64)
    union = total.sum(0) + total.sum(1) - intersection
    iou = intersection / np.maximum(union, 1.0)
    return {
        "global_miou": float(iou.mean()),
        "global_iou": {name: float(value) for name, value in zip(CLASS_NAMES, iou)},
        "image_mean_miou": float(np.mean(image_scores)),
        "image_scores": image_scores,
        "confusion": total.tolist(),
    }


def evaluate(cached, weight_new, bias):
    predictions, targets = [], []
    bias = np.asarray(bias, dtype=np.float32)[:, None, None]
    for old_probability, new_probability, target in cached:
        probability = (1.0 - weight_new) * old_probability + weight_new * new_probability
        prediction = (np.log(np.maximum(probability, 1e-7)) + bias).argmax(axis=0).astype(np.uint8)
        predictions.append(prediction)
        targets.append(target)
    return metrics(predictions, targets)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    old_model = load_model(args.old_checkpoint, device)
    new_model = load_model(args.new_checkpoint, device)
    cached = []
    for image_path, mask_path in collect_split(args.data_dir, "valid"):
        old_probability = predict_probabilities(old_model, image_path, args.old_patch_size, args.old_stride,
                                                 device, use_amp, args.tile_batch_size, True)
        new_probability = predict_probabilities(new_model, image_path, args.new_patch_size, args.new_stride,
                                                 device, use_amp, args.tile_batch_size, True)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        cached.append((old_probability, new_probability, target))
        print(f"cached {image_path.name}", flush=True)

    # Internal order is [Al3Ni, Primary Si, Al, Eutectic Si].
    # Keep the first pass small: each candidate still performs a full-res
    # argmax over all validation pixels. Fine-grained tuning is only useful
    # after this coarse search identifies the useful direction.
    weights = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
    biases = tuple(
        (0.0, 0.0, al, eutectic)
        for al in (-0.20, -0.10, 0.0)
        for eutectic in (-0.65, -0.55, -0.45, -0.35, -0.25, -0.15)
    )
    records = []
    for weight in weights:
        for bias in biases:
            result = evaluate(cached, float(weight), bias)
            records.append({"new_weight": float(weight), "bias_internal": bias, **result})
    best_image = max(records, key=lambda item: item["image_mean_miou"])
    best_global = max(records, key=lambda item: item["global_miou"])
    result = {"best_image_mean": best_image, "best_global": best_global,
              "records": records, "old_patch_size": args.old_patch_size, "old_stride": args.old_stride,
              "new_patch_size": args.new_patch_size, "new_stride": args.new_stride,
              "tta": True}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_image_mean": best_image, "best_global": best_global}, ensure_ascii=False), flush=True)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weight = best_image["new_weight"]
        bias = best_image["bias_internal"]
        test_images = sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png"))
        for image_path in test_images:
            old_probability = predict_probabilities(old_model, image_path, args.old_patch_size, args.old_stride,
                                                     device, use_amp, args.tile_batch_size, True)
            new_probability = predict_probabilities(new_model, image_path, args.new_patch_size, args.new_stride,
                                                     device, use_amp, args.tile_batch_size, True)
            probability = (1.0 - weight) * old_probability + weight * new_probability
            prediction = (np.log(np.maximum(probability, 1e-7)) + np.asarray(bias, dtype=np.float32)[:, None, None]).argmax(0)
            encode_mask(prediction.astype(np.uint8)).save(
                output_dir / image_path.name.replace("_image.png", "_mask.png"))
        print(f"wrote {len(test_images)} test masks to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--patch-size", type=int, default=None, help="legacy alias for both patch sizes")
    parser.add_argument("--stride", type=int, default=None, help="legacy alias for both strides")
    parser.add_argument("--old-patch-size", type=int, default=512)
    parser.add_argument("--old-stride", type=int, default=384)
    parser.add_argument("--new-patch-size", type=int, default=512)
    parser.add_argument("--new-stride", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.patch_size is not None:
        args.old_patch_size = args.new_patch_size = args.patch_size
    if args.stride is not None:
        args.old_stride = args.new_stride = args.stride
    main(args)
