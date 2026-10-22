#!/usr/bin/env python3
"""Soft ensemble and class-wise logit calibration for model98/model99."""

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


def predict_all(args, device, use_amp):
    model98 = load_model(args.checkpoint98, device)
    model99 = load_model(args.checkpoint99, device)
    samples = collect_split(args.data_dir, args.split)
    cached = []
    for image_path, mask_path in _local_progress(samples, 'ensemble_calibrate'):
        p98 = predict_probabilities(model98, image_path, args.patch_size, args.stride, device,
                                    use_amp, args.tile_batch_size, args.tta)
        p99 = predict_probabilities(model99, image_path, args.patch_size, args.stride, device,
                                    use_amp, args.tile_batch_size, args.tta)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        cached.append((p98, p99, target, image_path))
        print(f"processed {image_path.name}", flush=True)
    return cached


def sweep(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    cached = predict_all(args, device, use_amp)
    records = []
    for weight98 in args.weights:
        for al_bias in args.al_biases:
            for primary_bias in args.primary_biases:
                for eutectic_bias in args.eutectic_biases:
                    predictions, targets = [], []
                    biases = np.asarray([0.0, primary_bias, al_bias, eutectic_bias], dtype=np.float32)[:, None, None]
                    for p98, p99, target, _ in cached:
                        probabilities = weight98 * p98 + (1.0 - weight98) * p99
                        prediction = (np.log(np.maximum(probabilities, 1e-7)) + biases).argmax(axis=0).astype(np.uint8)
                        predictions.append(prediction)
                        targets.append(target)
                    result = metrics(predictions, targets)
                    records.append({"weight98": float(weight98), "al_bias": float(al_bias),
                                    "primary_bias": float(primary_bias), "eutectic_bias": float(eutectic_bias),
                                    **result})
    best_image_compat = max(records, key=lambda x: x["image_miou_compat"])
    best_image = max(records, key=lambda x: x["image_mean_miou"])
    result = {"best_image_compat": best_image_compat, "best_image_mean": best_image, "records": records,
              "tta": args.tta, "patch_size": args.patch_size, "stride": args.stride}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_image_compat": best_image_compat, "best_image_mean": best_image}, ensure_ascii=False), flush=True)


def predict_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    model98 = load_model(args.checkpoint98, device)
    model99 = load_model(args.checkpoint99, device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    biases = np.asarray([0.0, args.primary_bias, args.al_bias, args.eutectic_bias], dtype=np.float32)[:, None, None]
    images = sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png"))
    for image_path in images:
        p98 = predict_probabilities(model98, image_path, args.patch_size, args.stride, device,
                                    use_amp, args.tile_batch_size, args.tta)
        p99 = predict_probabilities(model99, image_path, args.patch_size, args.stride, device,
                                    use_amp, args.tile_batch_size, args.tta)
        probabilities = args.weight98 * p98 + (1.0 - args.weight98) * p99
        prediction = (np.log(np.maximum(probabilities, 1e-7)) + biases).argmax(axis=0).astype(np.uint8)
        encode_mask(prediction).save(output / image_path.name.replace("_image.png", "_mask.png"))
    print(f"wrote {len(images)} masks to {output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sweep", "predict-test"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint98", required=True)
    parser.add_argument("--checkpoint99", required=True)
    parser.add_argument("--output-json", default="outputs/model100_ensemble_calibration.json")
    parser.add_argument("--output-dir", default="outputs/model100_ensemble_calibration/test_masks")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--al-biases", type=float, nargs="+", default=[-0.8, -0.6, -0.4, -0.2, 0.0])
    parser.add_argument("--primary-biases", type=float, nargs="+", default=[-0.10, 0.0, 0.10])
    parser.add_argument("--eutectic-biases", type=float, nargs="+", default=[-0.10, 0.0, 0.10])
    parser.add_argument("--weight98", type=float, default=0.5)
    parser.add_argument("--al-bias", type=float, default=-0.60)
    parser.add_argument("--primary-bias", type=float, default=0.0)
    parser.add_argument("--eutectic-bias", type=float, default=0.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--split", default="valid")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "sweep":
        sweep(args)
    else:
        predict_test(args)
