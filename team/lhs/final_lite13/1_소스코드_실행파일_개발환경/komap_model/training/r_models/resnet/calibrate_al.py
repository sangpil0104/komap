#!/usr/bin/env python3
"""Validation calibration and test inference for the Eutectic Si class.

The corrected channel order is [Al3Ni, Primary Si, Al, Eutectic Si]. Adding a
scalar bias to the Eutectic Si logit is a cheap, validation-only calibration
that can reduce false positives without retraining the network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import autocast

from improve_five_causes import (
    CLASS_NAMES,
    NUM_CLASSES,
    collect_split,
    decode_mask,
    encode_mask,
    normalize_image,
    scaled_pair,
    tile_starts,
)
from train_resnet_base import UNetPlusPlusResNet34


def load_model(checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = UNetPlusPlusResNet34(pretrained=False).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    image_path: Path,
    patch_size: int,
    stride: int,
    device: torch.device,
    use_amp: bool,
    tile_batch_size: int,
    tta: bool,
) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), patch_size)
    height, width = scaled.shape[:2]
    ys, xs = tile_starts(height, patch_size, stride), tile_starts(width, patch_size, stride)
    padded_height, padded_width = max(height, ys[-1] + patch_size), max(width, xs[-1] + patch_size)
    padded = np.pad(scaled, ((0, padded_height - height), (0, padded_width - width), (0, 0)), mode="reflect")
    probability_sum = np.zeros((NUM_CLASSES, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    locations = [(top, left) for top in ys for left in xs]
    for start in range(0, len(locations), tile_batch_size):
        batch_locations = locations[start:start + tile_batch_size]
        inputs = torch.stack([
            normalize_image(padded[top:top + patch_size, left:left + patch_size])
            for top, left in batch_locations
        ]).to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            probabilities = torch.softmax(model(inputs), dim=1)
            if tta:
                views = [torch.flip(inputs, dims=(-1,)), torch.flip(inputs, dims=(-2,)),
                         torch.rot90(inputs, 1, dims=(-2, -1))]
                undo = [lambda x: torch.flip(x, dims=(-1,)), lambda x: torch.flip(x, dims=(-2,)),
                        lambda x: torch.rot90(x, -1, dims=(-2, -1))]
                for view, undo_fn in zip(views, undo):
                    probabilities = probabilities + undo_fn(torch.softmax(model(view), dim=1))
                probabilities = probabilities / 4.0
        probabilities = probabilities.float().cpu().numpy()
        for index, (top, left) in enumerate(batch_locations):
            probability_sum[:, top:top + patch_size, left:left + patch_size] += probabilities[index]
            weight_sum[top:top + patch_size, left:left + patch_size] += 1.0
    probabilities = probability_sum / np.maximum(weight_sum[None], 1e-6)
    probabilities = probabilities[:, :height, :width]
    # Resize each probability map back to the original image size before the
    # class decision. This avoids a nearest-neighbor label resize changing the
    # effect of the Al calibration at the original boundary.
    original_size = image.shape[1], image.shape[0]
    restored = np.stack([
        np.asarray(Image.fromarray(probabilities[class_id]).resize(original_size, Image.Resampling.BILINEAR),
                   dtype=np.float32)
        for class_id in range(NUM_CLASSES)
    ])
    return restored / np.maximum(restored.sum(axis=0, keepdims=True), 1e-7)


def confusion(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    bins = target.reshape(-1) * NUM_CLASSES + prediction.reshape(-1)
    return np.bincount(bins, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def score_confusions(confusions: Sequence[np.ndarray], image_scores: Sequence[Sequence[float]]) -> Dict[str, object]:
    total = np.sum(np.stack(confusions), axis=0)
    intersection = np.diag(total).astype(np.float64)
    union = total.sum(0) + total.sum(1) - intersection
    iou = intersection / np.maximum(union, 1.0)
    return {
        "global_miou": float(iou.mean()),
        "global_iou": {name: float(value) for name, value in zip(CLASS_NAMES, iou)},
        "image_mean_miou": float(np.mean(image_scores)),
        "image_std_miou": float(np.std(image_scores)),
        "confusion": total.tolist(),
    }


def score_prediction(prediction: np.ndarray, target: np.ndarray) -> float:
    current = confusion(prediction, target)
    intersection = np.diag(current).astype(np.float64)
    union = current.sum(0) + current.sum(1) - intersection
    return float(np.mean(intersection / np.maximum(union, 1.0)))


def sweep(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    model = load_model(args.checkpoint, device)
    samples = collect_split(args.data_dir, "valid")
    biases = np.arange(args.min_bias, args.max_bias + args.step * 0.5, args.step)
    confusions = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for _ in biases]
    image_scores: List[List[float]] = [[] for _ in biases]
    for image_path, mask_path in samples:
        probabilities = predict_probabilities(model, image_path, args.patch_size, args.stride, device,
                                               use_amp, args.tile_batch_size, args.tta)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        log_probabilities = np.log(np.maximum(probabilities, 1e-7))
        for index, bias in enumerate(biases):
            adjusted = log_probabilities.copy()
            adjusted[3] += float(bias)
            prediction = adjusted.argmax(axis=0).astype(np.uint8)
            confusions[index] += confusion(prediction, target)
            image_scores[index].append(score_prediction(prediction, target))
        print(f"processed {image_path.name}", flush=True)

    records = []
    for bias, current_confusion, current_scores in zip(biases, confusions, image_scores):
        metrics = score_confusions([current_confusion], current_scores)
        records.append({"eutectic_bias": float(bias), **metrics})
    best_global = max(records, key=lambda record: record["global_miou"])
    best_image = max(records, key=lambda record: record["image_mean_miou"])
    result = {"records": records, "best_global": best_global, "best_image_mean": best_image,
              "tta": args.tta, "patch_size": args.patch_size, "stride": args.stride}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_global": best_global, "best_image_mean": best_image}, ensure_ascii=False), flush=True)


def predict_test(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    model = load_model(args.checkpoint, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        probabilities = predict_probabilities(model, image_path, args.patch_size, args.stride, device,
                                               use_amp, args.tile_batch_size, args.tta)
        adjusted = np.log(np.maximum(probabilities, 1e-7))
        adjusted[3] += args.eutectic_bias
        prediction = adjusted.argmax(axis=0).astype(np.uint8)
        encode_mask(prediction).save(output_dir / image_path.name.replace("_image.png", "_mask.png"))
    print(f"wrote {len(list(output_dir.glob('*.png')))} masks to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sweep", "predict-test"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", default="outputs/calibrate_al/eutectic_bias_sweep.json")
    parser.add_argument("--output-dir", default="outputs/calibrate_al/test_masks_al_calibrated")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--min-bias", type=float, default=-0.60)
    parser.add_argument("--max-bias", type=float, default=0.40)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--eutectic-bias", "--al-bias", dest="eutectic_bias", type=float, default=0.0)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "sweep":
        sweep(args)
    else:
        predict_test(args)
