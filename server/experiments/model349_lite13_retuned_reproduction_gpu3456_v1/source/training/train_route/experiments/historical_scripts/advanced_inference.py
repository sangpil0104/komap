#!/usr/bin/env python3
"""Compare D4 TTA, Gaussian tile blending, and ensemble variants."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from hard_negative_finetune import load_model
from improve_five_causes import CLASS_NAMES, NUM_CLASSES, collect_split, decode_mask, encode_mask, normalize_image, scaled_pair, tile_starts


def transform_view(x: torch.Tensor, rotation: int, flip: bool) -> torch.Tensor:
    if flip:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, rotation, dims=(-2, -1))


def inverse_view(x: torch.Tensor, rotation: int, flip: bool) -> torch.Tensor:
    x = torch.rot90(x, -rotation, dims=(-2, -1))
    if flip:
        x = torch.flip(x, dims=(-1,))
    return x


def views(mode: str):
    if mode == "legacy":
        return [(0, False), (0, True), (2, True), (1, False)]
    if mode == "d4":
        return [(rotation, flip) for flip in (False, True) for rotation in range(4)]
    raise ValueError(mode)


def blend_window(patch_size: int, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones((patch_size, patch_size), dtype=np.float32)
    axis = np.linspace(-1.0, 1.0, patch_size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    window = np.exp(-(xx * xx + yy * yy) / (2.0 * 0.55 * 0.55)).astype(np.float32)
    return 0.15 + 0.85 * (window / window.max())


@torch.no_grad()
def predict_probabilities(model, image_path, patch_size, stride, device, use_amp, tile_batch_size,
                          tta_mode, blend_mode, scale=1.0) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    original_size = image.shape[1], image.shape[0]
    if abs(float(scale) - 1.0) > 1e-6:
        if float(scale) <= 0:
            raise ValueError("scale must be positive")
        new_size = (int(round(image.shape[1] * float(scale))),
                    int(round(image.shape[0] * float(scale))))
        image = np.asarray(Image.fromarray(image).resize(
            new_size, Image.Resampling.BICUBIC), dtype=np.uint8)
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), patch_size)
    height, width = scaled.shape[:2]
    ys, xs = tile_starts(height, patch_size, stride), tile_starts(width, patch_size, stride)
    padded_height, padded_width = max(height, ys[-1] + patch_size), max(width, xs[-1] + patch_size)
    padded = np.pad(scaled, ((0, padded_height - height), (0, padded_width - width), (0, 0)), mode="reflect")
    probability_sum = np.zeros((NUM_CLASSES, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    window = blend_window(patch_size, blend_mode)
    locations = [(top, left) for top in ys for left in xs]
    transforms = views(tta_mode)
    for start in range(0, len(locations), tile_batch_size):
        batch_locations = locations[start:start + tile_batch_size]
        inputs = torch.stack([
            normalize_image(padded[top:top + patch_size, left:left + patch_size])
            for top, left in batch_locations
        ]).to(device, non_blocking=True)
        probability_batch = None
        for rotation, flip in transforms:
            view = transform_view(inputs, rotation, flip)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                probabilities = torch.softmax(model(view), dim=1)
            probabilities = inverse_view(probabilities, rotation, flip)
            probability_batch = probabilities if probability_batch is None else probability_batch + probabilities
        probability_batch = (probability_batch / float(len(transforms))).float().cpu().numpy()
        for index, (top, left) in enumerate(batch_locations):
            probability_sum[:, top:top + patch_size, left:left + patch_size] += probability_batch[index] * window[None]
            weight_sum[top:top + patch_size, left:left + patch_size] += window
    probabilities = probability_sum / np.maximum(weight_sum[None], 1e-6)
    probabilities = probabilities[:, :height, :width]
    restored = np.stack([
        np.asarray(Image.fromarray(probabilities[class_id]).resize(original_size, Image.Resampling.BILINEAR), dtype=np.float32)
        for class_id in range(NUM_CLASSES)
    ])
    return restored / np.maximum(restored.sum(axis=0, keepdims=True), 1e-7)


def confusion(prediction, target):
    bins = target.reshape(-1) * NUM_CLASSES + prediction.reshape(-1)
    return np.bincount(bins, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics(predictions, targets):
    from local_image_metrics import summary
    return summary(predictions, targets, CLASS_NAMES)


def ensemble_probability(probabilities, weights):
    result = sum(weight * probability for weight, probability in zip(weights, probabilities))
    return result / np.maximum(result.sum(axis=0, keepdims=True), 1e-7)


def ensemble_logit(probabilities, weights):
    result = sum(weight * np.log(np.maximum(probability, 1e-7)) for weight, probability in zip(weights, probabilities))
    result -= result.max(axis=0, keepdims=True)
    result = np.exp(result)
    return result / np.maximum(result.sum(axis=0, keepdims=True), 1e-7)


def class_conditional(probabilities):
    # Give Al more weight from the boundary-refined model99 while keeping
    # texture-heavy Eutectic/Primary predictions closer to model98/model101.
    class_weights = np.asarray([
        [0.80, 0.05, 0.15],
        [0.75, 0.05, 0.20],
        [0.65, 0.05, 0.30],
        [0.50, 0.40, 0.10],
    ], dtype=np.float32)
    result = sum(class_weights[:, None, None, index] * probabilities[index]
                 for index in range(len(probabilities)))
    return result / np.maximum(result.sum(axis=0, keepdims=True), 1e-7)


def evaluate_candidate(cached, kind, weights, biases):
    predictions, targets = [], []
    bias = np.asarray(biases, dtype=np.float32)[:, None, None]
    for probabilities, target in cached:
        if kind == "probability":
            combined = ensemble_probability(probabilities, weights)
        elif kind == "logit":
            combined = ensemble_logit(probabilities, weights)
        elif kind == "class_conditional":
            combined = class_conditional(probabilities)
        else:
            raise ValueError(kind)
        prediction = (np.log(np.maximum(combined, 1e-7)) + bias).argmax(axis=0).astype(np.uint8)
        predictions.append(prediction)
        targets.append(target)
    return metrics(predictions, targets)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    specs = [
        ("model98", args.checkpoint98, 512, 384),
        ("model99", args.checkpoint99, 512, 384),
        ("model101", args.checkpoint101, 640, 512),
    ]
    samples = collect_split(args.data_dir, "valid")
    all_results = []
    for tta_mode, blend_mode in (("legacy", "uniform"), ("d4", "uniform"), ("d4", "gaussian")):
        cached = []
        model_probabilities = [[] for _ in specs]
        targets = []
        models = [load_model(checkpoint, device) for _, checkpoint, _, _ in specs]
        for image_path, mask_path in samples:
            for index, (_, _, patch_size, stride) in enumerate(specs):
                model_probabilities[index].append(
                    predict_probabilities(models[index], image_path, patch_size, stride, device, use_amp,
                                          4 if patch_size == 512 else 2, tta_mode, blend_mode))
            targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
            print(f"{tta_mode}/{blend_mode} processed {image_path.name}", flush=True)
        for index in range(len(samples)):
            cached.append(([model_probabilities[m][index] for m in range(len(specs))], targets[index]))
        candidate_specs = [
            ("probability", (0.75, 0.10, 0.15), (0.0, 0.0, -0.20, 0.10)),
            ("logit", (0.75, 0.10, 0.15), (0.0, 0.0, -0.20, 0.10)),
            ("class_conditional", (0.0, 0.0, 0.0), (0.0, 0.0, -0.20, 0.10)),
        ]
        for kind, weights, biases in candidate_specs:
            result = evaluate_candidate(cached, kind, weights, biases)
            all_results.append({"tta": tta_mode, "blend": blend_mode, "kind": kind,
                                "weights": weights, "biases": biases, **result})
            print(json.dumps(all_results[-1], ensure_ascii=False), flush=True)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    best = max(all_results, key=lambda x: x["image_miou_compat"])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best": best, "records": all_results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint98", required=True)
    parser.add_argument("--checkpoint99", required=True)
    parser.add_argument("--checkpoint101", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
