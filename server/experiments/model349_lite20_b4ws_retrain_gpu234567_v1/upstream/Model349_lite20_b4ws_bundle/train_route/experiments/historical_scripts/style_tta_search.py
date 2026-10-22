#!/usr/bin/env python3
"""Search label-free intensity-domain TTA for the high-resolution branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import blend_window, inverse_view, transform_view, views
from aspp768_addition import MATRIX, cleanup_primary, fit_stats
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask, encode_mask, normalize_image, scaled_pair, tile_starts
from restart_ensemble import metrics


def load_cache(data_dir, cache_dir):
    entries = []
    for image_path, mask_path in collect_split(data_dir, "valid"):
        loaded = np.load(Path(cache_dir) / f"{image_path.stem}.npz")
        entries.append({
            "name": image_path.name,
            "values": [loaded[f"p{i}"].astype(np.float32) for i in range(4)],
            "prior": loaded["prior"].astype(np.float32),
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
        })
    return entries


def transform_image(image, kind, train_mean, train_std):
    gray = image[..., 0].astype(np.float32)
    if kind == "standardize":
        value = (gray - gray.mean()) / max(gray.std(), 1e-5) * train_std + train_mean
    elif kind == "percentile":
        low, high = np.percentile(gray, (2.0, 98.0))
        value = (gray - low) / max(high - low, 1e-5) * 255.0
    else:
        raise ValueError(kind)
    value = np.clip(value, 0.0, 255.0).astype(np.uint8)
    return np.repeat(value[..., None], 3, axis=2)


@torch.no_grad()
def predict_array(model, image, patch_size, stride, device, use_amp, tile_batch_size, tta, blend):
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), patch_size)
    height, width = scaled.shape[:2]
    ys, xs = tile_starts(height, patch_size, stride), tile_starts(width, patch_size, stride)
    padded_height, padded_width = max(height, ys[-1] + patch_size), max(width, xs[-1] + patch_size)
    padded = np.pad(scaled, ((0, padded_height - height), (0, padded_width - width), (0, 0)), mode="reflect")
    probability_sum = np.zeros((4, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    locations = [(top, left) for top in ys for left in xs]
    transforms = views(tta)
    window = blend_window(patch_size, blend)
    for start in range(0, len(locations), tile_batch_size):
        batch_locations = locations[start:start + tile_batch_size]
        inputs = torch.stack([normalize_image(padded[top:top + patch_size, left:left + patch_size])
                              for top, left in batch_locations]).to(device, non_blocking=True)
        probability_batch = None
        for rotation, flip in transforms:
            view = transform_view(inputs, rotation, flip)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                value = torch.softmax(model(view), dim=1)
            value = inverse_view(value, rotation, flip)
            probability_batch = value if probability_batch is None else probability_batch + value
        probability_batch = (probability_batch / len(transforms)).float().cpu().numpy()
        for index, (top, left) in enumerate(batch_locations):
            probability_sum[:, top:top + patch_size, left:left + patch_size] += probability_batch[index] * window
            weight_sum[top:top + patch_size, left:left + patch_size] += window
    value = probability_sum / np.maximum(weight_sum[None], 1e-7)
    value = value[:, :height, :width]
    original_size = image.shape[1], image.shape[0]
    restored = np.stack([
        np.asarray(Image.fromarray(value[c]).resize(original_size, Image.Resampling.BILINEAR), dtype=np.float32)
        for c in range(4)
    ])
    return restored / np.maximum(restored.sum(axis=0, keepdims=True), 1e-7)


def evaluate(entries, styled, style_weight, threshold=128):
    predictions, targets = [], []
    bias = np.asarray([0.0, 0.0, 0.40, -0.55], dtype=np.float32)[:, None, None]
    for entry, style_value in zip(entries, styled):
        values = entry["values"]
        base = MATRIX * 0.40
        mixed = np.asarray([
            sum(base[c, m] * values[m][c] for m in range(3))
            + 0.60 * ((1.0 - style_weight) * values[3][c] + style_weight * style_value[c])
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        logits = np.log(np.maximum(mixed, 1e-7)) + 0.12 * entry["prior"] + bias
        prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, threshold)
        predictions.append(prediction); targets.append(entry["target"])
    return metrics(predictions, targets)


def evaluate_route(entries, styled, std_threshold, style_weight=0.75):
    predictions, targets = [], []
    bias = np.asarray([0.0, 0.0, 0.40, -0.50], dtype=np.float32)[:, None, None]
    for entry, style_value in zip(entries, styled):
        values = entry["values"]; use_style = entry["raw_std"] >= std_threshold
        high = (1.0 - style_weight) * values[3] + style_weight * style_value if use_style else values[3]
        base = MATRIX * 0.40
        mixed = np.asarray([
            sum(base[c, m] * values[m][c] for m in range(3)) + 0.60 * high[c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * entry["prior"] + bias
        predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128))
        targets.append(entry["target"])
    return metrics(predictions, targets)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    entries = load_cache(args.data_dir, args.cache_dir)
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.eval()
    samples = collect_split(args.data_dir, "valid")
    train_values = []
    for image_path, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_values.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([v[0] for v in train_values]))
    train_std = float(np.mean([v[1] for v in train_values]))
    records = []
    for kind in args.styles.split(","):
        styled = []
        for entry, (image_path, _) in zip(entries, samples):
            raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            entry["raw_std"] = float(raw[..., 0].astype(np.float32).std())
            transformed = transform_image(raw, kind, train_mean, train_std)
            styled.append(predict_array(model, transformed, 768, 576, device, use_amp, 1, "d4", "gaussian"))
            print(f"styled={kind} image={entry['name']}", flush=True)
        for weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60,
                       0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0):
            value = evaluate(entries, styled, weight)
            records.append({"style": kind, "style_weight": weight, **value})
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
        if kind == "standardize":
            route_records = []
            for threshold in (50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0):
                value = evaluate_route(entries, styled, threshold)
                route_records.append({"route": "std_threshold", "threshold": threshold, **value})
                print(json.dumps(route_records[-1], ensure_ascii=False), flush=True)
            route_best = max(route_records, key=lambda item: item["image_mean_miou"])
        else:
            route_records, route_best = [], None
    best_fixed = max(records, key=lambda item: item["image_mean_miou"])
    calibration = []
    for alpha in (0.09, 0.12, 0.15):
        for al_bias in (0.30, 0.40, 0.50, 0.60):
            for eut_bias in (-0.70, -0.60, -0.55, -0.50, -0.45):
                # Re-evaluate the selected style with the candidate bias/alpha.
                predictions, targets = [], []
                bias = np.asarray([0.0, 0.0, al_bias, eut_bias], dtype=np.float32)[:, None, None]
                for entry, style_value in zip(entries, styled):
                    values = entry["values"]; base = MATRIX * 0.40
                    mixed = np.asarray([
                        sum(base[c, m] * values[m][c] for m in range(3))
                        + 0.60 * ((1.0 - best_fixed["style_weight"]) * values[3][c]
                                  + best_fixed["style_weight"] * style_value[c])
                        for c in range(4)
                    ], dtype=np.float32)
                    mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
                    logits = np.log(np.maximum(mixed, 1e-7)) + alpha * entry["prior"] + bias
                    predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128))
                    targets.append(entry["target"])
                value = metrics(predictions, targets)
                calibration.append({"style": best_fixed["style"], "style_weight": best_fixed["style_weight"],
                                    "alpha": alpha, "biases": [0.0, 0.0, al_bias, eut_bias], **value})
    best_calibrated = max(calibration, key=lambda item: item["image_mean_miou"])
    best = max([best_fixed, best_calibrated], key=lambda item: item["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "best_fixed": best_fixed, "route_best": route_best,
                               "best_calibrated": best_calibrated, "records": records,
                               "calibration": calibration}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "best_fixed": best_fixed, "best_calibrated": best_calibrated}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--pair-cache", required=False); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-json", required=True); p.add_argument("--styles", default="standardize,percentile")
    p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
