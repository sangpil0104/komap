#!/usr/bin/env python3
"""Small, memory-safe route search on top of model 145."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from aspp768_addition import MATRIX, cleanup_primary
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image
from tent_style_search import adapt_model


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def make_prior(mixed, gray, mean, std, base_prior):
    z = (gray - mean) / max(std, 1e-5)
    stats = []
    for cid in range(4):
        weights = np.maximum(mixed[cid], 1e-7) ** 2.0
        value_mean = float((weights * z).sum() / weights.sum())
        value_std = float(np.sqrt((weights * (z - value_mean) ** 2).sum() / weights.sum()))
        stats.append((value_mean, max(value_std, .15)))
    adaptive = np.asarray([-0.5 * ((z - m) / s) ** 2 - np.log(s) for m, s in stats], dtype=np.float32)
    prior = base_prior.copy()
    prior[0] += 5.0 * (adaptive[0] - prior[0])
    return prior


def evaluate_route(items, threshold, low_high, high_high, pair_weight):
    predictions, targets = [], []
    for item in items:
        bright = item["mean"] >= threshold
        high_weight = high_high if bright else low_high
        base = MATRIX * (1.0 - high_weight - pair_weight)
        mixed = np.asarray([
            sum(base[c, m] * item["p"][m][c] for m in range(3))
            + high_weight * item["high"][c] + pair_weight * item["pair"][c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
        prior = make_prior(mixed, item["gray"], item["mean"], item["std"], item["base_prior"])
        logits = np.log(np.maximum(mixed, 1e-7)) + .09 * prior
        logits[2] += .40
        logits[3] += -.30 if bright else -.70
        predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128))
        targets.append(item["target"])
    return metrics(predictions, targets)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    samples = collect_split(args.data_dir, "valid")
    train_stats = []
    for path, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))
    pair = load_aspp(args.pair_checkpoint, device)
    items = []
    for image_path, mask_path in _local_progress(samples, 'route_small_search'):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        gray = raw[..., 0].astype(np.float32)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, raw, device, amp, 5, 1e-5)
        raw_high = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        style_high = predict_array(adapted, transform_image(raw, "standardize", train_mean, train_std),
                                   768, 576, device, amp, 1, "d4", "gaussian")
        items.append({"p": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
                      "pair": pair_value,
                      "high": .25 * raw_high + .75 * style_high if float(gray.std()) >= 50.0 else raw_high,
                      "base_prior": loaded["prior"].astype(np.float32), "gray": gray,
                      "mean": float(gray.mean()), "std": float(gray.std()),
                      "target": decode_mask(Image.open(mask_path).convert("RGB")), "name": image_path.name})
        del adapted, raw_high, style_high
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"cached {image_path.name}", flush=True)

    configs = [(165.0, .70, .60, pair) for pair in (.10, .15, .20)]
    configs += [(165.0, low, high, .15) for low in (.65, .70, .75) for high in (.55, .60, .65)]
    configs += [(threshold, .70, .60, .15) for threshold in (160.0, 165.0, 170.0, 175.0)]
    records = []
    for threshold, low, high, pair_weight in configs:
        value = evaluate_route(items, threshold, low, high, pair_weight)
        row = {"threshold": threshold, "low_high_weight": low, "high_high_weight": high,
               "pair_weight": pair_weight, **value}
        records.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    best = max(records, key=lambda x: x["image_mean_miou"])
    output = Path(args.output_json); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True); parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pair-checkpoint", required=True); parser.add_argument("--high-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    main(parser.parse_args())
