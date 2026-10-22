#!/usr/bin/env python3
"""Search mean-conditioned pairwise-source weights with dynamic high weight."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    samples = collect_split(args.data_dir, "valid")
    entries = []
    for image_path, mask_path in _local_progress(samples, 'tent_dynamic_pair_search'):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        entries.append({"values": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
                        "prior": loaded["prior"].astype(np.float32), "raw": raw,
                        "std": float(raw[..., 0].astype(np.float32).std()),
                        "mean": float(raw[..., 0].astype(np.float32).mean()),
                        "target": decode_mask(Image.open(mask_path).convert("RGB")),
                        "name": image_path.name})
    pair = load_aspp(args.pair_checkpoint, device)
    train_stats = []
    for image_path, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))

    cached = []
    for entry, (image_path, _) in zip(entries, samples):
        pair_value = predict_probabilities(pair, image_path, 768, 576, device,
                                            use_amp, 1, "d4", "gaussian")
        model = adapt_model(args.high_checkpoint, entry["raw"], device, use_amp,
                            args.steps, args.lr)
        raw_high = predict_array(model, entry["raw"], 768, 576, device,
                                 use_amp, 1, "d4", "gaussian")
        style = transform_image(entry["raw"], "standardize", train_mean, train_std)
        style_high = predict_array(model, style, 768, 576, device,
                                   use_amp, 1, "d4", "gaussian")
        high = (0.25 * raw_high + 0.75 * style_high
                if entry["std"] >= args.route_threshold else raw_high)
        cached.append({"values": entry["values"], "pair": pair_value, "high": high,
                       "prior": entry["prior"], "mean": entry["mean"],
                       "target": entry["target"], "name": entry["name"]})
        print(f"cached {entry['name']}", flush=True)

    records = []
    for threshold in (160.0, 165.0, 167.037086, 170.0, 175.0):
        for low_pair in (0.05, 0.10, 0.15):
            for high_pair in (0.05, 0.10, 0.15):
                predictions, targets = [], []
                for item in cached:
                    bright = item["mean"] >= threshold
                    high_weight = 0.60 if bright else 0.70
                    pair_weight = high_pair if bright else low_pair
                    base = MATRIX * (1.0 - high_weight - pair_weight)
                    mixed = np.asarray([
                        sum(base[c, m] * item["values"][m][c] for m in range(3))
                        + high_weight * item["high"][c] + pair_weight * item["pair"][c]
                        for c in range(4)
                    ], dtype=np.float32)
                    mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
                    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * item["prior"]
                    logits[2] += 0.40
                    logits[3] += (-0.30 if bright else -0.70)
                    predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128))
                    targets.append(item["target"])
                value = metrics(predictions, targets)
                records.append({"mean_threshold": threshold, "low_pair_weight": low_pair,
                                "high_pair_weight": high_pair, **value})
        print(f"searched threshold={threshold}", flush=True)
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--high-checkpoint", required=True); p.add_argument("--pair-checkpoint", required=True)
    p.add_argument("--output-json", required=True); p.add_argument("--steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--route-threshold", type=float, default=50.0)
    p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
