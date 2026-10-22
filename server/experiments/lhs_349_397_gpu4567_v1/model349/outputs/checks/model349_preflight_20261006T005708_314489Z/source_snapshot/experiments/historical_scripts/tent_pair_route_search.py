#!/usr/bin/env python3
"""Search pairwise-source weights after label-free TENT adaptation."""

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
    for image_path, mask_path in _local_progress(samples, 'tent_pair_route_search'):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        entries.append({
            "values": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
            "prior": loaded["prior"].astype(np.float32),
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
            "raw": raw,
            "std": float(raw[..., 0].astype(np.float32).std()),
            "name": image_path.name,
        })

    pair = load_aspp(args.pair_checkpoint, device)
    train_stats = []
    for image_path, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))

    pair_values = []
    raw_high_values, style_high_values = [], []
    for entry, (image_path, _) in zip(entries, samples):
        pair_values.append(predict_probabilities(
            pair, image_path, 768, 576, device, use_amp, 1, "d4", "gaussian"
        ))
        adapted = adapt_model(args.high_checkpoint, entry["raw"], device, use_amp,
                              args.steps, args.lr)
        raw_high_values.append(predict_array(
            adapted, entry["raw"], 768, 576, device, use_amp, 1, "d4", "gaussian"
        ))
        style_high_values.append(predict_array(
            adapted, transform_image(entry["raw"], "standardize", train_mean, train_std),
            768, 576, device, use_amp, 1, "d4", "gaussian"
        ))
        print(f"adapted {entry['name']}", flush=True)

    records = []
    bias = np.asarray([0.0, 0.0, 0.40, -0.50], dtype=np.float32)[:, None, None]
    for pair_weight in (0.0, 0.02, 0.05, 0.08, 0.10, 0.12):
        base_scale = 0.40 - pair_weight
        for threshold in (45.0, 50.0, 55.0, 60.0, 65.0):
            predictions, targets = [], []
            for entry, raw_high, style_high, pair_value in zip(
                entries, raw_high_values, style_high_values, pair_values
            ):
                high = (0.25 * raw_high + 0.75 * style_high
                        if entry["std"] >= threshold else raw_high)
                base = MATRIX * base_scale
                mixed = np.asarray([
                    sum(base[c, m] * entry["values"][m][c] for m in range(3))
                    + 0.60 * high[c] + pair_weight * pair_value[c]
                    for c in range(4)
                ], dtype=np.float32)
                mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
                logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * entry["prior"] + bias
                prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
                predictions.append(prediction)
                targets.append(entry["target"])
            value = metrics(predictions, targets)
            row = {"steps": args.steps, "lr": args.lr,
                   "pair_weight": pair_weight, "style_std_threshold": threshold, **value}
            records.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--high-checkpoint", required=True)
    parser.add_argument("--pair-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
