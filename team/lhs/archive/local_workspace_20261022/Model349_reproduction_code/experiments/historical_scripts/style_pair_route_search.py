#!/usr/bin/env python3
"""Combine intensity route with the pairwise-margin source."""

from __future__ import annotations

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


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"]); model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    entries, samples = [], collect_split(args.data_dir, "valid")
    for image_path, mask_path in samples:
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        entries.append({"values": [loaded[f"p{i}"].astype(np.float32) for i in range(4)],
                        "prior": loaded["prior"].astype(np.float32),
                        "target": decode_mask(Image.open(mask_path).convert("RGB")),
                        "raw": raw, "std": float(raw[..., 0].astype(np.float32).std()),
                        "name": image_path.name})
    high = load_aspp(args.high_checkpoint, device)
    pair = load_aspp(args.pair_checkpoint, device)
    stats = []
    for image_path, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        stats.append((float(gray.mean()), float(gray.std())))
    train_mean, train_std = float(np.mean([x[0] for x in stats])), float(np.mean([x[1] for x in stats]))
    high_style, pair_values = [], []
    for entry in entries:
        high_style.append(predict_array(high, transform_image(entry["raw"], "standardize", train_mean, train_std),
                                         768, 576, device, use_amp, 1, "d4", "gaussian"))
        pair_values.append(predict_probabilities(pair, samples[len(pair_values)][0], 768, 576,
                                                 device, use_amp, 1, "d4", "gaussian"))
        print(f"cached {entry['name']}", flush=True)
    records = []
    bias = np.asarray([0.0, 0.0, 0.40, -0.50], dtype=np.float32)[:, None, None]
    for pair_weight in (0.0, 0.02, 0.05, 0.08):
        old_scale = 0.40 - pair_weight
        for threshold in (45.0, 50.0, 55.0, 60.0, 65.0):
            predictions, targets = [], []
            for entry, styled, pair_value in zip(entries, high_style, pair_values):
                raw_high = entry["values"][3]
                high_value = 0.25 * raw_high + 0.75 * styled if entry["std"] >= threshold else raw_high
                values = entry["values"]; base = MATRIX * old_scale
                mixed = np.asarray([
                    sum(base[c, m] * values[m][c] for m in range(3))
                    + 0.60 * high_value[c] + pair_weight * pair_value[c]
                    for c in range(4)
                ], dtype=np.float32)
                mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
                logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * entry["prior"] + bias
                predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128))
                targets.append(entry["target"])
            value = metrics(predictions, targets)
            row = {"pair_weight": pair_weight, "style_std_threshold": threshold, **value}
            records.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--high-checkpoint", required=True); p.add_argument("--pair-checkpoint", required=True)
    p.add_argument("--output-json", required=True); p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
