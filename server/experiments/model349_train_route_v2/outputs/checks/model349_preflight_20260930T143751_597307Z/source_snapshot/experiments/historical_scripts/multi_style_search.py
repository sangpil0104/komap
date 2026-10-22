#!/usr/bin/env python3
"""Add intensity-standardized views to both 512px ASPP and 768px branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aspp768_addition import MATRIX, cleanup_primary
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image


def load_cache(data_dir, cache_dir):
    out = []
    for image_path, mask_path in collect_split(data_dir, "valid"):
        loaded = np.load(Path(cache_dir) / f"{image_path.stem}.npz")
        out.append({
            "values": [loaded[f"p{i}"].astype(np.float32) for i in range(4)],
            "prior": loaded["prior"].astype(np.float32),
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
            "name": image_path.name,
        })
    return out


def evaluate(entries, styled_aspp, styled_high, aspp_weight):
    predictions, targets = [], []
    bias = np.asarray([0.0, 0.0, 0.40, -0.50], dtype=np.float32)[:, None, None]
    for entry, p2_style, p3_style in zip(entries, styled_aspp, styled_high):
        values = entry["values"]
        base = MATRIX * 0.40
        p2 = (1.0 - aspp_weight) * values[2] + aspp_weight * p2_style
        p3 = 0.25 * values[3] + 0.75 * p3_style
        mixed = np.asarray([
            sum(base[c, m] * values[m][c] for m in range(3) if m != 2)
            + base[c, 2] * p2[c] + 0.60 * p3[c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * entry["prior"] + bias
        prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
        predictions.append(prediction); targets.append(entry["target"])
    return metrics(predictions, targets)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    entries = load_cache(args.data_dir, args.cache_dir)
    samples = collect_split(args.data_dir, "valid")
    high = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    high.load_state_dict(torch.load(args.high_checkpoint, map_location="cpu")["model"]); high.eval()
    aspp = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    aspp.load_state_dict(torch.load(args.aspp_checkpoint, map_location="cpu")["model"]); aspp.eval()
    stats = []
    for image_path, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        stats.append((float(gray.mean()), float(gray.std())))
    train_mean, train_std = float(np.mean([x[0] for x in stats])), float(np.mean([x[1] for x in stats]))
    styled_high, styled_aspp = [], []
    for entry, (image_path, _) in zip(entries, samples):
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        styled_high.append(predict_array(high, transform_image(raw, "standardize", train_mean, train_std),
                                         768, 576, device, use_amp, 1, "d4", "gaussian"))
        styled_aspp.append(predict_array(aspp, transform_image(raw, "standardize", train_mean, train_std),
                                         512, 384, device, use_amp, 2, "d4", "gaussian"))
        print(f"cached {entry['name']}", flush=True)
    records = []
    for aspp_weight in (0.0, 0.10, 0.20, 0.30, 0.40):
            value = evaluate(entries, styled_aspp, styled_high, aspp_weight)
            records.append({"high_ensemble_weight": 0.60, "high_style_weight": 0.75,
                            "aspp_style_weight": aspp_weight, **value})
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--aspp-checkpoint", required=True); p.add_argument("--high-checkpoint", required=True)
    p.add_argument("--output-json", required=True); p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
