#!/usr/bin/env python3
"""Search a small boundary-finetuned ASPP source in the current ensemble."""

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
from tent_style_search import adapt_model


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    samples = collect_split(args.data_dir, "valid")
    train_stats = []
    for p, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats])); train_std = float(np.mean([x[1] for x in train_stats]))
    pair = load_aspp(args.pair_checkpoint, device)
    boundary = load_aspp(args.boundary_checkpoint, device)
    items = []
    for image_path, mask_path in samples:
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        boundary_value = predict_probabilities(boundary, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, raw, device, amp, 3, 2e-5)
        raw_high = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        style_high = predict_array(adapted, transform_image(raw, "standardize", train_mean, train_std),
                                   768, 576, device, amp, 1, "d4", "gaussian")
        gray = raw[..., 0].astype(np.float32); mean = float(gray.mean()); std = float(gray.std())
        high = .25 * raw_high + .75 * style_high if std >= 50.0 else raw_high
        items.append({"p": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
                      "pair": pair_value, "high": high, "boundary": boundary_value,
                      "prior": loaded["prior"].astype(np.float32), "mean": mean,
                      "target": decode_mask(Image.open(mask_path).convert("RGB")), "name": image_path.name})
        print(f"cached {image_path.name}", flush=True)

    records = []
    for mode in ("replace_high", "replace_base"):
        for boundary_weight in (0.0, .03, .05, .08, .10, .15, .20):
            predictions, targets = [], []
            for item in items:
                bright = item["mean"] >= 165.0
                high_weight = .60 if bright else .70; pair_weight = .15
                if mode == "replace_high":
                    high_used = high_weight * (1.0 - boundary_weight)
                    base_weight = 1.0 - high_used - pair_weight - boundary_weight
                else:
                    high_used = high_weight
                    base_weight = 1.0 - high_used - pair_weight - boundary_weight
                base = MATRIX * max(base_weight, 0.0)
                mixed = np.asarray([
                    sum(base[c, m] * item["p"][m][c] for m in range(3))
                    + high_used * item["high"][c] + pair_weight * item["pair"][c]
                    + boundary_weight * item["boundary"][c]
                    for c in range(4)
                ], dtype=np.float32)
                mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
                logits = np.log(np.maximum(mixed, 1e-7)) + .09 * item["prior"]
                logits[2] += .40; logits[3] += -.30 if bright else -.70
                predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128))
                targets.append(item["target"])
            records.append({"mode": mode, "boundary_weight": boundary_weight, **metrics(predictions, targets)})
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--pair-checkpoint", required=True); p.add_argument("--boundary-checkpoint", required=True)
    p.add_argument("--high-checkpoint", required=True); p.add_argument("--output-json", required=True)
    p.add_argument("--no-amp", action="store_true"); main(p.parse_args())
