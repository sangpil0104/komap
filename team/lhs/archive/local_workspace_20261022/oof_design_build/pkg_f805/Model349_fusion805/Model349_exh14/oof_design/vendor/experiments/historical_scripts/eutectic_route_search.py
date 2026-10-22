#!/usr/bin/env python3
"""Search Eutectic calibration on the model 147 route."""

from __future__ import annotations

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
    model.load_state_dict(torch.load(path, map_location="cpu")["model"]); model.eval(); return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = device.type == "cuda"
    samples = collect_split(args.data_dir, "valid")
    train_stats = []
    for p, _ in collect_split(args.data_dir, "train"):
        g = np.asarray(Image.open(p).convert("L"), dtype=np.float32); train_stats.append((float(g.mean()), float(g.std())))
    train_mean = float(np.mean([x[0] for x in train_stats])); train_std = float(np.mean([x[1] for x in train_stats]))
    pair = load_aspp(args.pair_checkpoint, device); items = []
    for image_path, mask_path in samples:
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8); gray = raw[..., 0].astype(np.float32)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, raw, device, amp, 5, 1e-5)
        raw_high = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        style_high = predict_array(adapted, transform_image(raw, "standardize", train_mean, train_std),
                                   768, 576, device, amp, 1, "d4", "gaussian")
        high_weight = .55 if float(gray.mean()) >= 165.0 else .75
        high = .25 * raw_high + .75 * style_high if float(gray.std()) >= 50.0 else raw_high
        base = MATRIX * (1.0 - high_weight - .15)
        mixed = np.asarray([
            sum(base[c, m] * loaded[f"p{m}"].astype(np.float32)[c] for m in range(3))
            + high_weight * high[c] + .15 * pair_value[c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5); stats = []
        for cid in range(4):
            w = np.maximum(mixed[cid], 1e-7) ** 2.0
            m = float((w * z).sum() / w.sum()); s = float(np.sqrt((w * (z - m) ** 2).sum() / w.sum()))
            stats.append((m, max(s, .15)))
        adaptive = np.asarray([-0.5 * ((z - m) / s) ** 2 - np.log(s) for m, s in stats], dtype=np.float32)
        prior = loaded["prior"].astype(np.float32); prior[0] += 7.0 * (adaptive[0] - prior[0])
        logits_base = np.log(np.maximum(mixed, 1e-7)) + .09 * prior; logits_base[2] += .40
        items.append({"logits": logits_base, "mean": float(gray.mean()),
                      "target": decode_mask(Image.open(mask_path).convert("RGB")), "name": image_path.name})
        del adapted, raw_high, style_high; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
        print(f"cached {image_path.name}", flush=True)

    records = []
    for threshold in (160.0, 165.0, 167.037086, 170.0, 175.0):
        for low_bias in (-1.50, -1.30, -1.15, -1.0, -.85, -.70):
            for high_bias in (-.50, -.40, -.30, -.20, -.10, 0.0):
                predictions, targets = [], []
                for item in items:
                    logits = item["logits"].copy()
                    logits[3] += high_bias if item["mean"] >= threshold else low_bias
                    predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)); targets.append(item["target"])
                row = {"threshold": threshold, "low_eut_bias": low_bias, "high_eut_bias": high_bias, **metrics(predictions, targets)}
                records.append(row)
        print(f"searched threshold={threshold}", flush=True)
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--pair-checkpoint", required=True); p.add_argument("--high-checkpoint", required=True)
    p.add_argument("--output-json", required=True); main(p.parse_args())
