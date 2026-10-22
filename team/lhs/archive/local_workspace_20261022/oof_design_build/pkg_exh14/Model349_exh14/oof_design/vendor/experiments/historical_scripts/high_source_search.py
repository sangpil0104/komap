#!/usr/bin/env python3
"""Compare alternative high-resolution ASPP teachers in the current TENT route."""

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


def load_aspp(path: str, device: torch.device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    data_dir = Path(args.data_dir)
    train_stats = []
    for p, _ in collect_split(data_dir, "train"):
        gray = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))

    pair = load_aspp(args.pair_checkpoint, device)
    entries = []
    for image_path, mask_path in collect_split(data_dir, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        gray = raw[..., 0].astype(np.float32)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        entries.append({
            "path": image_path,
            "raw": raw,
            "gray": gray,
            "sources": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
            "prior": loaded["prior"].astype(np.float32),
            "pair": pair_value,
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
            "name": image_path.name,
        })
        print(f"cached {image_path.name}", flush=True)

    records = []
    for checkpoint in args.high_checkpoints:
        predictions, targets = [], []
        for entry in entries:
            raw = entry["raw"]
            adapted = adapt_model(checkpoint, raw, device, amp, 5, 1e-5)
            raw_high = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
            style = transform_image(raw, "standardize", train_mean, train_std)
            style_high = predict_array(adapted, style, 768, 576, device, amp, 1, "d4", "gaussian")
            gray = entry["gray"]
            mean = float(gray.mean())
            high_weight = .75 if mean < 165.0 else .55
            high = .25 * raw_high + .75 * style_high if float(gray.std()) >= 50.0 else raw_high
            base = MATRIX * (1.0 - high_weight - .15)
            mixed = np.asarray([
                sum(base[c, m] * entry["sources"][m][c] for m in range(3))
                + high_weight * high[c] + .15 * entry["pair"][c]
                for c in range(4)
            ], dtype=np.float32)
            mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
            z = (gray - mean) / max(float(gray.std()), 1e-5)
            stats = []
            for cid in range(4):
                weights = np.maximum(mixed[cid], 1e-7) ** 2.0
                weighted_mean = float((weights * z).sum() / weights.sum())
                weighted_std = float(np.sqrt((weights * (z - weighted_mean) ** 2).sum() / weights.sum()))
                stats.append((weighted_mean, max(weighted_std, .15)))
            adaptive = np.asarray([
                -0.5 * ((z - m) / s) ** 2 - np.log(s) for m, s in stats
            ], dtype=np.float32)
            beta = 3.0 if mean < 160.0 else 7.0
            prior = entry["prior"].copy()
            prior[0] += beta * (adaptive[0] - prior[0])
            logits = np.log(np.maximum(mixed, 1e-7)) + .09 * prior
            logits[2] += .40
            logits[3] += -1.0 if mean < 165.0 else -.30
            predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128))
            targets.append(entry["target"])
            del adapted, raw_high, style_high
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        value = metrics(predictions, targets)
        record = {"checkpoint": checkpoint, "steps": 5, "lr": 1e-5, **value}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--pair-checkpoint", required=True); p.add_argument("--high-checkpoints", nargs="+", required=True)
    p.add_argument("--output-json", required=True)
    main(p.parse_args())
