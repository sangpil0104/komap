#!/usr/bin/env python3
"""Compare label-free raw/style route criteria after TENT adaptation."""

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


def full_logits(entry, high, pair_value, pair_weight):
    base = MATRIX * (0.40 - pair_weight)
    mixed = np.asarray([
        sum(base[c, m] * entry["values"][m][c] for m in range(3))
        + 0.60 * high[c] + pair_weight * pair_value[c]
        for c in range(4)
    ], dtype=np.float32)
    mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * entry["prior"]
    logits[2] += 0.40; logits[3] -= 0.50
    return logits


def entropy(logits):
    shifted = logits - logits.max(axis=0, keepdims=True)
    p = np.exp(shifted); p /= np.maximum(p.sum(axis=0, keepdims=True), 1e-7)
    return float((-(p * np.log(np.maximum(p, 1e-7))).sum(axis=0)).mean())


def confidence(logits):
    shifted = logits - logits.max(axis=0, keepdims=True)
    p = np.exp(shifted); p /= np.maximum(p.sum(axis=0, keepdims=True), 1e-7)
    return float(p.max(axis=0).mean())


def area(logits):
    return np.bincount(logits.argmax(axis=0).reshape(-1), minlength=4).astype(np.float64) / logits.shape[1] / logits.shape[2]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    samples = collect_split(args.data_dir, "valid")
    entries = []
    for image_path, mask_path in _local_progress(samples, 'tent_route_criterion_search'):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        entries.append({"values": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
                        "prior": loaded["prior"].astype(np.float32), "raw": raw,
                        "std": float(raw[..., 0].astype(np.float32).std()),
                        "target": decode_mask(Image.open(mask_path).convert("RGB")),
                        "name": image_path.name})
    pair = load_aspp(args.pair_checkpoint, device)
    train_stats = []
    train_areas = []
    for image_path, mask_path in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
        train_areas.append(area(np.eye(4, dtype=np.float32)[:, decode_mask(Image.open(mask_path).convert("RGB"))]))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))
    target_area = np.mean(train_areas, axis=0)

    cached = []
    for entry, (image_path, _) in zip(entries, samples):
        pair_value = predict_probabilities(pair, image_path, 768, 576, device,
                                            use_amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, entry["raw"], device, use_amp,
                              args.steps, args.lr)
        raw_high = predict_array(adapted, entry["raw"], 768, 576, device,
                                 use_amp, 1, "d4", "gaussian")
        style_high = predict_array(
            adapted, transform_image(entry["raw"], "standardize", train_mean, train_std),
            768, 576, device, use_amp, 1, "d4", "gaussian")
        raw_logits = full_logits(entry, raw_high, pair_value, args.pair_weight)
        styled_logits = full_logits(entry, 0.25 * raw_high + 0.75 * style_high,
                                     pair_value, args.pair_weight)
        cached.append({"raw_logits": raw_logits, "style_logits": styled_logits,
                       "target": entry["target"], "std": entry["std"], "name": entry["name"]})
        print(f"cached {entry['name']}", flush=True)

    records = []

    def evaluate(name, choose_style):
        predictions, targets = [], []
        for item, use_style in zip(cached, choose_style):
            logits = item["style_logits"] if use_style else item["raw_logits"]
            predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128))
            targets.append(item["target"])
        value = metrics(predictions, targets)
        row = {"criterion": name, "style_count": int(sum(choose_style)), **value}
        records.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)

    evaluate("raw", [False] * len(cached))
    evaluate("style", [True] * len(cached))
    for threshold in (40.0, 45.0, 50.0, 55.0, 60.0, 65.0):
        evaluate(f"std>={threshold}", [x["std"] >= threshold for x in cached])

    entropy_delta = np.asarray([entropy(x["style_logits"]) - entropy(x["raw_logits"]) for x in cached])
    confidence_delta = np.asarray([confidence(x["style_logits"]) - confidence(x["raw_logits"]) for x in cached])
    area_distance_delta = np.asarray([
        np.abs(area(x["style_logits"]) - target_area).sum()
        - np.abs(area(x["raw_logits"]) - target_area).sum() for x in cached
    ])
    for threshold in np.linspace(-0.02, 0.02, 9):
        evaluate(f"entropy_delta<={threshold:.4f}", [x <= threshold for x in entropy_delta])
    for threshold in np.linspace(-0.01, 0.01, 9):
        evaluate(f"confidence_delta>={threshold:.4f}", [x >= threshold for x in confidence_delta])
    evaluate("area_prior_style", [x < 0 for x in area_distance_delta])

    diagnostics = [{"name": x["name"], "std": x["std"], "entropy_delta": float(ed),
                    "confidence_delta": float(cd), "area_distance_delta": float(ad)}
                   for x, ed, cd, ad in zip(cached, entropy_delta, confidence_delta, area_distance_delta)]
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records, "diagnostics": diagnostics},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--high-checkpoint", required=True); p.add_argument("--pair-checkpoint", required=True)
    p.add_argument("--output-json", required=True); p.add_argument("--steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--pair-weight", type=float, default=.10)
    p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
