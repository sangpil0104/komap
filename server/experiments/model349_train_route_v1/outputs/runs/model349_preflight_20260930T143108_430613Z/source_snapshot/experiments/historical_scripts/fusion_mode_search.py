#!/usr/bin/env python3
"""Compare probability, geometric-probability, and calibrated source fusion."""

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
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval(); return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = device.type == "cuda"
    data_dir = Path(args.data_dir)
    stats = []
    for p, _ in collect_split(data_dir, "train"):
        g = np.asarray(Image.open(p).convert("L"), dtype=np.float32); stats.append((float(g.mean()), float(g.std())))
    train_mean = float(np.mean([x[0] for x in stats])); train_std = float(np.mean([x[1] for x in stats]))
    pair = load_aspp(args.pair_checkpoint, device); items = []
    for image_path, mask_path in collect_split(data_dir, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8); gray = raw[..., 0].astype(np.float32)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, raw, device, amp, 5, 1e-5)
        raw_high = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        style_high = predict_array(adapted, transform_image(raw, "standardize", train_mean, train_std),
                                   768, 576, device, amp, 1, "d4", "gaussian")
        high = .25 * raw_high + .75 * style_high if float(gray.std()) >= 50.0 else raw_high
        high_weight = .75 if float(gray.mean()) < 165.0 else .55
        sources = [loaded[f"p{i}"].astype(np.float32) for i in range(3)]
        base = MATRIX * (1.0 - high_weight - .15)
        sources.append(high); pair_weight = .15
        weights = np.concatenate([
            base,
            np.full((4, 1), high_weight, dtype=np.float32),
            np.full((4, 1), pair_weight, dtype=np.float32),
        ], axis=1)
        sources.append(pair_value)
        items.append({"sources": sources, "weights": weights, "prior": loaded["prior"].astype(np.float32),
                      "gray": gray, "target": decode_mask(Image.open(mask_path).convert("RGB")),
                      "name": image_path.name})
        del adapted, raw_high, style_high
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
        print(f"cached {image_path.name}", flush=True)

    def evaluate(mode, temperature):
        predictions, targets = [], []
        for item in items:
            sources = [np.maximum(x, 1e-7) for x in item["sources"]]
            weights = item["weights"]
            if mode == "probability":
                mixed = np.asarray([sum(weights[c, m] * sources[m][c] for m in range(5)) for c in range(4)], np.float32)
            elif mode == "geometric":
                log_mixed = np.asarray([sum(weights[c, m] * np.log(sources[m][c]) for m in range(5)) for c in range(4)], np.float32)
                mixed = np.exp(log_mixed - log_mixed.max(0, keepdims=True))
            elif mode == "tempered_probability":
                tempered = [np.exp(np.log(x) / temperature) for x in sources]
                tempered = [x / np.maximum(x.sum(0, keepdims=True), 1e-7) for x in tempered]
                mixed = np.asarray([sum(weights[c, m] * tempered[m][c] for m in range(5)) for c in range(4)], np.float32)
            else:
                raise ValueError(mode)
            mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
            gray = item["gray"]; mean = float(gray.mean()); z = (gray - mean) / max(float(gray.std()), 1e-5)
            adaptive_stats = []
            for cid in range(4):
                w = np.maximum(mixed[cid], 1e-7) ** 2.0
                wm = float((w * z).sum() / w.sum()); ws = float(np.sqrt((w * (z - wm) ** 2).sum() / w.sum()))
                adaptive_stats.append((wm, max(ws, .15)))
            adaptive = np.asarray([-0.5 * ((z - m) / s) ** 2 - np.log(s) for m, s in adaptive_stats], np.float32)
            prior = item["prior"].copy(); beta = 3.0 if mean < 160.0 else 7.0
            prior[0] += beta * (adaptive[0] - prior[0])
            logits = np.log(np.maximum(mixed, 1e-7)) + .09 * prior
            logits[2] += .40; logits[3] += -1.0 if mean < 165.0 else -.30
            prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
            if float((prediction == 1).mean()) <= .01:
                mask = prediction == 1; alt = logits.copy(); alt[1] = -np.inf
                prediction[mask] = alt[:, mask].argmax(0).astype(np.uint8)
            predictions.append(prediction); targets.append(item["target"])
        return metrics(predictions, targets)

    records = []
    for mode in ("probability", "geometric"):
        records.append({"mode": mode, "temperature": 1.0, **evaluate(mode, 1.0)})
    for temperature in (.75, .85, 1.15, 1.30):
        records.append({"mode": "tempered_probability", "temperature": temperature,
                        **evaluate("tempered_probability", temperature)})
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "records": records}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--pair-checkpoint", required=True); p.add_argument("--high-checkpoint", required=True)
    p.add_argument("--output-json", required=True); main(p.parse_args())
