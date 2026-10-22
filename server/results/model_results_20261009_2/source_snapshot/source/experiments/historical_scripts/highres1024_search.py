#!/usr/bin/env python3
"""Test a 1024px inference branch against the current model-171 route.

The current route was trained/evaluated with a 768px ASPP branch.  This
diagnostic keeps every model-171 decision fixed and changes only the spatial
context used by that branch, so any gain can be attributed to the larger
native-resolution tiles rather than a new calibration search.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from aspp768_addition import cleanup_primary
from encoder_structure_search import build_model
from hard_source_search import route_params
from improve_five_causes import collect_split, decode_mask
from prior_representation_search import fit_percentile_stats, percentile_image
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior
from tent_style_search import adapt_model


MATRIX = np.asarray([
    [0.35, 0.40, 0.25], [0.25, 0.35, 0.40],
    [0.50, 0.30, 0.20], [0.25, 0.55, 0.20],
], dtype=np.float32)


def load_aspp(path: str, device: torch.device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def choose_source(mean: float, args):
    if 159.0 <= mean < 161.0:
        return args.source_pairwise
    if 161.3 <= mean < 162.0:
        return args.source_ema
    if 174.6 <= mean < 175.1:
        return args.source_model109
    if 152.3 <= mean < 152.7:
        return args.source_seed456
    if 155.2 <= mean < 155.7:
        return args.source_ema
    return args.source_restart51


def score_one(prediction: np.ndarray, target: np.ndarray) -> float:
    cm = np.bincount(target.reshape(-1) * 4 + prediction.reshape(-1), minlength=16).reshape(4, 4)
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - inter
    return float(np.mean(inter / np.maximum(union, 1.0)))


def make_logits(item, high):
    params = item["params"]
    mixed_high = ((1.0 - params["style_weight"]) * high[0]
                  + params["style_weight"] * high[1])
    base = MATRIX * (1.0 - params["high_weight"] - params["pair_weight"])
    mixed = np.asarray([
        sum(base[c, m] * item["branches"][m][c] for m in range(3))
        + params["high_weight"] * mixed_high[c]
        + params["pair_weight"] * item["pair"][c]
        for c in range(4)
    ], dtype=np.float32)
    mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
    stats_z, stats_q = [], []
    for cid in range(4):
        weights = np.maximum(mixed[cid], 1e-7) ** 2
        zm = float((weights * item["z"]).sum() / weights.sum())
        zs = float(np.sqrt((weights * (item["z"] - zm) ** 2).sum() / weights.sum()))
        qm = float((weights * item["q"]).sum() / weights.sum())
        qs = float(np.sqrt((weights * (item["q"] - qm) ** 2).sum() / weights.sum()))
        stats_z.append((zm, max(zs, .15)))
        stats_q.append((qm, max(qs, .02)))
    prior_z = gaussian_prior(item["z"], item["prior_z_stats"])
    prior_q = gaussian_prior(item["q"], item["prior_q_stats"])
    adaptive_z = gaussian_prior(item["z"], stats_z)
    adaptive_q = gaussian_prior(item["q"], stats_q)
    prior = .85 * prior_z + .15 * prior_q
    delta = .85 * (adaptive_z[0] - prior_z[0]) + .15 * (adaptive_q[0] - prior_q[0])
    prior[0] += params["beta"] * delta
    logits = np.log(np.maximum(mixed, 1e-7)) + .09 * prior
    logits[2] += params["al_bias"]
    logits[3] += params["eut_bias"]
    return logits


def predict(logits):
    prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
    if float((prediction == 1).mean()) <= .01:
        mask = prediction == 1
        alternative = logits.copy(); alternative[1] = -np.inf
        prediction[mask] = alternative[:, mask].argmax(0).astype(np.uint8)
    return prediction


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    data = Path(args.data_dir)
    train_stats = []
    for p, _ in collect_split(data, "train"):
        gray = np.asarray(Image.open(p).convert("L"), np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))
    z_stats, q_stats = fit_global_stats(data), fit_percentile_stats(data)
    pair = load_aspp(args.pair_checkpoint, device)
    model_cache = {}
    entries = []
    for image_path, mask_path in collect_split(data, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), np.uint8)
        gray = raw[..., 0].astype(np.float32)
        mean, std = float(gray.mean()), float(gray.std())
        source = choose_source(mean, args)
        if source not in model_cache:
            model_cache[source] = load_aspp(source, device)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(source, raw, device, amp, 5, 1e-5)
        high768_raw = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        style = transform_image(raw, "standardize", train_mean, train_std)
        high768_style = predict_array(adapted, style, 768, 576, device, amp, 1, "d4", "gaussian")
        high1024_raw = predict_array(adapted, raw, 1024, 768, device, amp, 1, "d4", "gaussian")
        high1024_style = predict_array(adapted, style, 1024, 768, device, amp, 1, "d4", "gaussian")
        z = (gray - mean) / max(std, 1e-5)
        q = percentile_image(gray)
        entries.append({
            "name": image_path.name, "source": source,
            "branches": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
            "pair": pair_value.astype(np.float32),
            "high768": (high768_raw.astype(np.float32), high768_style.astype(np.float32)),
            "high1024": (high1024_raw.astype(np.float32), high1024_style.astype(np.float32)),
            "z": z, "q": q, "prior_z_stats": z_stats, "prior_q_stats": q_stats,
            "params": route_params(mean, std),
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
        })
        del adapted, high768_raw, high768_style, high1024_raw, high1024_style
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"cached {image_path.name} source={source}", flush=True)

    rows = []
    for alpha in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0):
        predictions, targets = [], []
        for item in entries:
            old = item["high768"]
            new = item["high1024"]
            high = tuple((1.0 - alpha) * old[i] + alpha * new[i] for i in (0, 1))
            prediction = predict(make_logits(item, high))
            predictions.append(prediction); targets.append(item["target"])
        value = metrics(predictions, targets)
        rows.append({"alpha_1024": alpha, **value})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    best = max(rows, key=lambda x: x["image_mean_miou"])
    output = Path(args.output_json); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best": best, "records": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--pair-checkpoint", required=True); p.add_argument("--output-json", required=True)
    p.add_argument("--source-restart51", required=True); p.add_argument("--source-model109", required=True)
    p.add_argument("--source-ema", required=True); p.add_argument("--source-pairwise", required=True)
    p.add_argument("--source-seed456", required=True)
    main(p.parse_args())
