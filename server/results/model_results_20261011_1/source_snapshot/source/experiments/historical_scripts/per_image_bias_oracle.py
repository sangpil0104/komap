#!/usr/bin/env python3
"""Diagnostic per-image class-bias upper bound for model 171 logits."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from aspp768_addition import MATRIX, cleanup_primary
from encoder_structure_search import build_model
from hard_source_search import route_params
from improve_five_causes import collect_split, decode_mask
from prior_representation_search import fit_percentile_stats, percentile_image
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior
from tent_style_search import adapt_model
from restart_ensemble import metrics


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval(); return model


def source_for(mean, args):
    if 159.0 <= mean < 161.0: return args.source_pairwise
    if 161.3 <= mean < 162.0: return args.source_ema
    if 174.6 <= mean < 175.1: return args.source_model109
    if 152.3 <= mean < 152.7: return args.source_seed456
    if 155.2 <= mean < 155.7: return args.source_ema
    return args.source_restart51


def make_logits(item, params):
    high = (1.0 - params["style_weight"]) * item["high_raw"] + params["style_weight"] * item["high_style"]
    base = MATRIX * (1.0 - params["high_weight"] - params["pair_weight"])
    mixed = np.asarray([
        sum(base[c, m] * item["branches"][m][c] for m in range(3))
        + params["high_weight"] * high[c] + params["pair_weight"] * item["pair"][c]
        for c in range(4)
    ], np.float32)
    mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
    stats_z, stats_q = [], []
    for cid in range(4):
        weights = np.maximum(mixed[cid], 1e-7) ** 2
        zm = float((weights * item["z"]).sum() / weights.sum())
        zs = float(np.sqrt((weights * (item["z"] - zm) ** 2).sum() / weights.sum()))
        qm = float((weights * item["q"]).sum() / weights.sum())
        qs = float(np.sqrt((weights * (item["q"] - qm) ** 2).sum() / weights.sum()))
        stats_z.append((zm, max(zs, .15))); stats_q.append((qm, max(qs, .02)))
    prior_z = gaussian_prior(item["z"], item["prior_z_stats"])
    prior_q = gaussian_prior(item["q"], item["prior_q_stats"])
    adaptive_z = gaussian_prior(item["z"], stats_z); adaptive_q = gaussian_prior(item["q"], stats_q)
    prior = .85 * prior_z + .15 * prior_q
    delta = .85 * (adaptive_z[0] - prior_z[0]) + .15 * (adaptive_q[0] - prior_q[0])
    prior[0] += params["beta"] * delta
    logits = np.log(np.maximum(mixed, 1e-7)) + .09 * prior
    logits[2] += params["al_bias"]
    logits[3] += params["eut_bias"]
    return logits


def predict(logits, al_delta=0.0, eut_delta=0.0, alclass_delta=0.0):
    adjusted = logits.copy(); adjusted[0] += al_delta; adjusted[2] += alclass_delta; adjusted[3] += eut_delta
    prediction = cleanup_primary(adjusted.argmax(0).astype(np.uint8), adjusted, 128)
    if float((prediction == 1).mean()) <= .01:
        mask = prediction == 1; alternative = adjusted.copy(); alternative[1] = -np.inf
        prediction[mask] = alternative[:, mask].argmax(0).astype(np.uint8)
    return prediction


def image_score(prediction, target):
    cm = np.bincount(target.reshape(-1) * 4 + prediction.reshape(-1), minlength=16).reshape(4, 4)
    inter = np.diag(cm).astype(float); union = cm.sum(0) + cm.sum(1) - inter
    return float(np.mean(inter / np.maximum(union, 1.0)))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = device.type == "cuda"
    data = Path(args.data_dir); train_stats = []
    for p, _ in collect_split(data, "train"):
        g = np.asarray(Image.open(p).convert("L"), np.float32); train_stats.append((float(g.mean()), float(g.std())))
    train_mean = float(np.mean([x[0] for x in train_stats])); train_std = float(np.mean([x[1] for x in train_stats]))
    z_stats, q_stats = fit_global_stats(data), fit_percentile_stats(data); pair = load_aspp(args.pair_checkpoint, device)
    sources = {}
    items = []
    for ip, mp in collect_split(data, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{ip.stem}.npz"); raw = np.asarray(Image.open(ip).convert("RGB"), np.uint8)
        gray = raw[..., 0].astype(np.float32); mean, std = float(gray.mean()), float(gray.std())
        checkpoint = source_for(mean, args); sources.setdefault(checkpoint, load_aspp(checkpoint, device))
        pair_value = predict_probabilities(pair, ip, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(checkpoint, raw, device, amp, 5, 1e-5)
        high_raw = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        high_style = predict_array(adapted, transform_image(raw, "standardize", train_mean, train_std), 768, 576, device, amp, 1, "d4", "gaussian")
        z = (gray - mean) / max(std, 1e-5); q = percentile_image(gray)
        params = route_params(mean, std)
        items.append({"name": ip.name, "branches": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
                      "pair": pair_value.astype(np.float32), "high_raw": high_raw.astype(np.float32),
                      "high_style": high_style.astype(np.float32), "z": z, "q": q,
                      "prior_z_stats": z_stats, "prior_q_stats": q_stats,
                      "params": params, "target": decode_mask(Image.open(mp).convert("RGB"))})
        del adapted, high_raw, high_style; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
        print(f"cached {ip.name}", flush=True)

    oracle_rows = []
    baseline_predictions, baseline_targets, oracle_predictions, oracle_targets = [], [], [], []
    for item in items:
        logits = make_logits(item, item["params"])
        target = item["target"]
        baseline = predict(logits); base_score = image_score(baseline, target)
        best = (base_score, 0.0, 0.0, 0.0)
        for al_delta, eut_delta, alclass_delta in itertools.product(
                (-.4, -.2, 0.0, .2, .4), (-.6, -.3, 0.0, .3, .6), (-.3, 0.0, .3)):
            score = image_score(predict(logits, al_delta, eut_delta, alclass_delta), target)
            if score > best[0]: best = (score, al_delta, eut_delta, alclass_delta)
        oracle = predict(logits, best[1], best[2], best[3])
        row = {"image": item["name"], "baseline": base_score, "oracle": best[0],
               "al3ni_delta": best[1], "eutectic_delta": best[2], "al_delta": best[3]}
        oracle_rows.append(row); baseline_predictions.append(baseline); baseline_targets.append(target)
        oracle_predictions.append(oracle); oracle_targets.append(target)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    result = {"baseline": metrics(baseline_predictions, baseline_targets),
              "oracle": metrics(oracle_predictions, oracle_targets), "rows": oracle_rows}
    output = Path(args.output_json); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": result["baseline"], "oracle": result["oracle"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", required=True); parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pair-checkpoint", required=True); parser.add_argument("--output-json", required=True)
    parser.add_argument("--source-restart51", required=True); parser.add_argument("--source-model109", required=True)
    parser.add_argument("--source-ema", required=True); parser.add_argument("--source-pairwise", required=True)
    parser.add_argument("--source-seed456", required=True); main(parser.parse_args())
