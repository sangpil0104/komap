#!/usr/bin/env python3
"""Fine search branch weights around model 161's validation route."""
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
from improve_five_causes import collect_split, decode_mask
from prior_representation_search import fit_percentile_stats, percentile_image
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior
from tent_style_search import adapt_model


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    data = Path(args.data_dir)
    train_stats = []
    for path, _ in collect_split(data, "train"):
        gray = np.asarray(Image.open(path).convert("L"), np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))
    global_stats = fit_global_stats(data)
    percentile_stats = fit_percentile_stats(data)
    pair = load_aspp(args.pair_checkpoint, device)
    items = []
    for image_path, mask_path in collect_split(data, "valid"):
        cached = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), np.uint8)
        gray = raw[..., 0].astype(np.float32)
        pair_prob = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, raw, device, amp, 5, 1e-5)
        high_raw = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        high_style = predict_array(
            adapted, transform_image(raw, "standardize", train_mean, train_std),
            768, 576, device, amp, 1, "d4", "gaussian"
        )
        items.append({
            "branches": [cached[f"p{i}"].astype(np.float32) for i in range(3)],
            "pair": pair_prob.astype(np.float32),
            "high_raw": high_raw.astype(np.float32),
            "high_style": high_style.astype(np.float32),
            "gray": gray,
            "mean": float(gray.mean()),
            "std": float(gray.std()),
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
        })
        del adapted, high_raw, high_style
        gc.collect()
        torch.cuda.empty_cache()
        print(f"cached {image_path.name}", flush=True)

    records = []
    low_weights = (0.65, 0.70, 0.75, 0.80, 0.85)
    high_weights = (0.45, 0.50, 0.55, 0.60, 0.65)
    pair_weights = (0.10, 0.15, 0.20)
    for low_w, high_w, pair_w in itertools.product(low_weights, high_weights, pair_weights):
        if low_w + pair_w >= 0.98 or high_w + pair_w >= 0.98:
            continue
        predictions, targets = [], []
        for item in items:
            high = item["high_raw"]
            if item["std"] >= 50.0:
                high = 0.25 * item["high_raw"] + 0.75 * item["high_style"]
            high_w_img = high_w if item["mean"] >= 165.0 else low_w
            base = MATRIX * (1.0 - high_w_img - pair_w)
            branches = item["branches"]
            mixed = np.asarray([
                sum(base[c, m] * branches[m][c] for m in range(3))
                + high_w_img * high[c] + pair_w * item["pair"][c]
                for c in range(4)
            ], dtype=np.float32)
            mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
            gray = item["gray"]
            z = (gray - item["mean"]) / max(item["std"], 1e-5)
            q = percentile_image(gray)
            adaptive = []
            adaptive_q = []
            for cid in range(4):
                weights = np.maximum(mixed[cid], 1e-7) ** 2
                zm = float((weights * z).sum() / weights.sum())
                zs = float(np.sqrt((weights * (z - zm) ** 2).sum() / weights.sum()))
                qm = float((weights * q).sum() / weights.sum())
                qs = float(np.sqrt((weights * (q - qm) ** 2).sum() / weights.sum()))
                adaptive.append((zm, max(zs, .15)))
                adaptive_q.append((qm, max(qs, .02)))
            prior_z = gaussian_prior(z, global_stats)
            adaptive_z = gaussian_prior(z, adaptive)
            prior_q = gaussian_prior(q, percentile_stats)
            adaptive_qv = gaussian_prior(q, adaptive_q)
            prior = .85 * prior_z + .15 * prior_q
            delta = .85 * (adaptive_z[0] - prior_z[0]) + .15 * (adaptive_qv[0] - prior_q[0])
            prior[0] += (3 if item["mean"] < 160.0 else 7) * delta
            logits = np.log(np.maximum(mixed, 1e-7)) + .09 * prior
            logits[2] += .45
            logits[3] += -.3 if item["mean"] >= 165.0 else -1.0
            pred = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
            if float((pred == 1).mean()) <= .01:
                mask = pred == 1
                alternative = logits.copy()
                alternative[1] = -np.inf
                pred[mask] = alternative[:, mask].argmax(0).astype(np.uint8)
            predictions.append(pred)
            targets.append(item["target"])
        records.append({
            "low_high_weight": low_w,
            "high_high_weight": high_w,
            "pair_weight": pair_w,
            **metrics(predictions, targets),
        })
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pair-checkpoint", required=True)
    parser.add_argument("--high-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    main(parser.parse_args())
