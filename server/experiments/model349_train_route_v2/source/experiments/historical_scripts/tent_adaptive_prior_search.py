#!/usr/bin/env python3
"""Search label-free image-adaptive grayscale class priors."""

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


COLORS = ((25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204))


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def fit_global_stats(data_dir):
    values = [[] for _ in range(4)]
    for image_path, mask_path in collect_split(Path(data_dir), "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - gray.mean()) / max(float(gray.std()), 1e-5)
        mask = np.asarray(Image.open(mask_path).convert("RGB"))
        ids = np.full(mask.shape[:2], -1, dtype=np.int8)
        for cid, color in enumerate(COLORS):
            ids[np.all(mask == color, axis=2)] = cid
        rng = np.random.default_rng(sum(image_path.name.encode("utf-8")))
        for cid in range(4):
            current = z[ids == cid]
            if len(current):
                take = rng.choice(len(current), min(len(current), 10000), replace=False)
                values[cid].append(current[take])
    return [(float(np.concatenate(v).mean()), max(float(np.concatenate(v).std()), .15)) for v in values]


def gaussian_prior(z, stats):
    return np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                       for mean, std in stats], dtype=np.float32)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    samples = collect_split(args.data_dir, "valid")
    global_stats = fit_global_stats(args.data_dir)
    train_stats = []
    for p, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))
    pair = load_aspp(args.pair_checkpoint, device)
    cached = []
    for image_path, mask_path in _local_progress(samples, 'tent_adaptive_prior_search'):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        p0, p1, p2 = [loaded[f"p{i}"].astype(np.float32) for i in range(3)]
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        model = adapt_model(args.high_checkpoint, raw, device, amp, 3, 2e-5)
        raw_high = predict_array(model, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        style = transform_image(raw, "standardize", train_mean, train_std)
        style_high = predict_array(model, style, 768, 576, device, amp, 1, "d4", "gaussian")
        gray = raw[..., 0].astype(np.float32)
        mean = float(gray.mean())
        high_weight = .60 if mean >= 165 else .70
        high = .25 * raw_high + .75 * style_high if gray.std() >= 50 else raw_high
        pair_weight = .15
        base = MATRIX * (1.0 - high_weight - pair_weight)
        mixed = np.asarray([
            sum(base[c, m] * [p0, p1, p2][m][c] for m in range(3))
            + high_weight * high[c] + pair_weight * pair_value[c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
        z = (gray - mean) / max(float(gray.std()), 1e-5)
        global_prior = gaussian_prior(z, global_stats)
        adaptive_stats = []
        for cid in range(4):
            weights = np.maximum(mixed[cid], 1e-7) ** args.gamma
            weighted_mean = float((weights * z).sum() / weights.sum())
            weighted_std = float(np.sqrt((weights * (z - weighted_mean) ** 2).sum() / weights.sum()))
            adaptive_stats.append((weighted_mean, max(weighted_std, .15)))
        adaptive_prior = gaussian_prior(z, adaptive_stats)
        cached.append({"mixed": mixed, "global_prior": global_prior,
                       "adaptive_prior": adaptive_prior,
                       "target": decode_mask(Image.open(mask_path).convert("RGB")),
                       "mean": mean, "name": image_path.name})
        print(f"cached {image_path.name} adaptive_stats={adaptive_stats}", flush=True)

    records = []
    for beta in (0.0, .05, .10, .15, .20, .30, .50, .75, 1.0):
        for mode in ("all", "pair"):
            predictions, targets = [], []
            for item in cached:
                prior = item["global_prior"].copy()
                if mode == "all":
                    prior += beta * (item["adaptive_prior"] - item["global_prior"])
                else:
                    for cid in (0, 3):
                        prior[cid] += beta * (item["adaptive_prior"][cid] - item["global_prior"][cid])
                logits = np.log(np.maximum(item["mixed"], 1e-7)) + .09 * prior
                logits[2] += .40
                logits[3] += -.30 if item["mean"] >= 167.037086 else -.70
                prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
                predictions.append(prediction); targets.append(item["target"])
            value = metrics(predictions, targets)
            row = {"beta": beta, "mode": mode, "gamma": args.gamma, **value}
            records.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--high-checkpoint", required=True); p.add_argument("--pair-checkpoint", required=True)
    p.add_argument("--output-json", required=True); p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
