#!/usr/bin/env python3
"""Search label-free Al3Ni calibration on top of the model 139/140 mixture."""

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
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior, load_aspp
from tent_style_search import adapt_model


def load_unet(path, device):
    from calibrate_al import load_model
    return load_model(path, device)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    samples = collect_split(args.data_dir, "valid")
    data_dir = Path(args.data_dir)
    global_stats = fit_global_stats(data_dir)
    train_stats = []
    for p, _ in collect_split(data_dir, "train"):
        gray = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))
    old = load_unet(args.old_checkpoint, device)
    new = load_unet(args.new_checkpoint, device)
    aspp = load_aspp(args.aspp_checkpoint, device)
    pair = load_aspp(args.pair_checkpoint, device)
    items = []
    for image_path, mask_path in samples:
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        p0 = predict_probabilities(old, image_path, 640, 512, device, amp, 2, "d4", "gaussian")
        p1 = predict_probabilities(new, image_path, 512, 384, device, amp, 2, "d4", "gaussian")
        p2 = predict_probabilities(aspp, image_path, 512, 384, device, amp, 2, "d4", "gaussian")
        pp = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, raw, device, amp, 3, 2e-5)
        raw_high = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        style_high = predict_array(adapted, transform_image(raw, "standardize", train_mean, train_std),
                                   768, 576, device, amp, 1, "d4", "gaussian")
        gray = raw[..., 0].astype(np.float32)
        mean = float(gray.mean())
        high_weight = .60 if mean >= 165.0 else .70
        high = .25 * raw_high + .75 * style_high if float(gray.std()) >= 50.0 else raw_high
        base = MATRIX * (1.0 - high_weight - .15)
        mixed = np.asarray([
            sum(base[c, m] * [p0, p1, p2][m][c] for m in range(3))
            + high_weight * high[c] + .15 * pp[c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
        z = (gray - mean) / max(float(gray.std()), 1e-5)
        adaptive_stats = []
        for cid in range(4):
            w = np.maximum(mixed[cid], 1e-7)
            m = float((w * z).sum() / w.sum())
            s = float(np.sqrt((w * (z - m) ** 2).sum() / w.sum()))
            adaptive_stats.append((m, max(s, .15)))
        items.append({
            "name": image_path.name, "mean": mean,
            "mixed": mixed, "global": gaussian_prior(z, global_stats),
            "adaptive": gaussian_prior(z, adaptive_stats),
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
        })
        print(f"cached {image_path.name}", flush=True)

    # Compute each image/bias pair once, then combine image scores for the
    # mean-conditioned candidates. The previous nested implementation repeated
    # the full-resolution argmax thousands of times.
    records = []
    fixed_biases = (-.30, -.20, -.10, 0.0, .10, .20, .30, .40, .50, .60)
    thresholds = (160.0, 165.0, 167.037086, 170.0, 175.0)
    low_values = (0.0, .10, .20, .30, .40, .50)
    high_values = (-.10, 0.0, .10, .20, .30, .40)
    candidate_biases = sorted(set(fixed_biases + low_values + high_values))

    def image_confusion(prediction, target):
        bins = target.reshape(-1) * 4 + prediction.reshape(-1)
        return np.bincount(bins, minlength=16).reshape(4, 4)

    def summarise(selected):
        total = np.zeros((4, 4), dtype=np.int64)
        image_scores = []
        for confusion in selected:
            total += confusion
            inter = np.diag(confusion).astype(np.float64)
            union = confusion.sum(0) + confusion.sum(1) - inter
            image_scores.append(float(np.mean(inter / np.maximum(union, 1.0))))
        inter = np.diag(total).astype(np.float64)
        union = total.sum(0) + total.sum(1) - inter
        iou = inter / np.maximum(union, 1.0)
        return {"global_miou": float(iou.mean()),
                "global_iou": {n: float(v) for n, v in zip(
                    ("Al3Ni", "Primary Si", "Al", "Eutectic Si"), iou)},
                "image_mean_miou": float(np.mean(image_scores)),
                "image_scores": image_scores, "confusion": total.tolist()}

    score_cache = {}
    for prior_mode in ("global", "adaptive"):
        score_cache[prior_mode] = {}
        for bias in candidate_biases:
            confusions = []
            for item in items:
                logits = np.log(np.maximum(item["mixed"], 1e-7)) + .09 * item[prior_mode]
                logits[2] += .40
                logits[0] += bias
                logits[3] += -.30 if item["mean"] >= 167.037086 else -.70
                pred = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
                confusions.append(image_confusion(pred, item["target"]))
            score_cache[prior_mode][bias] = confusions

        for bias in fixed_biases:
            records.append({"prior": prior_mode, "mode": "fixed", "al_bias": bias,
                            **summarise(score_cache[prior_mode][bias])})
        for threshold in thresholds:
            for low in low_values:
                for high in high_values:
                    selected = [score_cache[prior_mode][high if item["mean"] >= threshold else low][idx]
                                for idx, item in enumerate(items)]
                    records.append({"prior": prior_mode, "mode": "mean", "threshold": threshold,
                                    "low_al_bias": low, "high_al_bias": high,
                                    **summarise(selected)})
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--old-checkpoint", required=True)
    p.add_argument("--new-checkpoint", required=True)
    p.add_argument("--aspp-checkpoint", required=True)
    p.add_argument("--pair-checkpoint", required=True)
    p.add_argument("--high-checkpoint", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
