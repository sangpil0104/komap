#!/usr/bin/env python3
"""Audit exact model-162 route by image and class."""
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
from improve_five_causes import collect_split, decode_mask, encode_mask
from prior_representation_search import fit_percentile_stats, percentile_image
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
    z_stats, q_stats = fit_global_stats(data), fit_percentile_stats(data)
    pair = load_aspp(args.pair_checkpoint, device)
    rows = []
    output = Path(args.output_dir); (output / "predictions").mkdir(parents=True, exist_ok=True)
    for image_path, mask_path in collect_split(data, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), np.uint8)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, raw, device, amp, 5, 1e-5)
        high_raw = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        high_style = predict_array(adapted, transform_image(raw, "standardize", train_mean, train_std),
                                   768, 576, device, amp, 1, "d4", "gaussian")
        gray = raw[..., 0].astype(np.float32); mean, std = float(gray.mean()), float(gray.std())
        high_weight = .85 if mean < 165.0 else .55
        high = .25 * high_raw + .75 * high_style if std >= 50.0 else high_raw
        base = MATRIX * (1.0 - high_weight - .10)
        mixed = np.asarray([
            sum(base[c, m] * loaded[f"p{m}"].astype(np.float32)[c] for m in range(3))
            + high_weight * high[c] + .10 * pair_value[c]
            for c in range(4)
        ], np.float32)
        mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
        z = (gray - mean) / max(std, 1e-5); q = percentile_image(gray)
        adaptive_z, adaptive_q = [], []
        for cid in range(4):
            weights = np.maximum(mixed[cid], 1e-7) ** 2
            zm = float((weights * z).sum() / weights.sum()); zs = float(np.sqrt((weights * (z - zm) ** 2).sum() / weights.sum()))
            qm = float((weights * q).sum() / weights.sum()); qs = float(np.sqrt((weights * (q - qm) ** 2).sum() / weights.sum()))
            adaptive_z.append((zm, max(zs, .15))); adaptive_q.append((qm, max(qs, .02)))
        prior_z, prior_q = gaussian_prior(z, z_stats), gaussian_prior(q, q_stats)
        adapt_z, adapt_q = gaussian_prior(z, adaptive_z), gaussian_prior(q, adaptive_q)
        prior = .85 * prior_z + .15 * prior_q
        delta = .85 * (adapt_z[0] - prior_z[0]) + .15 * (adapt_q[0] - prior_q[0])
        prior[0] += (3.0 if mean < 160.0 else 7.0) * delta
        logits = np.log(np.maximum(mixed, 1e-7)) + .09 * prior
        logits[2] += .45; logits[3] += -.3 if mean >= 165.0 else -1.0
        prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
        if float((prediction == 1).mean()) <= .01:
            mask = prediction == 1; alternative = logits.copy(); alternative[1] = -np.inf
            prediction[mask] = alternative[:, mask].argmax(0).astype(np.uint8)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        cm = np.bincount(target.reshape(-1) * 4 + prediction.reshape(-1), minlength=16).reshape(4, 4)
        inter = np.diag(cm).astype(np.float64); union = cm.sum(0) + cm.sum(1) - inter
        iou = inter / np.maximum(union, 1.0)
        encode_mask(prediction).save(output / "predictions" / image_path.name.replace("_image.png", "_mask.png"))
        rows.append({"name": image_path.name, "mean": mean, "std": std,
                     "image_mean_miou": float(iou.mean()), "iou": iou.tolist(),
                     "target_counts": np.bincount(target.reshape(-1), minlength=4).tolist(),
                     "prediction_counts": np.bincount(prediction.reshape(-1), minlength=4).tolist(),
                     "confusion": cm.tolist()})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        del adapted, high_raw, high_style; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    rows.sort(key=lambda x: x["image_mean_miou"])
    (output / "audit.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True); parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pair-checkpoint", required=True); parser.add_argument("--high-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True); main(parser.parse_args())
