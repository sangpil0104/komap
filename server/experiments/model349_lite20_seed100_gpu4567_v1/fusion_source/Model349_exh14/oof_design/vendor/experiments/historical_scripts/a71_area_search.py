#!/usr/bin/env python3
"""Search an absence-aware Primary-Si cleanup on the A71 hard case."""
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
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior
from tent_style_search import adapt_model

MATRIX = np.asarray([
    [0.35, 0.40, 0.25], [0.25, 0.35, 0.40],
    [0.50, 0.30, 0.20], [0.25, 0.55, 0.20],
], dtype=np.float32)


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval(); return model


def score(prediction, target):
    cm = np.bincount(target.reshape(-1) * 4 + prediction.reshape(-1), minlength=16).reshape(4, 4)
    inter = np.diag(cm).astype(float); union = cm.sum(0) + cm.sum(1) - inter
    return float(np.mean(inter / np.maximum(union, 1.0))), (inter / np.maximum(union, 1.0)).tolist()


def replace_class(prediction, logits, class_id, threshold):
    result = prediction.copy()
    mask = result == class_id
    if float(mask.mean()) <= threshold:
        alternatives = logits.copy(); alternatives[class_id] = -np.inf
        result[mask] = alternatives[:, mask].argmax(0).astype(np.uint8)
    return result


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = device.type == "cuda"
    data = Path(args.data_dir)
    train_stats = []
    for p, _ in collect_split(data, "train"):
        gray = np.asarray(Image.open(p).convert("L"), np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats])); train_std = float(np.mean([x[1] for x in train_stats]))
    z_stats, q_stats = fit_global_stats(data), fit_percentile_stats(data)
    pair = load_aspp(args.pair_checkpoint, device)
    item = None
    for image_path, mask_path in collect_split(data, "valid"):
        if "A71" not in image_path.name:
            continue
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), np.uint8)
        gray = raw[..., 0].astype(np.float32); mean = float(gray.mean()); std = float(gray.std())
        source = args.source_pairwise
        adapted = adapt_model(source, raw, device, amp, 5, 1e-5)
        high768_raw = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        style = transform_image(raw, "standardize", train_mean, train_std)
        high768_style = predict_array(adapted, style, 768, 576, device, amp, 1, "d4", "gaussian")
        high1024_raw = predict_array(adapted, raw, 1024, 768, device, amp, 1, "d4", "gaussian")
        high1024_style = predict_array(adapted, style, 1024, 768, device, amp, 1, "d4", "gaussian")
        high768 = high768_style
        high1024 = high1024_style
        high = .8 * high768 + .2 * high1024
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        params = route_params(mean, std)
        base = MATRIX * (1.0 - params["high_weight"] - params["pair_weight"])
        mixed = np.asarray([
            sum(base[c, m] * loaded[f"p{m}"].astype(np.float32)[c] for m in range(3))
            + params["high_weight"] * high[c] + params["pair_weight"] * pair_value[c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
        z = (gray - mean) / max(std, 1e-5); q = percentile_image(gray)
        stats_z, stats_q = [], []
        for cid in range(4):
            weights = np.maximum(mixed[cid], 1e-7) ** 2
            zm = float((weights * z).sum() / weights.sum()); zs = float(np.sqrt((weights * (z - zm) ** 2).sum() / weights.sum()))
            qm = float((weights * q).sum() / weights.sum()); qs = float(np.sqrt((weights * (q - qm) ** 2).sum() / weights.sum()))
            stats_z.append((zm, max(zs, .15))); stats_q.append((qm, max(qs, .02)))
        prior_z = gaussian_prior(z, z_stats); prior_q = gaussian_prior(q, q_stats)
        adaptive_z = gaussian_prior(z, stats_z); adaptive_q = gaussian_prior(q, stats_q)
        prior = .85 * prior_z + .15 * prior_q
        prior[0] += params["beta"] * (.85 * (adaptive_z[0] - prior_z[0]) + .15 * (adaptive_q[0] - prior_q[0]))
        logits = np.log(np.maximum(mixed, 1e-7)) + .09 * prior
        logits[2] += params["al_bias"]; logits[3] += params["eut_bias"]
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        raw_prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
        print(json.dumps({"image": image_path.name, "target_fraction": np.bincount(target.reshape(-1), minlength=4).tolist(),
                          "raw_fraction": np.bincount(raw_prediction.reshape(-1), minlength=4).tolist(),
                          "raw_score": score(raw_prediction, target)}, ensure_ascii=False), flush=True)
        rows = []
        for threshold in (0.0, .001, .002, .005, .01, .02, .03, .05, .08, .10, .15, .20, .30):
            pred = replace_class(raw_prediction, logits, 1, threshold)
            value, iou = score(pred, target)
            rows.append({"primary_area_threshold": threshold, "pred_fraction": np.bincount(pred.reshape(-1), minlength=4).tolist(),
                         "score": value, "iou": iou})
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        output = Path(args.output_json); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        del adapted, high768_raw, high768_style, high1024_raw, high1024_style
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
        return
    raise RuntimeError("A71 validation image not found")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--pair-checkpoint", required=True); p.add_argument("--source-pairwise", required=True); p.add_argument("--output-json", required=True)
    main(p.parse_args())
