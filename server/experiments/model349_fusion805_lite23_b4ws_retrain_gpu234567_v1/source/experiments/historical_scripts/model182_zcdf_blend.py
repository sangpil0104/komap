#!/usr/bin/env python3
"""Blend the model-181 raw/z/CDF branch into the exact model-180 route."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from advanced_inference import predict_probabilities as predict_rgb
from aspp768_addition import cleanup_primary
from encoder_structure_search import build_model
from hard_source_search import route_params
from highres1024_search import MATRIX, choose_source
from improve_five_causes import collect_split, decode_mask
from model181_raw_zcdf_aspp import TeamRepresentationASPP, predict_probabilities as predict_rep, score
from prior_representation_search import fit_percentile_stats, percentile_image
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior
from tent_style_search import adapt_model


def load_aspp(path: str, device: torch.device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def load_rep(path: str, device: torch.device):
    model = TeamRepresentationASPP(pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def mixed_probability(item, high):
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
    return mixed / np.maximum(mixed.sum(0, keepdims=True), 1e-7)


def calibrated_logits(item, mixed):
    stats_z, stats_q = [], []
    for class_id in range(4):
        weights = np.maximum(mixed[class_id], 1e-7) ** 2
        zm = float((weights * item["z"]).sum() / weights.sum())
        zs = float(np.sqrt((weights * (item["z"] - zm) ** 2).sum() / weights.sum()))
        qm = float((weights * item["q"]).sum() / weights.sum())
        qs = float(np.sqrt((weights * (item["q"] - qm) ** 2).sum() / weights.sum()))
        stats_z.append((zm, max(zs, 0.15)))
        stats_q.append((qm, max(qs, 0.02)))
    prior_z = gaussian_prior(item["z"], item["prior_z_stats"])
    prior_q = gaussian_prior(item["q"], item["prior_q_stats"])
    adaptive_z = gaussian_prior(item["z"], stats_z)
    adaptive_q = gaussian_prior(item["q"], stats_q)
    prior = 0.85 * prior_z + 0.15 * prior_q
    delta = 0.85 * (adaptive_z[0] - prior_z[0]) + 0.15 * (adaptive_q[0] - prior_q[0])
    prior[0] += item["params"]["beta"] * delta
    logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
    logits[2] += item["params"]["al_bias"]
    logits[3] += item["params"]["eut_bias"]
    return logits


def predict_from_logits(logits):
    refined = logits.copy()
    refined[1] = gaussian_filter(refined[1], sigma=0.50, mode="reflect")
    prediction = cleanup_primary(refined.argmax(0).astype(np.uint8), refined, 128)
    if float((prediction == 1).mean()) <= 0.01:
        mask = prediction == 1
        alternative = refined.copy()
        alternative[1] = -np.inf
        prediction[mask] = alternative[:, mask].argmax(0).astype(np.uint8)
    return prediction


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    data = Path(args.data_dir)
    global_z, global_q = fit_global_stats(data), fit_percentile_stats(data)
    train_values = []
    for image_path, _ in collect_split(data, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_values.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([value[0] for value in train_values]))
    train_std = float(np.mean([value[1] for value in train_values]))
    pair = load_aspp(args.pair_checkpoint, device)
    representation_model = load_rep(args.representation_checkpoint, device)
    source_paths = {
        "restart51": args.source_restart51,
        "model109": args.source_model109,
        "seed456": args.source_seed456,
        "pairwise": args.source_pairwise,
        "ema": args.source_ema,
    }
    source_args = argparse.Namespace(**{f"source_{key}": value for key, value in source_paths.items()})
    entries = []
    for image_path, mask_path in collect_split(data, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        gray = raw[..., 0].astype(np.float32)
        mean, std = float(gray.mean()), float(gray.std())
        source = choose_source(mean, source_args)
        pair_value = predict_rgb(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(source, raw, device, amp, 5, 1e-5)
        style = transform_image(raw, "standardize", train_mean, train_std)
        high = (
            (predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian"),
             predict_array(adapted, style, 768, 576, device, amp, 1, "d4", "gaussian")),
        )
        high_1024 = (
            predict_array(adapted, raw, 1024, 768, device, amp, 1, "d4", "gaussian"),
            predict_array(adapted, style, 1024, 768, device, amp, 1, "d4", "gaussian"),
        )
        high = tuple((1.0 - args.alpha_1024) * high[0][index]
                     + args.alpha_1024 * high_1024[index] for index in (0, 1))
        item = {
            "branches": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
            "pair": pair_value.astype(np.float32),
            "high": high,
            "extra": predict_rep(representation_model, image_path, 512, 384, device, amp, 2),
            "z": (gray - mean) / max(std, 1e-5),
            "q": percentile_image(gray),
            "prior_z_stats": global_z,
            "prior_q_stats": global_q,
            "params": route_params(mean, std),
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
            "name": image_path.name,
        }
        item["base"] = mixed_probability(item, high)
        entries.append(item)
        del adapted, high_1024
        gc.collect()
        torch.cuda.empty_cache() if device.type == "cuda" else None
        print(f"cached {image_path.name} source={source}", flush=True)

    records = []
    gammas = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
    for mode in ("all", "rare", "primary", "al"):
        for gamma in gammas:
            predictions, targets = [], []
            for item in entries:
                mixed = item["base"].copy()
                if mode == "all":
                    mixed = (1.0 - gamma) * mixed + gamma * item["extra"]
                else:
                    class_ids = {"rare": (0, 3), "primary": (1,), "al": (2,)}[mode]
                    for class_id in class_ids:
                        mixed[class_id] = ((1.0 - gamma) * mixed[class_id]
                                           + gamma * item["extra"][class_id])
                mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
                predictions.append(predict_from_logits(calibrated_logits(item, mixed)))
                targets.append(item["target"])
            value = score(predictions, targets)
            row = {"mode": mode, "gamma": gamma, **value}
            records.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    best = max(records, key=lambda row: row["image_mean_miou"])
    best_exclude = max(records, key=lambda row: row["image_mean_miou_exclude"])
    best_image_compat = max(records, key=lambda row: row["image_miou_compat"])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best": best, "best_exclude": best_exclude,
                                  "best_image_compat": best_image_compat, "records": records},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "best_exclude": best_exclude,
                      "best_image_compat": best_image_compat}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--representation-checkpoint", required=True)
    parser.add_argument("--pair-checkpoint", required=True)
    parser.add_argument("--source-restart51", required=True)
    parser.add_argument("--source-model109", required=True)
    parser.add_argument("--source-seed456", required=True)
    parser.add_argument("--source-pairwise", required=True)
    parser.add_argument("--source-ema", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--alpha-1024", type=float, default=0.20)
    main(parser.parse_args())
