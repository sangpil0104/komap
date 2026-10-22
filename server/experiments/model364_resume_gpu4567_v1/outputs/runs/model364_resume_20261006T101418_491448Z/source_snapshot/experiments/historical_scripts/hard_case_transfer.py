#!/usr/bin/env python3
"""Transfer the validated A71 hard-case route to other intensity bands."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from aspp768_addition import MATRIX
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask
from prior_representation_search import fit_percentile_stats, percentile_image
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior
from tent_style_search import adapt_model
from hard_case_route_search import route


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval(); return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"; data = Path(args.data_dir)
    train_stats = []
    for path, _ in collect_split(data, "train"):
        gray = np.asarray(Image.open(path).convert("L"), np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats])); train_std = float(np.mean([x[1] for x in train_stats]))
    z_stats, q_stats = fit_global_stats(data), fit_percentile_stats(data)
    pair = load_aspp(args.pair_checkpoint, device); items = []
    for image_path, mask_path in collect_split(data, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), np.uint8)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, raw, device, amp, 5, 1e-5)
        high_raw = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        high_style = predict_array(adapted, transform_image(raw, "standardize", train_mean, train_std),
                                   768, 576, device, amp, 1, "d4", "gaussian")
        gray = raw[..., 0].astype(np.float32); mean, std = float(gray.mean()), float(gray.std())
        z = (gray - mean) / max(std, 1e-5); q = percentile_image(gray)
        items.append({"name": image_path.name, "branches": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
                      "pair": pair_value.astype(np.float32), "high_raw": high_raw.astype(np.float32),
                      "high_style": high_style.astype(np.float32), "z": z, "q": q,
                      "prior_z": gaussian_prior(z, z_stats), "prior_q": gaussian_prior(q, q_stats),
                      "mean": mean, "std": std, "target": decode_mask(Image.open(mask_path).convert("RGB"))})
        del adapted, high_raw, high_style; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
        print(f"cached {image_path.name}", flush=True)

    base = {"high_weight": .85, "pair_weight": .10, "style_weight": .75,
            "al_bias": .45, "beta": 7.0, "eut_bias": -1.0}
    candidates = {
        "model166": {"high_weight": .70, "pair_weight": .05, "style_weight": 1.0,
                     "al_bias": .45, "beta": 10.0, "eut_bias": -1.5},
        "model166_less_eut": {"high_weight": .70, "pair_weight": .05, "style_weight": 1.0,
                               "al_bias": .45, "beta": 10.0, "eut_bias": -1.0},
    }
    ranges = ((159.0, 161.0), (161.0, 163.0), (164.0, 166.0), (173.0, 176.0))
    records = []
    for candidate_name, candidate in candidates.items():
        for low, high in ranges:
            predictions, targets = [], []
            for item in items:
                if low <= item["mean"] < high:
                    params = candidate
                else:
                    params = dict(base)
                    params["high_weight"] = .85 if item["mean"] < 165.0 else .55
                    params["style_weight"] = .75 if item["std"] >= 50.0 else 0.0
                    params["beta"] = 3.0 if item["mean"] < 160.0 else 7.0
                    params["eut_bias"] = -1.0 if item["mean"] < 165.0 else -.3
                predictions.append(route(item, **params)); targets.append(item["target"])
            records.append({"candidate": candidate_name, "range": [low, high], **metrics(predictions, targets)})
    output = Path(args.output_json); output.parent.mkdir(parents=True, exist_ok=True)
    best = max(records, key=lambda x: x["image_mean_miou"])
    output.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "records": records}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True); parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pair-checkpoint", required=True); parser.add_argument("--high-checkpoint", required=True)
    parser.add_argument("--output-json", required=True); main(parser.parse_args())
