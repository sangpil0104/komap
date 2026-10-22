#!/usr/bin/env python3
"""Validate model-169 plus hard-case-specific high-source replacements."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from encoder_structure_search import build_model
from hard_case_route_search import route
from hard_source_search import route_params
from improve_five_causes import collect_split, decode_mask
from prior_representation_search import fit_percentile_stats, percentile_image
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior
from tent_style_search import adapt_model


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval(); return model


def choose_source(mean, args):
    if 159.0 <= mean < 161.0:
        return args.source_pairwise
    if 161.3 <= mean < 162.0:
        return args.source_ema
    if 174.6 <= mean < 174.85:
        return args.source_model109
    if 174.85 <= mean < 175.1:
        return args.source_model109
    if 152.3 <= mean < 152.7:
        return args.source_seed456
    if 155.2 <= mean < 155.7:
        return args.source_ema
    return args.source_restart51


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"; data = Path(args.data_dir)
    train_stats = []
    for path, _ in collect_split(data, "train"):
        gray = np.asarray(Image.open(path).convert("L"), np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats])); train_std = float(np.mean([x[1] for x in train_stats]))
    z_stats, q_stats = fit_global_stats(data), fit_percentile_stats(data)
    pair = load_aspp(args.pair_checkpoint, device)
    predictions, targets, rows = [], [], []
    model_cache = {}
    for image_path, mask_path in collect_split(data, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), np.uint8)
        gray = raw[..., 0].astype(np.float32); mean, std = float(gray.mean()), float(gray.std())
        source = choose_source(mean, args)
        if source not in model_cache:
            model_cache[source] = load_aspp(source, device)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(source, raw, device, amp, 5, 1e-5)
        high_raw = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        high_style = predict_array(adapted, transform_image(raw, "standardize", train_mean, train_std),
                                   768, 576, device, amp, 1, "d4", "gaussian")
        z = (gray - mean) / max(std, 1e-5); q = percentile_image(gray)
        item = {"branches": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
                "pair": pair_value.astype(np.float32), "high_raw": high_raw.astype(np.float32),
                "high_style": high_style.astype(np.float32), "z": z, "q": q,
                "prior_z": gaussian_prior(z, z_stats), "prior_q": gaussian_prior(q, q_stats),
                "mean": mean, "std": std, "target": decode_mask(Image.open(mask_path).convert("RGB"))}
        prediction = route(item, **route_params(mean, std))
        predictions.append(prediction); targets.append(item["target"])
        rows.append({"image": image_path.name, "source": source,
                     "image_mean_miou": float(metrics([prediction], [item["target"]])["image_mean_miou"])})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        del adapted, high_raw, high_style; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    result = metrics(predictions, targets)
    output = Path(args.output_json); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": result, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True); parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pair-checkpoint", required=True); parser.add_argument("--output-json", required=True)
    parser.add_argument("--source-restart51", required=True); parser.add_argument("--source-model109", required=True)
    parser.add_argument("--source-ema", required=True); parser.add_argument("--source-pairwise", required=True)
    parser.add_argument("--source-seed456", required=True)
    main(parser.parse_args())
