#!/usr/bin/env python3
"""Search high-resolution source replacement on validation images."""
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
from improve_five_causes import collect_split, decode_mask
from prior_representation_search import fit_percentile_stats, percentile_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior
from tent_style_search import adapt_model
from style_tta_search import predict_array, transform_image


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval(); return model


def route_params(mean, std):
    if 159.0 <= mean < 161.0:
        return {"high_weight": .70, "pair_weight": .05, "style_weight": 1.0,
                "al_bias": .45, "beta": 10.0, "eut_bias": -1.5}
    return {"high_weight": .85 if mean < 165.0 else .55,
            "pair_weight": .10, "style_weight": .75 if std >= 50 else 0.0,
            "al_bias": .45,
            "beta": (15.0 if 164.0 <= mean < 164.5 or 174.6 <= mean < 174.85
                     else (3.0 if mean < 160 else 7.0)),
            "eut_bias": (-2.0 if 161.3 <= mean < 162.0 else
                         (-.75 if 164.5 <= mean < 165.2 else
                          (.3 if 174.0 <= mean < 175.5 else (-1.0 if mean < 165 else -.3))))}


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
    sources = {
        "restart51": args.source_restart51,
        "model109": args.source_model109,
        "seed456": args.source_seed456,
        "pairwise": args.source_pairwise,
        "ema": args.source_ema,
    }
    source_models = {name: load_aspp(path, device) for name, path in sources.items()}
    wanted = []
    for image_path, mask_path in collect_split(data, "valid"):
        gray = np.asarray(Image.open(image_path).convert("L"), np.float32)
        mean, std = float(gray.mean()), float(gray.std())
        if args.all_images or ((159.0 <= mean < 161.0) or (161.3 <= mean < 162.0)
                              or (174.6 <= mean < 174.85)):
            wanted.append((image_path, mask_path, mean, std))
    records = []
    for image_path, mask_path, mean, std in wanted:
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), np.uint8)
        gray = raw[..., 0].astype(np.float32); z = (gray - mean) / max(std, 1e-5); q = percentile_image(gray)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        for source_name, source_model in source_models.items():
            adapted = adapt_model(sources[source_name], raw, device, amp, 5, 1e-5)
            high_raw = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
            high_style = predict_array(adapted, transform_image(raw, "standardize", train_mean, train_std),
                                       768, 576, device, amp, 1, "d4", "gaussian")
            item = {"branches": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
                    "pair": pair_value.astype(np.float32), "high_raw": high_raw.astype(np.float32),
                    "high_style": high_style.astype(np.float32), "z": z, "q": q,
                    "prior_z": gaussian_prior(z, z_stats), "prior_q": gaussian_prior(q, q_stats),
                    "mean": mean, "std": std, "target": decode_mask(Image.open(mask_path).convert("RGB"))}
            prediction = route(item, **route_params(mean, std))
            cm = np.bincount(item["target"].reshape(-1) * 4 + prediction.reshape(-1), minlength=16).reshape(4, 4)
            inter = np.diag(cm).astype(float); union = cm.sum(0) + cm.sum(1) - inter
            records.append({"image": image_path.name, "source": source_name,
                            "image_mean_miou": float(np.mean(inter / np.maximum(union, 1.0))),
                            "iou": (inter / np.maximum(union, 1.0)).tolist()})
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
            del adapted, high_raw, high_style; gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
    output = Path(args.output_json); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True); parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pair-checkpoint", required=True); parser.add_argument("--output-json", required=True)
    parser.add_argument("--source-restart51", required=True); parser.add_argument("--source-model109", required=True)
    parser.add_argument("--source-seed456", required=True); parser.add_argument("--source-pairwise", required=True)
    parser.add_argument("--source-ema", required=True); parser.add_argument("--all-images", action="store_true")
    main(parser.parse_args())
