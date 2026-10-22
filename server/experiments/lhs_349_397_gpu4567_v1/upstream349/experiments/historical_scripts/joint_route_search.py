#!/usr/bin/env python3
"""Joint search of intensity-conditioned branch routing and Al3Ni prior strength."""

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
from improve_five_causes import collect_split, decode_mask
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image
from tent_style_search import adapt_model


def load_aspp(path: str, device: torch.device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    data_dir = Path(args.data_dir)
    train_stats = []
    for p, _ in collect_split(data_dir, "train"):
        gray = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))

    pair = load_aspp(args.pair_checkpoint, device)
    items = []
    for image_path, mask_path in collect_split(data_dir, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        gray = raw[..., 0].astype(np.float32)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, raw, device, amp, 5, 1e-5)
        raw_high = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        style_high = predict_array(
            adapted, transform_image(raw, "standardize", train_mean, train_std),
            768, 576, device, amp, 1, "d4", "gaussian",
        )
        items.append({
            "sources": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
            "pair": pair_value,
            "raw_high": raw_high,
            "style_high": style_high,
            "mean": float(gray.mean()),
            "std": float(gray.std()),
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
            "name": image_path.name,
        })
        del adapted, raw_high, style_high
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"cached {image_path.name}", flush=True)

    def evaluate(setting, stride=1):
        predictions, targets = [], []
        for item in items:
            mean = item["mean"]
            high_weight = setting["low_high_weight"] if mean < setting["route_threshold"] else setting["high_high_weight"]
            beta = setting["low_beta"] if mean < setting["beta_threshold"] else setting["high_beta"]
            high = item["raw_high"]
            if item["std"] >= 50.0:
                high = 0.25 * high + 0.75 * item["style_high"]
            if stride > 1:
                high = high[:, ::stride, ::stride]
            pair_weight = setting["pair_weight"]
            base = MATRIX * (1.0 - high_weight - pair_weight)
            mixed = np.asarray([
                sum(base[c, m] * item["sources"][m][c, ::stride, ::stride] for m in range(3))
                + high_weight * high[c] + pair_weight * item["pair"][c]
                for c in range(4)
            ], dtype=np.float32)
            mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
            prior = item["base_prior"][:, ::stride, ::stride].copy()
            prior[0] += beta * (item["adaptive"][0, ::stride, ::stride] - prior[0])
            logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
            logits[2] += 0.40
            logits[3] += -0.30 if mean >= 165.0 else -1.0
            predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128))
            targets.append(item["target"][::stride, ::stride])
        return metrics(predictions, targets)

    # Build the adaptive Al3Ni prior once per image using the current best route.
    # This keeps stage 1 focused on branch routing while stage 2 tests whether the
    # prior strength remains stable after routing changes.
    for item in items:
        mean = item["mean"]
        high_weight = .75 if mean < 165.0 else .55
        high = item["raw_high"] if item["std"] < 50.0 else .25 * item["raw_high"] + .75 * item["style_high"]
        base = MATRIX * (1.0 - high_weight - .15)
        mixed = np.asarray([
            sum(base[c, m] * item["sources"][m][c] for m in range(3))
            + high_weight * high[c] + .15 * item["pair"][c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
        gray = np.asarray(Image.open(data_dir / "valid" / "images" / item["name"]).convert("L"), dtype=np.float32)
        z = (gray - mean) / max(float(gray.std()), 1e-5)
        stats = []
        for cid in range(4):
            weights = np.maximum(mixed[cid], 1e-7) ** 2.0
            weighted_mean = float((weights * z).sum() / weights.sum())
            weighted_std = float(np.sqrt((weights * (z - weighted_mean) ** 2).sum() / weights.sum()))
            stats.append((weighted_mean, max(weighted_std, .15)))
        item["adaptive"] = np.asarray([
            -0.5 * ((z - m) / s) ** 2 - np.log(s) for m, s in stats
        ], dtype=np.float32)
        # The train-derived global prior is reconstructed from the fixed cached
        # branch by using the same per-image cache source as the prior search.
        # It is stored explicitly below to avoid changing the established metric.
        # This assignment is overwritten from the cache in the next loop.
        item["base_prior"] = None

    # The cache's prior is the train-derived global intensity prior.
    for item in items:
        cache = np.load(Path(args.cache_dir) / f"{item['name'].replace('_image.png', '_image.npz')}")
        item["base_prior"] = cache["prior"].astype(np.float32)

    records = []
    for route_threshold in (155.0, 160.0, 165.0, 170.0, 175.0):
        for low_high_weight in (.65, .70, .75, .80):
            for high_high_weight in (.45, .50, .55, .60):
                for pair_weight in (.10, .15, .20):
                    if low_high_weight + pair_weight >= 1.0 or high_high_weight + pair_weight >= 1.0:
                        continue
                    setting = {
                        "route_threshold": route_threshold,
                        "low_high_weight": low_high_weight,
                        "high_high_weight": high_high_weight,
                        "pair_weight": pair_weight,
                        "beta_threshold": 160.0,
                        "low_beta": 3.0,
                        "high_beta": 7.0,
                    }
                    row = {"stage": 1, **setting, **evaluate(setting, stride=4)}
                    records.append(row)
    top = sorted(records, key=lambda x: x["image_mean_miou"], reverse=True)[:8]
    print("stage1_best", json.dumps(top[:3], ensure_ascii=False), flush=True)

    for parent in top:
        for beta_threshold in (155.0, 160.0, 165.0):
            for low_beta in (1.0, 3.0, 5.0, 7.0):
                for high_beta in (5.0, 7.0, 9.0):
                    setting = {
                        "route_threshold": parent["route_threshold"],
                        "low_high_weight": parent["low_high_weight"],
                        "high_high_weight": parent["high_high_weight"],
                        "pair_weight": parent["pair_weight"],
                        "beta_threshold": beta_threshold,
                        "low_beta": low_beta,
                        "high_beta": high_beta,
                    }
                    row = {"stage": 2, **setting, **evaluate(setting, stride=4)}
                    records.append(row)

    # Re-score only the strongest low-resolution candidates at full resolution.
    # This preserves the established metric while keeping the broad search short.
    full_records = []
    for parent in sorted(records, key=lambda x: x["image_mean_miou"], reverse=True)[:20]:
        setting = {key: parent[key] for key in (
            "route_threshold", "low_high_weight", "high_high_weight", "pair_weight",
            "beta_threshold", "low_beta", "high_beta",
        )}
        full_records.append({"stage": 3, **setting, **evaluate(setting, stride=1)})
    records.extend(full_records)
    best = max(full_records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--pair-checkpoint", required=True); p.add_argument("--high-checkpoint", required=True)
    p.add_argument("--output-json", required=True)
    main(p.parse_args())
