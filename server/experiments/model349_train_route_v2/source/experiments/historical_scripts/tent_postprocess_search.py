#!/usr/bin/env python3
"""Search class-specific connected-component cleanup after TENT inference."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from advanced_inference import predict_probabilities
from aspp768_addition import MATRIX, cleanup_primary
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image
from tent_style_search import adapt_model


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def remove_small(prediction, logits, class_id, threshold):
    if threshold <= 0:
        return prediction
    labels, count = ndimage.label(prediction == class_id,
                                  structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    small = sizes[labels] < threshold
    small &= prediction == class_id
    if not np.any(small):
        return prediction
    alternative = logits.copy()
    alternative[class_id] = -np.inf
    result = prediction.copy()
    result[small] = alternative[:, small].argmax(axis=0).astype(np.uint8)
    return result


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    samples = collect_split(args.data_dir, "valid")
    entries = []
    for image_path, mask_path in _local_progress(samples, 'tent_postprocess_search'):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        entries.append({
            "values": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
            "prior": loaded["prior"].astype(np.float32),
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
            "raw": raw,
            "std": float(raw[..., 0].astype(np.float32).std()),
            "name": image_path.name,
        })
    pair = load_aspp(args.pair_checkpoint, device)
    train_stats = []
    for image_path, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))

    cached = []
    for entry, (image_path, _) in zip(entries, samples):
        pair_value = predict_probabilities(pair, image_path, 768, 576, device,
                                            use_amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, entry["raw"], device, use_amp,
                              args.steps, args.lr)
        raw_high = predict_array(adapted, entry["raw"], 768, 576, device,
                                 use_amp, 1, "d4", "gaussian")
        style_high = predict_array(
            adapted, transform_image(entry["raw"], "standardize", train_mean, train_std),
            768, 576, device, use_amp, 1, "d4", "gaussian"
        )
        high = (0.25 * raw_high + 0.75 * style_high
                if entry["std"] >= args.route_threshold else raw_high)
        base = MATRIX * (0.40 - args.pair_weight)
        mixed = np.asarray([
            sum(base[c, m] * entry["values"][m][c] for m in range(3))
            + 0.60 * high[c] + args.pair_weight * pair_value[c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        logits = np.log(np.maximum(mixed, 1e-7)) + args.alpha * entry["prior"]
        logits[2] += args.al_bias
        logits[3] += args.eut_bias
        prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
        cached.append((prediction, logits, entry["target"], entry["name"]))
        print(f"cached {entry['name']}", flush=True)

    records = []
    # First isolate each class, then test a small joint grid for the two dominant
    # confusion classes. Primary cleanup remains fixed at the validated 128 px.
    for class_id in (0, 3):
        for threshold in (0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            predictions, targets = [], []
            for prediction, logits, target, _ in cached:
                output = remove_small(prediction, logits, class_id, threshold)
                predictions.append(output); targets.append(target)
            value = metrics(predictions, targets)
            row = {"mode": "single", "class_id": class_id,
                   "threshold": threshold, **value}
            records.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)

    for al3ni_threshold in (0, 8, 32, 128, 512):
        for eutectic_threshold in (0, 8, 32, 128, 512, 1024):
            predictions, targets = [], []
            for prediction, logits, target, _ in cached:
                output = remove_small(prediction, logits, 0, al3ni_threshold)
                output = remove_small(output, logits, 3, eutectic_threshold)
                predictions.append(output); targets.append(target)
            value = metrics(predictions, targets)
            row = {"mode": "joint", "al3ni_threshold": al3ni_threshold,
                   "eutectic_threshold": eutectic_threshold, **value}
            records.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)

    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--high-checkpoint", required=True); p.add_argument("--pair-checkpoint", required=True)
    p.add_argument("--output-json", required=True); p.add_argument("--steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5); p.add_argument("--pair-weight", type=float, default=.10)
    p.add_argument("--route-threshold", type=float, default=50.0); p.add_argument("--alpha", type=float, default=.09)
    p.add_argument("--al-bias", type=float, default=.40); p.add_argument("--eut-bias", type=float, default=-.50)
    p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
