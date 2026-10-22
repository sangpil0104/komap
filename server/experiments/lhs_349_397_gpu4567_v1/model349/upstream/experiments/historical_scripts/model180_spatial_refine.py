#!/usr/bin/env python3
"""Spatial refinement search on the exact model-173 logits."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from advanced_inference import predict_probabilities
from aspp768_addition import cleanup_primary
from encoder_structure_search import build_model
from hard_source_search import route_params
from highres1024_search import choose_source, make_logits
from improve_five_causes import collect_split, decode_mask
from prior_representation_search import fit_percentile_stats, percentile_image
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats
from tent_style_search import adapt_model


def load_aspp(path: str, device: torch.device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def remove_small(prediction: np.ndarray, logits: np.ndarray, class_id: int, threshold: int) -> np.ndarray:
    if threshold <= 0:
        return prediction
    labels, count = ndimage.label(prediction == class_id, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return prediction
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    small = (prediction == class_id) & (sizes[labels] < threshold)
    if not small.any():
        return prediction
    alternative = logits.copy()
    alternative[class_id] = -np.inf
    output = prediction.copy()
    output[small] = alternative[:, small].argmax(axis=0).astype(np.uint8)
    return output


def predict_from_logits(logits: np.ndarray) -> np.ndarray:
    prediction = cleanup_primary(logits.argmax(axis=0).astype(np.uint8), logits, 128)
    # Keep the established A71 policy from model 173.
    if float((prediction == 1).mean()) <= 0.01:
        mask = prediction == 1
        alternative = logits.copy()
        alternative[1] = -np.inf
        prediction[mask] = alternative[:, mask].argmax(axis=0).astype(np.uint8)
    return prediction


def smooth_logits(logits: np.ndarray, sigma: float, mode: str) -> np.ndarray:
    if sigma <= 0:
        return logits
    output = logits.copy()
    if mode == "all":
        return ndimage.gaussian_filter(logits, sigma=(0.0, sigma, sigma), mode="reflect")
    class_ids = {"al3ni": (0,), "primary": (1,), "al": (2,), "eutectic": (3,),
                 "rare": (0, 3)}[mode]
    for class_id in class_ids:
        output[class_id] = ndimage.gaussian_filter(logits[class_id], sigma=sigma, mode="reflect")
    return output


def evaluate_logits(entries, transform, post=None):
    predictions, targets = [], []
    for entry in entries:
        logits = transform(entry["logits"])
        prediction = predict_from_logits(logits)
        if post is not None:
            for class_id, threshold in post:
                prediction = remove_small(prediction, logits, class_id, threshold)
        predictions.append(prediction)
        targets.append(entry["target"])
    return metrics(predictions, targets)


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
    source_paths = {"restart51": args.source_restart51, "model109": args.source_model109,
                    "seed456": args.source_seed456, "pairwise": args.source_pairwise,
                    "ema": args.source_ema}
    source_args = argparse.Namespace(**{f"source_{key}": value for key, value in source_paths.items()})

    entries = []
    for image_path, mask_path in collect_split(data, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        gray = raw[..., 0].astype(np.float32)
        mean, std = float(gray.mean()), float(gray.std())
        source = choose_source(mean, source_args)
        source_name = next(name for name, path in source_paths.items() if path == source)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(source, raw, device, amp, 5, 1e-5)
        style = transform_image(raw, "standardize", train_mean, train_std)
        high768 = (
            predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian"),
            predict_array(adapted, style, 768, 576, device, amp, 1, "d4", "gaussian"),
        )
        high1024 = (
            predict_array(adapted, raw, 1024, 768, device, amp, 1, "d4", "gaussian"),
            predict_array(adapted, style, 1024, 768, device, amp, 1, "d4", "gaussian"),
        )
        high = tuple((1.0 - args.alpha_1024) * high768[index]
                     + args.alpha_1024 * high1024[index] for index in (0, 1))
        item = {
            "branches": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
            "pair": pair_value.astype(np.float32), "high": high,
            "z": (gray - mean) / max(std, 1e-5), "q": percentile_image(gray),
            "prior_z_stats": global_z, "prior_q_stats": global_q,
            "params": route_params(mean, std),
        }
        logits = make_logits(item, high)
        entries.append({"name": image_path.name, "logits": logits.astype(np.float32),
                        "target": decode_mask(Image.open(mask_path).convert("RGB"))})
        del adapted, high768, high1024
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"cached {image_path.name} source={source_name}", flush=True)

    records = []
    baseline = evaluate_logits(entries, lambda logits: logits)
    records.append({"kind": "baseline", "sigma": 0.0, **baseline})
    print(json.dumps(records[-1], ensure_ascii=False), flush=True)

    for mode, sigma in itertools.product(("all", "al3ni", "primary", "al", "eutectic", "rare"),
                                          (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)):
        value = evaluate_logits(entries, lambda logits, s=sigma, m=mode: smooth_logits(logits, s, m))
        row = {"kind": "gaussian", "mode": mode, "sigma": sigma, **value}
        records.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)

    # Component cleanup is evaluated on the exact unsmoothed route and on the
    # strongest few smoothing candidates.  This keeps the search broad without
    # fitting a separate morphology rule to every validation image.
    candidates = sorted((row for row in records if row["kind"] == "gaussian"),
                        key=lambda row: row["image_mean_miou"], reverse=True)[:8]
    transforms = [("baseline", lambda logits: logits)]
    transforms.extend((f"{row['mode']}_{row['sigma']}",
                       lambda logits, s=row["sigma"], m=row["mode"]: smooth_logits(logits, s, m))
                      for row in candidates)
    for name, transform in transforms:
        for class_id in range(4):
            for threshold in (16, 32, 64, 128, 256, 512, 1024, 2048):
                value = evaluate_logits(entries, transform, ((class_id, threshold),))
                row = {"kind": "component", "transform": name,
                       "class_id": class_id, "threshold": threshold, **value}
                records.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    best = max(records, key=lambda row: row["image_mean_miou"])
    best_global = max(records, key=lambda row: row["global_miou"])
    output = Path(args.output_json); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"baseline": baseline, "best": best,
                                  "best_global": best_global, "records": records},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "best_global": best_global}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True); parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pair-checkpoint", required=True); parser.add_argument("--output-json", required=True)
    parser.add_argument("--source-restart51", required=True); parser.add_argument("--source-model109", required=True)
    parser.add_argument("--source-seed456", required=True); parser.add_argument("--source-pairwise", required=True)
    parser.add_argument("--source-ema", required=True); parser.add_argument("--alpha-1024", type=float, default=0.20)
    main(parser.parse_args())
