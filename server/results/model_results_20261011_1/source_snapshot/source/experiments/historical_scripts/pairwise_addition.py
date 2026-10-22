#!/usr/bin/env python3
"""Evaluate the pairwise-margin model as a fifth source in the 768px ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aspp768_addition import BIASES, MATRIX, cleanup_primary
from calibrate_al import predict_probabilities
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask, encode_mask
from restart_ensemble import metrics


def load_cache(data_dir, cache_dir):
    entries = []
    for image_path, mask_path in collect_split(data_dir, "valid"):
        loaded = np.load(Path(cache_dir) / f"{image_path.stem}.npz")
        entries.append({
            "name": image_path.name,
            "values": [loaded[f"p{i}"].astype(np.float32) for i in range(4)],
            "prior": loaded["prior"].astype(np.float32),
            "target": decode_mask(Image.open(mask_path).convert("RGB")),
        })
    return entries


def mix(values, pair, high_weight, pair_weight):
    old_scale = 1.0 - float(high_weight) - float(pair_weight)
    base = MATRIX * old_scale
    result = np.asarray([
        sum(base[c, m] * values[m][c] for m in range(3))
        + float(high_weight) * values[3][c]
        + float(pair_weight) * pair[c]
        for c in range(4)
    ], dtype=np.float32)
    return result / np.maximum(result.sum(axis=0, keepdims=True), 1e-7)


def evaluate(entries, pair_values, high_weight, pair_weight, alpha, biases, threshold=128):
    predictions, targets = [], []
    bias = np.asarray(biases, dtype=np.float32)[:, None, None]
    for entry, pair in zip(entries, pair_values):
        probability = mix(entry["values"], pair, high_weight, pair_weight)
        prior = entry["prior"]
        logits = np.log(np.maximum(probability, 1e-7)) + float(alpha) * prior + bias
        prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, threshold)
        predictions.append(prediction)
        targets.append(entry["target"])
    return metrics(predictions, targets)


@torch.no_grad()
def infer_model(model, samples, args, device, use_amp):
    model.eval()
    return [predict_probabilities(model, image_path, args.patch_size, args.stride, device,
                                  use_amp, args.tile_batch_size, True)
            for image_path, _ in samples]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    entries = load_cache(args.data_dir, args.cache_dir)
    valid_samples = collect_split(args.data_dir, "valid")
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    print(f"loaded={args.checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    pair_values = infer_model(model, valid_samples, args, device, use_amp)
    for entry, value in zip(entries, pair_values):
        entry["pair"] = value

    fixed_bias = [0.0, 0.0, 0.40, -0.55]
    records = []
    for high_weight in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        for pair_weight in (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25):
            if high_weight + pair_weight >= 0.98:
                continue
            result = evaluate(entries, pair_values, high_weight, pair_weight, 0.12, fixed_bias)
            records.append({"high_weight": high_weight, "pair_weight": pair_weight,
                            "alpha": 0.12, "biases": fixed_bias, **result})
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    best_weight = max(records, key=lambda item: item["image_mean_miou"])

    # Recalibrate only around the selected source mixture. The main selection
    # remains the fixed current calibration above, so this second pass is
    # clearly separated from the source-weight result.
    calibrated = []
    if args.fast:
        calibrated = [best_weight]
    else:
        for alpha in (0.06, 0.09, 0.12, 0.15, 0.18):
            for al3ni in (0.0, 0.2, 0.4):
                for al_bias in (0.20, 0.40, 0.60, 0.80):
                    for eutectic in (-0.80, -0.70, -0.60, -0.55, -0.50, -0.40, -0.30):
                        value = evaluate(entries, pair_values, best_weight["high_weight"],
                                         best_weight["pair_weight"], alpha,
                                         [al3ni, 0.0, al_bias, eutectic])
                        calibrated.append({"high_weight": best_weight["high_weight"],
                                           "pair_weight": best_weight["pair_weight"],
                                           "alpha": alpha, "biases": [al3ni, 0.0, al_bias, eutectic], **value})
    best_calibrated = max(calibrated, key=lambda item: item["image_mean_miou"])
    result = {"best_fixed_calibration": best_weight, "best_calibrated": best_calibrated,
              "records": records, "calibration_records": calibrated}
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_fixed_calibration": best_weight, "best_calibrated": best_calibrated},
                     ensure_ascii=False), flush=True)

    # Save the pairwise model's validation probabilities for reproducibility.
    np.savez_compressed(output.parent / "pair_valid_probabilities.npz",
                        **{f"p{i}": value.astype(np.float16) for i, value in enumerate(pair_values)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--stride", type=int, default=576)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--fast", action="store_true", help="skip expensive calibration grid")
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
