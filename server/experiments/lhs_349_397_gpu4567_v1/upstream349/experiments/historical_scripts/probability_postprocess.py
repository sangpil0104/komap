#!/usr/bin/env python3
"""Search small probability-map smoothing variants for the joint candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from calibrate_al import predict_probabilities
from hard_negative_finetune import load_model
from improve_five_causes import collect_split, decode_mask
from intensity_prior_sweep import fit_stats
from restart_ensemble import metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    stats = fit_stats(args.data_dir)
    old_model = load_model(args.old_checkpoint, device)
    new_model = load_model(args.new_checkpoint, device)
    cached = []
    for image_path, mask_path in collect_split(args.data_dir, "valid"):
        old_p = predict_probabilities(old_model, image_path, 640, 512, device, use_amp, 2, True)
        new_p = predict_probabilities(new_model, image_path, 512, 384, device, use_amp, 2, True)
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        prior = np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                            for mean, std in stats], dtype=np.float32)
        probability = (1.0 - 0.60) * old_p + 0.60 * new_p
        logits = np.log(np.maximum(probability, 1e-7)) + 0.07 * prior
        logits[3] += 0.07 * (1.0 - 1.0) * prior[3]
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        cached.append((logits, target))
        print(f"cached {image_path.name}", flush=True)

    records = []
    for sigma in (0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0):
        for eutectic_sigma in (sigma, 0.5, 1.0, 1.5):
            predictions, targets = [], []
            for logits, target in cached:
                smoothed = logits.copy()
                if sigma > 0:
                    for class_id in range(4):
                        current_sigma = eutectic_sigma if class_id == 3 else sigma
                        smoothed[class_id] = gaussian_filter(logits[class_id], current_sigma, mode="reflect")
                smoothed[2] += 0.10
                smoothed[3] -= 0.50
                predictions.append(smoothed.argmax(axis=0).astype(np.uint8))
                targets.append(target)
            result = metrics(predictions, targets)
            records.append({"sigma": sigma, "eutectic_sigma": eutectic_sigma, **result})
    best = max(records, key=lambda item: item["image_mean_miou"])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
