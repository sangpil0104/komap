#!/usr/bin/env python3
"""Train-only grayscale intensity prior for the mixed restart ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from calibrate_al import predict_probabilities
from hard_negative_finetune import load_model
from improve_five_causes import CLASS_NAMES, collect_split, decode_mask
from restart_ensemble import metrics


COLORS = ((25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204))


def fit_stats(data_dir):
    values = [[] for _ in range(4)]
    root = Path(data_dir)
    for image_path, mask_path in collect_split(root, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        mask = np.asarray(Image.open(mask_path).convert("RGB"))
        ids = np.full(mask.shape[:2], -1, dtype=np.int8)
        for class_id, color in enumerate(COLORS):
            ids[np.all(mask == color, axis=2)] = class_id
        rng = np.random.default_rng(sum(image_path.name.encode("utf-8")))
        for class_id in range(4):
            current = z[ids == class_id]
            if len(current):
                take = rng.choice(len(current), min(len(current), 10000), replace=False)
                values[class_id].append(current[take])
    stats = []
    for current in values:
        sample = np.concatenate(current)
        stats.append((float(sample.mean()), max(float(sample.std()), 0.15)))
    return stats


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    stats = fit_stats(args.data_dir)
    print(json.dumps({"train_z_stats": stats}, ensure_ascii=False), flush=True)
    old_model = load_model(args.old_checkpoint, device)
    new_model = load_model(args.new_checkpoint, device)
    cached = []
    for image_path, mask_path in collect_split(args.data_dir, "valid"):
        old_p = predict_probabilities(old_model, image_path, 640, 512, device, use_amp, 2, True)
        new_p = predict_probabilities(new_model, image_path, 512, 384, device, use_amp, 2, True)
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        prior = []
        for mean, std in stats:
            prior.append(-0.5 * ((z - mean) / std) ** 2 - np.log(std))
        cached.append((old_p, new_p, np.asarray(prior, dtype=np.float32),
                       decode_mask(Image.open(mask_path).convert("RGB"))))
        print(f"cached {image_path.name}", flush=True)

    records = []
    for model_weight in (0.45, 0.50, 0.55, 0.60):
        for alpha in (0.03, 0.05, 0.07):
            for eutectic_scale in (1.0, 1.5):
                for al_bias in (-0.10, 0.0, 0.10):
                    for eutectic_bias in (-0.50, -0.45, -0.40):
                        predictions, targets = [], []
                        bias = np.asarray([0.0, 0.0, al_bias, eutectic_bias], dtype=np.float32)[:, None, None]
                        for old_probability, new_probability, prior, target in cached:
                            probability = (1.0 - model_weight) * old_probability + model_weight * new_probability
                            logits = np.log(np.maximum(probability, 1e-7)) + alpha * prior
                            logits[3] += alpha * (eutectic_scale - 1.0) * prior[3]
                            predictions.append((logits + bias).argmax(axis=0).astype(np.uint8))
                            targets.append(target)
                        result = metrics(predictions, targets)
                        records.append({"model_weight": model_weight, "alpha": alpha,
                                        "eutectic_scale": eutectic_scale, "al_bias": al_bias,
                                        "eutectic_bias": eutectic_bias, **result})
    best = max(records, key=lambda item: item["image_mean_miou"])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best": best, "records": records, "stats": stats},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-weight", type=float, default=0.55)
    parser.add_argument("--al3ni-bias", type=float, default=0.0)
    parser.add_argument("--primary-bias", type=float, default=0.0)
    parser.add_argument("--al-bias", type=float, default=-0.10)
    parser.add_argument("--eutectic-bias", type=float, default=-0.55)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
