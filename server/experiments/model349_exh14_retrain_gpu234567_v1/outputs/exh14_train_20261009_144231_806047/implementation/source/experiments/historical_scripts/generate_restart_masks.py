#!/usr/bin/env python3
"""Generate valid/test masks for the current joint-search restart candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from calibrate_al import predict_probabilities
from hard_negative_finetune import load_model
from improve_five_causes import collect_split, decode_mask, encode_mask
from intensity_prior_sweep import fit_stats
from restart_ensemble import metrics


def predict_one(old_model, new_model, image_path, stats, args, device, use_amp):
    old_p = predict_probabilities(old_model, image_path, 640, 512, device, use_amp, 2, True)
    new_p = predict_probabilities(new_model, image_path, 512, 384, device, use_amp, 2, True)
    gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
    prior = []
    for mean, std in stats:
        prior.append(-0.5 * ((z - mean) / std) ** 2 - np.log(std))
    probability = (1.0 - args.model_weight) * old_p + args.model_weight * new_p
    logits = np.log(np.maximum(probability, 1e-7)) + args.alpha * np.asarray(prior, dtype=np.float32)
    logits[3] += args.alpha * (args.eutectic_scale - 1.0) * prior[3]
    bias = np.asarray([0.0, 0.0, args.al_bias, args.eutectic_bias], dtype=np.float32)[:, None, None]
    return (logits + bias).argmax(axis=0).astype(np.uint8)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    stats = fit_stats(args.data_dir)
    old_model = load_model(args.old_checkpoint, device)
    new_model = load_model(args.new_checkpoint, device)
    valid_predictions, targets = [], []
    valid_dir = Path(args.output_dir) / "valid_prediction"
    valid_dir.mkdir(parents=True, exist_ok=True)
    for image_path, mask_path in collect_split(args.data_dir, "valid"):
        prediction = predict_one(old_model, new_model, image_path, stats, args, device, use_amp)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        encode_mask(prediction).save(valid_dir / image_path.name.replace("_image.png", "_mask.png"))
        valid_predictions.append(prediction)
        targets.append(target)
        print(f"valid {image_path.name}", flush=True)
    result = metrics(valid_predictions, targets)
    test_dir = Path(args.output_dir) / "test_masks"
    test_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        prediction = predict_one(old_model, new_model, image_path, stats, args, device, use_amp)
        encode_mask(prediction).save(test_dir / image_path.name.replace("_image.png", "_mask.png"))
    manifest = {"params": vars(args), "metrics": result, "class_order": ["Al3Ni", "Primary Si", "Al", "Eutectic Si"]}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-weight", type=float, default=0.60)
    parser.add_argument("--alpha", type=float, default=0.07)
    parser.add_argument("--eutectic-scale", type=float, default=1.0)
    parser.add_argument("--al-bias", type=float, default=0.10)
    parser.add_argument("--eutectic-bias", type=float, default=-0.50)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
