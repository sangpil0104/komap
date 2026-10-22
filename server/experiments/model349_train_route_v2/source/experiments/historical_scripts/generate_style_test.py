#!/usr/bin/env python3
"""Generate test masks for the calibrated intensity-standardized style ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from aspp768_addition import MATRIX, fit_stats
from encoder_structure_search import build_model
from hard_negative_finetune import load_model
from improve_five_causes import encode_mask
from style_tta_search import predict_array, transform_image


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    old = load_model(args.old_checkpoint, device)
    new = load_model(args.new_checkpoint, device)
    aspp = load_aspp(args.aspp_checkpoint, device)
    high = load_aspp(args.high_checkpoint, device)
    train_values = []
    from improve_five_causes import collect_split
    for image_path, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_values.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_values]))
    train_std = float(np.mean([x[1] for x in train_values]))
    class_stats = fit_stats(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        original = predict_probabilities(old, image_path, 640, 512, device, use_amp, 2, "d4", "gaussian")
        newer = predict_probabilities(new, image_path, 512, 384, device, use_amp, 2, "d4", "gaussian")
        aspp_value = predict_probabilities(aspp, image_path, 512, 384, device, use_amp, 2, "d4", "gaussian")
        high_raw = predict_probabilities(high, image_path, 768, 576, device, use_amp, 1, "d4", "gaussian")
        styled = predict_array(high, transform_image(raw, "standardize", train_mean, train_std),
                               768, 576, device, use_amp, 1, "d4", "gaussian")
        high_value = 0.25 * high_raw + 0.75 * styled if raw[..., 0].astype(np.float32).std() >= args.route_threshold else high_raw
        base = MATRIX * 0.40
        values = [original, newer, aspp_value]
        mixed = np.asarray([sum(base[c, m] * values[m][c] for m in range(3)) + 0.60 * high_value[c]
                            for c in range(4)], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        gray = raw[..., 0].astype(np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        prior = np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                            for mean, std in class_stats], dtype=np.float32)
        logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
        logits[2] += 0.40; logits[3] -= 0.50
        encode_mask(logits.argmax(0).astype(np.uint8)).save(
            output_dir / image_path.name.replace("_image.png", "_mask.png"))
        print(f"wrote {image_path.name}", flush=True)
    print(f"test_masks={len(list(output_dir.glob('*.png')))} output={output_dir}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--old-checkpoint", required=True)
    p.add_argument("--new-checkpoint", required=True); p.add_argument("--aspp-checkpoint", required=True)
    p.add_argument("--high-checkpoint", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--route-threshold", type=float, default=50.0)
    p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
