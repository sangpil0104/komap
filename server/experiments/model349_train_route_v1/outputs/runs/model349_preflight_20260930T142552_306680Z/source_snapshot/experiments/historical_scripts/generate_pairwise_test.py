#!/usr/bin/env python3
"""Generate test masks for the best pairwise-margin fifth-source ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aspp768_addition import MATRIX, cleanup_primary, fit_stats
from calibrate_al import predict_probabilities
from encoder_structure_search import build_model
from hard_negative_finetune import load_model
from improve_five_causes import collect_split, encode_mask
from train_models_2_to_6 import UNetPlusPlusResNet34


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
    pair = load_aspp(args.pair_checkpoint, device)
    stats = fit_stats(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png"))
    for image_path in images:
        values = [
            predict_probabilities(old, image_path, 640, 512, device, use_amp, 2, True),
            predict_probabilities(new, image_path, 512, 384, device, use_amp, 2, True),
            predict_probabilities(aspp, image_path, 512, 384, device, use_amp, 2, True),
            predict_probabilities(high, image_path, 768, 576, device, use_amp, 1, True),
            predict_probabilities(pair, image_path, 768, 576, device, use_amp, 1, True),
        ]
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        prior = np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                            for mean, std in stats], dtype=np.float32)
        high_weight, pair_weight = 0.60, 0.05
        base = MATRIX * (1.0 - high_weight - pair_weight)
        mixed = np.asarray([
            sum(base[c, m] * values[m][c] for m in range(3))
            + high_weight * values[3][c] + pair_weight * values[4][c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        logits = np.log(np.maximum(mixed, 1e-7)) + 0.12 * prior
        logits[2] += 0.40
        logits[3] -= 0.55
        prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
        encode_mask(prediction).save(output_dir / image_path.name.replace("_image.png", "_mask.png"))
        print(f"wrote {image_path.name}", flush=True)
    print(f"test_masks={len(images)} output={output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--aspp-checkpoint", required=True)
    parser.add_argument("--high-checkpoint", required=True)
    parser.add_argument("--pair-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
