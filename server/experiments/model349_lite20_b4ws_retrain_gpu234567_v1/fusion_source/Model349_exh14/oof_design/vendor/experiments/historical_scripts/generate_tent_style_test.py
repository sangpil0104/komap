#!/usr/bin/env python3
"""Generate test masks with label-free TENT adaptation and style routing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from aspp768_addition import MATRIX, fit_stats
from calibrate_al import load_model as load_unet
from encoder_structure_search import build_model
from improve_five_causes import collect_split, encode_mask
from style_tta_search import predict_array, transform_image
from tent_style_search import adapt_model


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    old = load_unet(args.old_checkpoint, device)
    new = load_unet(args.new_checkpoint, device)
    aspp = load_aspp(args.aspp_checkpoint, device)
    stats = []
    for image_path, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        stats.append((float(gray.mean()), float(gray.std())))
    train_mean, train_std = float(np.mean([x[0] for x in stats])), float(np.mean([x[1] for x in stats]))
    class_stats = fit_stats(args.data_dir)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        # The low-resolution sources are kept fixed; only the 768px branch is adapted.
        p0 = predict_probabilities(old, image_path, 640, 512, device, use_amp, 2, "d4", "gaussian")
        p1 = predict_probabilities(new, image_path, 512, 384, device, use_amp, 2, "d4", "gaussian")
        p2 = predict_probabilities(aspp, image_path, 512, 384, device, use_amp, 2, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, raw, device, use_amp, args.steps, args.lr)
        raw_high = predict_array(adapted, raw, 768, 576, device, use_amp, 1, "d4", "gaussian")
        style_high = predict_array(adapted, transform_image(raw, "standardize", train_mean, train_std),
                                   768, 576, device, use_amp, 1, "d4", "gaussian")
        high = 0.25 * raw_high + 0.75 * style_high if raw[..., 0].astype(np.float32).std() >= args.route_threshold else raw_high
        base = MATRIX * 0.40
        mixed = np.asarray([sum(base[c, m] * [p0, p1, p2][m][c] for m in range(3)) + 0.60 * high[c]
                            for c in range(4)], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        gray = raw[..., 0].astype(np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        prior = np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                            for mean, std in class_stats], dtype=np.float32)
        logits = np.log(np.maximum(mixed, 1e-7)) + args.alpha * prior
        logits[2] += args.al_bias; logits[3] += args.eut_bias
        encode_mask(logits.argmax(0).astype(np.uint8)).save(
            output / image_path.name.replace("_image.png", "_mask.png"))
        print(f"wrote {image_path.name}", flush=True)
    print(f"test_masks={len(list(output.glob('*.png')))} output={output}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--old-checkpoint", required=True)
    p.add_argument("--new-checkpoint", required=True); p.add_argument("--aspp-checkpoint", required=True)
    p.add_argument("--high-checkpoint", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=2); p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--route-threshold", type=float, default=50.0); p.add_argument("--alpha", type=float, default=0.09)
    p.add_argument("--al-bias", type=float, default=0.40); p.add_argument("--eut-bias", type=float, default=-0.50)
    p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
