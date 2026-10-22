#!/usr/bin/env python3
"""Generate test masks for the TENT + pairwise-margin ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

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


def remove_small_eutectic(prediction, logits, threshold):
    if threshold <= 0:
        return prediction
    labels, count = ndimage.label(prediction == 3,
                                  structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    small = (sizes[labels] < threshold) & (prediction == 3)
    if not np.any(small):
        return prediction
    alternatives = logits.copy()
    alternatives[3] = -np.inf
    result = prediction.copy()
    result[small] = alternatives[:, small].argmax(axis=0).astype(np.uint8)
    return result


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    old = load_unet(args.old_checkpoint, device)
    new = load_unet(args.new_checkpoint, device)
    aspp = load_aspp(args.aspp_checkpoint, device)
    pair = load_aspp(args.pair_checkpoint, device)
    stats = []
    for image_path, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in stats]))
    train_std = float(np.mean([x[1] for x in stats]))
    class_stats = fit_stats(args.data_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        p0 = predict_probabilities(old, image_path, 640, 512, device, use_amp, 2, "d4", "gaussian")
        p1 = predict_probabilities(new, image_path, 512, 384, device, use_amp, 2, "d4", "gaussian")
        p2 = predict_probabilities(aspp, image_path, 512, 384, device, use_amp, 2, "d4", "gaussian")
        pp = predict_probabilities(pair, image_path, 768, 576, device, use_amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, raw, device, use_amp, args.steps, args.lr)
        raw_high = predict_array(adapted, raw, 768, 576, device, use_amp, 1, "d4", "gaussian")
        style_high = predict_array(
            adapted, transform_image(raw, "standardize", train_mean, train_std),
            768, 576, device, use_amp, 1, "d4", "gaussian"
        )
        image_std = float(raw[..., 0].astype(np.float32).std())
        high = 0.25 * raw_high + 0.75 * style_high if image_std >= args.route_threshold else raw_high
        gray = raw[..., 0].astype(np.float32)
        if args.weight_feature == "mean":
            high_weight = (args.high_high_weight if float(gray.mean()) >= args.weight_threshold
                           else args.low_high_weight)
        else:
            high_weight = args.high_weight
        base = MATRIX * (1.0 - high_weight - args.pair_weight)
        mixed = np.asarray([
            sum(base[c, m] * [p0, p1, p2][m][c] for m in range(3))
            + high_weight * high[c] + args.pair_weight * pp[c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        prior = np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                            for mean, std in class_stats], dtype=np.float32)
        logits = np.log(np.maximum(mixed, 1e-7)) + args.alpha * prior
        logits[2] += args.al_bias
        if args.bias_feature == "mean":
            logits[3] += (args.high_eut_bias if float(gray.mean()) >= args.bias_threshold
                          else args.low_eut_bias)
        else:
            logits[3] += args.eut_bias
        prediction = remove_small_eutectic(logits.argmax(0).astype(np.uint8), logits,
                                            args.eutectic_component_threshold)
        encode_mask(prediction).save(
            output / image_path.name.replace("_image.png", "_mask.png"))
        print(f"wrote {image_path.name}", flush=True)
    print(f"test_masks={len(list(output.glob('*.png')))} output={output}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--old-checkpoint", required=True)
    p.add_argument("--new-checkpoint", required=True)
    p.add_argument("--aspp-checkpoint", required=True)
    p.add_argument("--pair-checkpoint", required=True)
    p.add_argument("--high-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--pair-weight", type=float, default=0.12)
    p.add_argument("--high-weight", type=float, default=0.60)
    p.add_argument("--weight-feature", choices=("fixed", "mean"), default="fixed")
    p.add_argument("--weight-threshold", type=float, default=165.0)
    p.add_argument("--low-high-weight", type=float, default=0.60)
    p.add_argument("--high-high-weight", type=float, default=0.60)
    p.add_argument("--route-threshold", type=float, default=50.0)
    p.add_argument("--alpha", type=float, default=0.09)
    p.add_argument("--al-bias", type=float, default=0.40)
    p.add_argument("--eut-bias", type=float, default=-0.50)
    p.add_argument("--bias-feature", choices=("fixed", "mean"), default="fixed")
    p.add_argument("--bias-threshold", type=float, default=167.037086)
    p.add_argument("--low-eut-bias", type=float, default=-0.70)
    p.add_argument("--high-eut-bias", type=float, default=-0.30)
    p.add_argument("--eutectic-component-threshold", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
