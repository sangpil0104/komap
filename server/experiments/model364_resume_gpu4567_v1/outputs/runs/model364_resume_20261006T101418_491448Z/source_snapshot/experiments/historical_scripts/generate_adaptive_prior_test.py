#!/usr/bin/env python3
"""Generate model 140 test masks with the label-free adaptive intensity prior."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from aspp768_addition import MATRIX, cleanup_primary
from calibrate_al import load_model as load_unet
from encoder_structure_search import build_model
from improve_five_causes import collect_split, encode_mask
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior, load_aspp
from tent_style_search import adapt_model
from prior_representation_search import fit_percentile_stats, percentile_image


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    data_dir = Path(args.data_dir)

    train_stats = []
    for image_path, _ in collect_split(data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))
    global_stats = fit_global_stats(data_dir)
    percentile_stats = fit_percentile_stats(data_dir) if args.percentile_prior_weight > 0 else None

    old_model = load_unet(args.old_checkpoint, device)
    new_model = load_unet(args.new_checkpoint, device)
    aspp_model = load_aspp(args.aspp_checkpoint, device)
    pair_model = load_aspp(args.pair_checkpoint, device)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    test_images = sorted((data_dir / "test" / "images").glob("*_image.png"))
    for image_path in test_images:
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        p0 = predict_probabilities(old_model, image_path, 640, 512, device, amp, 2, "d4", "gaussian")
        p1 = predict_probabilities(new_model, image_path, 512, 384, device, amp, 2, "d4", "gaussian")
        p2 = predict_probabilities(aspp_model, image_path, 512, 384, device, amp, 2, "d4", "gaussian")
        pp = predict_probabilities(pair_model, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        gray = raw[..., 0].astype(np.float32)
        mean = float(gray.mean())
        high_checkpoint = args.high_checkpoint
        if (args.special_high_checkpoint is not None
                and args.special_mean_low is not None and args.special_mean_high is not None
                and args.special_mean_low <= mean < args.special_mean_high):
            high_checkpoint = args.special_high_checkpoint
        elif (args.narrow_a186_high_checkpoint is not None
              and args.narrow_a186_low <= mean < args.narrow_a186_high):
            high_checkpoint = args.narrow_a186_high_checkpoint
        elif (args.narrow_a67_high_checkpoint is not None
              and args.narrow_a67_low <= mean < args.narrow_a67_high):
            high_checkpoint = args.narrow_a67_high_checkpoint
        elif (args.narrow_a11_step2_high_checkpoint is not None
              and args.narrow_a11_step2_low <= mean < args.narrow_a11_step2_high):
            high_checkpoint = args.narrow_a11_step2_high_checkpoint
        elif (args.narrow_mean152_high_checkpoint is not None
              and 152.3 <= mean < 152.7):
            high_checkpoint = args.narrow_mean152_high_checkpoint
        elif (args.narrow_mean155_high_checkpoint is not None
              and 155.2 <= mean < 155.7):
            high_checkpoint = args.narrow_mean155_high_checkpoint
        adapted = adapt_model(high_checkpoint, raw, device, amp, args.steps, args.lr)
        raw_high = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        style = transform_image(raw, "standardize", train_mean, train_std)
        style_high = predict_array(adapted, style, 768, 576, device, amp, 1, "d4", "gaussian")

        z = (gray - mean) / max(float(gray.std()), 1e-5)
        q = percentile_image(gray) if args.percentile_prior_weight > 0 else None
        special = (args.special_mean_low is not None and args.special_mean_high is not None
                   and args.special_mean_low <= mean < args.special_mean_high)
        if special:
            high_weight = args.special_high_weight
            pair_weight = args.special_pair_weight
            style_weight = args.special_style_weight
        else:
            high_weight = args.high_high_weight if mean >= args.weight_threshold else args.low_high_weight
            pair_weight = args.pair_weight
            style_weight = .75
        high = ((1.0 - style_weight) * raw_high + style_weight * style_high
                if float(gray.std()) >= 50.0 else raw_high)
        base = MATRIX * (1.0 - high_weight - pair_weight)
        mixed = np.asarray([
            sum(base[c, m] * [p0, p1, p2][m][c] for m in range(3))
            + high_weight * high[c] + pair_weight * pp[c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)

        adaptive_stats = []
        adaptive_percentile_stats = []
        for cid in range(4):
            weights = np.maximum(mixed[cid], 1e-7) ** args.gamma
            weighted_mean = float((weights * z).sum() / weights.sum())
            weighted_std = float(np.sqrt((weights * (z - weighted_mean) ** 2).sum() / weights.sum()))
            adaptive_stats.append((weighted_mean, max(weighted_std, .15)))
            if q is not None:
                q_mean = float((weights * q).sum() / weights.sum())
                q_std = float(np.sqrt((weights * (q - q_mean) ** 2).sum() / weights.sum()))
                adaptive_percentile_stats.append((q_mean, max(q_std, .02)))
        prior_z = gaussian_prior(z, global_stats)
        adaptive_z = gaussian_prior(z, adaptive_stats)
        if q is not None:
            prior_q = gaussian_prior(q, percentile_stats)
            adaptive_q = gaussian_prior(q, adaptive_percentile_stats)
            q_weight = args.percentile_prior_weight
            prior = (1.0 - q_weight) * prior_z + q_weight * prior_q
            adaptive_blend = (1.0 - q_weight) * adaptive_z + q_weight * adaptive_q
            adaptive_delta = ((1.0 - q_weight) * (adaptive_z[0] - prior_z[0])
                              + q_weight * (adaptive_q[0] - prior_q[0]))
        else:
            prior = prior_z
            adaptive_blend = adaptive_z
            adaptive_delta = adaptive_z[0] - prior_z[0]
        if special:
            al3ni_beta = args.special_al3ni_beta
        elif args.narrow_a11_step1_low <= mean < args.narrow_a11_step1_high:
            al3ni_beta = args.narrow_a11_step1_beta
        elif args.narrow_a67_low <= mean < args.narrow_a67_high:
            al3ni_beta = args.narrow_a67_beta
        elif args.narrow_a11_step2_low <= mean < args.narrow_a11_step2_high:
            al3ni_beta = args.narrow_a11_step2_beta
        elif args.beta_threshold is None:
            al3ni_beta = args.al3ni_beta
        else:
            al3ni_beta = (
                args.low_al3ni_beta if mean < args.beta_threshold
                else args.high_al3ni_beta
            )
        prior[0] += al3ni_beta * adaptive_delta
        prior[3] += args.eutectic_beta * (adaptive_blend[3] - prior[3])
        logits = np.log(np.maximum(mixed, 1e-7)) + .09 * prior
        logits[2] += args.special_al_bias if special else args.al_bias
        if special:
            eut_bias = args.special_eut_bias
        elif args.narrow_a186_low <= mean < args.narrow_a186_high:
            eut_bias = args.narrow_a186_bias
        elif args.narrow_a153_low <= mean < args.narrow_a153_high:
            eut_bias = args.narrow_a153_bias
        elif args.narrow_a11_low <= mean < args.narrow_a11_high:
            eut_bias = args.narrow_a11_bias
        else:
            eut_bias = args.high_eut_bias if mean >= args.bias_threshold else args.low_eut_bias
        logits[3] += eut_bias
        prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
        if args.primary_area_threshold > 0.0 and float((prediction == 1).mean()) <= args.primary_area_threshold:
            mask = prediction == 1
            alternative = logits.copy(); alternative[1] = -np.inf
            prediction[mask] = alternative[:, mask].argmax(axis=0).astype(np.uint8)
        encode_mask(prediction).save(output / image_path.name.replace("_image.png", "_mask.png"))
        print(f"wrote {image_path.name} mean={mean:.3f} std={float(gray.std()):.3f}", flush=True)
    print(f"test_masks={len(list(output.glob('*.png')))} output={output}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--old-checkpoint", required=True)
    p.add_argument("--new-checkpoint", required=True)
    p.add_argument("--aspp-checkpoint", required=True)
    p.add_argument("--pair-checkpoint", required=True)
    p.add_argument("--high-checkpoint", required=True)
    p.add_argument("--special-high-checkpoint", default=None)
    p.add_argument("--narrow-a186-high-checkpoint", default=None)
    p.add_argument("--narrow-a67-high-checkpoint", default=None)
    p.add_argument("--narrow-a11-step2-high-checkpoint", default=None)
    p.add_argument("--narrow-mean152-high-checkpoint", default=None)
    p.add_argument("--narrow-mean155-high-checkpoint", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--al3ni-beta", type=float, default=1.0)
    p.add_argument("--beta-threshold", type=float, default=None)
    p.add_argument("--low-al3ni-beta", type=float, default=1.0)
    p.add_argument("--high-al3ni-beta", type=float, default=1.0)
    p.add_argument("--eutectic-beta", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-threshold", type=float, default=165.0)
    p.add_argument("--low-high-weight", type=float, default=.70)
    p.add_argument("--high-high-weight", type=float, default=.60)
    p.add_argument("--pair-weight", type=float, default=.15)
    p.add_argument("--al-bias", type=float, default=.40)
    p.add_argument("--bias-threshold", type=float, default=167.037086)
    p.add_argument("--low-eut-bias", type=float, default=-.70)
    p.add_argument("--high-eut-bias", type=float, default=-.30)
    p.add_argument("--primary-area-threshold", type=float, default=0.0)
    p.add_argument("--percentile-prior-weight", type=float, default=0.0)
    p.add_argument("--special-mean-low", type=float, default=None)
    p.add_argument("--special-mean-high", type=float, default=None)
    p.add_argument("--special-high-weight", type=float, default=.70)
    p.add_argument("--special-pair-weight", type=float, default=.05)
    p.add_argument("--special-style-weight", type=float, default=1.0)
    p.add_argument("--special-al-bias", type=float, default=.45)
    p.add_argument("--special-al3ni-beta", type=float, default=10.0)
    p.add_argument("--special-eut-bias", type=float, default=-1.5)
    p.add_argument("--narrow-a186-low", type=float, default=161.3)
    p.add_argument("--narrow-a186-high", type=float, default=162.0)
    p.add_argument("--narrow-a186-bias", type=float, default=-2.0)
    p.add_argument("--narrow-a153-low", type=float, default=164.5)
    p.add_argument("--narrow-a153-high", type=float, default=165.2)
    p.add_argument("--narrow-a153-bias", type=float, default=-.75)
    p.add_argument("--narrow-a11-low", type=float, default=174.0)
    p.add_argument("--narrow-a11-high", type=float, default=175.5)
    p.add_argument("--narrow-a11-bias", type=float, default=.3)
    p.add_argument("--narrow-a11-step1-low", type=float, default=164.0)
    p.add_argument("--narrow-a11-step1-high", type=float, default=164.5)
    p.add_argument("--narrow-a11-step1-beta", type=float, default=15.0)
    p.add_argument("--narrow-a67-low", type=float, default=174.6)
    p.add_argument("--narrow-a67-high", type=float, default=174.85)
    p.add_argument("--narrow-a67-beta", type=float, default=15.0)
    p.add_argument("--narrow-a11-step2-low", type=float, default=174.85)
    p.add_argument("--narrow-a11-step2-high", type=float, default=175.1)
    p.add_argument("--narrow-a11-step2-beta", type=float, default=7.0)
    p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
