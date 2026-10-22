#!/usr/bin/env python3
"""Generate test masks for the restart52 768-context ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aspp768_addition import BIASES, MATRIX, cleanup_primary, fit_stats, make_logits
from calibrate_al import predict_probabilities
from encoder_structure_search import build_model
from hard_negative_finetune import load_model
from improve_five_causes import encode_mask


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    old = load_model(args.old_checkpoint, device)
    new = load_model(args.new_checkpoint, device)
    aspp = build_model("aspp", pretrained=False, checkpoint=args.aspp_base_checkpoint).to(device)
    aspp.load_state_dict(torch.load(args.aspp_checkpoint, map_location="cpu")["model"]); aspp.eval()
    high = build_model("aspp", pretrained=False, checkpoint=args.high_base_checkpoint).to(device)
    high.load_state_dict(torch.load(args.high_checkpoint, map_location="cpu")["model"]); high.eval()
    stats = fit_stats(args.data_dir)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        values = [
            predict_probabilities(old, image_path, 640, 512, device, use_amp, 2, True),
            predict_probabilities(new, image_path, 512, 384, device, use_amp, 2, True),
            predict_probabilities(aspp, image_path, 512, 384, device, use_amp, 2, True),
            predict_probabilities(high, image_path, 768, 576, device, use_amp, 1, True),
        ]
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        prior = np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                            for mean, std in stats], dtype=np.float32)
        logits = make_logits(values, prior, args.highres_weight)
        desired_bias = np.asarray([args.al3ni_bias, args.primary_bias, args.al_bias, args.eutectic_bias], dtype=np.float32)
        logits += (args.prior_alpha - 0.09) * prior + (desired_bias - BIASES)[:, None, None]
        prediction = cleanup_primary(logits.argmax(axis=0).astype(np.uint8), logits)
        encode_mask(prediction).save(output / image_path.name.replace("_image.png", "_mask.png"))
        print(f"wrote {image_path.name}", flush=True)
    print(f"test_masks={len(list(output.glob('*.png')))} output={output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--old-checkpoint", required=True); parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--aspp-checkpoint", required=True); parser.add_argument("--aspp-base-checkpoint", required=True)
    parser.add_argument("--high-checkpoint", required=True); parser.add_argument("--high-base-checkpoint", required=True)
    parser.add_argument("--highres-weight", type=float, default=0.6); parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--prior-alpha", type=float, default=0.09)
    parser.add_argument("--al3ni-bias", type=float, default=0.0)
    parser.add_argument("--primary-bias", type=float, default=0.0)
    parser.add_argument("--al-bias", type=float, default=0.20)
    parser.add_argument("--eutectic-bias", type=float, default=-0.55)
    main(parser.parse_args())
