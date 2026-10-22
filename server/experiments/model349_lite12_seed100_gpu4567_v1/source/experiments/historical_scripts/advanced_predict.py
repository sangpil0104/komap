#!/usr/bin/env python3
"""Generate test masks for the best D4/Gaussian class-conditional ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import class_conditional, predict_probabilities
from hard_negative_finetune import load_model
from improve_five_causes import encode_mask


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    specs = [
        (args.checkpoint98, 512, 384, 4),
        (args.checkpoint99, 512, 384, 4),
        (args.checkpoint101, 640, 512, 2),
    ]
    models = [load_model(path, device) for path, _, _, _ in specs]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bias = np.asarray([0.0, args.primary_bias, args.al_bias, args.eutectic_bias], dtype=np.float32)[:, None, None]
    images = sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png"))
    for image_path in images:
        probabilities = []
        for model, (_, patch_size, stride, batch_size) in zip(models, specs):
            probabilities.append(predict_probabilities(model, image_path, patch_size, stride, device, use_amp,
                                                       batch_size, "d4", "gaussian"))
        combined = class_conditional(probabilities)
        prediction = (np.log(np.maximum(combined, 1e-7)) + bias).argmax(axis=0).astype(np.uint8)
        encode_mask(prediction).save(output / image_path.name.replace("_image.png", "_mask.png"))
    print(f"wrote {len(images)} masks to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint98", required=True)
    parser.add_argument("--checkpoint99", required=True)
    parser.add_argument("--checkpoint101", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-bias", type=float, default=0.0)
    parser.add_argument("--eutectic-bias", type=float, default=0.10)
    parser.add_argument("--al-bias", type=float, default=-0.20)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
