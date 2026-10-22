#!/usr/bin/env python3
"""Generate the conservative model117 ASPP-weight=0.20 test submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from advanced_inference import predict_probabilities
from ensemble_with_aspp import combine_class_conditional, load_aspp
from hard_negative_finetune import load_model
from improve_five_causes import encode_mask


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    specs = [
        (args.checkpoint98, 512, 384, 4),
        (args.checkpoint99, 512, 384, 4),
        (args.checkpoint101, 640, 512, 2),
        (args.checkpoint109, 512, 384, 4),
    ]
    models = [load_model(args.checkpoint98, device),
              load_model(args.checkpoint99, device),
              load_model(args.checkpoint101, device),
              load_aspp(args.checkpoint109, args.checkpoint98, device)]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bias = np.asarray([0.0, 0.0, -0.30, 0.10], dtype=np.float32)[:, None, None]
    images = sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png"))
    for image_path in images:
        values = []
        for model, (_, patch_size, stride, batch_size) in zip(models, specs):
            values.append(predict_probabilities(model, image_path, patch_size, stride, device, use_amp,
                                                batch_size, "d4", "gaussian"))
        combined = combine_class_conditional(values, 0.20)
        prediction = (np.log(np.maximum(combined, 1e-7)) + bias).argmax(axis=0).astype(np.uint8)
        encode_mask(prediction).save(output / image_path.name.replace("_image.png", "_mask.png"))
    print(f"wrote test_masks={len(images)} dir={output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint98", required=True)
    parser.add_argument("--checkpoint99", required=True)
    parser.add_argument("--checkpoint101", required=True)
    parser.add_argument("--checkpoint109", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
