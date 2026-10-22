#!/usr/bin/env python3
"""Evaluate model109 ASPP as an additional member of the current ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import metrics, predict_probabilities
from encoder_structure_search import build_model
from hard_negative_finetune import load_model
from improve_five_causes import CLASS_NAMES, collect_split, decode_mask, encode_mask


def load_aspp(checkpoint: str, base_checkpoint: str, device: torch.device):
    model = build_model("aspp", pretrained=False, checkpoint=base_checkpoint).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def combine_class_conditional(probabilities, aspp_weight: float):
    # Existing class-conditional matrix for model98/model99/model101, scaled
    # down to make room for the new ASPP member. The added model is deliberately
    # uniform across classes to avoid tuning a high-dimensional matrix on 20
    # validation images.
    # Rows follow the actual checkpoint channel order:
    # [Al3Ni, Primary Si, Al, Eutectic Si].  These are the old canonical
    # rows reordered to the official mask palette.
    base = np.asarray([
        [0.75, 0.05, 0.20],  # Al3Ni
        [0.80, 0.05, 0.15],  # Primary Si
        [0.50, 0.40, 0.10],  # Al
        [0.65, 0.05, 0.30],  # Eutectic Si
    ], dtype=np.float32)
    result = np.zeros_like(probabilities[0], dtype=np.float32)
    for class_id in range(4):
        for index in range(3):
            result[class_id] += (1.0 - aspp_weight) * base[class_id, index] * probabilities[index][class_id]
        result[class_id] += aspp_weight * probabilities[3][class_id]
    return result / np.maximum(result.sum(axis=0, keepdims=True), 1e-7)


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
    samples = collect_split(args.data_dir, "valid")
    cached = []
    for image_path, mask_path in samples:
        values = []
        for model, (_, patch_size, stride, batch_size) in zip(models, specs):
            values.append(predict_probabilities(model, image_path, patch_size, stride, device, use_amp,
                                                batch_size, "d4", "gaussian"))
        cached.append((values, decode_mask(Image.open(mask_path).convert("RGB"))))
        print(f"cached {image_path.name}", flush=True)

    records = []
    bias_configs = ((0.0, 0.0, -0.20), (0.10, 0.20, -0.20), (0.0, 0.10, -0.30))
    for aspp_weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        for primary_bias, eutectic_bias, al_bias in bias_configs:
            predictions, targets = [], []
            bias = np.asarray([0.0, primary_bias, al_bias, eutectic_bias], dtype=np.float32)[:, None, None]
            for probabilities, target in cached:
                combined = combine_class_conditional(probabilities, aspp_weight)
                prediction = (np.log(np.maximum(combined, 1e-7)) + bias).argmax(axis=0).astype(np.uint8)
                predictions.append(prediction)
                targets.append(target)
            result = metrics(predictions, targets)
            records.append({"aspp_weight": aspp_weight, "primary_bias": primary_bias,
                            "eutectic_bias": eutectic_bias, "al_bias": al_bias, **result})
    best = max(records, key=lambda item: item["global_miou"])
    robust = min((item for item in records if item["aspp_weight"] in (0.10, 0.15, 0.20)),
                 key=lambda item: -item["global_miou"])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best": best, "robust_weight_band_best": robust,
                                  "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "robust_weight_band_best": robust}, ensure_ascii=False), flush=True)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        bias = np.asarray([0.0, best["primary_bias"], best["al_bias"], best["eutectic_bias"]], dtype=np.float32)[:, None, None]
        for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
            values = []
            for model, (_, patch_size, stride, batch_size) in zip(models, specs):
                values.append(predict_probabilities(model, image_path, patch_size, stride, device, use_amp,
                                                    batch_size, "d4", "gaussian"))
            combined = combine_class_conditional(values, best["aspp_weight"])
            prediction = (np.log(np.maximum(combined, 1e-7)) + bias).argmax(axis=0).astype(np.uint8)
            encode_mask(prediction).save(output_dir / image_path.name.replace("_image.png", "_mask.png"))
        print(f"wrote test_masks={len(list(output_dir.glob('*.png')))} dir={output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint98", required=True)
    parser.add_argument("--checkpoint99", required=True)
    parser.add_argument("--checkpoint101", required=True)
    parser.add_argument("--checkpoint109", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
