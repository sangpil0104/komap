#!/usr/bin/env python3
"""Add scSE/Lovasz/EMA refinements to the current robust ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import metrics, predict_probabilities
from encoder_structure_search import build_model as build_structure_model
from ensemble_with_aspp import combine_class_conditional
from hard_negative_finetune import load_model
from improve_five_causes import collect_split, decode_mask, encode_mask
from resnet34_refinements import build_model as build_refined_model, load_compatible


def load_aspp(checkpoint, base_checkpoint, device):
    model = build_structure_model("aspp", pretrained=False, checkpoint=base_checkpoint).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def load_refined(candidate, checkpoint, base_checkpoint, device):
    model = build_refined_model(candidate)
    load_compatible(model, base_checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def combine_with_new(base_values, new_values, new_weight):
    base = combine_class_conditional(base_values, 0.20)
    if not new_values:
        return base
    new = sum(new_values) / float(len(new_values))
    result = (1.0 - new_weight) * base + new_weight * new
    return result / np.maximum(result.sum(axis=0, keepdims=True), 1e-7)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    base_specs = [
        (args.checkpoint98, 512, 384, 4),
        (args.checkpoint99, 512, 384, 4),
        (args.checkpoint101, 640, 512, 2),
        (args.checkpoint109, 512, 384, 4),
    ]
    base_models = [load_model(args.checkpoint98, device), load_model(args.checkpoint99, device),
                   load_model(args.checkpoint101, device), load_aspp(args.checkpoint109, args.checkpoint98, device)]
    refined_specs = [
        ("scse", args.checkpoint119), ("lovasz", args.checkpoint122), ("ema", args.checkpoint123)
    ]
    refined_models = [load_refined(name, checkpoint, args.checkpoint109, device)
                      for name, checkpoint in refined_specs]
    samples = collect_split(args.data_dir, "valid")
    cached = []
    for image_path, mask_path in samples:
        base_values = [predict_probabilities(model, image_path, patch_size, stride, device, use_amp,
                                              batch_size, "d4", "gaussian")
                       for model, (_, patch_size, stride, batch_size) in zip(base_models, base_specs)]
        new_values = [predict_probabilities(model, image_path, 512, 384, device, use_amp, 4, "d4", "gaussian")
                      for model in refined_models]
        cached.append((base_values, new_values, decode_mask(Image.open(mask_path).convert("RGB"))))
        print(f"cached {image_path.name}", flush=True)

    candidates = [("base", [])]
    for index, (name, _) in enumerate(refined_specs):
        candidates.append((name, [index]))
    candidates.append(("lovasz_scse_ema", [0, 1, 2]))
    records = []
    # Official semantic bias (Primary=0, Al3Ni=0, Eutectic=+0.10, Al=-0.30)
    # converted to checkpoint channel order [Al3Ni, Primary, Al, Eutectic].
    bias = np.asarray([0.0, 0.0, -0.30, 0.10], dtype=np.float32)[:, None, None]
    for name, indexes in candidates:
        for weight in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
            predictions, targets = [], []
            for base_values, new_values, target in cached:
                chosen = [new_values[index] for index in indexes]
                combined = combine_with_new(base_values, chosen, weight if indexes else 0.0)
                predictions.append((np.log(np.maximum(combined, 1e-7)) + bias).argmax(0).astype(np.uint8))
                targets.append(target)
            result = metrics(predictions, targets)
            records.append({"candidate": name, "new_weight": weight, **result})
    best = max(records, key=lambda item: item["global_miou"])
    output = Path(args.output_json); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)

    if args.output_dir:
        output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        selected = [new_values[index] for index in next(item[1] for item in candidates if item[0] == best["candidate"])]
        for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
            base_values = [predict_probabilities(model, image_path, patch_size, stride, device, use_amp,
                                                  batch_size, "d4", "gaussian")
                           for model, (_, patch_size, stride, batch_size) in zip(base_models, base_specs)]
            new_values = [predict_probabilities(model, image_path, 512, 384, device, use_amp, 4, "d4", "gaussian")
                          for model in refined_models]
            indexes = next(item[1] for item in candidates if item[0] == best["candidate"])
            combined = combine_with_new(base_values, [new_values[index] for index in indexes], best["new_weight"])
            encode_mask((np.log(np.maximum(combined, 1e-7)) + bias).argmax(0).astype(np.uint8)).save(
                output_dir / image_path.name.replace("_image.png", "_mask.png"))
        print(f"wrote test_masks={len(list(output_dir.glob('*.png')))} dir={output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint98", required=True)
    parser.add_argument("--checkpoint99", required=True)
    parser.add_argument("--checkpoint101", required=True)
    parser.add_argument("--checkpoint109", required=True)
    parser.add_argument("--checkpoint119", required=True)
    parser.add_argument("--checkpoint122", required=True)
    parser.add_argument("--checkpoint123", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
