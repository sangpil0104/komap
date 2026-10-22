#!/usr/bin/env python3
"""Label-free AdaBN domain adaptation for the native ResNet34+ASPP model."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask, normalize_image, scaled_pair, tile_starts
from restart_ensemble import metrics


def locations_for(image_path, patch_size, stride):
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    scaled, _ = scaled_pair(image, np.zeros(image.shape[:2], dtype=np.int64), patch_size)
    height, width = scaled.shape[:2]
    ys, xs = tile_starts(height, patch_size, stride), tile_starts(width, patch_size, stride)
    padded_height, padded_width = max(height, ys[-1] + patch_size), max(width, xs[-1] + patch_size)
    padded = np.pad(scaled, ((0, padded_height - height), (0, padded_width - width), (0, 0)), mode="reflect")
    return [padded[top:top + patch_size, left:left + patch_size] for top in ys for left in xs]


@torch.no_grad()
def update_bn(model, image_paths, device, use_amp, patch_size=512, stride=384, batch_size=2):
    """Update running BN statistics using unlabeled image tiles only."""
    model.train()
    patches = []
    for image_path in image_paths:
        patches.extend(locations_for(image_path, patch_size, stride))
    for start in range(0, len(patches), batch_size):
        current = patches[start:start + batch_size]
        # ASPP's global-pooling branch contains BatchNorm; skip a final
        # singleton batch because it has no valid variance estimate.
        if len(current) < 2:
            continue
        batch = torch.stack([normalize_image(value) for value in current]).to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            model(batch)
    model.eval()


def evaluate(model, samples, device, use_amp, bias=-0.55):
    predictions, targets = [], []
    for image_path, mask_path in samples:
        probability = predict_probabilities(model, image_path, 512, 384, device, use_amp, 2, "d4", "gaussian")
        logits = np.log(np.maximum(probability, 1e-7))
        logits[3] += bias
        predictions.append(logits.argmax(axis=0).astype(np.uint8))
        targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
    return metrics(predictions, targets)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    samples = collect_split(args.data_dir, "valid")
    image_paths = [image_path for image_path, _ in samples]
    model = build_model("aspp", pretrained=False, checkpoint=args.base_checkpoint).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model.eval()
    original = copy.deepcopy(model.state_dict())

    baseline = evaluate(model, samples, device, use_amp, args.bias)
    print(json.dumps({"mode": "baseline", **baseline}, ensure_ascii=False), flush=True)

    model.load_state_dict(original)
    update_bn(model, image_paths, device, use_amp, args.patch_size, args.stride, args.batch_size)
    global_adapted = evaluate(model, samples, device, use_amp, args.bias)
    print(json.dumps({"mode": "global_adabn", **global_adapted}, ensure_ascii=False), flush=True)

    per_image_predictions, targets = [], []
    for image_path, mask_path in samples:
        model.load_state_dict(original)
        update_bn(model, [image_path], device, use_amp, args.patch_size, args.stride, args.batch_size)
        probability = predict_probabilities(model, image_path, 512, 384, device, use_amp, 2, "d4", "gaussian")
        logits = np.log(np.maximum(probability, 1e-7)); logits[3] += args.bias
        per_image_predictions.append(logits.argmax(axis=0).astype(np.uint8))
        targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
        print(f"per-image adapted {image_path.name}", flush=True)
    per_image = metrics(per_image_predictions, targets)
    print(json.dumps({"mode": "per_image_adabn", **per_image}, ensure_ascii=False), flush=True)

    output = Path(args.output_json); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"baseline": baseline, "global_adabn": global_adapted,
                                  "per_image_adabn": per_image}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True); parser.add_argument("--output-json", required=True)
    parser.add_argument("--patch-size", type=int, default=512); parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2); parser.add_argument("--bias", type=float, default=-0.55)
    parser.add_argument("--no-amp", action="store_true"); main(parser.parse_args())
