#!/usr/bin/env python3
"""Test the 640px Al3Ni-weighted fine-tune as a fourth ensemble member."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from calibrate_al import predict_probabilities
from hard_negative_finetune import load_model
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask
from restart_ensemble import metrics


COLORS = ((25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204))
BASE_MATRIX = np.asarray([
    [0.35, 0.40, 0.25], [0.25, 0.35, 0.40],
    [0.50, 0.30, 0.20], [0.25, 0.55, 0.20],
], dtype=np.float32)


def fit_stats(data_dir):
    values = [[] for _ in range(4)]
    for image_path, mask_path in collect_split(Path(data_dir), "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        mask = np.asarray(Image.open(mask_path).convert("RGB"))
        ids = np.full(mask.shape[:2], -1, dtype=np.int8)
        for class_id, color in enumerate(COLORS):
            ids[np.all(mask == color, axis=2)] = class_id
        rng = np.random.default_rng(sum(image_path.name.encode("utf-8")))
        for class_id in range(4):
            current = z[ids == class_id]
            if len(current):
                take = rng.choice(len(current), min(len(current), 10000), replace=False)
                values[class_id].append(current[take])
    return [(float(np.concatenate(x).mean()), max(float(np.concatenate(x).std()), 0.15)) for x in values]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    old = load_model(args.old_checkpoint, device)
    new = load_model(args.new_checkpoint, device)
    aspp = build_model("aspp", pretrained=False, checkpoint=args.aspp_base_checkpoint).to(device)
    aspp.load_state_dict(torch.load(args.aspp_checkpoint, map_location="cpu")["model"])
    aspp.eval()
    highres = load_model(args.highres_checkpoint, device)
    stats = fit_stats(args.data_dir)
    cached = []
    for image_path, mask_path in collect_split(args.data_dir, "valid"):
        values = [
            predict_probabilities(old, image_path, 640, 512, device, use_amp, 2, True),
            predict_probabilities(new, image_path, 512, 384, device, use_amp, 2, True),
            predict_probabilities(aspp, image_path, 512, 384, device, use_amp, 2, True),
            predict_probabilities(highres, image_path, 640, 512, device, use_amp, 2, True),
        ]
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        prior = np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                            for mean, std in stats], dtype=np.float32)
        cached.append((values, prior, decode_mask(Image.open(mask_path).convert("RGB"))))
        print(f"cached {image_path.name}", flush=True)

    records = []
    for highres_weight in (0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25):
        predictions, targets = [], []
        matrix = BASE_MATRIX * (1.0 - highres_weight)
        for values, prior, target in cached:
            mixed = np.asarray([
                sum(matrix[c, m] * values[m][c] for m in range(3))
                + highres_weight * values[3][c]
                for c in range(4)
            ], dtype=np.float32)
            mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
            logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
            logits[3] += 0.09 * (0.75 - 1.0) * prior[3]
            logits += np.asarray([0.0, 0.0, 0.20, -0.55], dtype=np.float32)[:, None, None]
            predictions.append(logits.argmax(axis=0).astype(np.uint8))
            targets.append(target)
        records.append({"highres_weight": highres_weight, **metrics(predictions, targets)})
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    best = max(records, key=lambda x: x["image_mean_miou"])
    best_image_compat = max(records, key=lambda x: x["image_miou_compat"])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best": best, "best_image_compat": best_image_compat, "records": records},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "best_image_compat": best_image_compat}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--aspp-checkpoint", required=True)
    parser.add_argument("--aspp-base-checkpoint", required=True)
    parser.add_argument("--highres-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
