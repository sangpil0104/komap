#!/usr/bin/env python3
"""Apply the validated Primary cleanup to the global-best probability mix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from calibrate_al import predict_probabilities
from encoder_structure_search import build_model
from hard_negative_finetune import load_model
from improve_five_causes import collect_split, decode_mask, encode_mask
from restart_ensemble import metrics


COLORS = ((25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204))
MATRIX = np.asarray([
    [0.55, 0.40, 0.05], [0.30, 0.45, 0.25],
    [0.40, 0.20, 0.40], [0.25, 0.70, 0.05],
], dtype=np.float32)


def load_aspp(checkpoint, base_checkpoint, device):
    model = build_model("aspp", pretrained=False, checkpoint=base_checkpoint).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    model.eval()
    return model


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


def cleanup(prediction, logits, threshold):
    if threshold == 0:
        return prediction.copy()
    labels, count = ndimage.label(prediction == 1, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    small = sizes < threshold
    small[0] = False
    locations = small[labels]
    result = prediction.copy()
    alternative = logits.copy()
    alternative[1] = -np.inf
    result[locations] = alternative[:, locations].argmax(axis=0).astype(np.uint8)
    return result


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    old = load_model(args.old_checkpoint, device)
    new = load_model(args.new_checkpoint, device)
    aspp = load_aspp(args.aspp_checkpoint, args.aspp_base_checkpoint, device)
    stats = fit_stats(args.data_dir)
    cached = []
    for image_path, mask_path in collect_split(args.data_dir, "valid"):
        values = [
            predict_probabilities(old, image_path, 640, 512, device, use_amp, 2, True),
            predict_probabilities(new, image_path, 512, 384, device, use_amp, 2, True),
            predict_probabilities(aspp, image_path, 512, 384, device, use_amp, 2, True),
        ]
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        prior = np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                            for mean, std in stats], dtype=np.float32)
        mixed = np.asarray([sum(MATRIX[c, m] * values[m][c] for m in range(3))
                            for c in range(4)], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        logits = np.log(np.maximum(mixed, 1e-7)) + 0.13 * prior
        logits[3] += 0.13 * (0.40 - 1.0) * prior[3]
        logits += np.asarray([0.0, 0.0, 0.70, -0.65], dtype=np.float32)[:, None, None]
        cached.append((logits, decode_mask(Image.open(mask_path).convert("RGB"))))
        print(f"cached {image_path.name}", flush=True)

    records = []
    for threshold in (0, 16, 32, 64, 96, 128, 160, 192, 256):
        predictions, targets = [], []
        for logits, target in cached:
            raw = logits.argmax(axis=0).astype(np.uint8)
            predictions.append(cleanup(raw, logits, threshold))
            targets.append(target)
        records.append({"primary_threshold": threshold, **metrics(predictions, targets)})
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    best = max(records, key=lambda x: x["image_miou_compat"])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
            values = [
                predict_probabilities(old, image_path, 640, 512, device, use_amp, 2, True),
                predict_probabilities(new, image_path, 512, 384, device, use_amp, 2, True),
                predict_probabilities(aspp, image_path, 512, 384, device, use_amp, 2, True),
            ]
            mixed = np.asarray([sum(MATRIX[c, m] * values[m][c] for m in range(3)) for c in range(4)], dtype=np.float32)
            mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
            gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
            z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
            prior = np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                                for mean, std in stats], dtype=np.float32)
            logits = np.log(np.maximum(mixed, 1e-7)) + 0.13 * prior
            logits[3] += 0.13 * (0.40 - 1.0) * prior[3]
            logits += np.asarray([0.0, 0.0, 0.70, -0.65], dtype=np.float32)[:, None, None]
            prediction = cleanup(logits.argmax(axis=0).astype(np.uint8), logits, best["primary_threshold"])
            encode_mask(prediction).save(output_dir / image_path.name.replace("_image.png", "_mask.png"))
        print(f"wrote test_masks={len(list(output_dir.glob('*.png')))} dir={output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--aspp-checkpoint", required=True)
    parser.add_argument("--aspp-base-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
