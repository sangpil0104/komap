#!/usr/bin/env python3
"""Compare denser native-tile overlap against the current inference protocol."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from calibrate_al import predict_probabilities
from encoder_structure_search import build_model
from hard_negative_finetune import load_model
from improve_five_causes import collect_split, decode_mask
from restart_ensemble import metrics


COLORS = ((25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204))
MATRIX_IMAGE = np.asarray([
    [0.35, 0.40, 0.25], [0.25, 0.35, 0.40],
    [0.50, 0.30, 0.20], [0.25, 0.55, 0.20],
], dtype=np.float32)
MATRIX_GLOBAL = np.asarray([
    [0.55, 0.40, 0.05], [0.30, 0.45, 0.25],
    [0.40, 0.20, 0.40], [0.25, 0.70, 0.05],
], dtype=np.float32)


def load_aspp(checkpoint, base_checkpoint, device):
    model = build_model("aspp", pretrained=False, checkpoint=base_checkpoint).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    old = load_model(args.old_checkpoint, device)
    new = load_model(args.new_checkpoint, device)
    aspp = load_aspp(args.aspp_checkpoint, args.aspp_base_checkpoint, device)
    stats = fit_stats(args.data_dir)
    outputs = {}
    samples = collect_split(args.data_dir, "valid")
    for label, old_stride, new_stride in (("current", 512, 384), ("dense_new", 512, 256), ("dense_all", 384, 256)):
        cached = []
        for image_path, mask_path in _local_progress(samples, 'dense_overlap_search'):
            values = [
                predict_probabilities(old, image_path, 640, old_stride, device, use_amp, 2, True),
                predict_probabilities(new, image_path, 512, new_stride, device, use_amp, 4, True),
                predict_probabilities(aspp, image_path, 512, new_stride, device, use_amp, 4, True),
            ]
            gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
            z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
            prior = np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                                for mean, std in stats], dtype=np.float32)
            cached.append((values, prior, decode_mask(Image.open(mask_path).convert("RGB"))))
            print(f"{label} cached {image_path.name}", flush=True)

        result_set = []
        for name, matrix, alpha, scale, al_bias, eut_bias in (
            ("image", MATRIX_IMAGE, 0.09, 0.75, 0.20, -0.55),
            ("global", MATRIX_GLOBAL, 0.13, 0.40, 0.70, -0.65),
        ):
            predictions, targets = [], []
            bias = np.asarray([0.0, 0.0, al_bias, eut_bias], dtype=np.float32)[:, None, None]
            for values, prior, target in cached:
                mixed = np.asarray([sum(matrix[c, m] * values[m][c] for m in range(3))
                                    for c in range(4)], dtype=np.float32)
                mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
                logits = np.log(np.maximum(mixed, 1e-7)) + alpha * prior
                logits[3] += alpha * (scale - 1.0) * prior[3]
                predictions.append((logits + bias).argmax(axis=0).astype(np.uint8))
                targets.append(target)
            result_set.append({"calibration": name, **metrics(predictions, targets)})
        outputs[label] = result_set
        print(json.dumps({label: result_set}, ensure_ascii=False), flush=True)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(outputs, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--old-checkpoint", required=True)
    parser.add_argument("--new-checkpoint", required=True)
    parser.add_argument("--aspp-checkpoint", required=True)
    parser.add_argument("--aspp-base-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
