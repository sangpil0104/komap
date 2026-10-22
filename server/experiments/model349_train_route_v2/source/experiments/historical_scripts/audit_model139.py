#!/usr/bin/env python3
"""Audit per-image/class IoU for model 139's validation protocol."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from advanced_inference import predict_probabilities
from aspp768_addition import MATRIX, cleanup_primary
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask, encode_mask
from style_tta_search import predict_array, transform_image
from tent_style_search import adapt_model


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    samples = collect_split(args.data_dir, "valid")
    train_stats = []
    for p, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))
    pair = load_aspp(args.pair_checkpoint, device)
    results = []
    out_dir = Path(args.output_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    for image_path, mask_path in _local_progress(samples, 'audit_model139'):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        p0 = loaded["p0"].astype(np.float32)
        p1 = loaded["p1"].astype(np.float32)
        p2 = loaded["p2"].astype(np.float32)
        pp = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.high_checkpoint, raw, device, amp, 3, 2e-5)
        raw_high = predict_array(adapted, raw, 768, 576, device, amp, 1, "d4", "gaussian")
        style = transform_image(raw, "standardize", train_mean, train_std)
        style_high = predict_array(adapted, style, 768, 576, device, amp, 1, "d4", "gaussian")
        high = 0.25 * raw_high + 0.75 * style_high if raw[..., 0].astype(np.float32).std() >= 50 else raw_high
        mean = float(raw[..., 0].mean())
        high_weight = 0.60 if mean >= 165 else 0.70
        pair_weight = 0.15
        base = MATRIX * (1.0 - high_weight - pair_weight)
        mixed = np.asarray([
            sum(base[c, m] * [p0, p1, p2][m][c] for m in range(3))
            + high_weight * high[c] + pair_weight * pp[c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        gray = raw[..., 0].astype(np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        stats = [
            (float(np.mean(z)), 0.15),
        ]
        # Reuse the cache's fitted intensity prior by deriving it from the
        # original cached values' image geometry is not possible; fit class
        # statistics once from the train masks below.
        logits = np.log(np.maximum(mixed, 1e-7))
        # The current cache already includes prior logits, so recover the
        # calibrated prior-free logits only by using the same prior fit in the
        # companion implementation below.
        # This branch is replaced immediately after class statistics are fit.
        del stats, logits
        results.append((image_path, mask_path, raw, mixed))

    # Fit the same standardized grayscale class prior used by model 139.
    colors = ((25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204))
    values = [[] for _ in range(4)]
    for p, mp in collect_split(args.data_dir, "train"):
        image = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        z = (image - image.mean()) / max(float(image.std()), 1e-5)
        mask = np.asarray(Image.open(mp).convert("RGB"))
        ids = np.full(mask.shape[:2], -1, dtype=np.int8)
        for cid, color in enumerate(colors):
            ids[np.all(mask == color, axis=2)] = cid
        rng = np.random.default_rng(sum(p.name.encode("utf-8")))
        for cid in range(4):
            current = z[ids == cid]
            if len(current):
                take = rng.choice(len(current), min(len(current), 10000), replace=False)
                values[cid].append(current[take])
    class_stats = [(float(np.concatenate(v).mean()), max(float(np.concatenate(v).std()), .15)) for v in values]
    records = []
    for image_path, mask_path, raw, mixed in results:
        gray = raw[..., 0].astype(np.float32)
        z = (gray - gray.mean()) / max(float(gray.std()), 1e-5)
        prior = np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                            for mean, std in class_stats], dtype=np.float32)
        logits = np.log(np.maximum(mixed, 1e-7)) + .09 * prior
        logits[2] += .40
        logits[3] += -.30 if float(gray.mean()) >= 167.037086 else -.70
        prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        cm = np.bincount(target.reshape(-1) * 4 + prediction.reshape(-1), minlength=16).reshape(4, 4)
        inter = np.diag(cm).astype(np.float64)
        union = cm.sum(0) + cm.sum(1) - inter
        iou = inter / np.maximum(union, 1.0)
        stem = image_path.name.replace("_image.png", "")
        encode_mask(prediction).save(out_dir / "predictions" / f"{stem}_mask.png")
        records.append({"name": image_path.name, "image_mean_miou": float(iou.mean()),
                        "iou": iou.tolist(), "target_counts": np.bincount(target.reshape(-1), minlength=4).tolist(),
                        "prediction_counts": np.bincount(prediction.reshape(-1), minlength=4).tolist(),
                        "confusion": cm.tolist()})
    records.sort(key=lambda x: x["image_mean_miou"])
    (out_dir / "audit.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in records:
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--high-checkpoint", required=True); p.add_argument("--pair-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    main(p.parse_args())
