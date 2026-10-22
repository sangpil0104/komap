#!/usr/bin/env python3
"""Label-free TENT-style entropy adaptation for the high-resolution branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from aspp768_addition import MATRIX, cleanup_primary
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask, normalize_image, scaled_pair, tile_starts
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image


def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"]); model.eval()
    return model


def adapt_model(checkpoint, raw, device, use_amp, steps, lr):
    model = load_aspp(checkpoint, device)
    if steps <= 0:
        return model
    scaled, _ = scaled_pair(raw, np.zeros(raw.shape[:2], dtype=np.int64), 768)
    height, width = scaled.shape[:2]
    ys, xs = tile_starts(height, 768, 576), tile_starts(width, 768, 576)
    locations = [(ys[0], xs[0]), (ys[0], xs[-1]), (ys[-1], xs[0]), (ys[-1], xs[-1])]
    locations = list(dict.fromkeys(locations))
    inputs = torch.stack([normalize_image(scaled[top:top + 768, left:left + 768])
                          for top, left in locations]).to(device)
    params = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.train()
            if module.weight is not None:
                module.weight.requires_grad_(True); params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True); params.append(module.bias)
    # The ASPP image-pooling BN receives a 1x1 map; keep its source statistics.
    model.context.pool[2].eval()
    param_ids = {id(parameter) for parameter in params}
    for parameter in model.parameters():
        if id(parameter) not in param_ids:
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            probabilities = torch.softmax(model(inputs), dim=1)
            entropy = -(probabilities * torch.log(torch.clamp(probabilities, min=1e-7))).sum(1).mean()
        entropy.backward(); optimizer.step()
    model.eval()
    return model


def load_entries(data_dir, cache_dir):
    entries, samples = [], collect_split(data_dir, "valid")
    for image_path, mask_path in samples:
        loaded = np.load(Path(cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        entries.append({"values": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
                        "high_raw": loaded["p3"].astype(np.float32),
                        "prior": loaded["prior"].astype(np.float32),
                        "target": decode_mask(Image.open(mask_path).convert("RGB")),
                        "raw": raw, "std": float(raw[..., 0].astype(np.float32).std()),
                        "name": image_path.name})
    return entries


def evaluate(entries, raw_values, style_values, steps, threshold):
    predictions, targets = [], []
    bias = np.asarray([0.0, 0.0, 0.40, -0.50], dtype=np.float32)[:, None, None]
    for entry, raw_high, style_high in zip(entries, raw_values, style_values):
        high = 0.25 * raw_high + 0.75 * style_high if entry["std"] >= threshold else raw_high
        base = MATRIX * 0.40
        mixed = np.asarray([
            sum(base[c, m] * entry["values"][m][c] for m in range(3)) + 0.60 * high[c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * entry["prior"] + bias
        predictions.append(cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128))
        targets.append(entry["target"])
    return metrics(predictions, targets)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    entries = load_entries(args.data_dir, args.cache_dir)
    train_stats = []
    for image_path, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean, train_std = float(np.mean([x[0] for x in train_stats])), float(np.mean([x[1] for x in train_stats]))
    records = []
    for steps in (0, 1, 2):
        raw_values, style_values = [], []
        for entry in entries:
            if steps == 0:
                raw_values.append(entry["high_raw"]); style_values.append(entry["high_raw"])
            else:
                model = adapt_model(args.checkpoint, entry["raw"], device, use_amp, steps, args.lr)
                raw_values.append(predict_array(model, entry["raw"], 768, 576, device, use_amp, 1, "d4", "gaussian"))
                style_values.append(predict_array(model, transform_image(entry["raw"], "standardize", train_mean, train_std),
                                                  768, 576, device, use_amp, 1, "d4", "gaussian"))
            print(f"adapted steps={steps} image={entry['name']}", flush=True)
        for threshold in (45.0, 50.0, 55.0, 60.0, 65.0):
            value = evaluate(entries, raw_values, style_values, steps, threshold)
            row = {"steps": steps, "lr": args.lr, "threshold": threshold, **value}
            records.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--output-json", required=True)
    p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
