#!/usr/bin/env python3
"""Per-image source-free pseudo-label adaptation for the high ASPP branch."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from advanced_inference import predict_probabilities
from aspp768_addition import MATRIX, cleanup_primary
from encoder_structure_search import build_model
from improve_five_causes import collect_split, decode_mask, normalize_image, scaled_pair, tile_starts
from restart_ensemble import metrics
from style_tta_search import predict_array, transform_image
from tent_style_search import adapt_model


def load_aspp(path: str, device: torch.device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    model.eval()
    return model


def train_stats(data_dir):
    values = []
    for p, _ in collect_split(data_dir, "train"):
        gray = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        values.append((float(gray.mean()), float(gray.std())))
    return float(np.mean([x[0] for x in values])), float(np.mean([x[1] for x in values]))


def route_logits(entry, high, beta=7.0):
    gray = entry["gray"]
    mean = float(gray.mean())
    high_weight = .75 if mean < 165.0 else .55
    if float(gray.std()) >= 50.0:
        high = .25 * high[0] + .75 * high[1]
    else:
        high = high[0]
    base = MATRIX * (1.0 - high_weight - .15)
    mixed = np.asarray([
        sum(base[c, m] * entry["sources"][m][c] for m in range(3))
        + high_weight * high[c] + .15 * entry["pair"][c]
        for c in range(4)
    ], dtype=np.float32)
    mixed /= np.maximum(mixed.sum(0, keepdims=True), 1e-7)
    z = (gray - mean) / max(float(gray.std()), 1e-5)
    stats = []
    for cid in range(4):
        weights = np.maximum(mixed[cid], 1e-7) ** 2.0
        wm = float((weights * z).sum() / weights.sum())
        ws = float(np.sqrt((weights * (z - wm) ** 2).sum() / weights.sum()))
        stats.append((wm, max(ws, .15)))
    adaptive = np.asarray([-0.5 * ((z - m) / s) ** 2 - np.log(s) for m, s in stats], dtype=np.float32)
    prior = entry["prior"].copy()
    prior[0] += beta * (adaptive[0] - prior[0])
    logits = np.log(np.maximum(mixed, 1e-7)) + .09 * prior
    logits[2] += .40
    logits[3] += -1.0 if mean < 165.0 else -.30
    return logits, mixed


def pseudo_adapt(model, raw, teacher_prob, device, amp, steps, lr, confidence, mode):
    scaled, _ = scaled_pair(raw, np.zeros(raw.shape[:2], dtype=np.int64), 768)
    height, width = scaled.shape[:2]
    ys, xs = tile_starts(height, 768, 576), tile_starts(width, 768, 576)
    padded_height, padded_width = max(height, ys[-1] + 768), max(width, xs[-1] + 768)
    padded = np.pad(scaled, ((0, padded_height - height), (0, padded_width - width), (0, 0)), mode="reflect")
    teacher_labels = teacher_prob.argmax(axis=0).astype(np.uint8)
    teacher_conf = teacher_prob.max(axis=0).astype(np.float32)
    labels = np.asarray(Image.fromarray(teacher_labels).resize((width, height), Image.Resampling.NEAREST))
    conf = np.asarray(Image.fromarray(teacher_conf).resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    labels = np.pad(labels, ((0, padded_height - height), (0, padded_width - width)), mode="reflect")
    conf = np.pad(conf, ((0, padded_height - height), (0, padded_width - width)), mode="reflect")
    locations = [(ys[0], xs[0]), (ys[0], xs[-1]), (ys[-1], xs[0]), (ys[-1], xs[-1])]
    locations = list(dict.fromkeys(locations))
    inputs = torch.stack([
        normalize_image(padded[top:top + 768, left:left + 768]) for top, left in locations
    ]).to(device)
    targets = torch.from_numpy(np.stack([
        labels[top:top + 768, left:left + 768] for top, left in locations
    ]).astype(np.int64)).to(device)
    masks = torch.from_numpy(np.stack([
        conf[top:top + 768, left:left + 768] >= confidence for top, left in locations
    ])).to(device)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    if mode == "decoder":
        params = [p for name, p in model.named_parameters()
                  if name.startswith(("dec", "context", "final", "head"))]
        if not params:
            params = list(model.parameters())
    else:
        params = list(model.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            logits = model(inputs)
            loss_map = F.cross_entropy(logits, targets, reduction="none")
            loss = loss_map[masks].mean() if masks.any() else loss_map.mean()
        loss.backward()
        optimizer.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return float(loss.detach().cpu())


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    data_dir = Path(args.data_dir)
    train_mean, train_std = train_stats(data_dir)
    pair = load_aspp(args.pair_checkpoint, device)
    entries = []
    for image_path, mask_path in collect_split(data_dir, "valid"):
        loaded = np.load(Path(args.cache_dir) / f"{image_path.stem}.npz")
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        gray = raw[..., 0].astype(np.float32)
        pair_value = predict_probabilities(pair, image_path, 768, 576, device, amp, 1, "d4", "gaussian")
        entries.append({"path": image_path, "raw": raw, "gray": gray,
                        "sources": [loaded[f"p{i}"].astype(np.float32) for i in range(3)],
                        "cached_high": loaded["p3"].astype(np.float32),
                        "prior": loaded["prior"].astype(np.float32), "pair": pair_value,
                        "target": decode_mask(Image.open(mask_path).convert("RGB")), "name": image_path.name})
        print(f"cached {image_path.name}", flush=True)

    records = []
    for confidence in args.confidences:
        for mode in args.modes:
            for steps in args.steps:
                predictions, targets, losses = [], [], []
                for entry in entries:
                    model = load_aspp(args.high_checkpoint, device)
                    initial = entry["cached_high"]
                    teacher_logits, _ = route_logits(entry, (initial, initial), beta=7.0)
                    teacher_prob = np.exp(teacher_logits - teacher_logits.max(0, keepdims=True))
                    teacher_prob /= np.maximum(teacher_prob.sum(0, keepdims=True), 1e-7)
                    loss = pseudo_adapt(model, entry["raw"], teacher_prob, device, amp,
                                        steps, args.lr, confidence, mode)
                    raw_high = predict_array(model, entry["raw"], 768, 576, device, amp, 1, "d4", "gaussian")
                    style_high = predict_array(
                        model, transform_image(entry["raw"], "standardize", train_mean, train_std),
                        768, 576, device, amp, 1, "d4", "gaussian",
                    )
                    logits, _ = route_logits(entry, (raw_high, style_high), beta=7.0)
                    prediction = cleanup_primary(logits.argmax(0).astype(np.uint8), logits, 128)
                    predictions.append(prediction); targets.append(entry["target"]); losses.append(loss)
                    del model, raw_high, style_high
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                value = metrics(predictions, targets)
                row = {"confidence": confidence, "mode": mode, "steps": steps, "lr": args.lr,
                       "mean_pseudo_loss": float(np.mean(losses)), **value}
                records.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)

    best = max(records, key=lambda x: x["image_mean_miou"])
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--cache-dir", required=True)
    p.add_argument("--pair-checkpoint", required=True); p.add_argument("--high-checkpoint", required=True)
    p.add_argument("--output-json", required=True); p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--confidences", type=float, nargs="+", default=[.75, .85, .90])
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2])
    p.add_argument("--modes", nargs="+", default=["decoder", "all"])
    main(p.parse_args())
