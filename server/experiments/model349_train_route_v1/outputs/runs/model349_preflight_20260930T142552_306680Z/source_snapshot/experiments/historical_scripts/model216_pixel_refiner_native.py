#!/usr/bin/env python3
"""Train a train-only pixel residual calibrator on the native representation branch.

The segmentation network is frozen.  A small residual head learns how the
model-195 logits should move using only features available at test time:
class probabilities, grayscale representation, local contrast, gradient,
entropy, and normalized coordinates.  The zero-initialized head preserves the
teacher at alpha=0 and is evaluated with a conservative alpha sweep.
"""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import sobel, uniform_filter

from improve_five_causes import collect_split, decode_mask
from model181_raw_zcdf_aspp import TeamRepresentationASPP, predict_probabilities, load_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def features(image_path: Path, probabilities: np.ndarray) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    local_mean = uniform_filter(image, size=15, mode="reflect")
    local_sq = uniform_filter(image * image, size=15, mode="reflect")
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 1e-6))
    gradient = np.hypot(sobel(image, axis=0, mode="reflect"),
                        sobel(image, axis=1, mode="reflect"))
    gradient /= max(float(np.percentile(gradient, 99.0)), 1e-6)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-7))).sum(axis=0)
    height, width = image.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    xx = xx / max(width - 1, 1) - 0.5
    yy = yy / max(height - 1, 1) - 0.5
    values = np.concatenate([
        np.log(np.maximum(probabilities, 1e-7)),
        image[None], local_mean[None], local_std[None], gradient[None], entropy[None],
        xx[None], yy[None],
    ], axis=0)
    return values.transpose(1, 2, 0).reshape(-1, values.shape[0]).astype(np.float32)


class PixelResidual(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 96), nn.GELU(), nn.LayerNorm(96),
            nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 4),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)


@torch.no_grad()
def collect_probabilities(model, samples, args, device, amp):
    values = []
    for image_path, mask_path in _local_progress(samples, 'model216_pixel_refiner_native'):
        probability = predict_probabilities(
            model, image_path, args.patch_size, args.stride, device, amp, args.tile_batch_size)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        values.append((image_path, probability.astype(np.float32), target))
        print(f"cached {image_path.name}", flush=True)
    return values


def sample_training(cached, args):
    rng = np.random.default_rng(args.seed)
    xs, bases, ys = [], [], []
    for image_path, probability, target in cached:
        x = features(image_path, probability)
        base = np.log(np.maximum(probability, 1e-7)).transpose(1, 2, 0).reshape(-1, 4).astype(np.float32)
        labels = target.reshape(-1)
        selected = []
        for class_id in range(4):
            indices = np.flatnonzero(labels == class_id)
            if len(indices):
                selected.append(rng.choice(indices, min(args.samples_per_class, len(indices)), replace=False))
        indices = np.concatenate(selected)
        rng.shuffle(indices)
        xs.append(x[indices]); bases.append(base[indices]); ys.append(labels[indices])
    return np.concatenate(xs), np.concatenate(bases), np.concatenate(ys)


def train_refiner(x, base, y, args, device):
    model = PixelResidual(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_tensor, base_tensor = torch.from_numpy(x), torch.from_numpy(base)
    y_tensor = torch.from_numpy(y).long()
    weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    rng = np.random.default_rng(args.seed)
    for epoch in _local_progress(range(args.epochs), 'model216_pixel_refiner_native'):
        order = rng.permutation(len(y)); total = 0.0; count = 0
        model.train()
        for start in range(0, len(order), args.batch_size):
            index = torch.from_numpy(order[start:start + args.batch_size]).long()
            xb = x_tensor[index].to(device); bb = base_tensor[index].to(device)
            yb = y_tensor[index].to(device)
            delta = model(xb)
            loss = F.cross_entropy(bb + delta, yb, weight=weights)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.item()) * len(index); count += len(index)
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"refiner epoch={epoch + 1} loss={total / max(count, 1):.5f}", flush=True)
    return model


@torch.no_grad()
def apply_delta(model, image_path, probability, args, device):
    x = torch.from_numpy(features(image_path, probability)).to(device)
    values = []
    model.eval()
    for start in range(0, len(x), args.infer_batch_size):
        values.append(model(x[start:start + args.infer_batch_size]).cpu().numpy())
    return np.concatenate(values).reshape(probability.shape[1], probability.shape[2], 4).transpose(2, 0, 1)


def metrics(predictions, targets):
    from local_image_metrics import summary
    return summary(predictions, targets, CLASS_NAMES)


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    train_samples, valid_samples = collect_split(args.data_dir, "train"), collect_split(args.data_dir, "valid")
    model = TeamRepresentationASPP(pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint); model.eval()
    started = time.time()
    train_cached = collect_probabilities(model, train_samples, args, device, amp)
    valid_cached = collect_probabilities(model, valid_samples, args, device, amp)
    x, base, y = sample_training(train_cached, args)
    print(f"train features={x.shape} labels={y.shape}", flush=True)
    refiner = train_refiner(x, base, y, args, device)
    input_dim = int(x.shape[1])
    del x, base, y, train_cached
    valid_deltas = [(path, probability, target, apply_delta(refiner, path, probability, args, device))
                    for path, probability, target in valid_cached]
    records = []
    for alpha in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0):
        predictions = []
        targets = []
        for _, probability, target, delta in valid_deltas:
            logits = np.log(np.maximum(probability, 1e-7)) + alpha * delta
            predictions.append(logits.argmax(axis=0).astype(np.uint8)); targets.append(target)
        record = {"alpha": alpha, **metrics(predictions, targets)}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    best = max(records, key=lambda row: row[args.selection_metric])
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": refiner.state_dict(), "input_dim": input_dim},
               output / "refiner.pt")
    payload = {}
    for path, probability, _, delta in valid_deltas:
        value = np.exp(np.log(np.maximum(probability, 1e-7)) + best["alpha"] * delta)
        value /= np.maximum(value.sum(axis=0, keepdims=True), 1e-7)
        payload[path.stem] = value.astype(np.float16)
    np.savez_compressed(output / "valid_probs.npz", **payload)
    summary = {"best": best, "records": records, "checkpoint": args.checkpoint,
               "features": input_dim, "elapsed_seconds": time.time() - started,
               "selection_metric": args.selection_metric}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--stride", type=int, default=576)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=1800)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--infer-batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weights", type=float, nargs=4, default=[1.20, 1.10, 0.70, 1.80])
    parser.add_argument("--selection-metric", choices=("image_mean_miou", "image_mean_miou_exclude"),
                        default="image_mean_miou")
    parser.add_argument("--seed", type=int, default=216)
    main(parser.parse_args())
