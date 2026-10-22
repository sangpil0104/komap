#!/usr/bin/env python3
"""Fine-tune the 768px ASPP U-Net++ with a small boundary auxiliary loss."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from advanced_inference import predict_probabilities
from encoder_structure_search import build_model, loss_fn as base_loss
from improve_five_causes import NUM_CLASSES, RareAwarePatchDataset, collect_split, decode_mask, encode_mask
from restart_ensemble import metrics


def boundary_map(target):
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    dilated = F.max_pool2d(one_hot, 3, 1, 1)
    eroded = -F.max_pool2d(-one_hot, 3, 1, 1)
    return (dilated - eroded).amax(1).clamp(0.0, 1.0)


def pair_boundary_map(target):
    pair = ((target == 0) | (target == 3)).float()
    c0 = (target == 0).float()
    c3 = (target == 3).float()
    cross = ((F.max_pool2d(c0[:, None], 3, 1, 1) > 0) &
             (F.max_pool2d(c3[:, None], 3, 1, 1) > 0)).squeeze(1).float()
    return cross * pair


def loss_fn(logits, target, boundary_weight):
    normal = base_loss(logits, target)
    per_pixel = F.cross_entropy(
        logits, target, weight=torch.tensor([1.15, 1.45, .70, 2.00], device=logits.device),
        reduction="none")
    boundary = boundary_map(target)
    boundary_ce = (per_pixel * boundary).sum() / boundary.sum().clamp_min(1.0)
    pair = pair_boundary_map(target)
    pair_ce = (per_pixel * pair).sum() / pair.sum().clamp_min(1.0)
    return normal + float(boundary_weight) * boundary_ce + .05 * pair_ce


def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


@torch.no_grad()
def evaluate(model, samples, args, device, amp):
    model.eval()
    predictions, targets = [], []
    for image_path, mask_path in _local_progress(samples, 'aspp_boundary_finetune'):
        value = predict_probabilities(model, image_path, args.patch_size, args.val_stride,
                                      device, amp, args.tile_batch_size, "d4", "gaussian")
        predictions.append(np.log(np.maximum(value, 1e-7)).argmax(0).astype(np.uint8))
        targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
    return metrics(predictions, targets)


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    train = collect_split(args.data_dir, "train")
    valid = collect_split(args.data_dir, "valid")
    dataset = RareAwarePatchDataset(train, args.patch_size, args.patches_per_image, True, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"], strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=amp)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    best = -1.0; history = []; started = time.time()
    print(f"device={device} samples={len(dataset)} batch={args.batch_size} boundary={args.boundary_weight}", flush=True)
    for epoch in _local_progress(range(1, args.epochs + 1), 'aspp_boundary_finetune'):
        model.train(); freeze_bn(model); total = 0.0; count = 0; t0 = time.time()
        for images, masks in _local_progress(loader, 'aspp_boundary_finetune'):
            images = images.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                loss = loss_fn(model(images), masks, args.boundary_weight)
            if amp:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.item()) * images.shape[0]; count += images.shape[0]
        value = evaluate(model, valid, args, device, amp)
        row = {"epoch": epoch, "train_loss": total / max(count, 1), "valid": value,
               "seconds": time.time() - t0}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": value, "args": vars(args)}, output / "last.pt")
        print(f"epoch={epoch} loss={row['train_loss']:.5f} global={value['image_miou_compat']:.6f} image={value['image_mean_miou']:.6f} time={row['seconds']:.1f}s", flush=True)
        if value["image_mean_miou"] > best:
            best = value["image_mean_miou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": value, "args": vars(args)}, output / "best.pt")
    summary = {"best_image_mean_miou": best, "elapsed_seconds": time.time() - started}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--patch-size", type=int, default=768)
    p.add_argument("--val-stride", type=int, default=576); p.add_argument("--tile-batch-size", type=int, default=1)
    p.add_argument("--patches-per-image", type=int, default=4); p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-6); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--boundary-weight", type=float, default=.20); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    main(p.parse_args())
