#!/usr/bin/env python3
"""Deep-supervised ASPP U-Net++ fine-tune and auxiliary-logit search."""

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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from advanced_inference import predict_probabilities
from encoder_structure_search import ASPPUNetPP, loss_fn
from improve_five_causes import RareAwarePatchDataset, collect_split, decode_mask, encode_mask
from restart_ensemble import metrics


class DeepSupervisedASPP(ASPPUNetPP):
    def __init__(self):
        super().__init__(pretrained=False)
        self.aux1 = nn.Conv2d(64, 4, 1)
        self.aux2 = nn.Conv2d(64, 4, 1)
        self.aux_weight_1 = 0.0
        self.aux_weight_2 = 0.0

    def forward_all(self, x):
        input_size = x.shape[-2:]
        x0_0 = self.stem(x)
        x1_0 = self.enc1(self.pool(x0_0))
        x2_0 = self.enc2(x1_0)
        x3_0 = self.enc3(x2_0)
        x4_0 = self.context(self.enc4(x3_0))
        x3_1 = self.x3_1(torch.cat([self.up(x4_0, x3_0), x3_0], dim=1))
        x2_1 = self.x2_1(torch.cat([self.up(x3_0, x2_0), x2_0], dim=1))
        x1_1 = self.x1_1(torch.cat([self.up(x2_0, x1_0), x1_0], dim=1))
        x0_1 = self.x0_1(torch.cat([self.up(x1_0, x0_0), x0_0], dim=1))
        x2_2 = self.x2_2(torch.cat([self.up(x3_1, x2_0), x2_0, x2_1], dim=1))
        x1_2 = self.x1_2(torch.cat([self.up(x2_1, x1_0), x1_0, x1_1], dim=1))
        x0_2 = self.x0_2(torch.cat([self.up(x1_1, x0_0), x0_0, x0_1], dim=1))
        x1_3 = self.x1_3(torch.cat([self.up(x2_2, x1_0), x1_0, x1_1, x1_2], dim=1))
        x0_3 = self.x0_3(torch.cat([self.up(x1_2, x0_0), x0_0, x0_1, x0_2], dim=1))
        final = self.head(torch.cat([
            F.interpolate(x0_3, size=input_size, mode="bilinear", align_corners=False),
            F.interpolate(x0_0, size=input_size, mode="bilinear", align_corners=False),
        ], dim=1))
        aux1 = F.interpolate(self.aux1(x0_1), size=input_size, mode="bilinear", align_corners=False)
        aux2 = F.interpolate(self.aux2(x0_2), size=input_size, mode="bilinear", align_corners=False)
        return aux1, aux2, final

    def forward(self, x):
        aux1, aux2, final = self.forward_all(x)
        return final + self.aux_weight_1 * aux1 + self.aux_weight_2 * aux2


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def confusion(prediction, target):
    bins = target.reshape(-1) * 4 + prediction.reshape(-1)
    return np.bincount(bins, minlength=16).reshape(4, 4)


@torch.no_grad()
def collect_probabilities(model, samples, args, device, use_amp):
    model.eval()
    values, targets = [], []
    for image_path, mask_path in _local_progress(samples, 'deep_supervision_aspp'):
        values.append(predict_probabilities(model, image_path, args.patch_size, args.stride,
                                            device, use_amp, args.tile_batch_size, args.tta, args.blend))
        targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
    return values, targets


def evaluate(values, targets, bias):
    bias = np.asarray(bias, dtype=np.float32)[:, None, None]
    predictions = [(np.log(np.maximum(v, 1e-7)) + bias).argmax(0).astype(np.uint8) for v in values]
    return metrics(predictions, targets)


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, aux1_weight, aux2_weight):
    model.train()
    total, count = 0.0, 0
    for images, masks in _local_progress(loader, 'deep_supervision_aspp'):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            aux1, aux2, final = model.forward_all(images)
            value = loss_fn(final, masks) + aux1_weight * loss_fn(aux1, masks) + aux2_weight * loss_fn(aux2, masks)
        if use_amp:
            scaler.scale(value).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(value.item()) * images.shape[0]
        count += images.shape[0]
    return total / max(count, 1)


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    train_samples, valid_samples = collect_split(args.data_dir, "train"), collect_split(args.data_dir, "valid")
    dataset = RareAwarePatchDataset(train_samples, args.patch_size, args.patches_per_image, True, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    model = DeepSupervisedASPP().to(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    nn.init.zeros_(model.aux1.weight); nn.init.zeros_(model.aux1.bias)
    nn.init.zeros_(model.aux2.weight); nn.init.zeros_(model.aux2.bias)
    print(f"loaded={args.checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    start = time.time()
    print(f"device={device} patches={len(dataset)} batch={args.batch_size}", flush=True)
    for epoch in _local_progress(range(1, args.epochs + 1), 'deep_supervision_aspp'):
        t0 = time.time()
        aux_scale = 0.0 if epoch <= args.warmup_epochs else 1.0
        train_loss = train_one_epoch(model, loader, optimizer, scaler, device, use_amp,
                                     aux_scale * args.aux1_loss_weight, aux_scale * args.aux2_loss_weight)
        model.aux_weight_1 = model.aux_weight_2 = 0.0
        values, targets = collect_probabilities(model, valid_samples, args, device, use_amp)
        candidates = []
        for al in (0.20, 0.40, 0.60):
            for eut in (-0.70, -0.60, -0.55, -0.50, -0.40):
                candidates.append({"biases": [0.0, 0.0, al, eut], **evaluate(values, targets, [0.0, 0.0, al, eut])})
        valid = max(candidates, key=lambda x: x["image_mean_miou"])
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "valid": valid,
                  "aux_scale": aux_scale,
                  "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(record)
        (out / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": valid, "args": vars(args)}, out / "last.pt")
        print(f"epoch={epoch:02d}/{args.epochs} loss={train_loss:.5f} image_compat={valid['image_miou_compat']:.6f} "
              f"image={valid['image_mean_miou']:.6f} time={record['seconds']:.1f}s", flush=True)
        if valid["image_mean_miou"] > best:
            best = valid["image_mean_miou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": valid, "args": vars(args)}, out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu")["model"])
    calibration_records = []
    for a1 in (0.0, 0.10, 0.20):
        for a2 in (0.0, 0.10, 0.20):
            model.aux_weight_1, model.aux_weight_2 = a1, a2
            values, targets = collect_probabilities(model, valid_samples, args, device, use_amp)
            for bias in ([0.0, 0.0, 0.40, -0.55], [0.0, 0.0, 0.50, -0.55], [0.0, 0.0, 0.40, -0.50]):
                calibration_records.append({"aux1": a1, "aux2": a2, "biases": bias, **evaluate(values, targets, bias)})
    best_calibrated = max(calibration_records, key=lambda x: x["image_mean_miou"])
    model.aux_weight_1, model.aux_weight_2 = best_calibrated["aux1"], best_calibrated["aux2"]
    test_dir = out / "test_masks"
    test_dir.mkdir(exist_ok=True)
    bias = np.asarray(best_calibrated["biases"], dtype=np.float32)[:, None, None]
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        value = predict_probabilities(model, image_path, args.patch_size, args.stride, device, use_amp,
                                      args.tile_batch_size, args.tta, args.blend)
        encode_mask((np.log(np.maximum(value, 1e-7)) + bias).argmax(0).astype(np.uint8)).save(
            test_dir / image_path.name.replace("_image.png", "_mask.png"))
    summary = {"best_fixed_image": best, "best_calibrated": best_calibrated,
               "calibration_count": len(calibration_records), "elapsed_seconds": time.time() - start}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=384); p.add_argument("--tile-batch-size", type=int, default=2)
    p.add_argument("--patches-per-image", type=int, default=6); p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2); p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=8e-6); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--aux1-loss-weight", type=float, default=0.20); p.add_argument("--aux2-loss-weight", type=float, default=0.30)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--tta", choices=("legacy", "d4"), default="d4"); p.add_argument("--blend", choices=("uniform", "gaussian"), default="gaussian")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--no-amp", action="store_true")
    train(p.parse_args())
