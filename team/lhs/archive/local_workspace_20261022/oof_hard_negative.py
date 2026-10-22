#!/usr/bin/env python3
"""Group-aware OOF hard-negative mining followed by model98 fine-tuning."""

from __future__ import annotations

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

from advanced_inference import metrics, predict_probabilities
from hard_negative_finetune import HARD_CLASS_WEIGHTS, HardNegativeDataset, load_model, split_parameters
from improve_five_causes import NUM_CLASSES, RareAwarePatchDataset, collect_split, decode_mask, grouped_folds
from train_models_2_to_6 import UNetPlusPlusResNet34


def dice_loss(logits, target):
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    denominator = probabilities.sum(dims) + one_hot.sum(dims)
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def loss_fn(logits, target):
    return F.cross_entropy(logits, target, weight=HARD_CLASS_WEIGHTS.to(logits.device)) + dice_loss(logits, target)


def train_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total, count = 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            loss = loss_fn(model(images), masks)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.item()) * images.shape[0]
        count += images.shape[0]
    return total / max(count, 1)


def mine_oof(args, train_samples, device, use_amp):
    folds = grouped_folds(train_samples, args.folds, args.seed)
    centers = []
    fold_report = []
    for fold_id, valid_indices in enumerate(folds, 1):
        valid_set = set(valid_indices)
        fold_train = [sample for index, sample in enumerate(train_samples) if index not in valid_set]
        fold_valid = [train_samples[index] for index in valid_indices]
        model = UNetPlusPlusResNet34(pretrained=True).to(device)
        dataset = RareAwarePatchDataset(fold_train, args.patch_size, args.patches_per_image, True,
                                        args.seed + fold_id)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = GradScaler(enabled=use_amp)
        for epoch in range(args.fold_epochs):
            train_loss = train_epoch(model, loader, optimizer, scaler, device, use_amp)
            print(f"fold={fold_id} epoch={epoch + 1} loss={train_loss:.4f}", flush=True)
        model.eval()
        fold_centers = 0
        for image_path, mask_path in fold_valid:
            probabilities = predict_probabilities(model, image_path, args.patch_size, args.val_stride, device,
                                                   use_amp, args.tile_batch_size, "d4", "gaussian")
            adjusted = np.log(np.maximum(probabilities, 1e-7))
            adjusted[3] -= 0.20
            prediction = adjusted.argmax(axis=0)
            target = decode_mask(Image.open(mask_path).convert("RGB"))
            false_positive = np.argwhere((prediction == 3) & (target != 3))
            rng = np.random.default_rng(args.seed + fold_id * 1000 + fold_centers)
            if len(false_positive) > args.centers_per_image:
                selected = false_positive[rng.choice(len(false_positive), args.centers_per_image, replace=False)]
            else:
                selected = false_positive
            height, width = target.shape
            for y, x in selected:
                centers.append((image_path, mask_path, float(y) / max(height - 1, 1),
                                float(x) / max(width - 1, 1)))
            fold_centers += len(selected)
        fold_report.append({"fold": fold_id, "train_images": len(fold_train), "heldout_images": len(fold_valid),
                            "centers": fold_centers})
        print(f"fold={fold_id} oof_centers={fold_centers}", flush=True)
        del model, loader, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return centers, fold_report


@torch.no_grad()
def evaluate_single(model, samples, args, device, use_amp):
    predictions, targets = [], []
    for image_path, mask_path in samples:
        probabilities = predict_probabilities(model, image_path, args.patch_size, args.val_stride, device,
                                               use_amp, args.tile_batch_size, "d4", "gaussian")
        probabilities = np.log(np.maximum(probabilities, 1e-7))
        probabilities += np.asarray([0.0, 0.0, 0.10, -0.20], dtype=np.float32)[:, None, None]
        predictions.append(probabilities.argmax(axis=0).astype(np.uint8))
        targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
    return metrics(predictions, targets)


def finetune(args, centers, train_samples, valid_samples, device, use_amp):
    dataset = HardNegativeDataset(train_samples, centers, args.patch_size, args.base_patches_per_image,
                                  args.hard_repeats, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    model = load_model(args.base_checkpoint, device)
    encoder, decoder = split_parameters(model)
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": args.encoder_lr},
                                   {"params": decoder, "lr": args.decoder_lr}], weight_decay=1e-4)
    scaler = GradScaler(enabled=use_amp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, scaler, device, use_amp)
        validation = evaluate_single(model, valid_samples, args, device, use_amp)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid": validation})
        print(f"oof-ft epoch={epoch} loss={train_loss:.4f} global={validation['global_miou']:.4f} "
              f"iou=" + ",".join(f"{v:.3f}" for v in validation["global_iou"].values()), flush=True)
        if validation["global_miou"] > best:
            best = validation["global_miou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": validation, "args": vars(args)},
                       output_dir / "best.pt")
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return best


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    start = time.time()
    centers, fold_report = mine_oof(args, train_samples, device, use_amp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = [{"image": str(image), "mask": str(mask), "y": y, "x": x} for image, mask, y, x in centers]
    (output_dir / "oof_centers.json").write_text(json.dumps({"folds": fold_report, "centers": serializable},
                                                               ensure_ascii=False, indent=2), encoding="utf-8")
    best = finetune(args, centers, train_samples, valid_samples, device, use_amp)
    summary = {"folds": fold_report, "centers": len(centers), "best_single_model_miou": best,
               "elapsed_seconds": time.time() - start}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--val-stride", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--fold-epochs", type=int, default=4)
    parser.add_argument("--patches-per-image", type=int, default=4)
    parser.add_argument("--centers-per-image", type=int, default=8)
    parser.add_argument("--base-patches-per-image", type=int, default=4)
    parser.add_argument("--hard-repeats", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-6)
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
