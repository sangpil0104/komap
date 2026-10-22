#!/usr/bin/env python3
"""Fine-tune the native ResNet34-ASPP branch with eutectic soft-clDice loss.

The ordinary weighted CE + soft Dice objective remains the main objective;
soft-clDice is a small auxiliary term applied only to eutectic-Si pixels to
encourage continuity of fine eutectic features. Training reads train split
only; validation is used for checkpoint selection and branch measurement.
"""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from improve_five_causes import collect_split
from model181_raw_zcdf_aspp import (
    HARD_CLASS_WEIGHTS,
    RepresentationPatchDataset,
    dice_loss,
    evaluate,
    seed_everything,
)
from model239_native_stem_finetune import Stride1NativeStemASPP


def soft_erode(value: torch.Tensor) -> torch.Tensor:
    vertical = -F.max_pool2d(-value, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-value, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_dilate(value: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(value, 3, stride=1, padding=1)


def soft_open(value: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(value))


def soft_skeletonize(value: torch.Tensor, iterations: int) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    skeleton = F.relu(value - soft_open(value))
    for _ in range(iterations):
        value = soft_erode(value)
        delta = F.relu(value - soft_open(value))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def eutectic_cldice_loss(logits: torch.Tensor, target: torch.Tensor,
                         iterations: int) -> torch.Tensor:
    # Ignore samples without eutectic pixels: CE/Dice already penalize false
    # positives there, while a centerline ratio has no meaningful GT support.
    eutectic_target = (target == 3).unsqueeze(1).float()
    present = eutectic_target.sum((1, 2, 3)) > 0
    if not bool(present.any()):
        return logits.sum() * 0.0
    probability = torch.softmax(logits.float(), dim=1)[:, 3:4][present]
    truth = eutectic_target[present]
    pred_skeleton = soft_skeletonize(probability, iterations)
    true_skeleton = soft_skeletonize(truth, iterations)
    dims = (1, 2, 3)
    topological_precision = (pred_skeleton * truth).sum(dims) / (pred_skeleton.sum(dims) + 1e-6)
    topological_sensitivity = (true_skeleton * probability).sum(dims) / (true_skeleton.sum(dims) + 1e-6)
    cldice = (2.0 * topological_precision * topological_sensitivity + 1e-6) / (
        topological_precision + topological_sensitivity + 1e-6)
    return (1.0 - cldice).mean()


def train_epoch(model, loader, optimizer, scaler, device, use_amp,
                cldice_weight: float, skeleton_iterations: int) -> dict[str, float]:
    model.train()
    weights = HARD_CLASS_WEIGHTS.to(device)
    totals = {"total": 0.0, "hard": 0.0, "cldice": 0.0}
    count = 0
    for images, masks in _local_progress(loader, 'model249_eutectic_cldice_finetune'):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            hard = F.cross_entropy(logits, masks, weight=weights) + dice_loss(logits, masks)
        topology = eutectic_cldice_loss(logits, masks, skeleton_iterations)
        loss = hard + cldice_weight * topology
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
        batch_count = int(images.shape[0])
        totals["total"] += float(loss.detach()) * batch_count
        totals["hard"] += float(hard.detach()) * batch_count
        totals["cldice"] += float(topology.detach()) * batch_count
        count += batch_count
    return {key: value / max(count, 1) for key, value in totals.items()}


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    model = Stride1NativeStemASPP().to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)

    encoder, decoder, native = [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith(("native_stem.", "native_to_head.")):
            native.append(parameter)
        elif name.startswith(("stem.", "enc1.", "enc2.", "enc3.", "enc4.")):
            encoder.append(parameter)
        else:
            decoder.append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": args.encoder_lr},
        {"params": decoder, "lr": args.decoder_lr},
        {"params": native, "lr": args.native_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    dataset = RepresentationPatchDataset(
        train_samples, args.patch_size, args.patches_per_image, True, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=(args.num_workers > 0))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_score, best_state, best_record = -1.0, None, None
    history, start_time = [], time.time()
    print(f"device={device} train_images={len(train_samples)} patches={len(dataset)} "
          f"cldice_weight={args.cldice_weight} skeleton_iterations={args.skeleton_iterations}",
          flush=True)

    for epoch in _local_progress(range(1, args.epochs + 1), 'model249_eutectic_cldice_finetune'):
        losses = train_epoch(model, loader, optimizer, scaler, device, use_amp,
                             args.cldice_weight, args.skeleton_iterations)
        valid = evaluate(model, valid_samples, args, device, use_amp)
        scheduler.step()
        record = {"epoch": epoch, "losses": losses,
                  "selection": float(valid[args.selection_metric]), "valid": valid,
                  "seconds": time.time() - start_time}
        history.append(record)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"epoch": epoch, **losses,
                          "image_miou_compat": valid["image_miou_compat"],
                          "image_mean_miou": valid["image_mean_miou"],
                          "image_mean_miou_exclude": valid["image_mean_miou_exclude"]},
                         ensure_ascii=False), flush=True)
        if record["selection"] > best_score:
            best_score, best_state, best_record = (
                record["selection"], copy.deepcopy(model.state_dict()), record)
            torch.save({"model": best_state, "epoch": epoch, "metrics": valid,
                        "args": vars(args), "parent_checkpoint": str(args.checkpoint),
                        "auxiliary_loss": "eutectic soft-clDice"}, output / "best.pt")

    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    final = evaluate(model, valid_samples, args, device, use_amp, output_probs=True)
    probabilities = final.pop("probabilities")
    np.savez_compressed(output / "valid_probs.npz", **{
        path.stem: value.astype(np.float16)
        for (path, _), value in zip(valid_samples, probabilities)
    })
    summary = {"method": "weighted CE + Dice + eutectic soft-clDice",
               "selection_metric": args.selection_metric,
               "best_selection": best_score, "best_record": best_record,
               "final": final, "elapsed_seconds": time.time() - start_time,
               "parent_checkpoint": str(args.checkpoint)}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    del loader, dataset, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--cldice-weight", type=float, default=0.10)
    parser.add_argument("--skeleton-iterations", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=2e-6)
    parser.add_argument("--decoder-lr", type=float, default=6e-6)
    parser.add_argument("--native-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-stride", type=int, default=768)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--selection-metric", choices=("image_mean_miou",
                                                         "image_mean_miou_exclude"),
                        default="image_mean_miou")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=249)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
