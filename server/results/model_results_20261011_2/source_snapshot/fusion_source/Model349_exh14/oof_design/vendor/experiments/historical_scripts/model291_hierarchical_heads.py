#!/usr/bin/env python3
"""Hierarchical Al-vs-secondary and secondary-phase segmentation heads."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from hard_negative_finetune import HARD_CLASS_WEIGHTS
from improve_five_causes import NUM_CLASSES, collect_split
from model181_raw_zcdf_aspp import seed_everything
from model253_highres_scale_finetune import (
    HighResolutionPatchDataset,
    dice_loss,
    evaluate,
    freeze_batchnorm_stats,
)
from model239_native_stem_finetune import Stride1NativeStemASPP


class HierarchicalTeamASPP(Stride1NativeStemASPP):
    """Factor four phases into binary phase mask times conditional subtype."""

    def __init__(self) -> None:
        super().__init__()
        self.binary_head = nn.Conv2d(32, 2, kernel_size=1)
        self.phase_head = nn.Conv2d(32, 3, kernel_size=1)

    @torch.no_grad()
    def initialize_heads_from_flat_classifier(self) -> None:
        """Warm-start new heads from the parent's four-class linear classifier."""
        source = self.head[1]
        secondary = torch.tensor((0, 1, 3), device=source.weight.device)
        self.binary_head.weight[0].copy_(source.weight[secondary].mean(dim=0))
        self.binary_head.weight[1].copy_(source.weight[2])
        self.binary_head.bias[0].copy_(source.bias[secondary].mean())
        self.binary_head.bias[1].copy_(source.bias[2])
        self.phase_head.weight.copy_(source.weight[secondary])
        self.phase_head.bias.copy_(source.bias[secondary])

    def forward_all(self, x: torch.Tensor):
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

        decoded = F.interpolate(x0_3, size=input_size, mode="bilinear", align_corners=False)
        stem_skip = F.interpolate(x0_0, size=input_size, mode="bilinear", align_corners=False)
        joined = torch.cat([decoded, stem_skip], dim=1)
        head_block = self.head[0].block
        hidden = head_block[0](joined) + self.native_to_head(self.native_stem(x))
        hidden = head_block[2](head_block[1](hidden))
        hidden = head_block[5](head_block[4](head_block[3](hidden)))

        flat_logits = self.head[1](hidden)
        binary_logits = self.binary_head(hidden)
        phase_logits = self.phase_head(hidden)

        # Binary channel 0 is secondary, channel 1 is Al. Conditional phase
        # channels are [Al3Ni, Primary Si, Eutectic Si]; output class order is
        # [Al3Ni, Primary Si, Al, Eutectic Si].
        log_binary = F.log_softmax(binary_logits.float(), dim=1)
        log_phase = F.log_softmax(phase_logits.float(), dim=1)
        hierarchical = torch.stack((
            log_binary[:, 0] + log_phase[:, 0],
            log_binary[:, 0] + log_phase[:, 1],
            log_binary[:, 1],
            log_binary[:, 0] + log_phase[:, 2],
        ), dim=1)
        return flat_logits, binary_logits, phase_logits, hierarchical

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_all(x)[-1]


def phase_target_from_mask(target: torch.Tensor) -> torch.Tensor:
    # 255 is ignored for Al pixels, where the conditional phase is undefined.
    phase = torch.full_like(target, 255)
    phase[target == 0] = 0
    phase[target == 1] = 1
    phase[target == 3] = 2
    return phase


def objective(flat_logits: torch.Tensor, binary_logits: torch.Tensor,
              phase_logits: torch.Tensor, hierarchical_logits: torch.Tensor,
              target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    weights = HARD_CLASS_WEIGHTS.to(target.device)
    hier_loss = F.cross_entropy(hierarchical_logits, target, weight=weights) \
        + dice_loss(hierarchical_logits, target)
    binary_target = (target == 2).long()
    # Per-patch balancing lets the binary head learn the thin secondary/Al
    # interface without allowing the abundant Al pixels to dominate.
    counts = torch.bincount(binary_target.reshape(-1), minlength=2).float()
    binary_weights = counts.sum() / (2.0 * counts.clamp_min(1.0))
    binary_weights = binary_weights / binary_weights.mean().clamp_min(1e-6)
    binary_loss = F.cross_entropy(binary_logits.float(), binary_target,
                                  weight=binary_weights)
    phase_target = phase_target_from_mask(target)
    phase_weights = weights[torch.tensor((0, 1, 3), device=target.device)]
    if (phase_target != 255).any():
        phase_loss = F.cross_entropy(phase_logits.float(), phase_target,
                                     weight=phase_weights, ignore_index=255)
    else:
        phase_loss = phase_logits.float().sum() * 0.0
    flat_loss = F.cross_entropy(flat_logits, target, weight=weights) \
        + dice_loss(flat_logits, target)
    return (hier_loss + args.binary_loss_weight * binary_loss
            + args.phase_loss_weight * phase_loss
            + args.flat_aux_weight * flat_loss)


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
        torch.backends.cudnn.benchmark = True

    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    model = HierarchicalTeamASPP().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {"binary_head.weight", "binary_head.bias",
                       "phase_head.weight", "phase_head.bias"}
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint mismatch: {incompatible}")
    model.initialize_heads_from_flat_classifier()

    encoder, decoder, native, hierarchy = [], [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith(("binary_head.", "phase_head.")):
            hierarchy.append(parameter)
        elif name.startswith(("native_stem.", "native_to_head.")):
            native.append(parameter)
        elif name.startswith(("stem.", "enc1.", "enc2.", "enc3.", "enc4.")):
            encoder.append(parameter)
        else:
            decoder.append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": args.encoder_lr},
        {"params": decoder, "lr": args.decoder_lr},
        {"params": native, "lr": args.native_lr},
        {"params": hierarchy, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    dataset = HighResolutionPatchDataset(
        train_samples, args.patch_size, args.patches_per_image, 1.0,
        args.seed, global_representation=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=(args.num_workers > 0))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_score, best_state, best_record = -1.0, None, None
    history, start_time = [], time.time()
    print(f"device={device} train_images={len(train_samples)} valid_images={len(valid_samples)} "
          f"patches={len(dataset)} patch={args.patch_size}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        freeze_batchnorm_stats(model)
        total, count = 0.0, 0
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                flat, binary, phase, hierarchical = model.forward_all(images)
                loss = objective(flat, binary, phase, hierarchical, masks, args)
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
            total += float(loss.detach()) * images.shape[0]
            count += images.shape[0]

        valid = evaluate(model, valid_samples, args, device, use_amp)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": total / max(count, 1),
                  "global_miou": valid["global_miou"],
                  "image_mean_miou": valid["image_mean_miou"],
                  "image_mean_miou_exclude": valid["image_mean_miou_exclude"],
                  "global_iou": valid["global_iou"], "seconds": time.time() - start_time}
        history.append(record)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if record[args.selection_metric] > best_score:
            best_score, best_state, best_record = (
                float(record[args.selection_metric]), copy.deepcopy(model.state_dict()), record)
            torch.save({"model": best_state, "epoch": epoch, "metrics": valid,
                        "args": vars(args), "parent_checkpoint": str(args.checkpoint),
                        "architecture": "hierarchical binary + conditional phase heads"},
                       output / "best.pt")

    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    final = evaluate(model, valid_samples, args, device, use_amp, output_probs=True)
    probabilities = final.pop("probabilities")
    np.savez_compressed(output / "valid_probs.npz", **{
        image_path.stem: value.astype(np.float16)
        for (image_path, _), value in zip(valid_samples, probabilities)
    })
    summary = {"method": "hierarchical multi-head supervised fine-tuning",
               "selection_metric": args.selection_metric, "best_selection": best_score,
               "best_record": best_record, "final": final,
               "elapsed_seconds": time.time() - start_time,
               "parent_checkpoint": str(args.checkpoint),
               "factorization": "P(class)=P(secondary/Al)*P(phase|secondary)"}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "best_selection", "best_record", "final", "elapsed_seconds")},
        ensure_ascii=False), flush=True)
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
    parser.add_argument("--encoder-lr", type=float, default=5e-6)
    parser.add_argument("--decoder-lr", type=float, default=1.5e-5)
    parser.add_argument("--native-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--binary-loss-weight", type=float, default=0.5)
    parser.add_argument("--phase-loss-weight", type=float, default=0.5)
    parser.add_argument("--flat-aux-weight", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--val-stride", type=int, default=576)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--tta", choices=("none", "d4"), default="d4")
    parser.add_argument("--selection-metric", choices=("image_mean_miou", "image_mean_miou_exclude"),
                        default="image_mean_miou")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=291)
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
