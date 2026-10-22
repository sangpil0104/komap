#!/usr/bin/env python3
"""Fine-tune Model257 with a zero-initialized metallographic boundary module."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from improve_five_causes import collect_split, decode_mask
from model181_raw_zcdf_aspp import HARD_CLASS_WEIGHTS, dice_loss
from model239_native_stem_finetune import Stride1NativeStemASPP
from model253_highres_scale_finetune import HighResolutionPatchDataset, predict_scaled
from model288_region_classifier_cv import image_score


class BoundaryEnhancement(nn.Module):
    """Morphological edge cue + multi-dilation context with identity initialization."""

    def __init__(self, channels: int = 64):
        super().__init__()
        branch_channels = channels // 4
        self.dilated = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation,
                          dilation=dilation, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            for dilation in (1, 3, 5, 7)
        ])
        self.project = nn.Conv2d(branch_channels * 4, channels, 1, bias=False)
        self.project_norm = nn.BatchNorm2d(channels)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels * 2, branch_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, channels, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.project.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        eroded = -F.max_pool2d(-features, kernel_size=3, stride=1, padding=1)
        boundary = F.relu(features - eroded)
        boundary_enriched = features + boundary
        multi_scale = torch.cat([branch(boundary_enriched) for branch in self.dilated], dim=1)
        delta = self.project_norm(self.project(multi_scale))
        pooled_avg = F.adaptive_avg_pool2d(boundary_enriched, 1)
        pooled_max = F.adaptive_max_pool2d(boundary_enriched, 1)
        gate = self.channel_gate(torch.cat([pooled_avg, pooled_max], dim=1))
        return features + delta * gate


class BoundaryEnhancedASPP(Stride1NativeStemASPP):
    """Model257-compatible U-Net++ with shared boundary enhancement on fine decoder stages."""

    def __init__(self) -> None:
        super().__init__()
        self.boundary_enhancer = BoundaryEnhancement(channels=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x0_0 = self.stem(x)
        x1_0 = self.enc1(self.pool(x0_0))
        x2_0 = self.enc2(x1_0)
        x3_0 = self.enc3(x2_0)
        x4_0 = self.context(self.enc4(x3_0))

        x3_1 = self.x3_1(torch.cat([self.up(x4_0, x3_0), x3_0], dim=1))
        x2_1 = self.x2_1(torch.cat([self.up(x3_0, x2_0), x2_0], dim=1))
        x1_1 = self.x1_1(torch.cat([self.up(x2_0, x1_0), x1_0], dim=1))
        x0_1 = self.boundary_enhancer(
            self.x0_1(torch.cat([self.up(x1_0, x0_0), x0_0], dim=1)))
        x2_2 = self.x2_2(torch.cat([self.up(x3_1, x2_0), x2_0, x2_1], dim=1))
        x1_2 = self.x1_2(torch.cat([self.up(x2_1, x1_0), x1_0, x1_1], dim=1))
        x0_2 = self.boundary_enhancer(
            self.x0_2(torch.cat([self.up(x1_1, x0_0), x0_0, x0_1], dim=1)))
        x1_3 = self.x1_3(torch.cat([self.up(x2_2, x1_0), x1_0, x1_1, x1_2], dim=1))
        x0_3 = self.boundary_enhancer(
            self.x0_3(torch.cat([self.up(x1_2, x0_0), x0_0, x0_1, x0_2], dim=1)))

        decoded = F.interpolate(x0_3, size=input_size, mode="bilinear", align_corners=False)
        stem_skip = F.interpolate(x0_0, size=input_size, mode="bilinear", align_corners=False)
        joined = torch.cat([decoded, stem_skip], dim=1)
        head_block = self.head[0].block
        hidden = head_block[0](joined) + self.native_to_head(self.native_stem(x))
        hidden = head_block[2](head_block[1](hidden))
        hidden = head_block[5](head_block[4](head_block[3](hidden)))
        return self.head[1](hidden)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_base_checkpoint(model: BoundaryEnhancedASPP, checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    missing = [key for key in incompatible.missing_keys
               if not key.startswith("boundary_enhancer.")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint mismatch: missing={missing}, "
                           f"unexpected={incompatible.unexpected_keys}")
    return payload


@torch.no_grad()
def evaluate(model: nn.Module, samples, data_dir: Path, patch_size: int,
             stride: int, device: torch.device, use_amp: bool,
             save_probs: bool = False) -> dict:
    model.eval()
    scores = []
    predictions = {}
    for image_path, mask_path in _local_progress(samples, 'model301_boundary_enhanced_finetune'):
        probabilities = predict_scaled(model, image_path, patch_size, stride, 1.0,
                                       device, use_amp, "d4")
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        scores.append(image_score(probabilities.argmax(0).astype(np.uint8), target))
        if save_probs:
            predictions[image_path.stem] = probabilities.astype(np.float16)
    return {"image_mean_miou": float(np.mean(scores)),
            "image_scores": scores, "probabilities": predictions}


def score_archive(archive_path: Path, samples) -> dict:
    archive = np.load(archive_path)
    scores = []
    for image_path, mask_path in _local_progress(samples, 'model301_boundary_enhanced_finetune'):
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        prediction = archive[image_path.stem].astype(np.float32).argmax(0).astype(np.uint8)
        scores.append(image_score(prediction, target))
    return {"image_mean_miou": float(np.mean(scores)), "image_scores": scores}


def train_epoch(model, loader, optimizer, scaler, device, use_amp) -> float:
    model.train()
    total, count = 0.0, 0
    weights = HARD_CLASS_WEIGHTS.to(device)
    for images, masks in _local_progress(loader, 'model301_boundary_enhanced_finetune'):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = F.cross_entropy(logits, masks, weight=weights) + dice_loss(logits, masks)
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
    return total / max(count, 1)


def smoke_test() -> None:
    torch.manual_seed(301)
    baseline = Stride1NativeStemASPP().eval()
    torch.manual_seed(301)
    enhanced = BoundaryEnhancedASPP().eval()
    enhanced.load_state_dict(baseline.state_dict(), strict=False)
    sample = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        difference = float((baseline(sample) - enhanced(sample)).abs().max())
    if difference > 1e-6:
        raise RuntimeError(f"zero-init boundary module changed initial prediction: {difference}")
    print(json.dumps({"identity_init_max_logit_difference": difference,
                      "parameters": sum(p.numel() for p in enhanced.parameters())}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="outputs/model257_full_image_stats_1x/best.pt")
    parser.add_argument("--baseline-probs", default="outputs/model257_full_image_stats_1x/valid_probs.npz")
    parser.add_argument("--output-dir", default="outputs/model301_boundary_enhanced")
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--val-stride", type=int, default=864)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=257)
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    parser.add_argument("--fixed-final-epoch", action="store_true",
                        help="withhold validation masks until one final evaluation")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return

    started = time.time()
    seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index or 0)
        torch.backends.cudnn.benchmark = True

    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint)
    baseline_probs_path = Path(args.baseline_probs)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_samples = collect_split(data_dir, "train")
    valid_samples = collect_split(data_dir, "valid")
    baseline = None if args.fixed_final_epoch else score_archive(
        baseline_probs_path, valid_samples)

    model = BoundaryEnhancedASPP().to(device)
    payload = load_base_checkpoint(model, checkpoint_path)
    encoder, decoder, native, boundary = [], [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith("boundary_enhancer."):
            boundary.append(parameter)
        elif name.startswith(("native_stem.", "native_to_head.")):
            native.append(parameter)
        elif name.startswith(("stem.", "enc1.", "enc2.", "enc3.", "enc4.")):
            encoder.append(parameter)
        else:
            decoder.append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": 5e-6},
        {"params": decoder, "lr": 1.5e-5},
        {"params": native, "lr": 1e-4},
        {"params": boundary, "lr": 1e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    dataset = HighResolutionPatchDataset(
        train_samples, args.patch_size, args.patches_per_image, 1.0,
        args.seed, global_representation=True, metallography_augmentation=False,
        sampling_mode="class_focused")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0))
    history = []
    best_score, best_state, best_record = -1.0, None, None
    baseline_text = ("deferred" if baseline is None
                     else f"{baseline['image_mean_miou']*100:.4f}%")
    print(f"device={device} train_images={len(train_samples)} patches={len(dataset)} "
          f"baseline_valid={baseline_text} "
          f"checkpoint_epoch={payload.get('epoch')}", flush=True)
    for epoch in _local_progress(range(1, args.epochs + 1), 'model301_boundary_enhanced_finetune'):
        train_loss = train_epoch(model, loader, optimizer, scaler, device, use_amp)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss,
                  "seconds": time.time() - started}
        if not args.fixed_final_epoch:
            valid = evaluate(model, valid_samples, data_dir, args.patch_size,
                             args.val_stride, device, use_amp, save_probs=True)
            record["valid_image_mean_miou"] = valid["image_mean_miou"]
            record["valid_delta_pp_vs_model257"] = (
                valid["image_mean_miou"] - baseline["image_mean_miou"]) * 100.0
        history.append(record)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        if args.fixed_final_epoch and epoch == args.epochs:
            torch.save({"model": copy.deepcopy(model.state_dict()), "epoch": epoch,
                        "parent_checkpoint": str(checkpoint_path),
                        "architecture": "Model257 + shared multi-dilation boundary enhancement",
                        "validation_masks_used_for_selection": False}, output / "last.pt")
        elif not args.fixed_final_epoch and record["valid_image_mean_miou"] > best_score:
            best_score, best_record = record["valid_image_mean_miou"], record
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model": best_state, "epoch": epoch,
                        "parent_checkpoint": str(checkpoint_path),
                        "architecture": "Model257 + shared multi-dilation boundary enhancement",
                        "metrics": record}, output / "best.pt")

    if args.fixed_final_epoch:
        if not (output / "last.pt").is_file():
            raise RuntimeError("fixed final-epoch checkpoint was not saved")
        best_record = history[-1]
    else:
        if best_state is None:
            raise RuntimeError("no valid checkpoint selected")
        model.load_state_dict(best_state, strict=True)
    final = evaluate(model, valid_samples, data_dir, args.patch_size,
                     args.val_stride, device, use_amp, save_probs=True)
    if baseline is None:
        baseline = score_archive(baseline_probs_path, valid_samples)
    valid_probabilities = final.pop("probabilities")
    np.savez_compressed(output / "valid_probs.npz", **valid_probabilities)
    summary = {
        "model": 301,
        "method": "Model257 fine-tuned with shared boundary enhancement on U-Net++ fine decoder stages",
        "metric": "per-image mean of four class IoUs; empty union is 0",
        "loss": "weighted cross-entropy + Dice",
        "train_images": len(train_samples),
        "valid_masks_used_for_epoch_selection": not args.fixed_final_epoch,
        "checkpoint_selection": ("fixed final epoch" if args.fixed_final_epoch
                                  else "best validation image_mean_miou"),
        "baseline_model257_image_mean_miou": baseline["image_mean_miou"],
        "best_record": best_record,
        "final": {"image_mean_miou": final["image_mean_miou"],
                  "image_scores": final["image_scores"]},
        "elapsed_seconds": time.time() - started,
        "checkpoint_epoch": payload.get("epoch"),
        "artifacts": {"checkpoint": str(output /
                                          ("last.pt" if args.fixed_final_epoch else "best.pt")),
                      "valid_probs": str(output / "valid_probs.npz")},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "baseline_model257_image_mean_miou", "best_record", "final", "elapsed_seconds")},
        ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
