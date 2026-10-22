#!/usr/bin/env python3
"""Fine-tune an ASPP U-Net++ with an Al3Ni/Eutectic-Si pairwise margin.

The current validation audit shows that most of the low image-wise scores come
from confusion between internal channels 0 (Al3Ni) and 3 (Eutectic Si).  This
script keeps the normal weighted CE+Dice objective and adds a small symmetric
margin loss only on those two target classes.
"""

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

from advanced_inference import predict_probabilities
from encoder_structure_search import build_model, loss_fn as base_loss
from improve_five_causes import (
    NUM_CLASSES,
    RareAwarePatchDataset,
    collect_split,
    decode_mask,
    encode_mask,
)
from restart_ensemble import metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pairwise_margin_loss(logits: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    """Penalize insufficient separation between Al3Ni (0) and Eutectic Si (3)."""
    pair = (target == 0) | (target == 3)
    if not bool(pair.any()):
        return logits.sum() * 0.0
    other = torch.where(target == 0, torch.full_like(target, 3), torch.zeros_like(target))
    target_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    other_logit = logits.gather(1, other.unsqueeze(1)).squeeze(1)
    values = F.relu(float(margin) - (target_logit - other_logit))
    return values[pair].mean()


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, margin, pair_lambda):
    model.train()
    # 768px fits only as a micro-batch of one on the available P100. Freeze all
    # BN running statistics so the fine-tune does not corrupt pretrained stats;
    # affine BN parameters remain trainable.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    total, count = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            normal = base_loss(logits, masks)
            pair = pairwise_margin_loss(logits, masks, margin)
            loss = normal + float(pair_lambda) * pair
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


@torch.no_grad()
def collect_probabilities(model, samples, args, device, use_amp):
    model.eval()
    probabilities, targets = [], []
    for image_path, mask_path in samples:
        probabilities.append(predict_probabilities(
            model, image_path, args.patch_size, args.val_stride, device, use_amp,
            args.tile_batch_size, args.tta, args.blend,
        ))
        targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
    return probabilities, targets


def calibrated_metrics(probabilities, targets, biases):
    bias = np.asarray(biases, dtype=np.float32)[:, None, None]
    predictions = [(np.log(np.maximum(value, 1e-7)) + bias).argmax(0).astype(np.uint8)
                   for value in probabilities]
    return metrics(predictions, targets)


def evaluate_and_save_test(model, args, device, use_amp, bias, output_dir):
    test_dir = Path(output_dir) / "test_masks"
    test_dir.mkdir(parents=True, exist_ok=True)
    adjusted_bias = np.asarray(bias, dtype=np.float32)[:, None, None]
    for image_path in sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png")):
        value = predict_probabilities(model, image_path, args.patch_size, args.val_stride,
                                      device, use_amp, args.tile_batch_size, args.tta, args.blend)
        prediction = (np.log(np.maximum(value, 1e-7)) + adjusted_bias).argmax(0).astype(np.uint8)
        encode_mask(prediction).save(test_dir / image_path.name.replace("_image.png", "_mask.png"))
    return len(list(test_dir.glob("*.png")))


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    dataset = RareAwarePatchDataset(train_samples, args.patch_size, args.patches_per_image, True, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                        persistent_workers=(args.num_workers > 0))

    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    print(f"loaded={args.checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -1.0
    start = time.time()
    print(f"device={device} patches={len(dataset)} batch={args.batch_size} "
          f"pair_lambda={args.pair_lambda} margin={args.margin}", flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, loader, optimizer, scaler, device, use_amp,
                                     args.margin, args.pair_lambda)
        probabilities, targets = collect_probabilities(model, valid_samples, args, device, use_amp)
        # Keep a small calibration grid, with the pair classes independently
        # tunable because the auxiliary loss changes their relative margin.
        candidates = []
        for al3ni in (-0.10, 0.0, 0.10):
            for eutectic in (-0.70, -0.60, -0.55, -0.50, -0.40):
                value = calibrated_metrics(probabilities, targets, [al3ni, 0.0, 0.0, eutectic])
                candidates.append({"biases": [al3ni, 0.0, 0.0, eutectic], **value})
        calibrated = max(candidates, key=lambda item: item["image_mean_miou"])
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "valid": calibrated,
                  "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - epoch_start}
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": calibrated,
                    "args": vars(args)}, output_dir / "last.pt")
        print(f"epoch={epoch:02d}/{args.epochs} loss={train_loss:.5f} "
              f"global={calibrated['global_miou']:.6f} image={calibrated['image_mean_miou']:.6f} "
              f"bias={calibrated['biases']} time={record['seconds']:.1f}s", flush=True)
        score = float(calibrated["image_mean_miou"])
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": calibrated,
                        "args": vars(args)}, output_dir / "best.pt")

    best_payload = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["model"])
    probabilities, targets = collect_probabilities(model, valid_samples, args, device, use_amp)
    calibration = []
    for al3ni in np.arange(-0.20, 0.21, 0.05):
        for eutectic in np.arange(-0.90, -0.19, 0.05):
            value = calibrated_metrics(probabilities, targets, [float(al3ni), 0.0, 0.0, float(eutectic)])
            calibration.append({"biases": [float(al3ni), 0.0, 0.0, float(eutectic)], **value})
    best_calibrated = max(calibration, key=lambda item: item["image_mean_miou"])
    summary = {"pair_lambda": args.pair_lambda, "margin": args.margin, "epochs": args.epochs,
               "best_image_mean_miou": best_score, "best_calibrated": best_calibrated,
               "calibration": calibration, "checkpoint": str(output_dir / "best.pt"),
               "elapsed_seconds": time.time() - start}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "valid_probabilities.npz",
                        **{f"p{i}": value.astype(np.float16) for i, value in enumerate(probabilities)})
    n_test = 0  # no test prediction in the seed variability study
    print(json.dumps({"summary": summary, "test_masks": n_test}, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--val-stride", type=int, default=576)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--patches-per-image", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pair-lambda", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--tta", choices=("legacy", "d4"), default="d4")
    parser.add_argument("--blend", choices=("uniform", "gaussian"), default="gaussian")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
