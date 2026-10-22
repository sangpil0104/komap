#!/usr/bin/env python3
"""Train a complementary ASPP branch with rare-class and boundary losses.

The image-wise-IoU branch (model207) did not complement the current route.
This branch changes the optimization target instead: class-weighted pixel CE
is increased around annotated phase boundaries and a focal Tversky term gives
more weight to rare-class false negatives.
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
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from advanced_inference import predict_probabilities
from encoder_structure_search import build_model
from improve_five_causes import RareAwarePatchDataset, collect_split, decode_mask
from model181_raw_zcdf_aspp import score


CLASS_WEIGHTS = torch.tensor([1.30, 1.35, 0.65, 2.10], dtype=torch.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_parameters(model: torch.nn.Module):
    prefixes = ("stem.", "enc1.", "enc2.", "enc3.", "enc4.")
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith(prefixes) else decoder).append(parameter)
    return encoder, decoder


def freeze_encoder_bn(model: torch.nn.Module) -> None:
    for name, module in model.named_modules():
        if name.startswith(("stem", "enc1", "enc2", "enc3", "enc4")) \
                and isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def boundary_map(target: torch.Tensor) -> torch.Tensor:
    value = target.float().unsqueeze(1)
    padded = F.pad(value, (1, 1, 1, 1), mode="replicate")
    neighbors = (padded[:, :, 1:-1, :-2], padded[:, :, 1:-1, 2:],
                 padded[:, :, :-2, 1:-1], padded[:, :, 2:, 1:-1])
    edge = torch.zeros_like(value)
    for neighbor in neighbors:
        edge = torch.maximum(edge, (neighbor != value).float())
    return edge[:, 0]


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, 4).permute(0, 3, 1, 2).float()
    intersection = (probability * one_hot).sum(dim=(2, 3))
    denominator = probability.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    present = one_hot.sum(dim=(2, 3)) > 0
    return (1.0 - dice)[present].mean()


def focal_tversky_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, 4).permute(0, 3, 1, 2).float()
    tp = (probability * one_hot).sum(dim=(2, 3))
    fp = (probability * (1.0 - one_hot)).sum(dim=(2, 3))
    fn = ((1.0 - probability) * one_hot).sum(dim=(2, 3))
    # beta > alpha emphasizes false negatives, which dominate small phases.
    tversky = (tp + 1.0) / (tp + 0.30 * fp + 0.70 * fn + 1.0)
    present = one_hot.sum(dim=(2, 3)) > 0
    return ((1.0 - tversky).clamp_min(0.0) ** 0.75)[present].mean()


def imagewise_soft_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, 4).permute(0, 3, 1, 2).float()
    intersection = (probability * one_hot).sum(dim=(2, 3))
    union = probability.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3)) - intersection
    iou = (intersection + 1.0) / (union + 1.0)
    present = one_hot.sum(dim=(2, 3)) > 0
    per_image = (1.0 - iou).masked_fill(~present, 0.0).sum(1) / present.sum(1).clamp_min(1)
    return per_image.mean()


def loss_fn(logits: torch.Tensor, target: torch.Tensor, boundary_weight: float) -> torch.Tensor:
    weights = CLASS_WEIGHTS.to(logits.device)
    pixel_ce = F.cross_entropy(logits, target, weight=weights, reduction="none")
    edge = boundary_map(target)
    ce = (pixel_ce * (1.0 + float(boundary_weight) * edge)).mean()
    return (ce + soft_dice_loss(logits, target)
            + 0.40 * focal_tversky_loss(logits, target)
            + 0.30 * imagewise_soft_iou_loss(logits, target))


@torch.no_grad()
def evaluate(model, samples, args, device, use_amp):
    model.eval()
    probabilities, targets = [], []
    for image_path, mask_path in _local_progress(samples, 'model209_rare_boundary_finetune'):
        probabilities.append(predict_probabilities(
            model, image_path, args.patch_size, args.val_stride, device,
            use_amp, args.tile_batch_size, "d4", "gaussian"))
        targets.append(decode_mask(Image.open(mask_path).convert("RGB")))
    predictions = [np.log(np.maximum(value, 1e-7)).argmax(0).astype(np.uint8)
                   for value in probabilities]
    return score(predictions, targets), probabilities


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    dataset = RareAwarePatchDataset(
        train_samples, args.patch_size, args.patches_per_image, True, args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0))
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"], strict=False)
    encoder, decoder = split_parameters(model)
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": args.encoder_lr},
        {"params": decoder, "lr": args.decoder_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history, best_score, best_state = [], -1.0, None
    started = time.time()
    print(f"device={device} patches={len(dataset)} batch={args.batch_size} "
          f"boundary_weight={args.boundary_weight}", flush=True)
    for epoch in _local_progress(range(1, args.epochs + 1), 'model209_rare_boundary_finetune'):
        model.train()
        freeze_encoder_bn(model)
        total, count = 0.0, 0
        t0 = time.time()
        for images, masks in _local_progress(loader, 'model209_rare_boundary_finetune'):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                value = loss_fn(model(images), masks, args.boundary_weight)
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
        scheduler.step()
        valid, _ = evaluate(model, valid_samples, args, device, use_amp)
        record = {"epoch": epoch, "train_loss": total / max(count, 1),
                  "valid": valid, "seconds": time.time() - t0,
                  "lr": [group["lr"] for group in optimizer.param_groups]}
        history.append(record)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"epoch={epoch} loss={record['train_loss']:.5f} "
              f"image={valid['image_mean_miou']:.6f} "
              f"exclude={valid['image_mean_miou_exclude']:.6f} "
              f"global={valid['image_miou_compat']:.6f} time={record['seconds']:.1f}s", flush=True)
        if valid[args.selection_metric] > best_score:
            best_score = valid[args.selection_metric]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            torch.save({"model": best_state, "epoch": epoch,
                        "metrics": valid, "args": vars(args)}, output / "best.pt")
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    valid, probabilities = evaluate(model, valid_samples, args, device, use_amp)
    np.savez_compressed(
        output / "valid_probs.npz",
        **{path.stem: value.astype(np.float16)
           for (path, _), value in zip(valid_samples, probabilities)})
    summary = {"best_selection": best_score, "final": valid,
               "epochs": len(history), "elapsed_seconds": time.time() - started,
               "loss": "boundary-weighted CE + SoftDice + focal Tversky + image-wise soft IoU",
               "class_weights": CLASS_WEIGHTS.tolist(),
               "checkpoint": str(args.checkpoint)}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_selection": best_score, "final": valid},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--val-stride", type=int, default=576)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--encoder-lr", type=float, default=1e-6)
    parser.add_argument("--decoder-lr", type=float, default=8e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--boundary-weight", type=float, default=0.75)
    parser.add_argument("--selection-metric", choices=("image_mean_miou", "image_mean_miou_exclude"),
                        default="image_mean_miou_exclude")
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
