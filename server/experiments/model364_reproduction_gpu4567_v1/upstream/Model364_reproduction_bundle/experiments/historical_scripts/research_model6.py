#!/usr/bin/env python3
"""Paper-guided experiments for KoMaP's best segmentation family.

This runner uses NASA MicroNet microscopy-pretrained encoders with the
official segmentation_models_pytorch U-Net++ implementation. It also
implements the staged freeze/unfreeze protocol, focal+Dice loss, a
boundary-distance term, Lovasz-Softmax, TTA, and image-level 4-fold
ensemble inference.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from model1_unet_resnet34 import (
    CLASS_COLORS,
    CLASS_NAMES,
    NUM_CLASSES,
    decode_mask,
    encode_mask,
    seed_everything,
)


MICRO_MEAN = np.asarray([0.4723, 0.4599, 0.4468], dtype=np.float32)
MICRO_STD = np.asarray([0.1684, 0.1575, 0.1675], dtype=np.float32)
CLASS_WEIGHTS = torch.tensor([0.982, 1.630, 1.884, 0.660], dtype=torch.float32)


def make_samples(data_dir: str | Path, splits: Sequence[str]) -> List[Tuple[Path, Path]]:
    result: List[Tuple[Path, Path]] = []
    for split in splits:
        image_dir = Path(data_dir) / split / "images"
        for image_path in sorted(image_dir.glob("*_image.png")):
            mask_path = Path(data_dir) / split / "masks" / image_path.name.replace("_image.png", "_mask.png")
            if mask_path.exists():
                result.append((image_path, mask_path))
    if not result:
        raise FileNotFoundError(f"No samples under {data_dir} for {splits}")
    return result


class MicroDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, Path]], image_size: int, augment: bool,
                 paper_aug: bool = True) -> None:
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.augment = augment
        self.paper_aug = paper_aug

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BICUBIC)
        mask = Image.open(mask_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST)
        image_np = np.asarray(image, dtype=np.uint8)
        mask_np = decode_mask(mask)

        if self.augment:
            if random.random() < 0.5:
                image_np = np.flip(image_np, axis=1).copy()
                mask_np = np.flip(mask_np, axis=1).copy()
            if random.random() < 0.5:
                image_np = np.flip(image_np, axis=0).copy()
                mask_np = np.flip(mask_np, axis=0).copy()
            quarter_turns = random.randrange(4)
            if quarter_turns:
                image_np = np.rot90(image_np, quarter_turns).copy()
                mask_np = np.rot90(mask_np, quarter_turns).copy()

            if self.paper_aug:
                # Zoom-in crops emulate the 1-2x scale augmentation used in
                # the metallography paper while preserving mask geometry.
                zoom = random.uniform(1.0, 2.0)
                crop_side = max(16, int(self.image_size / zoom))
                top = random.randint(0, self.image_size - crop_side)
                left = random.randint(0, self.image_size - crop_side)
                image_crop = image_np[top:top + crop_side, left:left + crop_side]
                mask_crop = mask_np[top:top + crop_side, left:left + crop_side]
                image_np = np.asarray(Image.fromarray(image_crop).resize(
                    (self.image_size, self.image_size), Image.Resampling.BICUBIC), dtype=np.uint8)
                mask_np = np.asarray(Image.fromarray(mask_crop.astype(np.int64), mode="I").resize(
                    (self.image_size, self.image_size), Image.Resampling.NEAREST), dtype=np.int64)
                if random.random() < 0.7:
                    shift = random.uniform(-0.05, 0.05) * 255.0
                    image_np = np.clip(image_np.astype(np.float32) + shift, 0, 255).astype(np.uint8)

        image_float = image_np.astype(np.float32) / 255.0
        image_float = (image_float - MICRO_MEAN) / MICRO_STD
        return (torch.from_numpy(image_float.transpose(2, 0, 1)).float(),
                torch.from_numpy(mask_np).long())


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    denominator = probabilities.sum(dims) + one_hot.sum(dims)
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = CLASS_WEIGHTS.to(logits.device)
    ce = F.cross_entropy(logits, target, weight=weights, reduction="none")
    pt = torch.exp(-ce).clamp_min(1e-6)
    return (((1.0 - pt) ** 2.0) * ce).mean()


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1.0)
    if p > 1:
        jaccard[1:p] -= jaccard[:-1].clone()
    return jaccard


def lovasz_softmax(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = []
    for class_id in range(NUM_CLASSES):
        foreground = (labels == class_id).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - probs[:, class_id]).abs().reshape(-1)
        errors_sorted, permutation = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(foreground.reshape(-1)[permutation])))
    return torch.stack(losses).mean() if losses else probs.sum() * 0.0


def signed_distance_batch(target: torch.Tensor) -> torch.Tensor:
    target_np = target.detach().cpu().numpy()
    output = np.zeros((target_np.shape[0], NUM_CLASSES, target_np.shape[1], target_np.shape[2]), dtype=np.float32)
    for batch_id in range(target_np.shape[0]):
        for class_id in range(NUM_CLASSES):
            positive = target_np[batch_id] == class_id
            negative = ~positive
            if positive.any() and negative.any():
                signed = distance_transform_edt(negative) - distance_transform_edt(positive)
                scale = max(float(np.abs(signed).max()), 1.0)
                output[batch_id, class_id] = signed / scale
    return torch.from_numpy(output).to(target.device)


def boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    distance = signed_distance_batch(target)
    return (probabilities * distance).mean()


def calculate_loss(logits: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "focal":
        regional = focal_loss(logits, target)
    else:
        regional = F.cross_entropy(logits, target)
    result = regional + dice_loss(logits, target)
    if mode == "boundary_lovasz":
        probabilities = torch.softmax(logits, dim=1)
        result = result + 0.20 * lovasz_softmax(probabilities, target) + 0.10 * boundary_loss(logits, target)
    return result


def build_model(encoder: str, weights_path: str, device: torch.device) -> torch.nn.Module:
    model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=None,
                             in_channels=3, classes=NUM_CLASSES, activation=None).to(device)
    weights = torch.load(weights_path, map_location="cpu")
    model.encoder.load_state_dict(weights, strict=True)
    return model


def tta_probabilities(model: torch.nn.Module, images: torch.Tensor, use_amp: bool) -> torch.Tensor:
    views = [
        (images, lambda x: x),
        (torch.flip(images, (-1,)), lambda x: torch.flip(x, (-1,))),
        (torch.flip(images, (-2,)), lambda x: torch.flip(x, (-2,))),
        (torch.rot90(images, 1, (-2, -1)), lambda x: torch.rot90(x, -1, (-2, -1))),
    ]
    total = None
    for view, undo in views:
        with autocast(enabled=use_amp):
            p = torch.softmax(model(view), dim=1)
        p = undo(p)
        total = p if total is None else total + p
    return total / len(views)


@torch.no_grad()
def evaluate(model, loader, device, use_amp, loss_mode, tta=False) -> Dict[str, object]:
    model.eval()
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    total_loss = 0.0
    count = 0
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = calculate_loss(logits, masks, loss_mode)
        probabilities = tta_probabilities(model, images, use_amp) if tta else torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(1).reshape(-1).cpu()
        targets = masks.reshape(-1).cpu()
        bins = targets * NUM_CLASSES + predictions
        confusion += torch.bincount(bins, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
        total_loss += float(loss.item()) * images.shape[0]
        count += images.shape[0]
    intersection = torch.diag(confusion).float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - intersection
    iou = (intersection / union.clamp_min(1.0)).tolist()
    return {"loss": total_loss / max(count, 1),
            "iou": {name: float(score) for name, score in zip(CLASS_NAMES, iou)},
            "miou": float(np.mean(iou)), "tta": tta, "confusion": confusion.tolist()}


def make_loaders(train_samples, valid_samples, args):
    train_ds = MicroDataset(train_samples, args.image_size, True, args.paper_aug)
    valid_ds = MicroDataset(valid_samples, args.image_size, False, False)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=True, persistent_workers=(args.num_workers > 0))
    return (DataLoader(train_ds, shuffle=True, drop_last=False, **common),
            DataLoader(valid_ds, shuffle=False, drop_last=False, **common))


def parameter_groups(model, encoder_lr: float, decoder_lr: float, freeze_encoder: bool = False):
    if freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        return [{"params": [p for p in model.decoder.parameters() if p.requires_grad] +
                           [p for p in model.segmentation_head.parameters() if p.requires_grad],
                 "lr": decoder_lr}]
    for p in model.encoder.parameters():
        p.requires_grad = True
    return [{"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": list(model.decoder.parameters()) + list(model.segmentation_head.parameters()),
             "lr": decoder_lr}]


def train_one(model, train_loader, valid_loader, args, output_dir: Path, staged: bool) -> Dict[str, object]:
    device = next(model.parameters()).device
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(enabled=use_amp)
    history = []
    best_miou = -1.0
    best_epoch = -1
    start = time.time()

    schedules = []
    if staged:
        schedules.append(("freeze", args.freeze_epochs, True, args.decoder_lr))
    schedules.append(("finetune", args.finetune_epochs if staged else args.epochs,
                      False, args.decoder_lr))
    global_epoch = 0
    for stage_name, stage_epochs, freeze, stage_decoder_lr in schedules:
        if stage_epochs <= 0:
            continue
        groups = parameter_groups(model, args.encoder_lr, stage_decoder_lr, freeze_encoder=freeze)
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs)
        for stage_epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            model.train()
            total_loss = 0.0
            count = 0
            epoch_start = time.time()
            for images, masks in train_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    loss = calculate_loss(model(images), masks, args.loss_mode)
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
                total_loss += float(loss.item()) * images.shape[0]
                count += images.shape[0]
            valid = evaluate(model, valid_loader, device, use_amp, args.loss_mode, tta=False)
            scheduler.step()
            record = {"epoch": global_epoch, "stage": stage_name,
                      "train_loss": total_loss / max(count, 1), "valid": valid,
                      "epoch_seconds": time.time() - epoch_start}
            history.append(record)
            print(f"epoch={global_epoch:03d} stage={stage_name} loss={record['train_loss']:.4f} "
                  f"miou={valid['miou']:.4f} time={record['epoch_seconds']:.1f}s", flush=True)
            (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            if float(valid["miou"]) > best_miou:
                best_miou = float(valid["miou"])
                best_epoch = global_epoch
                torch.save({"model": model.state_dict(), "epoch": global_epoch,
                            "metrics": valid, "args": vars(args)}, output_dir / "best.pt")

    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    normal = evaluate(model, valid_loader, device, use_amp, args.loss_mode, tta=False)
    tta = evaluate(model, valid_loader, device, use_amp, args.loss_mode, tta=True)
    summary = {"best_normal_miou": best_miou, "best_epoch": best_epoch,
               "normal": normal, "tta": tta, "elapsed_seconds": time.time() - start,
               "best_checkpoint": str(output_dir / "best.pt"), "loss_mode": args.loss_mode,
               "encoder": args.encoder, "staged": staged}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


@torch.no_grad()
def predict_test(model, data_dir: str, output_dir: Path, args, use_tta: bool = True):
    device = next(model.parameters()).device
    use_amp = device.type == "cuda" and not args.no_amp
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    test_images = sorted((Path(data_dir) / "test" / "images").glob("*_image.png"))
    for image_path in test_images:
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        resized = image.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
        image_np = np.asarray(resized, dtype=np.float32) / 255.0
        image_np = (image_np - MICRO_MEAN) / MICRO_STD
        tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        probabilities = tta_probabilities(model, tensor, use_amp) if use_tta else torch.softmax(model(tensor), dim=1)
        prediction = probabilities.argmax(1)[0].cpu().numpy().astype(np.uint8)
        encode_mask(prediction).resize(original_size, Image.Resampling.NEAREST).save(
            output_dir / image_path.name.replace("_image.png", "_mask.png"))
    print(f"wrote {len(test_images)} test masks to {output_dir}", flush=True)


def run_single(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_samples = make_samples(args.data_dir, ["train"])
    valid_samples = make_samples(args.data_dir, ["valid"])
    train_loader, valid_loader = make_loaders(train_samples, valid_samples, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args.encoder, args.weights_path, device)
    print(f"device={device} encoder={args.encoder} train={len(train_samples)} valid={len(valid_samples)} "
          f"loss={args.loss_mode} staged={args.staged}", flush=True)
    train_one(model, train_loader, valid_loader, args, output_dir, args.staged)
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    predict_test(model, args.data_dir, output_dir / "test_masks", args, use_tta=True)


def fold_splits(samples, seed: int):
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    folds = [indices[i::4] for i in range(4)]
    for fold in range(4):
        valid_ids = set(folds[fold])
        train = [samples[i] for i in indices if i not in valid_ids]
        valid = [samples[i] for i in folds[fold]]
        yield fold, train, valid


def run_four_fold(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_samples = make_samples(args.data_dir, ["train", "valid"])
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fold_summaries = []
    probability_sum = None
    image_names = sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png"))
    for fold, train_samples, valid_samples in fold_splits(all_samples, args.seed):
        print(f"FOLD {fold + 1}/4 train={len(train_samples)} valid={len(valid_samples)}", flush=True)
        fold_args = argparse.Namespace(**vars(args))
        fold_args.output_dir = str(root / f"fold{fold + 1}")
        fold_args.freeze_epochs = args.fold_freeze_epochs
        fold_args.finetune_epochs = args.fold_finetune_epochs
        fold_args.staged = True
        train_loader, valid_loader = make_loaders(train_samples, valid_samples, fold_args)
        out = Path(fold_args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model = build_model(args.encoder, args.weights_path, device)
        summary = train_one(model, train_loader, valid_loader, fold_args, out, staged=True)
        fold_summaries.append(summary)
        checkpoint = torch.load(out / "best.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        fold_probabilities = []
        use_amp = device.type == "cuda" and not args.no_amp
        for image_path in image_names:
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            resized = image.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
            image_np = np.asarray(resized, dtype=np.float32) / 255.0
            image_np = (image_np - MICRO_MEAN) / MICRO_STD
            tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            fold_probabilities.append(tta_probabilities(model, tensor, use_amp)[0].cpu().numpy())
        fold_probabilities = np.stack(fold_probabilities)
        probability_sum = fold_probabilities if probability_sum is None else probability_sum + fold_probabilities
        del model, train_loader, valid_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    probability_mean = probability_sum / 4.0
    test_output = root / "test_masks"
    test_output.mkdir(parents=True, exist_ok=True)
    for image_path, probabilities in zip(image_names, probability_mean):
        original_size = Image.open(image_path).size
        prediction = probabilities.argmax(axis=0).astype(np.uint8)
        encode_mask(prediction).resize(original_size, Image.Resampling.NEAREST).save(
            test_output / image_path.name.replace("_image.png", "_mask.png"))
    result = {"folds": fold_summaries,
              "mean_normal_miou": float(np.mean([x["normal"]["miou"] for x in fold_summaries])),
              "mean_tta_miou": float(np.mean([x["tta"]["miou"] for x in fold_summaries])),
              "test_masks": len(image_names), "test_output": str(test_output)}
    (root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", choices=["resnet34", "efficientnet-b4"], default="resnet34")
    parser.add_argument("--weights-path", required=True)
    parser.add_argument("--loss-mode", choices=["standard", "focal", "boundary_lovasz"], default="standard")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--four-fold", action="store_true")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--fold-freeze-epochs", type=int, default=4)
    parser.add_argument("--fold-finetune-epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paper-aug", action="store_true", default=True)
    parser.add_argument("--no-paper-aug", dest="paper_aug", action="store_false")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.four_fold:
        run_four_fold(args)
    else:
        run_single(args)
