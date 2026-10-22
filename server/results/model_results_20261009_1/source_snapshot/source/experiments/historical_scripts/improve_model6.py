#!/usr/bin/env python3
"""Fine-tuning and inference improvements for the best KoMaP model.

The starting point is model 6: ResNet34 + compact U-Net++ decoder.
Experiments keep the existing train/valid split so that changes are
comparable with the recorded 74.44% validation mIoU.

Supported improvements:
  - discriminative low learning-rate fine-tuning from model 6's best.pt
  - class-balanced Cross Entropy + Dice loss
  - mild microscopy-safe brightness/contrast augmentation
  - horizontal/vertical/90-degree TTA for validation and test inference
"""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from model1_unet_resnet34 import (
    CLASS_COLORS,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    decode_mask,
    encode_mask,
    seed_everything,
)
from train_models_2_to_6 import UNetPlusPlusResNet34


# Inverse-square-root frequency weights from the 70 training masks. They are
# deliberately milder than inverse-frequency weights to avoid suppressing Al.
CLASS_WEIGHTS = torch.tensor([0.982, 1.630, 1.884, 0.660], dtype=torch.float32)


class FineTuneDataset(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int, aug_mode: str = "base") -> None:
        self.root = Path(root) / split
        self.image_size = int(image_size)
        self.aug_mode = aug_mode
        self.image_paths = sorted((self.root / "images").glob("*_image.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {self.root / 'images'}")
        self.mask_paths = [
            self.root / "masks" / path.name.replace("_image.png", "_mask.png")
            for path in self.image_paths
        ]
        for image_path, mask_path in zip(self.image_paths, self.mask_paths):
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image_np = np.asarray(image, dtype=np.uint8)
        mask_np = decode_mask(mask)

        if self.aug_mode != "none":
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

            if self.aug_mode == "safe_plus":
                # Mild photometric noise only; it does not alter mask geometry.
                if random.random() < 0.5:
                    contrast = random.uniform(0.90, 1.10)
                    brightness = random.uniform(-0.04, 0.04) * 255.0
                    image_np = np.clip((image_np.astype(np.float32) - 127.5) * contrast + 127.5 + brightness,
                                       0.0, 255.0).astype(np.uint8)
                if random.random() < 0.20:
                    noise = np.random.normal(0.0, 2.5, image_np.shape).astype(np.float32)
                    image_np = np.clip(image_np.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

        image_float = image_np.astype(np.float32) / 255.0
        image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image_float.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_np).long()
        return image_tensor, mask_tensor


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * target_one_hot).sum(dims)
    denominator = probs.sum(dims) + target_one_hot.sum(dims)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


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


def boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    target_np = target.detach().cpu().numpy()
    distances = np.zeros((target_np.shape[0], NUM_CLASSES, target_np.shape[1], target_np.shape[2]), dtype=np.float32)
    for batch_id in range(target_np.shape[0]):
        for class_id in range(NUM_CLASSES):
            positive = target_np[batch_id] == class_id
            negative = ~positive
            if positive.any() and negative.any():
                signed = distance_transform_edt(negative) - distance_transform_edt(positive)
                distances[batch_id, class_id] = signed / max(float(np.abs(signed).max()), 1.0)
    distance = torch.from_numpy(distances).to(target.device)
    return (probabilities * distance).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "weighted":
        regional = F.cross_entropy(logits, target, weight=CLASS_WEIGHTS.to(logits.device))
    elif mode == "focal":
        regional = focal_loss(logits, target)
    else:
        regional = F.cross_entropy(logits, target)
    result = regional + dice_loss(logits, target)
    if mode == "boundary_lovasz":
        result = result + 0.20 * lovasz_softmax(torch.softmax(logits, dim=1), target) + 0.10 * boundary_loss(logits, target)
    return result


def tta_logits(model: torch.nn.Module, images: torch.Tensor, use_amp: bool) -> torch.Tensor:
    """Average probabilities from geometry-preserving flip/rotation views."""
    views = [
        (images, lambda x: x),
        (torch.flip(images, dims=(-1,)), lambda x: torch.flip(x, dims=(-1,))),
        (torch.flip(images, dims=(-2,)), lambda x: torch.flip(x, dims=(-2,))),
        (torch.rot90(images, 1, dims=(-2, -1)), lambda x: torch.rot90(x, -1, dims=(-2, -1))),
    ]
    probability_sum = None
    for view, undo in views:
        with autocast(enabled=use_amp):
            probabilities = torch.softmax(model(view), dim=1)
        probabilities = undo(probabilities)
        probability_sum = probabilities if probability_sum is None else probability_sum + probabilities
    return probability_sum / float(len(views))


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    loss_mode: str,
    use_tta: bool = False,
) -> Dict[str, object]:
    from local_image_metrics import summary
    model.eval()
    predictions, targets = [], []
    total_loss = count = 0
    for images, masks in _local_progress(loader, 'improve_model6'):
        images, masks = images.to(device), masks.to(device)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = segmentation_loss(logits, masks, loss_mode)
        probs = tta_logits(model, images, use_amp) if use_tta else torch.softmax(logits, 1)
        predictions.extend(probs.argmax(1).cpu().numpy())
        targets.extend(masks.cpu().numpy())
        total_loss += float(loss) * images.shape[0]
        count += images.shape[0]
    result = summary(predictions, targets, CLASS_NAMES)
    result.update(loss=total_loss / max(count, 1), tta=use_tta)
    return result


def load_model(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    model = UNetPlusPlusResNet34(pretrained=False).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    return model


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer, scheduler, epoch: int,
                    metrics: Dict[str, object], args: argparse.Namespace) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "class_names": CLASS_NAMES,
            "class_colors": CLASS_COLORS,
        },
        path,
    )


def split_parameters(model: torch.nn.Module):
    encoder_prefixes = ("stem.", "enc1.", "enc2.", "enc3.", "enc4.")
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith(encoder_prefixes) else decoder).append(parameter)
    return encoder, decoder


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_ds = FineTuneDataset(args.data_dir, "train", args.image_size, args.aug_mode)
    valid_ds = FineTuneDataset(args.data_dir, "valid", args.image_size, "none")
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    valid_loader = DataLoader(valid_ds, shuffle=False, drop_last=False, **common)
    model = load_model(args.checkpoint, device)
    scaler = GradScaler(enabled=use_amp)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, object]] = []
    best_miou = -1.0
    stale = 0
    start = time.time()
    print(f"device={device} train={len(train_ds)} valid={len(valid_ds)} loss={args.loss_mode} aug={args.aug_mode}", flush=True)
    print(f"encoder_lr={args.encoder_lr} decoder_lr={args.decoder_lr} tta_selection={args.select_tta}", flush=True)

    schedules = []
    if args.staged:
        schedules.append(("freeze", args.freeze_epochs, True))
        schedules.append(("finetune", args.finetune_epochs, False))
    else:
        schedules.append(("finetune", args.epochs, False))

    global_epoch = 0
    for stage_name, stage_epochs, freeze_encoder in schedules:
        encoder, decoder = split_parameters(model)
        if freeze_encoder:
            for parameter in encoder:
                parameter.requires_grad = False
            optimizer = torch.optim.AdamW([{"params": decoder, "lr": args.decoder_lr}],
                                          weight_decay=args.weight_decay)
        else:
            for parameter in encoder:
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(
                [{"params": encoder, "lr": args.encoder_lr}, {"params": decoder, "lr": args.decoder_lr}],
                weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(stage_epochs, 1))
        stale = 0
        for _ in _local_progress(range(stage_epochs), 'improve_model6'):
            global_epoch += 1
            model.train()
            total_loss = 0.0
            count = 0
            epoch_start = time.time()
            for images, masks in _local_progress(train_loader, 'improve_model6'):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    loss = segmentation_loss(model(images), masks, args.loss_mode)
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
            valid = evaluate_model(model, valid_loader, device, use_amp, args.loss_mode, use_tta=args.select_tta)
            scheduler.step()
            record = {"epoch": global_epoch, "stage": stage_name,
                      "train_loss": total_loss / max(count, 1), "valid": valid,
                      "lr_encoder": optimizer.param_groups[0]["lr"],
                      "lr_decoder": optimizer.param_groups[-1]["lr"],
                      "epoch_seconds": time.time() - epoch_start}
            history.append(record)
            (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, global_epoch, valid, args)
            miou = float(valid["miou"])
            print(f"epoch={global_epoch:03d} stage={stage_name} train_loss={record['train_loss']:.4f} "
                  f"miou={miou:.4f} iou=" + ", ".join(f"{v:.3f}" for v in valid["iou"].values()) +
                  f" time={record['epoch_seconds']:.1f}s", flush=True)
            if miou > best_miou:
                best_miou = miou
                stale = 0
                save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, global_epoch, valid, args)
            else:
                stale += 1
                if stage_name == "finetune" and stale >= args.patience:
                    print(f"early stopping after {args.patience} stale epochs", flush=True)
                    break
    summary = {"best_miou": best_miou, "epochs_completed": len(history),
               "elapsed_seconds": time.time() - start, "best_checkpoint": str(out_dir / "best.pt"),
               "loss_mode": args.loss_mode, "aug_mode": args.aug_mode,
               "class_weights": CLASS_WEIGHTS.tolist(), "select_tta": args.select_tta,
               "staged": args.staged}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


@torch.no_grad()
def evaluate_checkpoint(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    valid_ds = FineTuneDataset(args.data_dir, "valid", args.image_size, "none")
    loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    model = load_model(args.checkpoint, device)
    normal = evaluate_model(model, loader, device, use_amp, "standard", use_tta=False)
    tta = evaluate_model(model, loader, device, use_amp, "standard", use_tta=True)
    print(json.dumps({"normal": normal, "tta": tta}, ensure_ascii=False), flush=True)


@torch.no_grad()
def predict_test(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    model = load_model(args.checkpoint, device)
    model.eval()
    test_dir = Path(args.data_dir) / "test" / "images"
    output_dir = Path(args.output_dir) / "test_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(test_dir.glob("*_image.png")):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        resized = image.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
        image_np = np.asarray(resized, dtype=np.float32) / 255.0
        image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        probabilities = tta_logits(model, tensor, use_amp) if args.tta else None
        if probabilities is None:
            with autocast(enabled=use_amp):
                probabilities = torch.softmax(model(tensor), dim=1)
        prediction = probabilities.argmax(1)[0].cpu().numpy().astype(np.uint8)
        encode_mask(prediction).resize(original_size, Image.Resampling.NEAREST).save(
            output_dir / image_path.name.replace("_image.png", "_mask.png"))
    print(f"wrote {len(list(output_dir.glob('*.png')))} test masks to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--select-tta", action="store_true")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-mode", choices=["standard", "weighted", "focal", "boundary_lovasz"], default="standard")
    parser.add_argument("--aug-mode", choices=["base", "safe_plus"], default="base")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        evaluate_checkpoint(args)
    elif args.predict_test:
        predict_test(args)
    else:
        train(args)
