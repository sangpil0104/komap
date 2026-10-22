#!/usr/bin/env python3
"""Unlabeled test-domain adaptation with teacher pseudo-label distillation.

Only test images are used as unlabeled target-domain inputs.  A frozen ASPP
teacher creates high-confidence pseudo labels on fixed native crops; a student
initialized from the teacher is adapted on contrast/brightness views while
the encoder and its BN statistics remain frozen.
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
from torch.utils.data import DataLoader, Dataset

from advanced_inference import predict_probabilities
from encoder_structure_search import build_model
from improve_five_causes import collect_split, crop_bounds, decode_mask, normalize_image
from model181_raw_zcdf_aspp import score


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


def make_style(image: np.ndarray, rng: random.Random) -> np.ndarray:
    contrast = rng.uniform(0.94, 1.06)
    brightness = rng.uniform(-0.025, 0.025) * 255.0
    return np.clip((image.astype(np.float32) - 127.5) * contrast
                   + 127.5 + brightness, 0.0, 255.0).astype(np.uint8)


class TargetPatchDataset(Dataset):
    def __init__(self, image_paths, patch_size: int, patches_per_image: int, seed: int):
        self.items = []
        for image_path in image_paths:
            image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            height, width = image.shape[:2]
            if min(height, width) < patch_size:
                scale = float(patch_size) / float(min(height, width))
                new_size = (max(patch_size, int(round(width * scale))),
                            max(patch_size, int(round(height * scale))))
                image = np.asarray(Image.fromarray(image).resize(
                    new_size, Image.Resampling.BICUBIC), dtype=np.uint8)
                height, width = image.shape[:2]
            # Deterministic coverage of the unlabeled target image.
            grid_y = np.linspace(0.18, 0.82, 2)
            grid_x = np.linspace(0.12, 0.88, max(1, patches_per_image // 2))
            for index in range(patches_per_image):
                y_fraction = float(grid_y[index // len(grid_x) % len(grid_y)])
                x_fraction = float(grid_x[index % len(grid_x)])
                center_y = int(round(y_fraction * max(height - 1, 1)))
                center_x = int(round(x_fraction * max(width - 1, 1)))
                top, left = crop_bounds(height, width, patch_size, center_y, center_x)
                crop = image[top:top + patch_size, left:left + patch_size]
                rng = random.Random(seed + len(self.items) * 37)
                style = make_style(crop, rng)
                self.items.append((normalize_image(crop), normalize_image(style)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        base, style = self.items[index]
        return base, style, index


class PseudoDataset(Dataset):
    def __init__(self, target: TargetPatchDataset, pseudo, confidence):
        self.target = target
        self.pseudo = pseudo
        self.confidence = confidence

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        _, style, _ = self.target[index]
        return style, self.pseudo[index], self.confidence[index]


@torch.no_grad()
def make_pseudo_labels(model, target, batch_size, device, use_amp):
    loader = DataLoader(target, batch_size=batch_size, shuffle=False, num_workers=0)
    pseudo, confidence = [None] * len(target), [None] * len(target)
    model.eval()
    for base, _, indices in _local_progress(loader, 'model211_test_time_distill'):
        base = base.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            probability = torch.softmax(model(base), dim=1)
        conf, label = probability.max(dim=1)
        for row, index in enumerate(indices.tolist()):
            pseudo[index] = label[row].cpu().to(torch.uint8)
            confidence[index] = conf[row].cpu().to(torch.float16)
    return pseudo, confidence


@torch.no_grad()
def evaluate(model, samples, args, device, use_amp):
    model.eval()
    probabilities, targets = [], []
    for image_path, mask_path in _local_progress(samples, 'model211_test_time_distill'):
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
    test_images = sorted((Path(args.data_dir) / "test" / "images").glob("*_image.png"))
    if not test_images:
        raise FileNotFoundError("no test images found")
    target = TargetPatchDataset(
        test_images, args.patch_size,
        args.patches_per_image, args.seed)
    teacher = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    teacher.load_state_dict(payload["model"], strict=False)
    pseudo, confidence = make_pseudo_labels(
        teacher, target, args.batch_size, device, use_amp)
    student = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    student.load_state_dict(payload["model"], strict=False)
    encoder, decoder = split_parameters(student)
    for parameter in encoder:
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(decoder, lr=args.decoder_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(PseudoDataset(target, pseudo, confidence), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=(args.num_workers > 0))
    scaler = GradScaler(enabled=use_amp)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history, best_score, best_state = [], -1.0, None
    started = time.time()
    print(f"device={device} target_patches={len(target)} threshold={args.confidence_threshold}", flush=True)
    for epoch in _local_progress(range(1, args.epochs + 1), 'model211_test_time_distill'):
        student.train()
        freeze_encoder_bn(student)
        total, count = 0.0, 0
        t0 = time.time()
        for images, labels, confidence_batch in _local_progress(loader, 'model211_test_time_distill'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            confidence_batch = confidence_batch.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = student(images)
                pixel_loss = F.cross_entropy(logits, labels, reduction="none")
                mask = confidence_batch >= args.confidence_threshold
                if mask.any():
                    pseudo_ce = pixel_loss[mask].mean()
                else:
                    pseudo_ce = pixel_loss.mean()
                probability = torch.softmax(logits, dim=1)
                entropy = -(probability * torch.log(probability.clamp_min(1e-7))).sum(1).mean()
                value = pseudo_ce + args.entropy_weight * entropy
            if use_amp:
                scaler.scale(value).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(decoder, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                value.backward()
                torch.nn.utils.clip_grad_norm_(decoder, 1.0)
                optimizer.step()
            total += float(value.item()) * images.shape[0]
            count += images.shape[0]
        scheduler.step()
        valid, _ = evaluate(student, valid_samples, args, device, use_amp)
        record = {"epoch": epoch, "loss": total / max(count, 1), "valid": valid,
                  "seconds": time.time() - t0}
        history.append(record)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"epoch={epoch} loss={record['loss']:.5f} "
              f"image={valid['image_mean_miou']:.6f} "
              f"exclude={valid['image_mean_miou_exclude']:.6f} "
              f"image_compat={valid['image_miou_compat']:.6f} time={record['seconds']:.1f}s", flush=True)
        if valid[args.selection_metric] > best_score:
            best_score = valid[args.selection_metric]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in student.state_dict().items()}
            torch.save({"model": best_state, "epoch": epoch,
                        "metrics": valid, "args": vars(args)}, output / "best.pt")
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    student.load_state_dict(best_state)
    valid, probabilities = evaluate(student, valid_samples, args, device, use_amp)
    np.savez_compressed(output / "valid_probs.npz",
                        **{path.stem: value.astype(np.float16)
                           for (path, _), value in zip(valid_samples, probabilities)})
    summary = {"best_selection": best_score, "final": valid,
               "epochs": len(history), "elapsed_seconds": time.time() - started,
               "method": "test-domain high-confidence pseudo-label distillation + entropy",
               "target_images": len(test_images), "target_patches": len(target),
               "checkpoint": str(args.checkpoint)}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_selection": best_score, "final": valid}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--val-stride", type=int, default=576)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    parser.add_argument("--decoder-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--confidence-threshold", type=float, default=0.72)
    parser.add_argument("--entropy-weight", type=float, default=0.02)
    parser.add_argument("--selection-metric", choices=("image_mean_miou", "image_mean_miou_exclude"),
                        default="image_mean_miou_exclude")
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
