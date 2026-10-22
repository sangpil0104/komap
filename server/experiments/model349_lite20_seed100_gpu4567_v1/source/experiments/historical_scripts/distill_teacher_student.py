#!/usr/bin/env python3
"""Teacher-ensemble distillation into a ResNet34 U-Net++ student.

Only train images are used to build teacher targets.  Validation remains a
strict evaluation split.  The student receives hard CE+Dice supervision and
temperature-scaled KL supervision from the current native-resolution teacher.
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

from calibrate_al import predict_probabilities, score_prediction
from advanced_inference import predict_probabilities as current_predict_probabilities
from aspp768_addition import MATRIX
from encoder_structure_search import build_model
from hard_negative_finetune import HARD_CLASS_WEIGHTS, load_model
from improve_five_causes import (
    CLASS_COLORS,
    NUM_CLASSES,
    collect_split,
    crop_bounds,
    decode_mask,
    encode_mask,
    load_pair,
    normalize_image,
    scaled_pair,
)
from train_models_2_to_6 import UNetPlusPlusResNet34
from prior_representation_search import fit_percentile_stats, percentile_image
from style_tta_search import predict_array, transform_image
from tent_adaptive_prior_search import fit_global_stats, gaussian_prior
from tent_style_search import adapt_model


TEACHER_MATRIX = np.asarray([
    [0.35, 0.40, 0.25], [0.25, 0.35, 0.40],
    [0.50, 0.30, 0.20], [0.25, 0.55, 0.20],
], dtype=np.float32)
TEACHER_BIAS = np.asarray([0.0, 0.0, 0.20, -0.55], dtype=np.float32)
COLORS = ((25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204))


def fit_stats(data_dir):
    values = [[] for _ in range(4)]
    for image_path, mask_path in collect_split(Path(data_dir), "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        mask = np.asarray(Image.open(mask_path).convert("RGB"))
        ids = np.full(mask.shape[:2], -1, dtype=np.int8)
        for class_id, color in enumerate(COLORS):
            ids[np.all(mask == color, axis=2)] = class_id
        rng = np.random.default_rng(sum(image_path.name.encode("utf-8")))
        for class_id in range(4):
            current = z[ids == class_id]
            if len(current):
                take = rng.choice(len(current), min(len(current), 10000), replace=False)
                values[class_id].append(current[take])
    return [(float(np.concatenate(x).mean()), max(float(np.concatenate(x).std()), 0.15)) for x in values]


def build_teacher_cache(args, cache_dir, device, use_amp):
    cache_dir.mkdir(parents=True, exist_ok=True)
    old = load_model(args.teacher_old, device)
    new = load_model(args.teacher_new, device)
    aspp = build_model("aspp", pretrained=False, checkpoint=args.teacher_aspp_base).to(device)
    aspp.load_state_dict(torch.load(args.teacher_aspp, map_location="cpu")["model"])
    aspp.eval()
    stats = fit_stats(args.data_dir)
    for image_path, _ in collect_split(args.data_dir, "train"):
        output_path = cache_dir / f"{image_path.stem}.npy"
        if output_path.exists():
            continue
        old_p = predict_probabilities(old, image_path, 640, 512, device, use_amp, 2, True)
        new_p = predict_probabilities(new, image_path, 512, 384, device, use_amp, 2, True)
        aspp_p = predict_probabilities(aspp, image_path, 512, 384, device, use_amp, 2, True)
        mixed = np.asarray([
            sum(TEACHER_MATRIX[class_id, model_id] * values[class_id]
                for model_id, values in enumerate((old_p, new_p, aspp_p)))
            for class_id in range(NUM_CLASSES)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
        prior = np.asarray([-0.5 * ((z - mean) / std) ** 2 - np.log(std)
                            for mean, std in stats], dtype=np.float32)
        logits = np.log(np.maximum(mixed, 1e-7)) + 0.09 * prior
        logits[3] += 0.09 * (0.75 - 1.0) * prior[3]
        logits += TEACHER_BIAS[:, None, None]
        np.save(output_path, logits.astype(np.float16))
        print(f"teacher cached {image_path.name}", flush=True)
    return stats


def build_current_teacher_cache(args, cache_dir, device, use_amp):
    """Cache soft targets from the current best label-free route.

    This is intentionally separate from the historical model-101 cache.  It
    uses the current high-resolution TENT branch, pairwise branch, conditional
    intensity prior, and the model-162 branch weights, but does not apply hard
    connected-component cleanup to the soft targets.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    old = load_model(args.teacher_old, device)
    new = load_model(args.teacher_new, device)
    aspp = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    aspp.load_state_dict(torch.load(args.teacher_aspp, map_location="cpu")["model"])
    aspp.eval()
    pair = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    pair.load_state_dict(torch.load(args.teacher_pair, map_location="cpu")["model"])
    pair.eval()
    train_stats = []
    for image_path, _ in collect_split(args.data_dir, "train"):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        train_stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in train_stats]))
    train_std = float(np.mean([x[1] for x in train_stats]))
    z_stats = fit_global_stats(args.data_dir)
    q_stats = fit_percentile_stats(args.data_dir)
    for image_path, _ in collect_split(args.data_dir, "train"):
        output_path = cache_dir / f"{image_path.stem}.npy"
        if output_path.exists():
            continue
        raw = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        p0 = current_predict_probabilities(old, image_path, 640, 512, device, use_amp, 2, "d4", "gaussian")
        p1 = current_predict_probabilities(new, image_path, 512, 384, device, use_amp, 2, "d4", "gaussian")
        p2 = current_predict_probabilities(aspp, image_path, 512, 384, device, use_amp, 2, "d4", "gaussian")
        pair_value = current_predict_probabilities(pair, image_path, 768, 576, device, use_amp, 1, "d4", "gaussian")
        adapted = adapt_model(args.teacher_high, raw, device, use_amp, 5, 1e-5)
        high_raw = predict_array(adapted, raw, 768, 576, device, use_amp, 1, "d4", "gaussian")
        high_style = predict_array(
            adapted, transform_image(raw, "standardize", train_mean, train_std),
            768, 576, device, use_amp, 1, "d4", "gaussian"
        )
        gray = raw[..., 0].astype(np.float32)
        mean = float(gray.mean())
        std = float(gray.std())
        high_weight = .85 if mean < 165.0 else .55
        high = .25 * high_raw + .75 * high_style if std >= 50.0 else high_raw
        base = MATRIX * (1.0 - high_weight - .10)
        mixed = np.asarray([
            sum(base[c, m] * (p0, p1, p2)[m][c] for m in range(3))
            + high_weight * high[c] + .10 * pair_value[c]
            for c in range(4)
        ], dtype=np.float32)
        mixed /= np.maximum(mixed.sum(axis=0, keepdims=True), 1e-7)
        z = (gray - mean) / max(std, 1e-5)
        q = percentile_image(gray)
        adaptive_z, adaptive_q = [], []
        for class_id in range(4):
            weights = np.maximum(mixed[class_id], 1e-7) ** 2.0
            zm = float((weights * z).sum() / weights.sum())
            zs = float(np.sqrt((weights * (z - zm) ** 2).sum() / weights.sum()))
            qm = float((weights * q).sum() / weights.sum())
            qs = float(np.sqrt((weights * (q - qm) ** 2).sum() / weights.sum()))
            adaptive_z.append((zm, max(zs, .15)))
            adaptive_q.append((qm, max(qs, .02)))
        prior_z = gaussian_prior(z, z_stats)
        prior_q = gaussian_prior(q, q_stats)
        adaptive_z_value = gaussian_prior(z, adaptive_z)
        adaptive_q_value = gaussian_prior(q, adaptive_q)
        prior = .85 * prior_z + .15 * prior_q
        delta = .85 * (adaptive_z_value[0] - prior_z[0]) + .15 * (adaptive_q_value[0] - prior_q[0])
        prior[0] += (3.0 if mean < 160.0 else 7.0) * delta
        logits = np.log(np.maximum(mixed, 1e-7)) + .09 * prior
        logits[2] += .45
        logits[3] += -.3 if mean >= 165.0 else -1.0
        np.save(output_path, logits.astype(np.float16))
        print(f"current teacher cached {image_path.name}", flush=True)
        del adapted, high_raw, high_style
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return z_stats


class DistillDataset(Dataset):
    def __init__(self, samples, cache_dir, patch_size, patches_per_image, seed):
        self.samples = list(samples)
        self.cache_dir = Path(cache_dir)
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.seed = int(seed)

    def __len__(self):
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index % len(self.samples)]
        image, mask = load_pair(image_path, mask_path)
        image, mask = scaled_pair(image, mask, self.patch_size)
        teacher = np.asarray(np.load(self.cache_dir / f"{image_path.stem}.npy"), dtype=np.float32)
        height, width = mask.shape
        if teacher.shape[1:] != (height, width):
            teacher = np.stack([
                np.asarray(Image.fromarray(teacher[class_id]).resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
                for class_id in range(NUM_CLASSES)
            ])
        rng = random.Random(self.seed + index * 1009 + random.randrange(1_000_000))
        draw = rng.random()
        cumulative = 0.0
        focus_class = None
        for class_id, probability in ((3, 0.45), (1, 0.25), (0, 0.20), (2, 0.10)):
            cumulative += probability
            if draw <= cumulative:
                focus_class = class_id
                break
        ys, xs = np.where(mask == focus_class)
        if len(xs) == 0:
            center_y, center_x = height // 2, width // 2
        else:
            chosen = rng.randrange(len(xs))
            center_y, center_x = int(ys[chosen]), int(xs[chosen])
        top, left = crop_bounds(height, width, self.patch_size, center_y, center_x)
        image, mask = image[top:top + self.patch_size, left:left + self.patch_size], mask[top:top + self.patch_size, left:left + self.patch_size]
        teacher = teacher[:, top:top + self.patch_size, left:left + self.patch_size]
        if rng.random() < 0.5:
            image, mask, teacher = np.flip(image, 1).copy(), np.flip(mask, 1).copy(), np.flip(teacher, 2).copy()
        if rng.random() < 0.5:
            image, mask, teacher = np.flip(image, 0).copy(), np.flip(mask, 0).copy(), np.flip(teacher, 1).copy()
        if rng.random() < 0.5:
            k = rng.randrange(4)
            image, mask, teacher = np.rot90(image, k).copy(), np.rot90(mask, k).copy(), np.rot90(teacher, k, axes=(1, 2)).copy()
        if rng.random() < 0.45:
            contrast = rng.uniform(0.94, 1.06)
            brightness = rng.uniform(-0.025, 0.025) * 255.0
            image = np.clip((image.astype(np.float32) - 127.5) * contrast + 127.5 + brightness,
                            0.0, 255.0).astype(np.uint8)
        return normalize_image(image), torch.from_numpy(mask).long(), torch.from_numpy(teacher).float()


def dice_loss(logits, target):
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    intersection = (probs * one_hot).sum((0, 2, 3))
    denominator = probs.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def evaluate(model, samples, args, device, use_amp):
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    image_scores = []
    for image_path, mask_path in _local_progress(samples, 'distill_teacher_student'):
        probability = predict_probabilities(model, image_path, args.patch_size, args.val_stride,
                                             device, use_amp, args.tile_batch_size, True)
        logits = np.log(np.maximum(probability, 1e-7))
        logits[3] += args.eval_eutectic_bias
        prediction = logits.argmax(axis=0).astype(np.uint8)
        target = decode_mask(Image.open(mask_path).convert("RGB"))
        bins = target.reshape(-1) * NUM_CLASSES + prediction.reshape(-1)
        current = np.bincount(bins, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
        confusion += current
        image_scores.append(score_prediction(prediction, target))
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = intersection / np.maximum(union, 1.0)
    return {"image_miou_compat": float(iou.mean()), "image_mean_miou": float(np.mean(image_scores)),
            "image_class_iou": {str(i): float(v) for i, v in enumerate(iou)}, "confusion": confusion.tolist()}


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "teacher_cache"
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    if args.teacher_mode == "current":
        build_current_teacher_cache(args, cache_dir, device, use_amp)
    else:
        build_teacher_cache(args, cache_dir, device, use_amp)
    dataset = DistillDataset(train_samples, cache_dir, args.patch_size, args.patches_per_image, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    model = load_model(args.student_checkpoint, device)
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith(("stem.", "enc1.", "enc2.", "enc3.", "enc4.")) else decoder).append(parameter)
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": args.encoder_lr},
                                   {"params": decoder, "lr": args.decoder_lr}], weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)
    best = -1.0
    history = []
    start = time.time()
    for epoch in _local_progress(range(1, args.epochs + 1), 'distill_teacher_student'):
        model.train()
        total, count = 0.0, 0
        for images, masks, teacher_logits in _local_progress(loader, 'distill_teacher_student'):
            images, masks, teacher_logits = images.to(device), masks.to(device), teacher_logits.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                student_logits = model(images)
                hard = F.cross_entropy(student_logits, masks, weight=HARD_CLASS_WEIGHTS.to(device)) + dice_loss(student_logits, masks)
                temperature = args.temperature
                teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
                # `batchmean` sums all H*W pixels and would make the KL term
                # thousands of times larger than the hard segmentation loss.
                distill = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                                   teacher_soft, reduction="none").sum(dim=1).mean() * (temperature ** 2)
                loss = hard + args.distill_weight * distill
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
        valid = evaluate(model, valid_samples, args, device, use_amp)
        record = {"epoch": epoch, "train_loss": total / max(count, 1), "valid": valid}
        history.append(record)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": valid, "args": vars(args)}, output_dir / "last.pt")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if valid["image_mean_miou"] > best:
            best = valid["image_mean_miou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": valid, "args": vars(args)}, output_dir / "best.pt")
    (output_dir / "summary.json").write_text(json.dumps({"best_image_mean_miou": best, "elapsed_seconds": time.time() - start}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_image_mean_miou": best, "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--teacher-old", required=True)
    parser.add_argument("--teacher-new", required=True)
    parser.add_argument("--teacher-aspp", required=True)
    parser.add_argument("--teacher-aspp-base", required=True)
    parser.add_argument("--teacher-pair")
    parser.add_argument("--teacher-high")
    parser.add_argument("--teacher-mode", choices=("historical", "current"), default="historical")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--val-stride", type=int, default=384)
    parser.add_argument("--tile-batch-size", type=int, default=4)
    parser.add_argument("--patches-per-image", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-6)
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--eval-eutectic-bias", type=float, default=-0.55)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
