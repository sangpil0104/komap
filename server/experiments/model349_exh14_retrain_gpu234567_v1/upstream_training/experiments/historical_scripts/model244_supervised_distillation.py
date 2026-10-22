#!/usr/bin/env python3
"""Train a model239 student with hard labels plus model242 EMA soft targets."""

from __future__ import annotations

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


def train_epoch(student, teacher, loader, optimizer, scaler, device, use_amp,
                temperature: float, distill_weight: float) -> float:
    student.train()
    teacher.eval()
    weights = HARD_CLASS_WEIGHTS.to(device)
    total_loss, count = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), autocast(enabled=use_amp):
            teacher_logits = teacher(images)
            teacher_prob = torch.softmax(teacher_logits / temperature, dim=1)
        with autocast(enabled=use_amp):
            student_logits = student(images)
            hard_loss = F.cross_entropy(student_logits, masks, weight=weights) + dice_loss(student_logits, masks)
            student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
            soft_loss = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(1).mean()
            loss = hard_loss + distill_weight * (temperature ** 2) * soft_loss
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.detach()) * images.shape[0]
        count += images.shape[0]
    return total_loss / max(count, 1)


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")

    student = Stride1NativeStemASPP().to(device)
    student_payload = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
    student.load_state_dict(student_payload["model"], strict=True)
    teacher = Stride1NativeStemASPP().to(device)
    teacher_payload = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher.load_state_dict(teacher_payload["model"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    encoder, decoder, native = [], [], []
    for name, parameter in student.named_parameters():
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
    print(f"device={device} distill_weight={args.distill_weight} "
          f"temperature={args.temperature} steps_per_epoch={len(loader)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(student, teacher, loader, optimizer, scaler, device,
                                 use_amp, args.temperature, args.distill_weight)
        valid = evaluate(student, valid_samples, args, device, use_amp)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss,
                  "selection": float(valid[args.selection_metric]),
                  "valid": valid, "seconds": time.time() - start_time}
        history.append(record)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"epoch": epoch, "train_loss": train_loss,
                          "selection": record["selection"],
                          "global_miou": valid["global_miou"],
                          "image_mean_miou": valid["image_mean_miou"],
                          "image_mean_miou_exclude": valid["image_mean_miou_exclude"]},
                         ensure_ascii=False), flush=True)
        if record["selection"] > best_score:
            best_score, best_state, best_record = (
                record["selection"], copy.deepcopy(student.state_dict()), record)
            torch.save({"model": best_state, "epoch": epoch, "metrics": valid,
                        "args": vars(args), "teacher": str(args.teacher_checkpoint)},
                       output / "best.pt")

    if best_state is None:
        raise RuntimeError("no validation checkpoint selected")
    student.load_state_dict(best_state, strict=True)
    final = evaluate(student, valid_samples, args, device, use_amp, output_probs=True)
    probability_list = final.pop("probabilities")
    np.savez_compressed(output / "valid_probs.npz", **{
        path.stem: value.astype(np.float16)
        for (path, _), value in zip(valid_samples, probability_list)
    })
    summary = {"method": "supervised CE+Dice with EMA-teacher temperature-scaled KL",
               "distill_weight": args.distill_weight, "temperature": args.temperature,
               "best_selection": best_score, "best_record": best_record,
               "final": final, "elapsed_seconds": time.time() - start_time}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_selection": best_score, "final": final},
                     ensure_ascii=False), flush=True)
    del loader, dataset, student, teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--distill-weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--encoder-lr", type=float, default=5e-6)
    parser.add_argument("--decoder-lr", type=float, default=1.5e-5)
    parser.add_argument("--native-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-stride", type=int, default=576)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--selection-metric", choices=("image_mean_miou",
                                                         "image_mean_miou_exclude"),
                        default="image_mean_miou")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=244)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
