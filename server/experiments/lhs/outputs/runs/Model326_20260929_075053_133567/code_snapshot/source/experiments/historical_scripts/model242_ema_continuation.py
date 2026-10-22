#!/usr/bin/env python3
"""Continue model239 with EMA 0.99 and compare raw/EMA checkpoints on valid."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
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


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
        # Copy BN running statistics and integer counters, which are already
        # population estimates rather than trainable parameters.
        for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def train_epoch(model, ema, loader, optimizer, scaler, device, use_amp, decay):
    model.train()
    total_loss, count = 0.0, 0
    weights = HARD_CLASS_WEIGHTS.to(device)
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, masks, weight=weights) + dice_loss(logits, masks)
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
        update_ema(ema, model, decay)
        total_loss += float(loss.detach()) * images.shape[0]
        count += images.shape[0]
    return total_loss / max(count, 1)


def metrics_only(record: dict) -> dict:
    return {key: value for key, value in record.items() if key != "confusion"}


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    model = Stride1NativeStemASPP().to(device)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(source["model"], strict=True)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)

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
    best_raw = {"score": -1.0, "state": None, "record": None}
    best_ema = {"score": -1.0, "state": None, "record": None}
    history, start_time = [], time.time()
    print(f"device={device} epochs={args.epochs} ema_decay={args.ema_decay} "
          f"steps_per_epoch={len(loader)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, ema, loader, optimizer, scaler,
                                 device, use_amp, args.ema_decay)
        raw_metrics = evaluate(model, valid_samples, args, device, use_amp)
        ema_metrics = evaluate(ema, valid_samples, args, device, use_amp)
        scheduler.step()
        record = {
            "epoch": epoch, "train_loss": train_loss,
            "raw": metrics_only(raw_metrics), "ema": metrics_only(ema_metrics),
            "seconds": time.time() - start_time,
        }
        history.append(record)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "epoch": epoch, "train_loss": train_loss,
            "raw_image_mean_miou": raw_metrics["image_mean_miou"],
            "ema_image_mean_miou": ema_metrics["image_mean_miou"],
            "raw_global_miou": raw_metrics["global_miou"],
            "ema_global_miou": ema_metrics["global_miou"],
        }, ensure_ascii=False), flush=True)

        if raw_metrics[args.selection_metric] > best_raw["score"]:
            best_raw = {"score": float(raw_metrics[args.selection_metric]),
                        "state": copy.deepcopy(model.state_dict()), "record": record}
            torch.save({"model": best_raw["state"], "epoch": epoch,
                        "metrics": raw_metrics, "args": vars(args),
                        "ema_decay": args.ema_decay, "weights": "raw"},
                       output / "best_raw.pt")
        if ema_metrics[args.selection_metric] > best_ema["score"]:
            best_ema = {"score": float(ema_metrics[args.selection_metric]),
                        "state": copy.deepcopy(ema.state_dict()), "record": record}
            torch.save({"model": best_ema["state"], "epoch": epoch,
                        "metrics": ema_metrics, "args": vars(args),
                        "ema_decay": args.ema_decay, "weights": "ema"},
                       output / "best_ema.pt")

    finals = {}
    for name, selected in (("raw", best_raw), ("ema", best_ema)):
        if selected["state"] is None:
            raise RuntimeError(f"no {name} checkpoint selected")
        model.load_state_dict(selected["state"], strict=True)
        result = evaluate(model, valid_samples, args, device, use_amp, output_probs=True)
        probability_list = result.pop("probabilities")
        np.savez_compressed(output / f"valid_probs_{name}.npz", **{
            path.stem: value.astype(np.float16)
            for (path, _), value in zip(valid_samples, probability_list)
        })
        finals[name] = {"best_selection": selected["score"],
                        "best_record": selected["record"], "final": result}

    summary = {"ema_decay": args.ema_decay, "selection_metric": args.selection_metric,
               "elapsed_seconds": time.time() - start_time, "results": finals}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {
        "image_mean_miou": value["final"]["image_mean_miou"],
        "global_miou": value["final"]["global_miou"],
        "image_mean_miou_exclude": value["final"]["image_mean_miou_exclude"],
    } for name, value in finals.items()}, ensure_ascii=False), flush=True)
    del loader, dataset, model, ema
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
    parser.add_argument("--ema-decay", type=float, default=0.99)
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
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
