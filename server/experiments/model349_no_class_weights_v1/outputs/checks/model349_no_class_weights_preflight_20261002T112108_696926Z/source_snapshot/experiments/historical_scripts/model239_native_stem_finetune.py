#!/usr/bin/env python3
"""Fine-tune model195 with a zero-init stride-1, full-resolution feature stem."""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

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
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from improve_five_causes import collect_split
from model181_raw_zcdf_aspp import (
    RepresentationPatchDataset,
    TeamRepresentationASPP,
    evaluate,
    seed_everything,
    train_epoch,
)


class Stride1NativeStemASPP(TeamRepresentationASPP):
    """Keep the pretrained path intact; inject full-resolution features at the head."""

    def __init__(self) -> None:
        super().__init__(pretrained=False)
        self.native_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.native_to_head = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(self.native_to_head.weight)

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
        return self.head[1](hidden)


def load_compatible(model: nn.Module, checkpoint: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    # Only the two intentionally new modules may be missing.
    allowed_prefixes = ("native_stem.", "native_to_head.")
    unexpected = [key for key in incompatible.missing_keys
                  if not key.startswith(allowed_prefixes)]
    if unexpected or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={unexpected}, "
                           f"unexpected={incompatible.unexpected_keys}")
    print(f"loaded={checkpoint} new_modules={len(incompatible.missing_keys)}", flush=True)


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    model = Stride1NativeStemASPP().to(device)
    load_compatible(model, args.checkpoint)

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

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = RepresentationPatchDataset(
        train_samples, args.patch_size, args.patches_per_image, True, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=(args.num_workers > 0))
    best_score, best_state, best_record = -1.0, None, None
    history = []
    start_time = time.time()
    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,} "
          f"train_patches={len(dataset)}", flush=True)

    for epoch in _local_progress(range(1, args.epochs + 1), 'model239_native_stem_finetune'):
        train_loss = train_epoch(model, loader, optimizer, scaler, device, use_amp)
        valid = evaluate(model, valid_samples, args, device, use_amp)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss,
                  "selection": float(valid[args.selection_metric]), "valid": valid,
                  "seconds": time.time() - start_time}
        history.append(record)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"epoch": epoch, "train_loss": train_loss,
                          "selection": record["selection"],
                          "image_miou_compat": valid["image_miou_compat"],
                          "image_mean_miou": valid["image_mean_miou"],
                          "image_mean_miou_exclude": valid["image_mean_miou_exclude"]},
                         ensure_ascii=False), flush=True)
        if record["selection"] > best_score:
            best_score, best_state, best_record = record["selection"], copy.deepcopy(model.state_dict()), record
            torch.save({"model": best_state, "epoch": epoch, "metrics": valid,
                        "args": vars(args), "architecture": "zero-init stride-1 native stem"},
                       output / "best.pt")

    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    final = evaluate(model, valid_samples, args, device, use_amp, output_probs=True)
    probabilities = final.pop("probabilities")
    payload = {path.stem: value.astype(np.float16)
               for (path, _), value in zip(valid_samples, probabilities)}
    np.savez_compressed(output / "valid_probs.npz", **payload)
    summary = {"selection_metric": args.selection_metric, "best_selection": best_score,
               "best_record": best_record, "final": final,
               "elapsed_seconds": time.time() - start_time,
               "evaluation": {"patch_size": args.patch_size, "stride": args.val_stride,
                             "tta": "D4", "blend": "gaussian"}}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
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
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--decoder-lr", type=float, default=3e-5)
    parser.add_argument("--native-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-stride", type=int, default=576)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--selection-metric", choices=("image_mean_miou",
                                                         "image_mean_miou_exclude"),
                        default="image_mean_miou")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
