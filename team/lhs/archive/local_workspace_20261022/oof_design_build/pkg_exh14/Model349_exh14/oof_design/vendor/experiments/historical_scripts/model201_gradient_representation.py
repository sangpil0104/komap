#!/usr/bin/env python3
"""Train a complementary 4-channel texture-aware representation branch.

The existing raw/z-score/CDF branch has no explicit local texture channel.
This branch adds a normalized high-frequency residual (gray minus a Gaussian
blur), while importing the model-195 weights for the first three channels.
The fourth stem channel starts at zero and is learned during native-patch
fine-tuning, so the branch remains compatible with the established decoder.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

import model181_raw_zcdf_aspp as base
from improve_five_causes import collect_split


def representation4(image: np.ndarray) -> np.ndarray:
    gray = image[..., 0].astype(np.float32) / 255.0
    z = (gray - float(gray.mean())) / max(float(gray.std()), 1e-5)
    z = np.clip(z, -4.0, 4.0) / 3.0
    flat = gray.reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(order.size, dtype=np.float32)
    ranks[order] = np.arange(order.size, dtype=np.float32)
    cdf = (ranks / max(order.size - 1, 1)).reshape(gray.shape)
    smooth = gaussian_filter(gray, sigma=1.0, mode="reflect")
    texture = gray - smooth
    texture = (texture - float(texture.mean())) / max(float(texture.std()), 1e-5)
    texture = np.clip(texture, -4.0, 4.0) / 3.0
    values = np.stack([
        (gray - 0.5) / 0.25,
        z,
        (cdf - 0.5) / 0.288675,
        texture,
    ], axis=0)
    return values.astype(np.float32)


class FourChannelTeamRepresentationASPP(base.TeamRepresentationASPP):
    def __init__(self) -> None:
        super().__init__(pretrained=False)
        old = self.stem[0]
        new = nn.Conv2d(
            4, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=(old.bias is not None),
        )
        with torch.no_grad():
            new.weight[:, :3].copy_(old.weight)
            new.weight[:, 3].zero_()
            if old.bias is not None and new.bias is not None:
                new.bias.copy_(old.bias)
        self.stem[0] = new


def load_compatible(model: nn.Module, checkpoint: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    state = dict(payload["model"])
    state.pop("stem.0.weight", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded={checkpoint} missing={missing} unexpected={unexpected}", flush=True)


def main(args: argparse.Namespace) -> None:
    base.representation = representation4
    base.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    train_samples = collect_split(args.data_dir, "train")
    valid_samples = collect_split(args.data_dir, "valid")
    dataset = base.RepresentationPatchDataset(
        train_samples, args.patch_size, args.patches_per_image, True, args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    model = FourChannelTeamRepresentationASPP().to(device)
    load_compatible(model, args.checkpoint)
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith(("stem.", "enc1.", "enc2.", "enc3.", "enc4."))
         else decoder).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": args.encoder_lr},
        {"params": decoder, "lr": args.decoder_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_score, best_state, best_record = -1.0, None, None
    history = []
    start = time.time()
    print(f"device={device} dataset={len(dataset)} params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)
    eval_args = argparse.Namespace(
        patch_size=args.eval_patch_size,
        val_stride=args.eval_stride,
        tile_batch_size=args.tile_batch_size,
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = base.train_epoch(model, loader, optimizer, scaler, device, use_amp)
        valid = base.evaluate(model, valid_samples, eval_args, device, use_amp)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "valid": valid,
                  "seconds": time.time() - start}
        history.append(record)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        selection = float(valid[args.selection_metric])
        print(json.dumps({"epoch": epoch, "train_loss": train_loss,
                          "selection": selection,
                          "global_miou": valid["global_miou"],
                          "image_mean_miou": valid["image_mean_miou"],
                          "image_mean_miou_exclude": valid["image_mean_miou_exclude"]},
                         ensure_ascii=False), flush=True)
        if selection > best_score:
            best_score, best_record = selection, record
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"model": best_state, "epoch": epoch,
                        "metrics": valid, "args": vars(args)}, output / "best.pt")
    if best_state is None:
        raise RuntimeError("no best checkpoint")
    model.load_state_dict(best_state)
    final = base.evaluate(model, valid_samples, eval_args, device, use_amp, output_probs=True)
    probabilities = final.pop("probabilities")
    np.savez_compressed(
        output / "valid_probs.npz",
        **{path.stem: value.astype(np.float16)
           for (path, _), value in zip(valid_samples, probabilities)},
    )
    summary = {
        "selection_metric": args.selection_metric,
        "best_selection": best_score,
        "best_record": best_record,
        "final": final,
        "representation": "raw_gray+per_image_zscore+empirical_cdf+normalized_texture",
        "texture": "gray - gaussian_filter(sigma=1.0)",
        "elapsed_seconds": time.time() - start,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--val-stride", type=int, default=576)
    parser.add_argument("--eval-patch-size", type=int, default=768)
    parser.add_argument("--eval-stride", type=int, default=576)
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selection-metric", choices=("image_mean_miou", "image_mean_miou_exclude"),
                        default="image_mean_miou")
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--no-amp", action="store_true")
    main(parser.parse_args())
