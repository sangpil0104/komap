#!/usr/bin/env python3
"""MatSSL-inspired contrastive domain adaptation of the ResNet34 encoder.

Only train-split images are read. Their masks, validation images, and test images
are not used by this self-supervised stage. The adapted encoder is intended to
be transferred back into the model-195 segmentation checkpoint.
"""

from __future__ import annotations
from local_image_metrics import progress as _local_progress

import argparse
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from improve_five_causes import collect_split
from model181_raw_zcdf_aspp import TeamRepresentationASPP, load_checkpoint, representation


ENCODER_PREFIXES = ("stem.", "enc1.", "enc2.", "enc3.", "enc4.")
STAGE_CHANNELS = (64, 64, 128, 256, 512)


class MetallographyPairDataset(Dataset):
    """On-the-fly patch pairs with texture-preserving microscopy transforms."""

    def __init__(self, samples, patch_size: int, patches_per_image: int, seed: int):
        self.paths = [image_path for image_path, _ in samples]
        if not self.paths:
            raise ValueError("train split has no images")
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.seed = int(seed)
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths) * self.patches_per_image

    def _read_gray(self, image_index: int) -> np.ndarray:
        if image_index in self.cache:
            self.cache.move_to_end(image_index)
            return self.cache[image_index]
        gray = np.asarray(Image.open(self.paths[image_index]).convert("RGB"),
                           dtype=np.uint8)[..., 0]
        self.cache[image_index] = gray
        while len(self.cache) > 12:
            self.cache.popitem(last=False)
        return gray

    def _base_patch(self, gray: np.ndarray, rng: random.Random) -> np.ndarray:
        side = self.patch_size
        height, width = gray.shape
        if height < side or width < side:
            scale = max(side / height, side / width)
            resized = Image.fromarray(gray).resize(
                (int(round(width * scale)), int(round(height * scale))),
                Image.Resampling.BICUBIC)
            gray = np.asarray(resized, dtype=np.uint8)
            height, width = gray.shape
        crop_side = max(1, int(round(side * rng.uniform(0.80, 1.0))))
        top = rng.randrange(max(1, height - crop_side + 1))
        left = rng.randrange(max(1, width - crop_side + 1))
        patch = gray[top:top + crop_side, left:left + crop_side]
        return np.asarray(Image.fromarray(patch).resize(
            (side, side), Image.Resampling.BICUBIC), dtype=np.uint8)

    def _view(self, patch: np.ndarray, rng: random.Random) -> torch.Tensor:
        side = self.patch_size
        crop_side = max(1, int(round(side * rng.uniform(0.75, 1.0))))
        top = rng.randrange(max(1, side - crop_side + 1))
        left = rng.randrange(max(1, side - crop_side + 1))
        view = patch[top:top + crop_side, left:left + crop_side]
        view = np.asarray(Image.fromarray(view).resize(
            (side, side), Image.Resampling.BICUBIC), dtype=np.uint8)

        if rng.random() < 0.5:
            view = np.flip(view, axis=1).copy()
        if rng.random() < 0.5:
            view = np.flip(view, axis=0).copy()
        turns = rng.randrange(4)
        if turns:
            view = np.rot90(view, turns).copy()

        value = view.astype(np.float32) / 255.0
        contrast = rng.uniform(0.90, 1.10)
        brightness = rng.uniform(-0.04, 0.04)
        gamma = rng.uniform(0.90, 1.10)
        value = np.clip((value ** gamma - 0.5) * contrast + 0.5 + brightness,
                        0.0, 1.0)
        view = np.round(value * 255.0).astype(np.uint8)
        if rng.random() < 0.12:
            view = np.asarray(Image.fromarray(view).filter(
                ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.55))),
                dtype=np.uint8)

        rgb = np.repeat(view[..., None], 3, axis=2)
        return torch.from_numpy(representation(rgb))

    def __getitem__(self, index: int):
        image_index = index % len(self.paths)
        rng = random.Random(self.seed + index * 1009 + random.randrange(1_000_000))
        patch = self._base_patch(self._read_gray(image_index), rng)
        return self._view(patch, rng), self._view(patch, rng)


class GatedProjection(nn.Module):
    """GAP + learnable gated fusion of all ResNet stages + SimCLR projector."""

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.gates = nn.ParameterList([
            nn.Parameter(torch.ones(channels)) for channels in STAGE_CHANNELS
        ])
        fused_channels = sum(STAGE_CHANNELS)
        self.projector = nn.Sequential(
            nn.Linear(fused_channels, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),
        )

    def forward(self, stage_features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        pooled = [F.adaptive_avg_pool2d(value, 1).flatten(1)
                  for value in stage_features]
        fused = torch.cat([
            torch.sigmoid(value) * gate.unsqueeze(0)
            for value, gate in zip(pooled, self.gates)
        ], dim=1)
        return self.projector(fused)


def encoder_stages(model: TeamRepresentationASPP, images: torch.Tensor):
    x0 = model.stem(images)
    x1 = model.enc1(model.pool(x0))
    x2 = model.enc2(x1)
    x3 = model.enc3(x2)
    x4 = model.enc4(x3)
    return x0, x1, x2, x3, x4


def ntxent_loss(embeddings: torch.Tensor, batch_size: int, temperature: float) -> torch.Tensor:
    values = F.normalize(embeddings.float(), dim=1)
    similarities = (values @ values.T) / temperature
    similarities.fill_diagonal_(torch.finfo(similarities.dtype).min)
    targets = (torch.arange(2 * batch_size, device=values.device) + batch_size)
    targets.remainder_(2 * batch_size)
    return F.cross_entropy(similarities, targets)


def save_payload(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # The contrastive stage deliberately sees train images only, never masks.
    train_samples = collect_split(args.data_dir, "train")
    dataset = MetallographyPairDataset(
        train_samples, args.patch_size, args.patches_per_image, args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0))

    model = TeamRepresentationASPP(pretrained=False).to(device)
    load_checkpoint(model, args.init_checkpoint)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(ENCODER_PREFIXES))
    head = GatedProjection(args.projection_dim).to(device)

    encoder_parameters = [parameter for parameter in model.parameters()
                          if parameter.requires_grad]
    head_parameters = list(head.parameters())
    optimizer = torch.optim.SGD(
        [{"params": encoder_parameters, "lr": args.lr},
         {"params": head_parameters, "lr": args.lr}],
        momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = GradScaler(enabled=use_amp)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_loss = float("inf")
    start_epoch = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["encoder_model"], strict=False)
        head.load_state_dict(payload["head"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        history = payload["history"]
        best_loss = payload["best_loss"]
        start_epoch = payload["epoch"]

    print(json.dumps({
        "device": str(device), "train_images": len(train_samples),
        "pairs_per_epoch": len(dataset), "steps_per_epoch": len(loader),
        "patch_size": args.patch_size, "batch_size_pairs": args.batch_size,
        "epochs": args.epochs, "init_checkpoint": args.init_checkpoint,
    }), flush=True)

    started = time.time()
    for epoch in _local_progress(range(start_epoch + 1, args.epochs + 1), 'model236_matssl_inspired_ssl'):
        model.train()
        head.train()
        total_loss, examples = 0.0, 0
        epoch_start = time.time()
        for first, second in _local_progress(loader, 'model236_matssl_inspired_ssl'):
            first = first.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            batch_size = first.shape[0]
            optimizer.zero_grad(set_to_none=True)
            images = torch.cat([first, second], dim=0)
            with autocast(enabled=use_amp):
                stages = encoder_stages(model, images)
                embeddings = head(stages)
                loss = ntxent_loss(embeddings, batch_size, args.temperature)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder_parameters + head_parameters, 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder_parameters + head_parameters, 5.0)
                optimizer.step()
            total_loss += float(loss.detach()) * batch_size
            examples += batch_size

        scheduler.step()
        mean_loss = total_loss / max(examples, 1)
        record = {
            "epoch": epoch, "nt_xent": mean_loss,
            "lr": [group["lr"] for group in optimizer.param_groups],
            "seconds": time.time() - epoch_start,
        }
        history.append(record)
        encoder_state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if key.startswith(ENCODER_PREFIXES)
        }
        payload = {
            "epoch": epoch, "encoder_model": model.state_dict(),
            "encoder": encoder_state, "head": head.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "history": history,
            "best_loss": min(best_loss, mean_loss),
            "args": vars(args),
        }
        save_payload(output / "last.pt", payload)
        if mean_loss < best_loss:
            best_loss = mean_loss
            payload["best_loss"] = best_loss
            save_payload(output / "best.pt", payload)
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "epoch": epoch, "nt_xent": mean_loss,
            "best_nt_xent": best_loss, "lr": record["lr"],
            "seconds": record["seconds"],
            "elapsed_seconds": time.time() - started,
        }), flush=True)

    summary = {
        "method": "MatSSL-inspired ResNet34 gated multi-stage NT-Xent",
        "training_split": "train only; masks unused in SSL stage",
        "train_images": len(train_samples), "patches_per_image": args.patches_per_image,
        "patch_size": args.patch_size, "batch_size_pairs": args.batch_size,
        "epochs": len(history), "best_nt_xent": best_loss,
        "elapsed_seconds": time.time() - started,
        "encoder_checkpoint": str(output / "best.pt"),
        "history": history,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("method", "train_images", "epochs", "best_nt_xent", "elapsed_seconds")}),
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-image", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=236)
    main(parser.parse_args())
