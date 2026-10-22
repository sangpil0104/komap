#!/usr/bin/env python3
"""Model403: Model392 recipe plus a class-balanced local supervised contrastive loss.

This trainer uses only the training split supplied by the fixed specimen-group
OOF protocol. It deliberately has no validation/test data loader.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "model336_hrnet_widthnorm_candidate"
sys.path.insert(0, str(SOURCE_ROOT))

from komap_seg.checkpoint import save_checkpoint  # noqa: E402
from komap_seg.dataset import KomapPatchDataset, modal_image_width, seed_worker  # noqa: E402
from komap_seg.losses import BoundaryHeadLoss, ComboLoss  # noqa: E402
from komap_seg.model import build_model  # noqa: E402


def supervised_local_contrastive(
    projected: torch.Tensor,
    masks: torch.Tensor,
    samples_per_class: int = 16,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Class-balanced supervised contrastive loss on decoder-grid pixels.

    Labels are nearest-neighbour downsampled to the decoder grid. At most
    `samples_per_class` locations are sampled for each present class in each
    image, preventing Al-dominated patches from dominating the pair set.
    """
    if samples_per_class < 2 or temperature <= 0:
        raise ValueError("need >=2 samples/class and a positive temperature")
    labels_map = F.interpolate(
        masks[:, None].float(), size=projected.shape[-2:], mode="nearest"
    )[:, 0].long()
    feature_map = F.normalize(projected.float(), dim=1)
    sampled_features: list[torch.Tensor] = []
    sampled_labels: list[torch.Tensor] = []
    for batch_index in range(projected.shape[0]):
        flat_labels = labels_map[batch_index].flatten()
        flat_features = feature_map[batch_index].permute(1, 2, 0).reshape(-1, feature_map.shape[1])
        for class_id in torch.unique(flat_labels):
            indices = torch.nonzero(flat_labels == class_id, as_tuple=False).flatten()
            if indices.numel() < 2:
                continue
            if indices.numel() > samples_per_class:
                indices = indices[torch.randperm(indices.numel(), device=indices.device)[:samples_per_class]]
            sampled_features.append(flat_features[indices])
            sampled_labels.append(torch.full_like(indices, int(class_id)))

    if not sampled_features:
        return projected.float().sum() * 0.0
    embeddings = torch.cat(sampled_features, dim=0)
    labels = torch.cat(sampled_labels, dim=0)
    if embeddings.shape[0] < 2:
        return projected.float().sum() * 0.0

    logits = (embeddings @ embeddings.T) / temperature
    diagonal = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(diagonal, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positives = (labels[:, None] == labels[None, :]) & ~diagonal
    positive_count = positives.sum(dim=1)
    anchors = positive_count > 0
    if not anchors.any():
        return projected.float().sum() * 0.0
    positive_log_prob = log_prob.masked_fill(~positives, 0.0).sum(dim=1)
    return -(positive_log_prob[anchors] / positive_count[anchors]).mean()


class ProjectionHead(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 128, output_channels: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def build_projection_head(input_channels: int, device: torch.device) -> ProjectionHead:
    """Initialize the auxiliary head without perturbing Model392's RNG stream.

    Model392's DataLoader shuffling and legacy worker seeds use the global RNG.
    Constructing the new head inside a fork preserves those streams while
    retaining the head's deterministic initialization for the current seed.
    """
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        return ProjectionHead(input_channels).to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="One fixed OOF fold root with train/images and train/masks")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--supcon-weight", type=float, default=0.0062)
    parser.add_argument("--samples-per-class", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.supcon_weight < 0:
        raise ValueError("epochs must be positive and supcon weight nonnegative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = data_root / "train" / "images"
    mask_dir = data_root / "train" / "masks"
    target_width = modal_image_width(image_dir)
    dataset = KomapPatchDataset(
        image_dir,
        mask_dir,
        patch_size=512,
        input_scale=2.0,
        target_width=target_width,
        samples_per_image=12,
        seed=args.seed,
        rng_mode="legacy",
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        drop_last=True,
        pin_memory=True,
    )

    model_config = {
        "architecture": "UnetPlusPlus",
        "encoder_name": "tu-hrnet_w48",
        "boundary_head": True,
    }
    preprocessing = {
        "patch_size": 512,
        "input_scale": 2.0,
        "target_width": target_width,
        "normalize_width": True,
    }
    model = build_model(**model_config, encoder_weights="imagenet").to(device)
    decoder_channels = model.segmentation_head[0].in_channels
    projector = build_projection_head(decoder_channels, device)
    segmentation_loss = ComboLoss().to(device)
    boundary_loss = BoundaryHeadLoss().to(device)

    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    decoder_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 3e-4},
            {"params": decoder_parameters, "lr": 3e-4},
            {"params": projector.parameters(), "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    state_path = output_dir / "train_state.pt"
    start_epoch = 1

    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        saved_args = dict(state["args"])
        current_args = dict(vars(args))
        for key in ("output_dir", "resume"):
            saved_args.pop(key, None)
            current_args.pop(key, None)
        if saved_args != current_args:
            raise ValueError(f"resume arguments differ: saved={saved_args}, current={current_args}")
        model.load_state_dict(state["model"])
        projector.load_state_dict(state["projector"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        dataset.rng.setstate(state["dataset_rng_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])

    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(
        f"Model403 train-only OOF: images={len(dataset.images)} steps/epoch={len(loader)} "
        f"target_width={target_width} supcon_weight={args.supcon_weight}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        dataset.set_epoch(epoch)
        model.train()
        projector.train()
        epoch_total = epoch_seg = epoch_boundary = epoch_supcon = 0.0
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, masks) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                encoded = model.encoder(images)
                decoded = model.decoder(encoded)
                logits = model.segmentation_head(decoded)
                boundary_logits = model.boundary_head(decoded)
                projected = projector(decoded)
                loss_seg = segmentation_loss(logits, masks)
                loss_boundary = boundary_loss(boundary_logits, masks)
                loss_supcon = supervised_local_contrastive(
                    projected, masks, args.samples_per_class, args.temperature
                )
                loss = loss_seg + loss_boundary + args.supcon_weight * loss_supcon
            scaler.scale(loss / 2).backward()
            update = (batch_index + 1) % 2 == 0 or batch_index + 1 == len(loader)
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(projector.parameters()), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_total += loss.item()
            epoch_seg += loss_seg.item()
            epoch_boundary += loss_boundary.item()
            epoch_supcon += loss_supcon.item()
            if (batch_index + 1) % 50 == 0 or batch_index + 1 == len(loader):
                print(
                    f"epoch={epoch:02d}/{args.epochs} step={batch_index+1}/{len(loader)} "
                    f"total={epoch_total/(batch_index+1):.4f} seg={epoch_seg/(batch_index+1):.4f} "
                    f"boundary={epoch_boundary/(batch_index+1):.4f} "
                    f"supcon={epoch_supcon/(batch_index+1):.4f}",
                    flush=True,
                )
        scheduler.step()
        denominator = max(1, len(loader))
        print(
            f"epoch={epoch:02d} done total={epoch_total/denominator:.4f} "
            f"seg={epoch_seg/denominator:.4f} boundary={epoch_boundary/denominator:.4f} "
            f"supcon={epoch_supcon/denominator:.4f} elapsed={time.time()-start:.1f}s",
            flush=True,
        )
        save_checkpoint(output_dir / "last.pt", model, model_config, preprocessing)
        torch.save(
            {
                "epoch": epoch,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "projector": {key: value.detach().cpu() for key, value in projector.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "dataset_rng_state": dataset.rng.getstate(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "args": vars(args),
            },
            state_path,
        )
    print(f"finished: {output_dir / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
