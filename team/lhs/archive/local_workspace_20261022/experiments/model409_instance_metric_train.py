#!/usr/bin/env python3
"""Model409 paired fine-tuning with instance-local feature contrast."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "model336_hrnet_widthnorm_candidate"
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from komap_seg.checkpoint import save_checkpoint  # noqa: E402
from komap_seg.dataset import KomapPatchDataset, modal_image_width, seed_worker  # noqa: E402
from komap_seg.losses import BoundaryHeadLoss, ComboLoss  # noqa: E402
from komap_seg.model import build_model  # noqa: E402
from model409_instance_metric_loss_v2 import (  # noqa: E402
    ProjectionHead,
    build_component_pairs,
    component_local_metric_loss,
)


def build_projector(input_channels: int, device: torch.device, seed: int) -> ProjectionHead:
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed + 409_000)
        return ProjectionHead(input_channels).to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--metric-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs != 8 or args.metric_weight < 0:
        raise ValueError("Model409 fixes eight epochs and a nonnegative metric weight")
    if args.arm == "control" and args.metric_weight != 0:
        raise ValueError("control requires a zero metric weight")
    if args.arm == "candidate" and args.metric_weight <= 0:
        raise ValueError("candidate requires the locked positive metric weight")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    init_checkpoint = Path(args.init_checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    model_config = dict(initial["model_config"])
    preprocessing = dict(initial["preprocessing_config"])
    if model_config != {
        "architecture": "UnetPlusPlus",
        "encoder_name": "tu-hrnet_w48",
        "boundary_head": True,
    }:
        raise ValueError(f"unexpected parent architecture: {model_config}")
    if (preprocessing.get("patch_size") != 512
            or preprocessing.get("input_scale") != 2.0
            or not preprocessing.get("normalize_width")):
        raise ValueError(f"unexpected parent preprocessing: {preprocessing}")

    train_images = data_root / "train" / "images"
    train_masks = data_root / "train" / "masks"
    target_width = modal_image_width(train_images)
    if target_width != preprocessing.get("target_width"):
        raise ValueError("outer-train modal width differs from parent checkpoint")
    dataset = KomapPatchDataset(
        train_images,
        train_masks,
        patch_size=512,
        input_scale=2.0,
        target_width=target_width,
        samples_per_image=12,
        seed=args.seed,
        rng_mode="per_item",
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

    model = build_model(**model_config, encoder_weights=None).to(device)
    missing, unexpected = model.load_state_dict(initial["state_dict"], strict=True)
    if missing or unexpected:
        raise ValueError(f"parent checkpoint mismatch: {missing}, {unexpected}")
    del initial
    projector = build_projector(model.segmentation_head[0].in_channels, device, args.seed)

    segmentation_loss = ComboLoss().to(device)
    boundary_loss = BoundaryHeadLoss().to(device)
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    decoder_parameters = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 3e-5},
            {"params": decoder_parameters, "lr": 1e-4},
            {"params": projector.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    state_path = output_dir / "train_state.pt"
    start_epoch = 1

    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"resume requested without {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        saved_args = dict(state["args"])
        current_args = dict(vars(args))
        for key in ("output_dir", "resume"):
            saved_args.pop(key, None)
            current_args.pop(key, None)
        if saved_args != current_args:
            raise ValueError(f"resume args differ: {saved_args} != {current_args}")
        model.load_state_dict(state["model"], strict=True)
        projector.load_state_dict(state["projector"], strict=True)
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
    elif state_path.exists() or (output_dir / "last.pt").exists():
        raise FileExistsError(f"refusing to overwrite a partial Model409 output: {output_dir}")

    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(
        f"Model409 arm={args.arm}: images={len(dataset.images)} steps/epoch={len(loader)} "
        f"target_width={target_width} metric_weight={args.metric_weight}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        dataset.set_epoch(epoch)
        model.train()
        projector.train()
        epoch_total = epoch_seg = epoch_boundary = epoch_metric = 0.0
        active_components = 0
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images_cpu, masks_cpu) in enumerate(loader):
            pairs = build_component_pairs(masks_cpu)
            images = images_cpu.to(device, non_blocking=True)
            masks = masks_cpu.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                encoded = model.encoder(images)
                decoded = model.decoder(encoded)
                logits = model.segmentation_head(decoded)
                boundary_logits = model.boundary_head(decoded)
                projected = projector(decoded)
                loss_seg = segmentation_loss(logits, masks)
                loss_boundary = boundary_loss(boundary_logits, masks)
                loss_metric = component_local_metric_loss(projected, pairs)
                total = loss_seg + loss_boundary + args.metric_weight * loss_metric
            scaler.scale(total / 2).backward()
            update = (batch_index + 1) % 2 == 0 or batch_index + 1 == len(loader)
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(projector.parameters()), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            epoch_total += float(total.detach().item())
            epoch_seg += float(loss_seg.detach().item())
            epoch_boundary += float(loss_boundary.detach().item())
            epoch_metric += float(loss_metric.detach().item())
            active_components += pairs.count
            if (batch_index + 1) % 50 == 0 or batch_index + 1 == len(loader):
                print(
                    f"epoch={epoch:02d}/{args.epochs} step={batch_index+1}/{len(loader)} "
                    f"components={active_components} total={epoch_total/(batch_index+1):.4f} "
                    f"seg={epoch_seg/(batch_index+1):.4f} "
                    f"boundary={epoch_boundary/(batch_index+1):.4f} "
                    f"local_metric={epoch_metric/(batch_index+1):.4f}",
                    flush=True,
                )
        scheduler.step()
        denominator = max(1, len(loader))
        print(
            f"epoch={epoch:02d} done total={epoch_total/denominator:.4f} "
            f"seg={epoch_seg/denominator:.4f} boundary={epoch_boundary/denominator:.4f} "
            f"local_metric={epoch_metric/denominator:.4f} "
            f"eligible_pairs={active_components} elapsed={time.time()-start:.1f}s",
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
