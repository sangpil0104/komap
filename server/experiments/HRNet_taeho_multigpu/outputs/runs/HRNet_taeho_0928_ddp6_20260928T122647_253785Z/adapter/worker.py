#!/usr/bin/env python3
"""One worker per GPU; global batch 8, then distributed native-image evaluation."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from runtime import (
    ROOT,
    atomic,
    comparison,
    image_metrics,
    load_config,
    reserve_result,
    save_json,
    write_report,
)
from compat import model_config, preflight, load_checkpoint
from distributed_core import (
    TrainForward,
    augmentation,
    batch_sizes,
    broadcast_batch,
    global_loss,
)
from komap_seg.dataset import KomapPatchDataset, KomapFullDataset, seed_worker
from komap_seg.inference import predict_probs_tta
from komap_seg.model import build_model
from komap_seg.palette import PALETTE


def broadcast_object(value, rank):
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def gather_rows(rows, world):
    gathered = [None] * world
    dist.all_gather_object(gathered, rows)
    return sorted(
        [row for group in gathered for row in group], key=lambda row: row["image_id"]
    )


@torch.inference_mode()
def validate(network, cfg, rank, world, device, checkpoint=None):
    """Independent images on each GPU; aggregate image scores with equal weight."""
    network.eval()
    data = Path(cfg["data_root"])
    ds = KomapFullDataset(data / "valid/images", data / "valid/masks")
    full = checkpoint is not None
    result = None
    if full:
        if rank == 0:
            result = reserve_result(cfg["model_date_tag"])
            (result / "predictions").mkdir()
            (result / "valid_mask_comparison").mkdir()
        result = Path(broadcast_object(str(result), rank))
    rows = []
    with tqdm(
        total=len(ds),
        desc="valid tile+flip4" if full else "quick valid native1x",
        disable=rank != 0,
        leave=full,
    ) as progress:
        for start in range(0, len(ds), world):
            index = start + rank
            if index < len(ds):
                sample = ds[index]
                if full:
                    prob = predict_probs_tta(
                        network,
                        sample["image"],
                        patch_size=cfg["patch_size"],
                        overlap=cfg["overlap"],
                        batch_size=cfg["inference_batch_size"],
                        input_scale=cfg["input_scale"],
                        tta=cfg["tta"],
                        device=device,
                    )
                    pred = prob.argmax(0).numpy().astype(np.uint8)
                else:
                    x = sample["image"][None].to(device)
                    h, w = x.shape[-2:]
                    x = torch.nn.functional.pad(
                        x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect"
                    )
                    pred = network(x)[0, :, :h, :w].argmax(0).cpu().numpy()
                target = sample["mask"].numpy()
                row = {
                    "image_id": sample["stem"],
                    "height": sample["height"],
                    "width": sample["width"],
                    **image_metrics(pred, target),
                }
                rows.append(row)
                if full:
                    Image.fromarray(PALETTE[pred]).save(
                        result / "predictions" / f'{sample["stem"]}_mask.png'
                    )
                    original = np.array(
                        Image.open(
                            data / "valid/images" / f'{sample["stem"]}_image.png'
                        ).convert("RGB")
                    )
                    comparison(
                        result / "valid_mask_comparison" / f'{sample["stem"]}.png',
                        original,
                        pred,
                        target,
                        row["image_miou"],
                    )
            dist.barrier()
            progress.update(min(world, len(ds) - start))
    rows = gather_rows(rows, world)
    if len(rows) != len(ds) or len({row["image_id"] for row in rows}) != len(ds):
        raise ValueError("Validation images missing or duplicated")
    score = float(np.mean([row["image_miou"] for row in rows]))
    if full and rank == 0:
        summary = write_report(
            result, rows, checkpoint, {k: v for k, v in cfg.items() if k != "variant"}
        )
        summary["execution"] = execution_info(cfg, world)
        save_json(result / "summary.json", summary)
        print(
            f'{cfg["model_name"]}: image mIoU={score*100:.6f}%; results={result}',
            flush=True,
        )
    network.train()
    return score, result


def execution_info(cfg, world):
    return {
        "mode": "DDP + SyncBatchNorm",
        "world_size": world,
        "visible_physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "global_batch_size": cfg["batch_size"],
        "local_batch_sizes": batch_sizes(cfg["batch_size"], world),
        "loss_reduction": "global_batch_statistics",
        "variant": cfg["variant"],
    }


def distributed_model(cfg, rank, device, pretrained):
    network = build_model(
        **model_config(cfg),
        encoder_weights=cfg["encoder_weights"] if pretrained and rank == 0 else None,
    )
    network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network).to(device)
    wrapper = TrainForward(network, cfg["boundary_head"])
    return DDP(
        wrapper,
        device_ids=[device.index],
        output_device=device.index,
        gradient_as_bucket_view=True,
    )


def train(cfg, rank, world, device):
    output = None
    if rank == 0:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = ROOT / "outputs/runs" / f'{cfg["model_name"]}_0928_ddp{world}_{stamp}'
        output.mkdir(parents=True, exist_ok=False)
        preflight(cfg, output / "preflight.json")
        save_json(
            output / "config.json", {k: v for k, v in cfg.items() if k != "variant"}
        )
        save_json(output / "execution.json", execution_info(cfg, world))
        shutil.copytree(
            ROOT / "source",
            output / "source",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        (output / "adapter").mkdir()
        for name in [
            "runtime.py",
            "compat.py",
            "distributed_core.py",
            "worker.py",
            "run_distributed.sh",
            "source_manifest.json",
            "model_metadata.json",
            "requirements.txt",
        ]:
            shutil.copy2(ROOT / name, output / "adapter" / name)
        save_json(output / "status.json", {"status": "initializing"})
        save_json(
            ROOT / f'outputs/latest_{cfg["variant"]}.json',
            {"status": "initializing", "run_dir": str(output)},
        )
    output = Path(broadcast_object(str(output), rank))
    torch.manual_seed(cfg["seed"])
    wrapped = distributed_model(cfg, rank, device, pretrained=True)
    network = wrapped.module.network
    loader = None
    if rank == 0:
        data = Path(cfg["data_root"])
        ds = KomapPatchDataset(
            data / "train/images",
            data / "train/masks",
            patch_size=cfg["patch_size"],
            input_scale=cfg["input_scale"],
            samples_per_image=cfg["samples_per_image"],
            seed=cfg["seed"],
            transform=augmentation(cfg["variant"]),
        )
        kwargs = {"multiprocessing_context": "spawn"} if cfg["num_workers"] else {}
        loader = DataLoader(
            ds,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            worker_init_fn=seed_worker,
            drop_last=True,
            pin_memory=True,
            **kwargs,
        )
    steps = broadcast_object(len(loader) if rank == 0 else None, rank)
    if not steps:
        raise ValueError("Empty training loader")
    encoder_ids = {id(p) for p in network.encoder.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": network.encoder.parameters(),
                "lr": cfg["lr"] * cfg["encoder_lr_mult"],
            },
            {
                "params": [p for p in network.parameters() if id(p) not in encoder_ids],
                "lr": cfg["lr"],
            },
        ],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )
    start = time.monotonic()
    updates = 0
    if rank == 0:
        print(
            f'Run={output}; {steps} steps/epoch; global batch={cfg["batch_size"]}; local={batch_sizes(cfg["batch_size"], world)}',
            flush=True,
        )
    with tqdm(
        total=cfg["epochs"] * steps,
        desc=cfg["model_name"],
        unit="update",
        disable=rank != 0,
    ) as progress:
        for epoch in range(1, cfg["epochs"] + 1):
            wrapped.train()
            iterator = iter(loader) if rank == 0 else None
            epoch_loss = 0.0
            for _ in range(steps):
                x, y = broadcast_batch(iterator, cfg, rank, world, device)
                optimizer.zero_grad()
                logits, boundary = wrapped(x)
                loss = global_loss(logits, boundary, y, cfg["boundary_head_weight"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                value = float(loss.detach())
                epoch_loss += value
                updates += 1
                progress.update()
                if rank == 0:
                    progress.set_postfix(
                        epoch=f'{epoch}/{cfg["epochs"]}', loss=f"{value:.4f}"
                    )
                del x, y, logits, boundary, loss
            scheduler.step()
            record = {
                "epoch": epoch,
                "optimizer_updates": updates,
                "train_loss": epoch_loss / steps,
                "learning_rates": scheduler.get_last_lr(),
                "wall_seconds": time.monotonic() - start,
            }
            if cfg["val_every"] and (
                epoch % cfg["val_every"] == 0 or epoch == cfg["epochs"]
            ):
                record["quick_native1x_image_miou"], _ = validate(
                    network, cfg, rank, world, device
                )
            if rank == 0:
                with (output / "history.jsonl").open("a") as f:
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                state = {
                    "model_config": model_config(cfg),
                    "state_dict": network.state_dict(),
                    "experiment_config": cfg,
                    "epoch": epoch,
                    "optimizer_updates": updates,
                    "model_name": cfg["model_name"],
                    "execution": execution_info(cfg, world),
                }
                atomic(output / "last.pt", lambda p: torch.save(state, p))
                del state
                save_json(output / "status.json", {"status": "training", **record})
                tqdm.write(json.dumps(record, allow_nan=False))
            dist.barrier()
    if rank == 0:
        save_json(
            output / "status.json",
            {
                "status": "training_complete_evaluation_pending",
                "optimizer_updates": updates,
            },
        )
    # Release gradients and Adam states before inference; keep the trained network.
    optimizer.zero_grad(set_to_none=True)
    del loader, iterator, optimizer, scheduler, wrapped
    torch.cuda.empty_cache()
    score, result = validate(network, cfg, rank, world, device, output / "last.pt")
    if rank == 0:
        complete = {
            "status": "completed",
            "run_dir": str(output),
            "result_dir": str(result),
            "checkpoint": str(output / "last.pt"),
            "image_miou": score,
            "optimizer_updates": updates,
            "wall_seconds": time.monotonic() - start,
        }
        save_json(output / "completed.json", complete)
        save_json(output / "status.json", complete)
        save_json(ROOT / f'outputs/latest_{cfg["variant"]}.json', complete)
    dist.barrier()


def smoke(cfg, rank, world, device):
    """Two synthetic backward passes to check actual DDP reducer and SyncBN."""
    torch.manual_seed(0)
    wrapped = distributed_model(cfg, rank, device, pretrained=False)
    before = wrapped.module.network.segmentation_head[0].weight.detach().clone()
    shape = (
        batch_sizes(cfg["batch_size"], world)[rank],
        3,
        cfg["patch_size"],
        cfg["patch_size"],
    )
    started = time.monotonic()
    if rank == 0:
        # Exercise the same four-worker spawn and global-batch broadcast as training.
        synthetic = torch.utils.data.TensorDataset(
            torch.randn(2 * cfg["batch_size"], 3, cfg["patch_size"], cfg["patch_size"]),
            torch.randint(
                0, 4, (2 * cfg["batch_size"], cfg["patch_size"], cfg["patch_size"])
            ),
        )
        loader = DataLoader(
            synthetic,
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            multiprocessing_context="spawn" if cfg["num_workers"] else None,
        )
        iterator = iter(loader)
    else:
        iterator = None
    for step in range(2):
        wrapped.zero_grad(set_to_none=True)
        x, y = broadcast_batch(iterator, cfg, rank, world, device)
        logits, boundary = wrapped(x)
        loss = global_loss(logits, boundary, y, cfg["boundary_head_weight"])
        loss.backward()
        if not torch.isfinite(loss) or not all(
            torch.isfinite(p.grad).all().item()
            for p in wrapped.parameters()
            if p.grad is not None
        ):
            raise FloatingPointError("Nonfinite distributed gradients")
        if logits.shape != (shape[0], 4, *shape[-2:]):
            raise ValueError("Unexpected model output")
        del x, y, logits, boundary
    assert torch.equal(before, wrapped.module.network.segmentation_head[0].weight)
    item = {
        "rank": rank,
        "local_batch_size": shape[0],
        "loss": float(loss.detach()),
        "peak_gpu_memory_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "wall_seconds": time.monotonic() - started,
    }
    stats = [None] * world
    dist.all_gather_object(stats, item)
    if rank == 0:
        report = {
            "status": "passed",
            "execution": execution_info(cfg, world),
            "data": "synthetic_only",
            "backward_passes": 2,
            "optimizer_updates": 0,
            "parameters": sum(p.numel() for p in wrapped.parameters()),
            "ranks": stats,
        }
        save_json(ROOT / "reports/smoke.json", report)
        print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=[
            "train",
            "smoke",
            "check-loss",
            "check-evaluation",
            "preflight",
            "evaluate",
        ],
    )
    p.add_argument("--variant", choices=["baseline", "no_gaussian"], default="baseline")
    p.add_argument("--config")
    p.add_argument("--checkpoint")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config or ROOT / "configs" / f"{args.variant}.json")
    expected = (
        "HRNet_taeho" if args.variant == "baseline" else "HRNet_taeho_no_gaussian"
    )
    if cfg["model_name"] != expected:
        p.error("Variant and configuration model name differ")
    cfg["variant"] = args.variant
    rank, world, local_rank = (
        int(os.environ.get(k, default))
        for k, default in [("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]
    )
    batch_sizes(cfg["batch_size"], world)
    torch.set_num_threads(cfg["cpu_threads"])
    if args.command == "preflight":
        if rank == 0:
            preflight(cfg, ROOT / f"reports/preflight_{args.variant}.json")
            print("Preflight passed; no training started.", flush=True)
        return
    if args.command == "train" and not args.execute:
        if rank == 0:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "execution": execution_info(cfg, world),
                        "config": cfg,
                    },
                    indent=2,
                )
            )
        return
    if "RANK" not in os.environ:
        p.error("Use run_distributed.sh to start GPU workers")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cfg["device"] = str(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=15), device_id=device)
    try:
        if args.command == "train":
            train(cfg, rank, world, device)
        elif args.command == "smoke":
            smoke(cfg, rank, world, device)
        elif args.command == "check-loss":
            from verify_distributed import verify

            verify(cfg, rank, world, device)
        elif args.command == "check-evaluation":
            from verify_distributed import verify_evaluation

            verify_evaluation(cfg, rank, world, device, validate)
        elif args.command == "evaluate":
            if not args.checkpoint:
                p.error("--checkpoint is required")
            if rank == 0:
                preflight(cfg, ROOT / f"reports/preflight_{args.variant}.json")
            dist.barrier()
            network = load_checkpoint(args.checkpoint, cfg)
            validate(network, cfg, rank, world, device, Path(args.checkpoint).resolve())
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
