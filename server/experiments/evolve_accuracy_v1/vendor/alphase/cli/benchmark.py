"""Measure real optimizer updates without retaining trained model weights."""
import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from alphase.data.splits import select_rows
from alphase.engine.state import EMA, optimizer_for
from alphase.engine.trainer import data_loader, move_batch, setup_device
from alphase.evaluation.evaluate import evaluate
from alphase.losses.segmentation import segmentation_loss
from alphase.models.factory import build_model
from alphase.utils import amp_context, environment, json_hash, load_config, seed_all, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cnn_unet.yaml")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--output")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    if args.updates < 1 or args.warmup < 0:
        parser.error("updates >= 1 and warmup >= 0 required")
    cfg = load_config(args.config, args.set)
    device, rank, world, accumulation = setup_device(cfg)
    seed_all(cfg["seed"])
    rows, valid = select_rows(cfg, "official")
    model = build_model(cfg).to(device)
    ema = EMA(model, cfg["train"]["ema_decay"])
    timings = []
    try:
        for si, stage in enumerate(cfg["train"]["stages"]):
            if hasattr(model, "set_backbone_trainable"):
                model.set_backbone_trainable(stage["backbone_last_n"])
            optimizer, scheduler = optimizer_for(model, cfg, stage)
            net = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                      find_unused_parameters=hasattr(model, "backbone")) if world > 1 else model
            loader = data_loader(rows, cfg, stage, si, rank, world)
            iterator = iter(loader)
            samples = []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for step in range(args.warmup+args.updates):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if world > 1:
                    dist.barrier()
                started = time.monotonic()
                net.train()
                optimizer.zero_grad(set_to_none=True)
                for micro in range(accumulation):
                    batch = move_batch(next(iterator), device)
                    sync = net.no_sync() if world > 1 and micro < accumulation-1 else nullcontext()
                    with sync:
                        with amp_context(device, cfg["train"]["precision"]):
                            output = net(batch["detail"], batch["dino"], batch["box"])
                        loss, _ = segmentation_loss(output, batch["target"], cfg["loss"])
                        (loss/accumulation).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                ema.update(model)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = torch.tensor(time.monotonic()-started, device=device)
                if world > 1:
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                if step >= args.warmup:
                    samples.append(float(elapsed))
                if rank == 0:
                    print(f"{stage['name']}: {step+1}/{args.warmup+args.updates} · {float(elapsed):.3f} seconds/update", flush=True)
            memory = torch.tensor(torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0., device=device)
            if world > 1:
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            timings.append(dict(name=stage["name"], updates=stage["max_updates"], seconds_per_update=float(np.mean(samples)),
                                median_seconds=float(np.median(samples)), measured_update_seconds=samples,
                                peak_gpu_memory_gb=float(memory)))
            del iterator, loader, net, optimizer, scheduler, batch, output, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
        started = time.monotonic()
        # Time the full validation set, including image loading and metrics.
        evaluate(model, valid, cfg, device, distributed=world>1, label="Benchmark validation")
        if world > 1:
            dist.barrier()
        validation_seconds = 2*(time.monotonic()-started)  # raw + EMA
        training_seconds = sum(s["seconds_per_update"]*s["updates"] for s in timings)
        checks = sum(math.ceil(s["max_updates"]/cfg["train"]["eval_every_updates"]) for s in cfg["train"]["stages"])
        total = training_seconds + checks*validation_seconds
        if rank == 0:
            result = dict(config_hash=json_hash(cfg), environment=environment(), world_size=world,
                          effective_batch=cfg["train"]["effective_batch_size"], stages=timings,
                          validation_seconds=validation_seconds, validation_checks=checks,
                          estimated_training_seconds=training_seconds, estimated_total_seconds=total,
                          planning_range_hours=[total*.8/3600, total*1.3/3600],
                          note="Short sampled benchmark. Range is a planning margin, not a statistical confidence interval; checkpoint I/O and competing jobs may add time. No benchmark weights are retained.")
            out = args.output or f"artifacts/benchmark_{cfg['experiment']}_{world}gpu.json"
            write_json(out, result)
            print(json.dumps(result, indent=2), flush=True)
            print(f"Benchmark saved: {out}", flush=True)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
