from __future__ import annotations

from contextlib import nullcontext
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import eta, read_json, records, write_json
from .data import Patches
from .inference import autocast
from .loss import reconstruction_loss, segmentation_loss
from .model import SmallUNet, transfer_encoder
from .parallel import DistributedUpdateIndices, WeightedDataset


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


class Context:
    def __init__(self, cfg, cpu=False):
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.local = int(os.environ.get("LOCAL_RANK", 0))
        if not cpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA required for actual experiments; use prepare for read-only checks")
        self.device = torch.device("cpu" if cpu else f"cuda:{self.local}")
        if not cpu:
            torch.cuda.set_device(self.local)
        torch.set_num_threads(cfg["train"]["cpu_threads"])
        if self.world > 1:
            dist.init_process_group("gloo" if cpu else "nccl")

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def broadcast(self, value):
        packet = [value]
        if self.world > 1:
            dist.broadcast_object_list(packet, src=0)
        return packet[0]

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def maximum(self, value):
        item = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        if self.world > 1:
            dist.all_reduce(item, op=dist.ReduceOp.MAX)
        return item.item()

    def close(self):
        if self.world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def budget_remaining(a_seconds, ssl_seconds, b_seconds):
    return max(0.0, ssl_seconds + b_seconds - a_seconds)


def learning_rate_factor(step, planned_steps, options):
    warmup = min(options["warmup"], max(1, planned_steps // 10))
    fraction = min(1., max(0., (step - warmup) / max(1, planned_steps - warmup - 1)))
    if step < warmup:
        return (step + 1) / warmup
    floor = options["min_lr_ratio"]
    return floor + (1 - floor) * (1 + math.cos(math.pi * fraction)) / 2


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def result_for(state, checkpoint):
    return {"checkpoint": str(checkpoint), "updates": state["update"],
            "compute_seconds": state["compute_seconds"], "wall_seconds": state["wall_seconds"],
            "world_size": state["world_size"], "stage": state["stage"],
            "selection": "fixed updates or training-compute budget; never validation selection"}


def run_stage(suite, cfg, ctx, name, initial, task="seg", encoder_checkpoint=None,
              parent_checkpoint=None, budget_seconds=None, planned_steps=None, overall_start=None,
              future_work=None):
    """Train one stage. All ranks execute; only rank 0 logs and writes files.

    `compute_seconds` is summed per-update elapsed compute time on the slowest
    rank. It excludes logging, checkpoint writes, and validation. C extends A
    until A+C reaches B's SSL+finetune budget, to within one optimizer update.
    """
    directory = Path(suite) / name
    if (directory / "completed.json").exists():
        return read_json(directory / "completed.json")
    options = cfg["train"]
    fixed_steps = cfg["ssl"]["updates"] if task == "ssl" else options["updates"]
    maximum_steps = cfg["control"]["max_extra_updates"] if budget_seconds is not None else fixed_steps
    planned_steps = max(1, planned_steps or fixed_steps)
    if ctx.rank == 0:
        directory.mkdir(exist_ok=True)
    ctx.barrier()
    seed_all(cfg["seed"])
    raw = SmallUNet(outputs=1 if task == "ssl" else 4, freeze_bn=cfg["model"]["freeze_encoder_bn_stats"])
    initial_state = load_checkpoint(initial)["model"]
    if task == "ssl":
        # Same encoder and shared decoder initialization as A; only the 1-channel head differs.
        raw.load_state_dict({k: v for k, v in initial_state.items() if not k.startswith("head.")}, strict=False)
    else:
        raw.load_state_dict(initial_state)
    del initial_state
    if encoder_checkpoint:
        transfer_encoder(raw, load_checkpoint(encoder_checkpoint)["model"])
    raw.to(ctx.device)
    encoder_ids = {id(p) for p in raw.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": raw.encoder.parameters(), "lr": options["encoder_lr"]},
        {"params": [p for p in raw.parameters() if id(p) not in encoder_ids], "lr": options["decoder_lr"]}],
        weight_decay=options["weight_decay"])
    sample_offset = 0
    if parent_checkpoint:
        parent = load_checkpoint(parent_checkpoint)
        raw.load_state_dict(parent["model"])
        optimizer.load_state_dict(parent["optimizer"])
        sample_offset = parent["update"] * options["effective_batch"]
        del parent
    start, compute_seconds, previous_wall = 0, 0., 0.
    seed_all(cfg["seed"] + 10000 + ctx.rank)
    if (directory / "latest.json").exists():
        saved = Path(read_json(directory / "latest.json")["checkpoint"])
        state = load_checkpoint(saved)
        if state["world_size"] != ctx.world:
            raise ValueError("Resume requires the same GPU count")
        raw.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"][ctx.rank], ctx.device)
        start, compute_seconds, previous_wall = state["update"], state["compute_seconds"], state["wall_seconds"]
        if state["finished"]:
            result = result_for(state, saved)
            if ctx.rank == 0:
                write_json(directory / "completed.json", result)
            ctx.barrier()
            return result
        del state
    model = DDP(raw, device_ids=[ctx.local] if ctx.device.type == "cuda" else None) if ctx.world > 1 else raw
    dataset = WeightedDataset(Patches(records(cfg)["train"], cfg, task, maximum_steps, sample_offset))
    sampler = DistributedUpdateIndices(maximum_steps, options["effective_batch"], ctx.rank, ctx.world, start)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=options["workers"],
                        persistent_workers=options["workers"] > 0, pin_memory=ctx.device.type == "cuda",
                        generator=torch.Generator().manual_seed(cfg["seed"] + ctx.rank))
    iterator = iter(loader)
    rounds = math.ceil(options["effective_batch"] / ctx.world)
    started = time.monotonic()
    total_display = planned_steps if budget_seconds is not None else maximum_steps
    progress = tqdm(total=total_display, initial=start, desc=name, unit="update",
                    disable=ctx.rank != 0 or not __import__('sys').stderr.isatty())
    if ctx.rank == 0:
        print(f"[{name}] task={task} updates={maximum_steps} "
              f"compute_budget={budget_seconds} world={ctx.world} effective_batch={options['effective_batch']}", flush=True)
    raw.train()
    last_path = None
    finished = False
    for step in range(start, maximum_steps):
        factor = learning_rate_factor(step, planned_steps, options)
        for group, base in zip(optimizer.param_groups, (options["encoder_lr"], options["decoder_lr"])):
            group["lr"] = base * factor
        # Prefetch all microbatches outside the measured compute interval.
        batches = [next(iterator) for _ in range(rounds)]
        ctx.sync()
        compute_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        value = torch.zeros((), device=ctx.device)
        for micro, batch in enumerate(batches):
            x, target, weight, real = (v.to(ctx.device, non_blocking=True) for v in batch)
            sync = model.no_sync() if ctx.world > 1 and micro < rounds - 1 else nullcontext()
            with sync:
                with autocast(ctx.device, options["precision"]):
                    logits = model(x)
                    loss = reconstruction_loss(logits, target, weight) if task == "ssl" else segmentation_loss(logits, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss in {name}")
                (loss * real[0] * ctx.world / options["effective_batch"]).backward()
            value += loss.detach() * real[0] / options["effective_batch"]
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        ctx.sync()
        compute_seconds += ctx.maximum(time.monotonic() - compute_started)
        if ctx.world > 1:
            dist.all_reduce(value)
        update = step + 1
        finished = compute_seconds >= budget_seconds if budget_seconds is not None else update >= maximum_steps
        stop = bool(ctx.maximum((Path(suite) / "SAVE_AND_STOP").exists()))
        if ctx.rank == 0 and (update % options["log_every"] == 0 or finished or stop):
            elapsed = previous_wall + time.monotonic() - started
            average_wall = (time.monotonic() - started) / (update - start)
            remaining_steps = ((max(0, budget_seconds - compute_seconds) / max(compute_seconds / update, 1e-9))
                               if budget_seconds is not None else maximum_steps - update)
            remaining = remaining_steps * average_wall
            entry = {"stage": name, "update": update, "loss": value.item(), "elapsed_seconds": elapsed,
                     "compute_seconds": compute_seconds, "eta_seconds": remaining,
                     "finish_estimate": eta(remaining), "encoder_lr": optimizer.param_groups[0]["lr"]}
            with (directory / "training.jsonl").open("a") as handle:
                handle.write(__import__('json').dumps(entry) + "\n")
            total_elapsed = time.monotonic() - overall_start if overall_start else elapsed
            future = future_work or {}
            future_time = future.get("updates", 0) * future.get("seconds_per_update", average_wall)
            future_time += future.get("evaluations", 0) * future.get("evaluation_seconds", 0)
            suite_remaining = remaining + future_time
            qualification = " (evaluation time not measured yet)" if future.get("evaluations", 0) and not future.get("evaluation_seconds") else ""
            print(f"[{name}] {update}/{total_display} loss={value.item():.5f} elapsed={elapsed:.1f}s "
                  f"remaining~{remaining:.1f}s finish~{eta(remaining)} suite_elapsed={total_elapsed:.1f}s "
                  f"suite_total_estimate~{total_elapsed+suite_remaining:.1f}s "
                  f"suite_finish~{eta(suite_remaining)}{qualification}", flush=True)
        progress.update(1)
        if update % options["checkpoint_every"] == 0 or finished or stop or update == maximum_steps:
            states = [None] * ctx.world
            if ctx.world > 1:
                dist.all_gather_object(states, rng_state(ctx.device))
            else:
                states[0] = rng_state(ctx.device)
            last_path = directory / "checkpoints" / f"update_{update:06d}.pt"
            if ctx.rank == 0:
                last_path.parent.mkdir(exist_ok=True)
                if last_path.exists():
                    raise FileExistsError(last_path)
                state = {"model": raw.state_dict(), "optimizer": optimizer.state_dict(), "rng": states,
                         "update": update, "world_size": ctx.world, "stage": name, "task": task,
                         "compute_seconds": compute_seconds, "wall_seconds": previous_wall + time.monotonic() - started,
                         "finished": finished, "config": cfg, "initial": str(initial),
                         "encoder_checkpoint": str(encoder_checkpoint) if encoder_checkpoint else None,
                         "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
                         "budget_seconds": budget_seconds, "planned_steps": planned_steps}
                tmp = last_path.with_suffix(".tmp")
                torch.save(state, tmp)
                tmp.replace(last_path)
                write_json(directory / "latest.json", {"checkpoint": str(last_path)}, replace=True)
                if finished:
                    write_json(directory / "completed.json", result_for(state, last_path))
            ctx.barrier()
        if finished or stop:
            break
    progress.close()
    del iterator, loader, model, raw, optimizer
    if ctx.device.type == "cuda":
        torch.cuda.empty_cache()
    if finished:
        return read_json(directory / "completed.json")
    if (Path(suite) / "SAVE_AND_STOP").exists():
        return None
    raise RuntimeError(f"{name} reached max_extra_updates without matching compute budget; inspect run, do not report as completed")
