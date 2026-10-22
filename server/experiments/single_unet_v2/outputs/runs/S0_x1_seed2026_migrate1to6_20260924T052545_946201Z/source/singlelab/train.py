from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import dataset_signature, read_json, records, sources, write_json
from .data import PatchDataset
from .inference import autocast
from .loss import segmentation_loss
from .model import EfficientUNet
from .parallel import DistributedUpdateIndices, WeightedDataset


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(run, resume=False):
    """Called only by an explicit train/suite command, never by preflight or tests."""
    run = Path(run).resolve()
    cfg = read_json(run / "config.json")
    world, rank, local = (int(os.environ.get(k, default)) for k, default in
                          (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; CPU verification is available with preflight/tests")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(cfg["train"]["cpu_threads"])
    if world > 1:
        dist.init_process_group("nccl")
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    previous_handler = signal.signal(signal.SIGUSR1, request_stop)
    try:
        if (run / "completed.json").exists():
            raise ValueError("Completed runs are immutable; start a new run instead")
        if sources() != read_json(run / "source_hashes.json"):
            raise ValueError("Experiment source changed since run creation")
        if dataset_signature(cfg, verify=(rank == 0)) != read_json(run / "dataset.json"):
            raise ValueError("Dataset changed since run creation")
        batch = cfg["train"]["effective_batch"]
        accumulation = math.ceil(batch / world)
        seed_all(cfg["seed"])
        raw = EfficientUNet(cfg, pretrained=not resume).to(device)
        model = DistributedDataParallel(raw, device_ids=[local]) if world > 1 else raw
        encoder_ids = {id(p) for p in raw.encoder.parameters()}
        options = cfg["train"]
        groups = [dict(params=list(raw.encoder.parameters()), lr=options["encoder_lr"]),
                  dict(params=[p for p in raw.parameters() if id(p) not in encoder_ids], lr=options["decoder_lr"])]
        optimizer = torch.optim.AdamW(groups, weight_decay=options["weight_decay"])
        start, previous_elapsed = 0, 0.
        seed_all(cfg["seed"] + 10_000 + rank)
        if resume:
            checkpoint = run / read_json(run / "latest.json")["checkpoint"]
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            raw.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start, previous_elapsed = state["update"], state["elapsed_seconds"]
            if state["world_size"] == world:
                restore_rng(state["rng"][rank])
            else:
                migration = read_json(run / "migration.json") if (run / "migration.json").is_file() else {}
                if (migration.get("target_world_size") != world or migration.get("imported_update") != start
                        or migration.get("source_world_size") != state["world_size"]):
                    raise ValueError("GPU-count changes require an explicit imported migration run")
                if rank < state["world_size"]:
                    restore_rng(state["rng"][rank])
                else:
                    seed_all((cfg["seed"] + 10_000 + rank + 1_000_003 * start) % 2**32)
            del state
        elif (run / "latest.json").exists():
            raise ValueError("This run has a checkpoint; use --resume")
        dataset = WeightedDataset(PatchDataset(records(cfg)["train"], cfg))
        sampler = DistributedUpdateIndices(options["updates"], batch, rank, world, start)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=options["workers"],
                            pin_memory=True, persistent_workers=options["workers"] > 0,
                            generator=torch.Generator().manual_seed(cfg["seed"] + rank))
        iterator = iter(loader)
        updates, started = options["updates"], time.monotonic()
        stopping = False
        progress = tqdm(total=updates, initial=start, desc=f"{cfg['experiment']} x{cfg['scale']} seed{cfg['seed']}",
                        unit="update", dynamic_ncols=True, disable=rank != 0)
        raw.train()
        if rank == 0:
            write_json(run / "worker.json", dict(pid=os.getpid(), world_size=world,
                       cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "all")))
        for step in range(start, updates):
            warmup = options["warmup"]
            fraction = min(1., max(0., (step - warmup) / max(1, updates - warmup - 1)))
            factor = ((step + 1) / warmup if step < warmup else options["min_lr_ratio"] +
                      (1 - options["min_lr_ratio"]) * (1 + math.cos(math.pi * fraction)) / 2)
            for group, base in zip(optimizer.param_groups, (options["encoder_lr"], options["decoder_lr"])):
                group["lr"] = base * factor
            optimizer.zero_grad(set_to_none=True)
            values = torch.zeros(3, device=device)
            for micro in range(accumulation):
                x, target, edge, real = (t.to(device, non_blocking=True) for t in next(iterator))
                sync = model.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
                with sync:
                    with autocast(device, options["precision"]):
                        logits = model(x)
                        loss, parts = segmentation_loss(logits, target, edge, cfg["loss"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    (loss * real[0] * world / batch).backward()
                values += torch.stack((loss.detach(), parts["ce"], parts["dice"])) * real[0] / batch
            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if world > 1:
                dist.all_reduce(values)
            update = step + 1
            should_stop = torch.tensor(int(stop_requested or (run / "SAVE_AND_STOP").exists()), device=device)
            if world > 1:
                dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
            stopping = bool(should_stop.item())
            progress.update(1)
            if rank == 0 and (update % options["log_every"] == 0 or update == updates):
                elapsed = previous_elapsed + time.monotonic() - started
                eta = (time.monotonic() - started) / (update - start) * (updates - update)
                entry = dict(update=update, loss=values[0].item(), ce=values[1].item(), dice=values[2].item(),
                             encoder_lr=optimizer.param_groups[0]["lr"], elapsed_seconds=elapsed, eta_seconds=eta)
                with (run / "training.jsonl").open("a") as handle:
                    handle.write(json.dumps(entry) + "\n")
                progress.set_postfix(loss=f"{entry['loss']:.4f}", lr=f"{entry['encoder_lr']:.1e}")
            if update % options["checkpoint_every"] == 0 or update == updates or stopping:
                states = [None] * world
                if world > 1:
                    dist.all_gather_object(states, rng_state())
                else:
                    states[0] = rng_state()
                if rank == 0:
                    relative = f"checkpoints/update_{update:06d}.pt"
                    path = run / relative
                    path.parent.mkdir(exist_ok=True)
                    if path.exists():
                        raise FileExistsError(f"Will not replace existing checkpoint: {path}")
                    state = dict(model=raw.state_dict(), optimizer=optimizer.state_dict(), rng=states,
                                 update=update, world_size=world, elapsed_seconds=previous_elapsed + time.monotonic() - started,
                                 config=cfg, torch_version=str(torch.__version__))
                    temporary = path.with_suffix(".tmp")
                    torch.save(state, temporary)
                    temporary.replace(path)
                    write_json(run / "latest.json", dict(checkpoint=relative, update=update))
                if world > 1:
                    dist.barrier()
            if stopping:
                if rank == 0:
                    write_json(run / "paused.json", dict(update=update, world_size=world,
                               reason="save-and-stop requested; completed optimizer update saved"))
                break
        progress.close()
        if rank == 0 and not stopping:
            write_json(run / "completed.json", dict(updates=updates, seed=cfg["seed"], scale=cfg["scale"],
                       elapsed_seconds=previous_elapsed + time.monotonic() - started,
                       selection="last fixed-budget checkpoint, no validation selection"))
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
