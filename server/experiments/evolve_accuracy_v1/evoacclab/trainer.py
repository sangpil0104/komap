"""Ported from experiments/hundred_v1/hundredlab/trainer.py (import paths only; same stage/EMA/
checkpoint semantics: best.pt never regresses past the untouched parent's update-0 score,
best_trained.pt tracks the best score actually reached during training)."""
from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from alphase.data.audit import verify_manifest
from alphase.data.dataset import RankIndices
from evoacclab.dataset import EvolveDataset as TrainDataset
from alphase.data.palette import PALETTE
from alphase.data.splits import select_rows
from evoacclab.progress import Progress, duration
from alphase.engine.state import EMA, restore_rng, rng_state, save_checkpoint
from alphase.evaluation.evaluate import evaluate
from evoacclab.models import build_model, configure_trainable
from evoacclab.optimization import optimizer_for
from alphase.utils import amp_context, code_hash, environment, json_hash, seed_all, sha256, write_json


def setup_device(cfg):
    torch.set_num_threads(cfg["train"].get("cpu_threads", 4))
    world, rank, local = (int(os.environ.get(k, default)) for k, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        kwargs = {"device_id": device} if device.type == "cuda" else {}
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=30), **kwargs)
    effective = cfg["train"]["effective_batch_size"]
    if cfg["train"]["micro_batch_size"] != 1 or effective % world:
        raise ValueError("micro_batch_size=1 required for variable-size images; effective batch must be divisible by GPU count")
    return device, rank, world, effective//world


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def data_loader(rows, cfg, stage, si, rank, world, start=0):
    dataset = TrainDataset(rows, cfg, stage, si)
    sampler = RankIndices(len(dataset), rank, world, start*cfg["train"]["effective_batch_size"])
    return DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=cfg["train"].get("num_workers", 2),
                      pin_memory=torch.cuda.is_available(), persistent_workers=cfg["train"].get("num_workers", 2)>0)


def train(cfg, split="official", run=None, resume=None, overwrite=False, stop_after=None):
    if cfg["augmentation"].get("scales", [1.0]) != [1.0]:
        raise ValueError("Training scale augmentation is not enabled; keep scales=[1.0]")
    device, rank, world, accumulation = setup_device(cfg)
    seed_all(cfg["seed"])
    run = Path(run or f"runs/{cfg['experiment']}/{split}").resolve()
    resume_path = Path(resume).resolve() if resume else None
    if resume_path and resume_path.parent != run:
        raise ValueError("Resume into the checkpoint's original run directory")
    if run.exists() and any(run.iterdir()) and not resume and not overwrite:
        raise FileExistsError(f"Run already exists: {run}. Use --resume {run/'last.pt'} or a new --run-dir.")
    if rank == 0:
        if run.exists() and any(run.iterdir()) and overwrite and not resume:
            backup = run.with_name(run.name+f".backup-{time.time_ns()}")
            run.rename(backup)
        run.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    train_rows, valid_rows = select_rows(cfg, split)
    fingerprint = verify_manifest(cfg["data"]["manifest"])
    config_hash = json_hash(cfg)
    saved = torch.load(resume_path, map_location="cpu", weights_only=False) if resume else None
    if saved and (saved["config_hash"] != config_hash or saved["dataset_hash"] != fingerprint or saved["split"] != split or saved["world_size"] != world):
        raise ValueError("Resume rejected: configuration, dataset, split or GPU count changed")
    if saved and saved["completed"]:
        raise ValueError("This run is already complete")
    model = build_model(cfg, load_pretrained=not bool(saved)).to(device)
    if saved:
        model.load_state_dict(saved["model"])
    ema = EMA(model, cfg["train"]["ema_decay"])
    if saved:
        ema.module.load_state_dict(saved["ema_model"])
        ema.updates = saved["ema_updates"]
    update = saved["optimizer_step"] if saved else 0
    global_best = saved["best_metric"] if saved else -1.
    selected_weights = saved["selected_weights"] if saved else "raw"
    trained_best = saved.get("trained_best_metric", -1.) if saved else -1.
    start_stage = saved["stage_index"] if saved else 0
    progress = Progress(run, cfg, world, saved["elapsed_seconds"] if saved else 0) if rank == 0 else None
    writer = SummaryWriter(str(run/"tensorboard"), purge_step=update+1 if saved else None) if rank == 0 else None
    if rank == 0:
        OmegaConf.save(OmegaConf.create(cfg), run/"resolved_config.yaml")
        metadata = environment()
        metadata.update(code_sha256=code_hash(), dataset_sha256=fingerprint, config_hash=config_hash, split=split,
                        train_ids=[r["image_id"] for r in train_rows], validation_ids=[r["image_id"] for r in valid_rows],
                        world_size=world, effective_batch=cfg["train"]["effective_batch_size"],
                        folds_sha256=sha256(cfg["data"]["folds"]) if Path(cfg["data"]["folds"]).exists() else None,
                        pretrained_sha256=None)
        write_json(run/"environment.json", metadata)
        print(f"Run: {run}\nDevice: {device}, GPUs: {world}, accumulation: {accumulation}\nTrain: {len(train_rows)}, validation: {len(valid_rows)}", flush=True)
    bar = tqdm(total=sum(s["max_updates"] for s in cfg["train"]["stages"]), initial=update, desc="Optimizer updates", disable=rank != 0, dynamic_ncols=True)
    stage_step, si = 0, start_stage
    try:
        for si, stage in enumerate(cfg["train"]["stages"]):
            if si < start_stage:
                continue
            continuing = bool(saved and si == start_stage)
            if si > 0 and not continuing:
                previous = torch.load(run/f"stage_{si-1}_best.pt", map_location="cpu", weights_only=False)
                model.load_state_dict(previous["ema_model"] if previous["stage_selected_weights"] == "ema" else previous["model"])
                ema = EMA(model, cfg["train"]["ema_decay"])
                del previous
            configure_trainable(model, cfg, stage)
            optimizer, scheduler = optimizer_for(model, cfg, stage)
            if continuing:
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                restore_rng(saved["random_states"][rank])
            stage_step = saved["stage_step"] if continuing else 0
            stage_best = saved["stage_best_metric"] if continuing else -1.
            stage_selected = saved["stage_selected_weights"] if continuing else "raw"
            net = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                      find_unused_parameters=False) if world > 1 else model
            loader = data_loader(train_rows, cfg, stage, si, rank, world, stage_step)
            iterator = iter(loader)
            saved = None
            timings = []

            def checkpoint(completed=False, best=False, stage_improved=False, trained_improved=False, winner="raw"):
                states = [None]*world
                if world > 1:
                    dist.all_gather_object(states, rng_state())
                else:
                    states[0] = rng_state()
                if rank == 0:
                    content = dict(model=model.state_dict(), ema_model=ema.module.state_dict(), optimizer=optimizer.state_dict(),
                                   scheduler=scheduler.state_dict(), scaler=None, optimizer_step=update, stage_step=stage_step,
                                   stage_index=si, stage_best_metric=stage_best, stage_selected_weights=stage_selected,
                                   best_metric=global_best, selected_weights=selected_weights, ema_updates=ema.updates, trained_best_metric=trained_best,
                                   random_states=states, resolved_config=cfg, config_hash=config_hash,
                                   dataset_hash=fingerprint, train_ids=[r["image_id"] for r in train_rows],
                                   validation_ids=[r["image_id"] for r in valid_rows], split=split, fold_id=split,
                                   palette=PALETTE.tolist(), world_size=world, elapsed_seconds=progress.elapsed,
                                   trainable_parameter_groups=[g["names"] for g in optimizer.param_groups],
                                   backbone_commit=metadata["backbone_commit"], pretrained_sha256=metadata["pretrained_sha256"],
                                   code_sha256=metadata["code_sha256"], folds_sha256=metadata["folds_sha256"], completed=completed)
                    save_checkpoint(run/"last.pt", content)
                    if best:
                        shutil.copyfile(run/"last.pt", run/"best.pt")
                    if trained_improved:
                        trained_content = dict(content, selected_weights=winner, best_metric=trained_best)
                        save_checkpoint(run/"best_trained.pt", trained_content)
                    if stage_improved:
                        shutil.copyfile(run/"last.pt", run/f"stage_{si}_best.pt")
                if world > 1:
                    dist.barrier()

            if rank == 0:
                progress.write(si, stage_step, update, best_miou=global_best if global_best >= 0 else None)
            if si == 0 and update == 0 and not resume:
                initial = evaluate(model, valid_rows, cfg, device, run/"initial_validation", distributed=world>1, label="Initial model")
                global_best = initial[f"mIoU_{cfg['evaluation']['empty_union']}"]
                selected_weights = "raw"
                if rank == 0:
                    write_json(run/"initial_score.json", dict(image_mIoU=global_best, optimizer_step=0))
                checkpoint(best=True)
            while stage_step < stage["max_updates"]:
                started = time.monotonic()
                net.train()
                optimizer.zero_grad(set_to_none=True)
                total_loss = torch.zeros((), device=device)
                for micro in range(accumulation):
                    batch = move_batch(next(iterator), device)
                    sync = net.no_sync() if world > 1 and micro < accumulation-1 else nullcontext()
                    with sync:
                        with amp_context(device, cfg["train"]["precision"]):
                            output = net(batch["detail"], batch["dino"], batch["box"])
                        from evoacclab.losses import segmentation_loss
                        loss, _ = segmentation_loss(output, batch["target"], cfg["loss"])
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Non-finite loss at update {update}")
                        (loss/accumulation).backward()
                        total_loss += loss.detach()/accumulation
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                ema.update(model)
                if world > 1:
                    dist.all_reduce(total_loss)
                    total_loss /= world
                value = float(total_loss)
                update += 1
                stage_step += 1
                timings.append(time.monotonic()-started)
                bar.update(1)
                if rank == 0 and (stage_step == 1 or update % cfg["train"].get("log_every_updates", 10) == 0):
                    rate = sum(timings[-20:])/len(timings[-20:])
                    progress.write(si, stage_step, update, seconds_per_update=rate, loss=value,
                                   best_miou=global_best if global_best >= 0 else None,
                                   gpu_memory_gb=torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0)
                    event = dict(update=update, stage=stage["name"], loss=value, lr=optimizer.param_groups[0]["lr"], elapsed_seconds=progress.elapsed)
                    with open(run/"metrics.jsonl", "a") as file:
                        file.write(json.dumps(event)+"\n")
                    writer.add_scalar("train/loss", value, update)
                    writer.add_scalar("train/lr", event["lr"], update)
                    writer.add_scalar("time/remaining_hours", (progress.last["remaining_seconds"] or 0)/3600, update)
                    bar.set_postfix(stage=stage["name"], loss=f"{value:.4f}", best=f"{global_best:.2f}", ETA=duration(progress.last["remaining_seconds"]))
                evaluate_now = stage_step % cfg["train"]["eval_every_updates"] == 0 or stage_step == stage["max_updates"]
                is_best, stage_improved, trained_improved = False, False, False
                winner = "raw"
                if evaluate_now and valid_rows:
                    if rank == 0:
                        progress.write(si, stage_step, update, state="validating")
                    eval_start = time.monotonic()
                    scores = {}
                    for label, candidate in (("raw", model), ("ema", ema.module)):
                        scores[label] = evaluate(candidate, valid_rows, cfg, device, run/"validation"/label, distributed=world>1, label=f"Valid {label}")
                    metric_key = f"mIoU_{cfg['evaluation']['empty_union']}"
                    winner = max(scores, key=lambda key: scores[key][metric_key])
                    score = scores[winner][metric_key]
                    stage_improved, is_best = score > stage_best, score > global_best
                    trained_improved = score > trained_best
                    if trained_improved: trained_best = score
                    if stage_improved:
                        stage_best, stage_selected = score, winner
                    if is_best:
                        global_best, selected_weights = score, winner
                    if rank == 0:
                        progress.validation_seconds = time.monotonic()-eval_start
                        progress.write(si, stage_step, update, best_miou=global_best, validation_miou=score, validation_weights=winner)
                        event = dict(update=update, stage=stage["name"], validation_miou=score, validation_raw=scores["raw"][metric_key], validation_ema=scores["ema"][metric_key])
                        with open(run/"metrics.jsonl", "a") as file:
                            file.write(json.dumps(event)+"\n")
                        for label in scores:
                            writer.add_scalar(f"validation/{label}_miou", scores[label][metric_key], update)
                        writer.flush()
                        tqdm.write(f"Validation update {update}: raw={scores['raw'][metric_key]:.3f}, EMA={scores['ema'][metric_key]:.3f}, best={global_best:.3f}")
                elif evaluate_now:
                    stage_improved = True
                    stage_selected = "raw"
                    if si == len(cfg["train"]["stages"])-1 and stage_step == stage["max_updates"]:
                        is_best, selected_weights = True, "raw"
                stopping = stop_after is not None and update >= stop_after
                completed = si == len(cfg["train"]["stages"])-1 and stage_step == stage["max_updates"]
                if evaluate_now or stopping or update % cfg["train"].get("checkpoint_every_updates", 100) == 0:
                    checkpoint(completed=completed, best=is_best, stage_improved=stage_improved, trained_improved=trained_improved, winner=winner)
                if stopping and not completed:
                    if rank == 0:
                        progress.write(si, stage_step, update, state="paused")
                    return run
            del iterator, loader, net
        if rank == 0:
            progress.write(si, stage_step, update, state="completed", best_miou=global_best if global_best >= 0 else None)
            print(f"Completed. Checkpoint: {run/'best.pt'}", flush=True)
    except BaseException as exc:
        if rank == 0:
            progress.write(si, stage_step, update, state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        bar.close()
        if writer:
            writer.close()
        if world > 1:
            dist.destroy_process_group()
    return run
