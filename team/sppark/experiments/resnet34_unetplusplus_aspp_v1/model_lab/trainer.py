from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from model_lab.models import build_model
from model_lab.dataset import KomapTrainDataset
from model_lab.losses import CombinedLoss
from model_lab.evaluation import evaluate_dataset
from model_lab.workspace import get_dataset_samples, write_json, OUTPUTS_DIR


class ModelEMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.99) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            cur_state = model.state_dict()
            for k in self.shadow:
                if cur_state[k].dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(cur_state[k], alpha=1.0 - self.decay)
                else:
                    self.shadow[k].copy_(cur_state[k])

    def apply_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def train_runner(
    cfg: dict[str, Any],
    smoke: bool = False,
    resume_path: str | None = None,
    stop_after: int | None = None,
) -> None:
    """Multi-stage trainer for ResNet34 U-Net++ + residual ASPP."""
    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    rank = get_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    seed = int(cfg.get("seed", 2026)) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir = OUTPUTS_DIR / "runs"
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "resolved_config.json", cfg)

    # Initialize model
    model = build_model(cfg).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    raw_model = model.module if hasattr(model, "module") else model
    ema = ModelEMA(raw_model, decay=float(cfg.get("train", {}).get("ema_decay", 0.99)))

    # Loss function
    loss_cfg = cfg.get("loss", {})
    criterion = CombinedLoss(
        num_classes=cfg.get("model", {}).get("num_classes", 4),
        ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        class_weights=loss_cfg.get("class_weights", None),
    ).to(device)

    # Dataset samples
    train_samples = get_dataset_samples("train")
    valid_samples = get_dataset_samples("valid")

    stages = cfg.get("train", {}).get("stages", [])
    if smoke:
        # Override stages for smoke testing
        stages = [
            {"name": "smoke_s0", "mode": "patch", "core_size": 256, "max_updates": 2, "warmup_updates": 1, "encoder_lr": 1e-4, "decoder_lr": 3e-4},
            {"name": "smoke_s1", "mode": "full_frame", "core_size": None, "max_updates": 1, "warmup_updates": 1, "encoder_lr": 1e-5, "decoder_lr": 3e-5},
        ]

    effective_batch = int(cfg.get("train", {}).get("effective_batch_size", 8))
    micro_batch = int(cfg.get("train", {}).get("micro_batch_size", 1))
    accum_steps = max(1, effective_batch // (world_size * micro_batch))

    eval_every = 1 if smoke else int(cfg.get("train", {}).get("eval_every_updates", 200))
    best_val_miou = 0.0
    best_weights_type = "raw"
    global_update = 0
    start_time = time.time()

    # Resume if requested
    if resume_path and Path(resume_path).is_file():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        ema.shadow = ckpt["ema_model"]
        global_update = ckpt.get("global_update", 0)
        best_val_miou = ckpt.get("best_val_miou", 0.0)
        if rank == 0:
            print(f"Resumed from {resume_path} at update {global_update} with best mIoU {best_val_miou:.2f}%")

    total_updates = sum(s["max_updates"] for s in stages)
    pbar = tqdm(total=total_updates, initial=global_update, desc="Total Progress", dynamic_ncols=True) if rank == 0 else None

    # Multi-stage training loop
    for stage_idx, stage in enumerate(stages):
        stage_name = stage["name"]
        stage_max_updates = stage["max_updates"]
        enc_lr = float(stage.get("encoder_lr", 1e-4))
        dec_lr = float(stage.get("decoder_lr", 3e-4))
        warmup_updates = int(stage.get("warmup_updates", 50))

        # Dataset & Loader for this stage
        train_ds = KomapTrainDataset(train_samples, cfg, stage, stage_index=stage_idx)
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        loader = DataLoader(
            train_ds,
            batch_size=micro_batch,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        # Optimizer with differential learning rates
        param_groups = raw_model.get_parameter_groups(
            encoder_lr=enc_lr,
            decoder_lr=dec_lr,
            weight_decay=float(cfg.get("train", {}).get("weight_decay", 1e-4)),
        )
        optimizer = torch.optim.AdamW(param_groups)

        # Stage updates counter
        stage_update = 0
        loader_iter = iter(loader)
        running_loss = 0.0
        micro_step = 0

        while stage_update < stage_max_updates:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            # Cosine LR schedule with warmup
            progress = min(1.0, max(0.0, stage_update / max(1, stage_max_updates)))
            lr_factor = (
                float(stage_update + 1) / float(max(1, warmup_updates))
                if stage_update < warmup_updates
                else 0.5 * (1.0 + np.cos(np.pi * progress))
            )
            for i, pg in enumerate(optimizer.param_groups):
                base_lr = enc_lr if i == 0 else dec_lr
                pg["lr"] = max(1e-7, base_lr * lr_factor)

            # Forward pass with BF16
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                loss_dict = criterion(outputs, masks)
                loss = loss_dict["loss"] / accum_steps

            loss.backward()
            running_loss += loss_dict["loss"].item()
            micro_step += 1

            if micro_step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(raw_model)

                stage_update += 1
                global_update += 1
                if pbar:
                    pbar.update(1)

                # Validation checkpointing
                if stage_update % eval_every == 0 or stage_update == stage_max_updates:
                    avg_loss = running_loss / max(1, eval_every * accum_steps)
                    running_loss = 0.0

                    if rank == 0:
                        # Evaluate raw model
                        raw_metrics = evaluate_dataset(raw_model, valid_samples, d4=False, device=device)
                        # Evaluate EMA model
                        eval_model = copy.deepcopy(raw_model)
                        ema.apply_to(eval_model)
                        ema_metrics = evaluate_dataset(eval_model, valid_samples, d4=False, device=device)

                        use_ema = ema_metrics["image_mean_miou"] >= raw_metrics["image_mean_miou"]
                        current_val_miou = ema_metrics["image_mean_miou"] if use_ema else raw_metrics["image_mean_miou"]
                        val_type = "EMA" if use_ema else "Raw"

                        is_best = current_val_miou > best_val_miou
                        if is_best:
                            best_val_miou = current_val_miou
                            best_weights_type = val_type

                        status_line = (
                            f"[{stage_name}] Upd: {global_update} | Loss: {avg_loss:.4f} | "
                            f"Val {val_type}: {current_val_miou:.2f}% (Best: {best_val_miou:.2f}%)"
                        )
                        if pbar:
                            pbar.set_postfix_str(status_line)
                        else:
                            print(status_line)

                        # Save checkpoints
                        ckpt_data = {
                            "global_update": global_update,
                            "stage": stage_name,
                            "stage_update": stage_update,
                            "model": raw_model.state_dict(),
                            "ema_model": ema.shadow,
                            "best_val_miou": best_val_miou,
                            "best_weights_type": best_weights_type,
                            "config": cfg,
                            "elapsed_seconds": time.time() - start_time,
                        }
                        torch.save(ckpt_data, run_dir / "last.pt")
                        if is_best:
                            best_state = ema.shadow if use_ema else raw_model.state_dict()
                            best_ckpt = dict(ckpt_data, selected_model=best_state, selected_type=val_type)
                            torch.save(best_ckpt, run_dir / "best.pt")

            if stop_after is not None and global_update >= stop_after:
                break

        if stop_after is not None and global_update >= stop_after:
            break

    if pbar:
        pbar.close()

    if rank == 0:
        print(f"\nTraining completed! Best Validation mIoU: {best_val_miou:.2f}% ({best_weights_type})")
        print(f"Checkpoints saved in {run_dir}")

