"""User-started training only. Indexed sampling, stage-aware resume, raw/EMA validation."""

from copy import deepcopy
from pathlib import Path
import math
import random
import shutil
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from tqdm import tqdm
from .io import (
    ROOT,
    local_path,
    digest,
    ids_hash,
    sha256,
    json_write,
    jsonl_write,
    jsonl_read,
    torch_save,
)
from .config import preflight, save_config, inference_hash
from .data import TrainSampler, dataset_rows, verify_rows, model_inputs
from .models import load_model
from .losses import losses
from .contracts import rgb_to_ids, confusion_matrix, summarize_images
from .inference import predict_probs


class EMA:
    """Parameters EMA; every buffer (including BN counters) copied after each update."""

    def __init__(self, model, decay):
        self.model = deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for dst, src in zip(self.model.parameters(), model.parameters()):
            dst.lerp_(src.detach(), 1 - self.decay)
        for dst, src in zip(self.model.buffers(), model.buffers()):
            dst.copy_(src.detach())
        self.updates += 1


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def stage_at(cfg, step):
    t = cfg["training"]
    if t["recipe"] == "scratch_comparable":
        return "scratch"
    return (
        "A"
        if cfg["model"]["new_branches"]["residual_head"] and step < t["stage_a_updates"]
        else "B"
    )


def optimizer_for(model, cfg, step):
    t = cfg["training"]
    stage = stage_at(cfg, step)
    model.set_base_frozen(stage == "A")
    if stage == "scratch":
        groups = [
            {"params": model.parameters(), "lr": t["learning_rate"], "name": "all"}
        ]
        length = t["total_optimizer_updates"]
        warm = t["warmup_updates"]
    else:
        new = [
            p for name, p in model.named_parameters() if not name.startswith("base.")
        ]
        if stage == "A":
            groups = [{"params": new, "lr": t["stage_a_new_lr"], "name": "new"}]
            length = t["stage_a_updates"]
            warm = 0
        else:
            groups = [
                {
                    "params": model.base.encoder.parameters(),
                    "lr": t["stage_b_encoder_lr"],
                    "name": "encoder",
                },
                {
                    "params": list(model.base.decoder.parameters())
                    + list(model.base.segmentation_head.parameters()),
                    "lr": t["stage_b_decoder_lr"],
                    "name": "decoder_head",
                },
            ]
            if new:
                groups.append({"params": new, "lr": t["stage_b_new_lr"], "name": "new"})
            length = t["stage_b_updates"] if new else t["total_optimizer_updates"]
            warm = t["stage_b_warmup_updates"]
    if any(
        not list(group["params"])
        for group in groups
        if isinstance(group["params"], list)
    ):
        raise ValueError("Empty optimizer group")
    opt = torch.optim.AdamW(groups, weight_decay=t["weight_decay"])
    if stage == "scratch":
        factor = (
            lambda it: min(1.0, (it + 1) / max(warm, 1))
            * 0.5
            * (1 + math.cos(math.pi * min(it, length) / length))
        )
    elif stage == "A":
        factor = lambda it: 1.0
    else:
        factor = (
            lambda it: min(1.0, (it + 1) / max(warm, 1))
            * 0.5
            * (1 + math.cos(math.pi * min(it, length) / length))
        )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    return opt, sched, stage


@torch.inference_mode()
def evaluate_model(model, cfg, images, device):
    cms = []
    for row in tqdm(
        images,
        desc="valid full-image " + ("D4" if cfg["evaluation"]["d4"] else "no-TTA"),
        leave=False,
    ):
        gray = np.array(Image.open(row["image_path"]).convert("L"))
        target = rgb_to_ids(np.array(Image.open(row["mask_path"])))
        pred = predict_probs(model, gray, cfg, device).argmax(0).cpu().numpy()
        cms.append(confusion_matrix(target, pred))
    return summarize_images(cms)


def train(cfg, resume=None, device=None):
    cfg = deepcopy(cfg)
    cfg["execution"]["dry_run"] = False
    preflight(cfg, training=True)
    device = device or cfg["execution"]["device"]
    t = cfg["training"]
    train_rows = dataset_rows(cfg, "train")
    verify_rows(train_rows, "train")
    valid_rows = dataset_rows(cfg, "valid") if t["validation_enabled"] else []
    if valid_rows:
        verify_rows(valid_rows, "valid")
    run = (
        local_path(cfg["run"]["output_root"])
        / f'{cfg["run"]["name"]}_seed{cfg["run"]["seed"]}'
    )
    if resume:
        state = torch.load(local_path(resume), map_location="cpu", weights_only=True)
        if state.get("kind") != "alphase_training_v1":
            raise ValueError(
                "resume needs a full training checkpoint, not legacy weights"
            )
        if digest(state["config"]) != digest(cfg):
            raise ValueError("Resolved config changed across resume")
        if Path(state["run_dir"]) != run.resolve():
            raise ValueError("Resume run directory mismatch")
    else:
        state = None
        run.mkdir(parents=True, exist_ok=False)
        save_config(run / "resolved.yaml", cfg)
        shutil.copytree(
            ROOT / "alphase",
            run / "source/alphase",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(
            ROOT / "requirements.lock.txt", run / "source/requirements.lock.txt"
        )
        json_write(run / "source/input_hashes.json", jsonl_or_audit_hashes())
    torch.set_num_threads(cfg["execution"]["cpu_threads"])
    seed_all(cfg["run"]["seed"])
    checkpoint = t["initialize_from"] if t["recipe"] == "finetune_existing" else None
    model = load_model(
        cfg,
        None if state else checkpoint,
        device,
        imagenet=not state and checkpoint is None,
    )
    sampler = TrainSampler(cfg, train_rows)
    mining_cfg = deepcopy(cfg)
    mining_cfg["inference"]["d4"] = cfg["sampling"]["hard_mining_d4"]

    def mine_training_image(gray):
        was_training = model.training
        try:
            return predict_probs(model, gray, mining_cfg, device).cpu().numpy()
        finally:
            model.train(was_training)

    sampler.mining_callback = mine_training_image
    step = state["step"] if state else 0
    # A last checkpoint may sit exactly at A/B boundary, before the new optimizer exists.
    saved_stage_step = (
        max(0, step - 1) if state and state["stage"] != stage_at(cfg, step) else step
    )
    opt, sched, stage = optimizer_for(model, cfg, saved_stage_step)
    ema = EMA(model, t["ema_decay"]) if t["ema_enabled"] else None
    best = {"raw": -1.0, "ema": -1.0}
    wall_before = 0.0
    samples_seen = 0
    split_hashes = {
        "train": ids_hash([r["image_id"] for r in train_rows]),
        "valid": ids_hash([r["image_id"] for r in valid_rows]),
    }
    if state:
        if state["split_hashes"] != split_hashes:
            raise ValueError("Splits changed across resume")
        model.load_state_dict(state["model"], strict=True)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        sampler.load_state_dict(state["sampler"])
        best = state["best"]
        wall_before = state["wall_seconds"]
        samples_seen = state["samples_seen"]
        if ema:
            ema.model.load_state_dict(state["ema"], strict=True)
            ema.updates = state["ema_updates"]
        restore_rng(state["rng"])
    started = time.monotonic()
    history = run / "history.jsonl"
    if state and history.exists():
        # Discard log rows beyond the restored checkpoint before replaying updates.
        jsonl_write(history, [r for r in jsonl_read(history) if r["step"] <= step])

    def snapshot():
        return {
            "kind": "alphase_training_v1",
            "run_dir": str(run.resolve()),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "amp_scaler": None,
            "amp_policy": "BF16 without GradScaler; nonfinite grads fail before step",
            "ema": ema.model.state_dict() if ema else None,
            "ema_updates": ema.updates if ema else 0,
            "step": step,
            "stage": stage,
            "seed": cfg["run"]["seed"],
            "rng": rng_state(),
            "sampler": sampler.state_dict(),
            "config": cfg,
            "config_hash": digest(cfg),
            "split_hashes": split_hashes,
            "training_ids": [r["image_id"] for r in train_rows],
            "checkpoint_selection_ids": [r["image_id"] for r in valid_rows],
            "best": best.copy(),
            "wall_seconds": wall_before + time.monotonic() - started,
            "samples_seen": samples_seen,
            "parent_checkpoint_sha256": (
                sha256(local_path(checkpoint)) if checkpoint else None
            ),
        }

    with tqdm(
        total=t["total_optimizer_updates"],
        initial=step,
        desc=f'{cfg["run"]["name"]} seed{cfg["run"]["seed"]}',
        unit="update",
    ) as progress:
        while step < t["total_optimizer_updates"]:
            if stage_at(cfg, step) != stage:
                opt, sched, stage = optimizer_for(model, cfg, step)
            model.train()
            opt.zero_grad(set_to_none=True)
            value = 0.0
            for micro in range(t["accumulation"]):
                batch = sampler.batch(step, micro, device)
                with torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=torch.device(device).type == "cuda",
                ):
                    output = model(**model_inputs(batch))
                loss = losses(output, batch, cfg)["total"] / t["accumulation"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at update{step}")
                loss.backward()
                value += float(loss.detach())
                samples_seen += len(batch["image_ids"])
                del batch, output, loss
            nn.utils.clip_grad_norm_(
                model.parameters(), t["grad_clip"], error_if_nonfinite=True
            )
            opt.step()
            sched.step()
            step += 1
            if ema:
                ema.update(model)
            record = {
                "step": step,
                "loss": value,
                "stage": stage,
                "lr": sched.get_last_lr(),
                "samples_seen": samples_seen,
                "wall_seconds": wall_before + time.monotonic() - started,
            }
            validate_now = bool(valid_rows) and (
                step % t["validation_every_updates"] == 0
                or step == t["total_optimizer_updates"]
            )
            if validate_now:
                for name, net in [("raw", model)] + (
                    [("ema", ema.model)] if ema else []
                ):
                    score = evaluate_model(net, cfg, valid_rows, device)[
                        "val_image_miou"
                    ]
                    record["val_image_miou_" + name] = score
                    if score > best[name]:
                        best[name] = score
                        torch_save(
                            run
                            / (
                                "best_image_miou.pt"
                                if name == "raw"
                                else "best_ema_image_miou.pt"
                            ),
                            snapshot(),
                        )
            with history.open("a") as f:
                import json

                f.write(json.dumps(record, allow_nan=False) + "\n")
            if (
                step % t["checkpoint_every_updates"] == 0
                or validate_now
                or step == t["total_optimizer_updates"]
            ):
                torch_save(run / "last.pt", snapshot())
            progress.update()
            progress.set_postfix(
                loss=f"{value:.4f}",
                stage=stage,
                miou=f"{max(best.values())*100:.3f}%" if valid_rows else "fixed-update",
            )
    selected = (
        run
        / (
            "best_ema_image_miou.pt"
            if best["ema"] > best["raw"]
            else "best_image_miou.pt"
        )
        if valid_rows
        else run / "last.pt"
    )
    json_write(
        run / "completed.json",
        {
            "step": step,
            "selected_checkpoint": str(selected),
            "best": best,
            "samples_seen": samples_seen,
            "wall_seconds": wall_before + time.monotonic() - started,
        },
    )
    if valid_rows and t["save_results"]:
        del model, opt, sched, ema
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        export_evaluation(
            cfg, run, selected, device, use_ema=selected.name.startswith("best_ema")
        )
    return run


def jsonl_or_audit_hashes():
    import json

    return json.loads((ROOT / "reports/audit.json").read_text())["input_hashes"]


def export_evaluation(cfg, run, checkpoint, device, use_ema=False):
    from .cache import cache_model
    from .evaluation import report_predictions, reserve_results, write_csv

    checkpoint = Path(checkpoint)
    inference_checkpoint = checkpoint
    if use_ema:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        inference_checkpoint = Path(run) / "selected_ema_weights.pt"
        if not inference_checkpoint.exists():
            torch_save(inference_checkpoint, {"model": state["ema"]})
    rows = dataset_rows(cfg, "valid")
    train_ids = [r["image_id"] for r in dataset_rows(cfg, "train")]
    manifest = cache_model(
        cfg,
        inference_checkpoint,
        rows,
        Path(run) / f"evaluation_{checkpoint.stem}",
        device,
        train_ids,
        [r["image_id"] for r in rows],
    )
    result = reserve_results()
    summary = report_predictions(manifest, cfg["paths"]["dataset_manifest"], result)
    json_write(
        Path(run) / "result.json",
        {"result_dir": str(result), "val_image_miou": summary["val_image_miou"]},
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    a71 = next(
        (
            r["image_miou"]
            for r in summary["per_image"]
            if r["image_id"] == "A71_step2_sample1"
        ),
        None,
    )
    write_csv(
        result / "experiments.csv",
        [
            {
                "experiment_id": cfg["run"]["name"],
                "seed": cfg["run"]["seed"],
                "parent_checkpoint": cfg["training"]["initialize_from"],
                "training_recipe": cfg["training"]["recipe"],
                "train_ids_hash": ids_hash(train_ids),
                "valid_ids_hash": ids_hash([r["image_id"] for r in rows]),
                "geometry_profile": cfg["geometry"]["profile"],
                "branch_flags": cfg["model"]["new_branches"],
                "sampler_config_hash": digest(cfg["sampling"]),
                "loss_config_hash": digest(cfg["loss"]),
                "optimizer_updates": state.get("step"),
                "selection_metric": "val_image_miou",
                "selection_protocol_hash": digest(cfg["evaluation"]),
                "selected_step": state.get("step"),
                "checkpoint_hash": sha256(checkpoint),
                "inference_config_hash": inference_hash(cfg),
                "val_image_miou": summary["val_image_miou"],
                "val_global_miou": summary["val_global_miou"],
                "mean_iou_Al3Ni": summary["mean_class_iou"][1],
                "mean_iou_EutecticSi": summary["mean_class_iou"][3],
                "A71_image_miou_diagnostic": a71,
                "status": "RUN_COMPLETED",
            }
        ],
    )
    return summary
