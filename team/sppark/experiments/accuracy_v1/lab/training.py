from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from alphase.data.splits import select_rows
from alphase.engine.trainer import train
from alphase.utils import code_hash, json_hash, sha256, write_json
from lab.workspace import (BASELINE, SOURCE, WORKSPACE, experiment_source_hash, load_experiment,
                           read_json, run_dir, verify_preservation)


@contextmanager
def change_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def training_config(name, overrides=(), pretrained=None, smoke=False):
    cfg = load_experiment(name, overrides, pretrained)
    if smoke:
        cfg["experiment"] = name + "_smoke"
        cfg["train"].update(eval_every_updates=1, checkpoint_every_updates=1, log_every_updates=1, num_workers=1)
        for stage in cfg["train"]["stages"]:
            stage["max_updates"] = 1
            stage["warmup_updates"] = 1
    if cfg["train"]["effective_batch_size"] != 8:
        raise ValueError("Keep effective_batch_size=8 for these controlled experiments")
    return cfg


def run_training(args):
    rank = int(os.environ.get("RANK", 0))
    cfg = training_config(args.experiment, args.overrides, args.pretrained, args.smoke)
    folder = run_dir(args.experiment, args.split, args.smoke)
    verify_preservation(full=False)
    if args.experiment == "dino_large":
        checkpoint = cfg["model"].get("pretrained_checkpoint")
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError("DINOv3 official ViT-L weights are required: add --pretrained /path/to/original_filename.pth")
        if "dinov3_vitl16_" not in Path(checkpoint).name:
            raise ValueError("DINO-L expects its original official dinov3_vitl16_*.pth filename")
    if folder.exists() and any(folder.iterdir()) and not args.resume:
        raise FileExistsError(f"Experiment already exists: {folder}. Use --resume; no overwrite option is provided.")
    resume = folder/"last.pt" if args.resume else None
    if resume and not resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint missing: {resume}")
    # All paths are absolute. The pinned source's environment recorder can find
    # its own unchanged third_party/dinov3 commit from this working directory.
    with change_directory(SOURCE):
        result = train(cfg, split=args.split, run=folder, resume=resume, stop_after=args.stop_after)
    if rank == 0:
        write_json(folder/"experiment_provenance.json", dict(
            experiment=args.experiment, smoke_test=args.smoke, split=args.split,
            baseline_checkpoint_sha256=read_json(BASELINE/"run/evaluation/provenance.json")["checkpoint_sha256"],
            baseline_model_source_sha256=code_hash(), experiment_source_sha256=experiment_source_hash(),
            config_hash=json_hash(cfg), dataset_manifest_sha256=sha256(BASELINE/"data/manifest.csv"),
            completed_at=time.time(), pretrained_sha256=sha256(cfg["model"]["pretrained_checkpoint"]) if cfg["model"].get("pretrained_checkpoint") else None))
    return result
