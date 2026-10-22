from __future__ import annotations

import argparse
import os
import subprocess
import sys
from itertools import product

import torch
from tqdm import tqdm

from alphase.data.audit import verify_manifest
from alphase.utils import json_hash, write_json
from auglab.workspace import (ROOT, NAMES, SPLITS, DEFAULT_EXPERIMENTS, config, output_path,
                              read_json, run_dir, verify_preservation)


def launch_train(args):
    cfg = config(args.experiment, args.seed, args.smoke)
    if args.worker:
        from auglab.training import run_training
        return run_training(args)
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Launch this command once; it starts distributed workers itself")
    if args.gpus > torch.cuda.device_count():
        raise ValueError(f"Requested {args.gpus} GPUs, visible: {torch.cuda.device_count()}")
    folder = run_dir(args.experiment, args.seed, args.split, args.smoke)
    if folder.exists() and any(folder.iterdir()) and not args.resume:
        raise FileExistsError(f"Run exists: {folder}; use --resume")
    cmd = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.gpus}",
           str(ROOT / "run.py"), "train", args.experiment, "--seed", str(args.seed), "--split", args.split,
           "--gpus", str(args.gpus), "--worker"]
    if args.smoke:
        cmd.append("--smoke")
    if args.resume:
        cmd.append("--resume")
    if args.stop_after is not None:
        cmd.extend(["--stop-after", str(args.stop_after)])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("OMP_NUM_THREADS", "4")
    subprocess.run(cmd, env=env, check=True)


def suite(args):
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Run suite once; it starts workers itself")
    if args.gpus > torch.cuda.device_count():
        raise ValueError("Not enough visible GPUs")
    print("Preservation:", verify_preservation(), flush=True)
    jobs = list(product(args.experiments, args.seeds))
    if len(jobs) != len(set(jobs)):
        raise ValueError("Duplicate experiment/seed in suite")
    completed = []
    from auglab.evaluation import evaluate_run, compare
    # One completion includes training plus plain/D4 validation. First suite ETA
    # needs a completed job; each inner update bar estimates its own remaining time.
    with tqdm(total=len(jobs), desc="Experiments (train + evaluate)", unit="experiment", dynamic_ncols=True) as bar:
        for name, seed in jobs:
            bar.set_postfix_str(f"{name} seed={seed}")
            cfg = config(name, seed, args.smoke)
            folder = run_dir(name, seed, "official", args.smoke)
            last = folder / "last.pt"
            done = False
            if last.is_file():
                state = torch.load(last, map_location="cpu", weights_only=False)
                if state["config_hash"] != json_hash(cfg) or state["dataset_hash"] != verify_manifest(cfg["data"]["manifest"]):
                    raise ValueError(f"Existing run has different code/config/data: {folder}")
                done = state["completed"]
                if not done and state["world_size"] != args.gpus:
                    raise ValueError("Resume with the original GPU count")
                del state
            if not done:
                launch_train(argparse.Namespace(experiment=name, seed=seed, split="official", smoke=args.smoke,
                    gpus=args.gpus, worker=False, resume=last.exists(), stop_after=None))
            else:
                tqdm.write(f"Completed training reused: {name} seed={seed}")
            for d4 in (False, True):
                evaluate_run(name, seed, smoke=args.smoke, d4=d4)
            completed.append(dict(experiment=name, seed=seed))
            write_json(output_path("smoke_suite_status.json" if args.smoke else "suite_status.json"),
                       dict(state="completed" if len(completed) == len(jobs) else "running", completed=completed,
                            total=len(jobs), terminal_progress=True))
            bar.update(1)
    if not args.smoke:
        compare()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Augmentation experiments; terminal tqdm only, no web monitor")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("experiment", choices=NAMES)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--split", choices=SPLITS, default="official")
    train.add_argument("--gpus", choices=[1, 2, 4, 8], type=int, default=8)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-after", type=int)
    train.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    group = sub.add_parser("suite")
    group.add_argument("--gpus", choices=[1, 2, 4, 8], type=int, default=8)
    group.add_argument("--seeds", type=int, nargs="+", default=[2026])
    group.add_argument("--experiments", choices=NAMES, nargs="+", default=list(DEFAULT_EXPERIMENTS))
    group.add_argument("--smoke", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("experiment", choices=NAMES)
    ev.add_argument("--seed", type=int, default=2026)
    ev.add_argument("--split", choices=SPLITS, default="official")
    ev.add_argument("--d4", action="store_true")
    ev.add_argument("--smoke", action="store_true")
    sub.add_parser("compare")
    sub.add_parser("verify")
    sub.add_parser("list")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            if args.stop_after is not None and args.stop_after < 1:
                raise ValueError("stop-after must be positive")
            launch_train(args)
        elif args.command == "suite":
            suite(args)
        elif args.command == "evaluate":
            from auglab.evaluation import evaluate_run
            evaluate_run(args.experiment, args.seed, args.split, args.smoke, args.d4)
        elif args.command == "compare":
            from auglab.evaluation import compare
            compare()
        elif args.command == "verify":
            print(verify_preservation())
        else:
            for name in NAMES:
                print(name, "(optional control)" if name == "control" else "(default suite)")
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
