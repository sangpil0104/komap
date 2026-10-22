from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from contextlib import contextmanager
from itertools import product

import torch
from tqdm import tqdm
from alphase.data.audit import verify_manifest
from alphase.utils import json_hash, write_json
from refinelab.workspace import ROOT, config, output_path, plan, read_json, run_dir, verify


@contextmanager
def lock(name):
    path = output_path("locks", name + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is using {name}") from None
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def controller_check(gpus=None):
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Launch once; this runner starts the GPU workers")
    if gpus is not None and gpus > torch.cuda.device_count():
        raise ValueError(f"Requested {gpus} GPUs; visible {torch.cuda.device_count()}")


def prepare():
    from refinelab.mining import ensure_mining
    with lock("mining"):
        return ensure_mining()


def launch_train(args):
    if args.worker:
        from refinelab.training import train
        return train(args)
    controller_check(args.gpus)
    cfg = config(args.experiment, args.seed, args.smoke)
    with lock(f"train-{args.experiment}-{args.seed}-{'smoke' if args.smoke else 'real'}"):
        folder = run_dir(args.experiment, args.seed, args.smoke)
        if folder.exists() and any(folder.iterdir()) and not args.resume:
            raise FileExistsError(f"Run exists: {folder}; use --resume")
        if cfg["refinement"]["hard"]:
            prepare()
        cmd = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={args.gpus}",
               str(ROOT / "run.py"), "train", args.experiment, "--seed", str(args.seed), "--gpus", str(args.gpus), "--worker"]
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
    controller_check(args.gpus)
    jobs = list(product(args.experiments, args.seeds))
    if len(set(jobs)) != len(jobs) or any(seed < 0 for _, seed in jobs):
        raise ValueError("Duplicate jobs or invalid seed")
    from refinelab.evaluation import evaluate_run, compare
    with lock("suite-smoke" if args.smoke else "suite"):
        print("기존 파일 보존 검사:", verify(), flush=True)
        if any(plan()[name].get("hard") for name, _ in jobs):
            prepare()
        completed = []
        status_path = output_path("smoke_suite_status.json" if args.smoke else "suite_status.json")
        with tqdm(total=len(jobs), desc="Sequential experiments", unit="experiment", dynamic_ncols=True) as bar:
            for name, seed in jobs:
                cfg = config(name, seed, args.smoke)
                folder = run_dir(name, seed, args.smoke)
                last = folder / "last.pt"
                done = False
                if last.is_file():
                    saved = torch.load(last, map_location="cpu", weights_only=False)
                    if saved["config_hash"] != json_hash(cfg) or saved["dataset_hash"] != verify_manifest(cfg["data"]["manifest"]):
                        raise ValueError(f"Existing run has different code/config/data: {folder}")
                    done = saved["completed"]
                    if not done and saved["world_size"] != args.gpus:
                        raise ValueError("Resume with the same GPU count")
                    del saved
                bar.set_postfix_str(f"{name} seed={seed}")
                write_json(status_path, dict(state="running", active=name, completed=completed, total=len(jobs)))
                if not done:
                    launch_train(argparse.Namespace(experiment=name, seed=seed, smoke=args.smoke, gpus=args.gpus,
                                                    worker=False, resume=last.exists(), stop_after=None))
                else:
                    tqdm.write(f"이미 학습 완료: {name}; 결과를 재사용합니다 (오류 아님)")
                for d4 in (False, True):
                    evaluate_run(name, seed, args.smoke, d4)
                completed.append(dict(experiment=name, seed=seed))
                write_json(status_path, dict(state="completed" if len(completed) == len(jobs) else "running",
                                             completed=completed, total=len(jobs), terminal_only=True))
                if not args.smoke:
                    compare()
                bar.update(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="10 isolated refinement experiments; terminal tqdm only")
    sub = parser.add_subparsers(dest="command", required=True)
    names = list(plan())
    p = sub.add_parser("train")
    p.add_argument("experiment", choices=names)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--gpus", type=int, choices=[1, 2, 4, 8], default=8)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p = sub.add_parser("suite")
    p.add_argument("--gpus", type=int, choices=[1, 2, 4, 8], default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[2026])
    p.add_argument("--experiments", choices=names, nargs="+", default=names)
    p.add_argument("--smoke", action="store_true")
    for command in ("evaluate", "predict"):
        p = sub.add_parser(command)
        p.add_argument("experiment", choices=names)
        p.add_argument("--seed", type=int, default=2026)
        p.add_argument("--d4", action="store_true")
        p.add_argument("--smoke", action="store_true")
    for command in ("prepare", "compare", "verify", "list"):
        sub.add_parser(command)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            if args.stop_after is not None and args.stop_after < 1:
                raise ValueError("stop-after must be positive")
            launch_train(args)
        elif args.command == "suite":
            suite(args)
        elif args.command == "prepare":
            controller_check()
            verify(full=False)
            prepare()
        elif args.command in ("evaluate", "predict"):
            controller_check()
            from refinelab.evaluation import evaluate_run, predict_test
            (evaluate_run if args.command == "evaluate" else predict_test)(args.experiment, args.seed, args.smoke, args.d4)
        elif args.command == "compare":
            from refinelab.evaluation import compare
            compare()
        elif args.command == "verify":
            print(verify())
        else:
            for name, entry in plan().items():
                print(f"{name}: {entry['description']}")
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")
