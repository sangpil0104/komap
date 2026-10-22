#!/usr/bin/env python3
"""Local entry point. Training requires an explicit --execute argument."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from runtime import ROOT, DATA, audit, read_json, write_json


def gpu_list(value):
    result = [int(v) for v in value.split(",")]
    if not result or len(result) != len(set(result)) or any(g not in range(6) for g in result):
        raise ValueError("GPU list must contain distinct physical IDs from 0 through 5")
    return result


def terminate_jobs(jobs):
    for p, _, _ in jobs.values():
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
    for p, _, handle in jobs.values():
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        handle.close()


def train(args):
    from tqdm import tqdm
    gpus = gpu_list(args.gpus)
    check = audit(args.data)
    config = read_json(ROOT / "configs/model326.json")
    unsupported = {"accum": 1, "ema": 0, "cldice_w": 0, "grad_ckpt": False,
                   "no_valid_eval": True, "target_width": 0}
    if any(config[k] != v for k, v in unsupported.items()):
        raise ValueError("This runner implements the recorded Model326 recipe; unsupported configuration override")
    print(json.dumps({"training": "enabled" if args.execute else "DRY RUN: no training",
                      "gpu_pool": gpus, "concurrent_folds": min(4, len(gpus)),
                      "steps_per_fold": config["iters"], "batch_per_fold": config["bs"],
                      "folds": check["folds"]}, indent=2), flush=True)
    if not args.execute:
        return
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    # Children retain this descriptor so an abruptly killed parent does not allow duplicate training.
    lock = (outputs / "model326.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another Model326 run is active; check outputs/logs (no new training started)")
    run = outputs / "runs" / ("Model326_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    shutil.copy2(ROOT / "configs/folds.json", run / "folds.json")
    write_json(run / "preflight.json", check)
    snapshot = run / "code_snapshot"
    snapshot.mkdir()
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in (".py", ".sh", ".json", ".md", ".txt"):
            shutil.copy2(p, snapshot / p.name)
    shutil.copytree(ROOT / "configs", snapshot / "configs")
    shutil.copytree(ROOT / "source", snapshot / "source", ignore=shutil.ignore_patterns("__pycache__"))
    (run / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    print(f"RUN_DIRECTORY={run}", flush=True)
    write_json(outputs / "latest_model326.json", {"run": str(run)})
    pending, available, jobs = deque(range(1, 5)), deque(gpus), {}
    started = time.monotonic()
    progress = tqdm(total=4 * config["iters"], desc="Model326 four folds", unit="step", mininterval=10)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        while pending or jobs:
            while pending and available:
                fold, gpu = pending.popleft(), available.popleft()
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                           MKL_NUM_THREADS="4", NO_ALBUMENTATIONS_UPDATE="1", PYTHONUNBUFFERED="1")
                handle = (run / f"fold{fold}.log").open("w")
                cmd = [sys.executable, "-u", str(ROOT / "run.py"), "_fold326", "--run", str(run),
                       "--fold", str(fold), "--data", str(args.data)]
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                        start_new_session=True, pass_fds=(lock.fileno(),))
                jobs[fold] = (proc, gpu, handle)
                print(f"fold{fold}: GPU {gpu}, PID {proc.pid}, log {run / f'fold{fold}.log'}", flush=True)
            for fold, (proc, gpu, handle) in list(jobs.items()):
                code = proc.poll()
                if code is None:
                    continue
                handle.close()
                if code:
                    raise RuntimeError(f"fold{fold} failed ({code}); see {run / f'fold{fold}.log'}")
                del jobs[fold]
                available.append(gpu)
            states = [read_json(p) for p in sorted(run.glob("fold*/progress.json"))]
            count = sum(s.get("step", 0) for s in states)
            progress.update(max(0, count - progress.n))
            progress.set_postfix_str(" ".join(f"f{s['fold']}:{s['phase']}" for s in states))
            write_json(run / "status.json", {"phase": "fold_training_and_oof", "states": states,
                                              "elapsed_seconds": time.monotonic() - started})
            if pending or jobs:
                time.sleep(5)
        progress.close()
        from model326 import aggregate_oof
        aggregate_oof(run, args.data)
        print("Four folds and OOF complete. Starting validation ensemble inference.", flush=True)
        write_json(run / "status.json", {"phase": "validation", "elapsed_seconds": time.monotonic() - started})
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpus[0]), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "run.py"), "evaluate326", "--run", str(run),
                                 "--data", str(args.data)], env=env, cwd=ROOT, start_new_session=True,
                                pass_fds=(lock.fileno(),))
        # Keep evaluation in the same lifecycle/lock as training; its tqdm is visible in the nohup log.
        jobs[0] = (proc, gpus[0], open(os.devnull, "w"))
        code = proc.wait()
        if code:
            raise RuntimeError(f"Validation failed ({code}); completed weights remain in {run}")
        jobs.pop(0)[2].close()
        write_json(run / "status.json", {"phase": "complete", "elapsed_seconds": time.monotonic() - started})
        print(f"COMPLETE: {run}", flush=True)
    except BaseException as error:
        terminate_jobs(jobs)
        write_json(run / "status.json", {"phase": "failed_or_interrupted", "error": str(error),
                                          "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        progress.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "check338", "prepare326", "train326", "_fold326", "evaluate326", "evaluate338"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=DATA)
        if name in ("check338", "evaluate338"):
            p.add_argument("--config", type=Path, default=ROOT / "configs/model338.json")
        if name == "check338":
            p.add_argument("--deep", action="store_true", help="also load/validate all supplied artifacts")
        if name in ("_fold326", "evaluate326"):
            p.add_argument("--run", type=Path, required=True)
        if name == "_fold326":
            p.add_argument("--fold", type=int, choices=range(1, 5), required=True)
        if name == "train326":
            p.add_argument("--gpus", default=os.environ.get("LHS_GPUS", "0,1,2,3,4,5"))
            p.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "check":
        from model338 import artifact_status, route_functions
        info = audit(args.data)
        route_functions()
        info["model338"] = artifact_status(read_json(ROOT / "configs/model338.json"))
        write_json(ROOT / "reports/preflight.json", info)
        print(json.dumps({"dataset_and_source": "OK", "folds": info["folds"], "model338": info["model338"]}, indent=2))
    elif args.command == "check338":
        from model338 import artifact_status, load_inputs
        config = read_json(args.config)
        info = artifact_status(config)
        if args.deep and info["files_present"]:
            load_inputs(config, args.data)
            info["content_validation"] = "passed"
        print(json.dumps(info, indent=2))
        return 0 if info["files_present"] else 2
    elif args.command == "prepare326":
        from runtime import common
        model = common.build_model("unet", "efficientnet-b4", pretrained=True)
        print(f"ImageNet initialization cached; parameters={sum(p.numel() for p in model.parameters()):,}; training not started")
    elif args.command == "train326":
        train(args)
    elif args.command == "_fold326":
        from model326 import train_fold
        train_fold(args.run, args.fold, args.data)
    elif args.command == "evaluate326":
        from model326 import evaluate
        print(f"RESULT_DIRECTORY={evaluate(args.run, args.data)}", flush=True)
    elif args.command == "evaluate338":
        from model338 import evaluate
        print(f"RESULT_DIRECTORY={evaluate(args.config, args.data)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
