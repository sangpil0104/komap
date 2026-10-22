from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch

from alphase.utils import json_hash, write_json
from lab.inference import ensemble, evaluate_one
from lab.report import compare_results
from lab.training import training_config
from lab.workspace import WORKSPACE, output_path, plan, require_manifest, run_dir, verify_preservation


def launch_training(arguments, gpus):
    if gpus not in (1, 2, 4, 8):
        raise ValueError("GPU count must be 1, 2, 4, or 8")
    available = torch.cuda.device_count()
    if gpus > available:
        raise ValueError(f"Requested {gpus} GPUs, but only {available} are visible")
    command = [sys.executable, "-B"]
    if gpus > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"]
    command += [str(WORKSPACE/"run.py"), *arguments]
    print("Launching:", " ".join(command), flush=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(command, cwd=Path.cwd(), env=environment, check=True)


def completed_run(name, split="official", smoke=False):
    folder = run_dir(name, split, smoke)
    last = folder/"last.pt"
    if not last.exists():
        return False
    state = torch.load(last, map_location="cpu", weights_only=False)
    cfg = training_config(name, smoke=smoke)
    if state["config_hash"] != json_hash(cfg) or state["dataset_hash"] != require_manifest() or state["split"] != split:
        raise ValueError(f"Stored experiment settings/data differ from this suite: {folder}")
    if state["completed"] and not (folder/"best.pt").exists():
        raise ValueError(f"Completed run has no best checkpoint: {folder}")
    return state["completed"]


def run_suite(gpus=8):
    from lab.monitor import estimated_seconds
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Run suite as one Python process; it starts distributed workers itself")
    verify_preservation(full=True)
    journal = dict(state="running", completed=[], gpus=gpus, estimated_seconds=estimated_seconds(gpus), phase="학습 준비")
    journal_path = output_path("suite_status.json")
    write_json(journal_path, journal)
    try:
        for name in plan()["default_suite"]:
            journal.update(active_experiment=name, phase=f"{name} 학습")
            write_json(journal_path, journal)
            folder = run_dir(name)
            if completed_run(name):
                print(f"Reusing completed experiment: {name}", flush=True)
            else:
                arguments = ["train", name, "--gpus", str(gpus)]
                if (folder/"last.pt").exists():
                    arguments.append("--resume")
                launch_training(arguments, gpus)
            for d4 in (False, True):
                journal["phase"] = f"{name} 검증 · D4={d4}"
                write_json(journal_path, journal)
                evaluate_one(name, d4=d4)
            journal["completed"].append(name)
            write_json(journal_path, journal)
            compare_results()
        journal.update(active_experiment="baseline", phase="기존 모델 기준 검증")
        write_json(journal_path, journal)
        evaluate_one("baseline", d4=False)
        evaluate_one("baseline", d4=True)
        evaluate_one("baseline", split="test", d4=True)
        for split in ("valid", "test"):
            journal.update(active_experiment="seed_ensemble", phase=f"확률 앙상블 · {split}")
            write_json(journal_path, journal)
            ensemble(WORKSPACE/"configs/ensemble_seeds.yaml", split=split)
        compare_results()
        journal["preservation"] = verify_preservation(full=True)
        journal.update(state="completed", active_experiment="완료", phase="완료")
    except BaseException as exc:
        journal.update(state="interrupted", error=str(exc))
        raise
    finally:
        write_json(journal_path, journal)
