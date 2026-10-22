"""One independent process per authorized GPU; subprocess failures propagate."""

import os
import subprocess
import sys
import time
from pathlib import Path
from tqdm import tqdm
from .io import ROOT, json_write


def run_jobs(jobs, gpus, execute=False):
    if not gpus or not set(gpus) <= set(range(6)):
        raise ValueError("Only physical GPUs0–5 allowed")
    plan = [
        {
            "name": j["name"],
            "command": [sys.executable, "-m", "alphase.cli", *j["args"]],
            "gpu_round_robin": gpus[i % len(gpus)],
        }
        for i, j in enumerate(jobs)
    ]
    if not execute:
        for row in plan:
            print(row)
        print(f"DRY RUN: {len(jobs)} jobs, GPUs {gpus}. Add --execute to start.")
        return plan
    pending = list(jobs)
    active = {}
    errors = []
    with tqdm(total=len(jobs), desc="GPU jobs", unit="job") as bar:
        try:
            while pending or active:
                for gpu in gpus:
                    if gpu not in active and pending:
                        job = pending.pop(0)
                        log = Path(job["log"])
                        log.parent.mkdir(parents=True, exist_ok=True)
                        stream = log.open("a")
                        env = os.environ.copy()
                        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                        env["PYTHONUNBUFFERED"] = "1"
                        env["OMP_NUM_THREADS"] = "1"
                        cmd = [
                            sys.executable,
                            "-m",
                            "alphase.cli",
                            *job["args"],
                            "--execute",
                            "--device",
                            "cuda:0",
                        ]
                        proc = subprocess.Popen(
                            cmd,
                            cwd=ROOT,
                            env=env,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                        )
                        active[gpu] = (proc, stream, job)
                for gpu, (proc, stream, job) in list(active.items()):
                    code = proc.poll()
                    if code is not None:
                        stream.close()
                        del active[gpu]
                        bar.update()
                        if code:
                            errors.append((job["name"], code, str(job["log"])))
                            pending.clear()
                bar.set_postfix(active=",".join(map(str, active)), failed=len(errors))
                time.sleep(0.5)
        finally:
            for proc, stream, job in active.values():
                if proc.poll() is None:
                    proc.terminate()
                proc.wait()
                stream.close()
    if errors:
        raise RuntimeError(f"GPU jobs failed: {errors}")
    return plan
