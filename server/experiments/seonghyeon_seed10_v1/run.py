"""Prepare or launch ten independent seeds on physical GPUs 0 through 5."""
import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.metadata
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

from report import allocate_results, suite_report
from runtime import duration, read_json, suite_eta, write_json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
GPUS = list(range(6))
SEEDS = list(range(10))
WEIGHTS = HERE / "assets" / "efficientnet_b4_imagenet_encoder.pt"
BASE = dict(data=str(PROJECT / "dataset"), encoder_weights=str(WEIGHTS),
            iters=16000, crop=1536, bs=2, target_width=4000, eval_every=1000, workers=6)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment():
    names = ["torch", "torchvision", "segmentation-models-pytorch", "timm",
             "albumentations", "opencv-python-headless", "numpy", "pillow", "tqdm"]
    return dict(python=sys.version, executable=sys.executable,
                packages={n: importlib.metadata.version(n) for n in names})


def prepare():
    """Download/verify pretrained encoder on CPU, never train."""
    import torch
    from vendor.common import build_model
    torch.set_num_threads(2)
    WEIGHTS.parent.mkdir(exist_ok=True)
    if WEIGHTS.exists():
        meta = read_json(WEIGHTS.with_suffix(".json"))
        if meta["sha256"] != sha256(WEIGHTS):
            raise ValueError("Prepared encoder checksum mismatch")
        print(f"[prepare] verified cached encoder: {WEIGHTS}", flush=True)
        return
    print("[prepare] loading ImageNet EfficientNet-B4 encoder on CPU (no training)", flush=True)
    model = build_model("unet", "efficientnet-b4", pretrained=True)
    tmp = WEIGHTS.with_suffix(".tmp")
    torch.save(model.encoder.state_dict(), tmp)
    tmp.replace(WEIGHTS)
    write_json(WEIGHTS.with_suffix(".json"), dict(sha256=sha256(WEIGHTS),
               source="segmentation_models_pytorch efficientnet-b4 encoder_weights=imagenet",
               environment=environment()))
    print(f"[prepare] ready: {WEIGHTS}", flush=True)


def preflight():
    import numpy as np
    from PIL import Image
    from vendor.common import list_split, load_mask
    import torch
    print("[preflight] checking packages, source hashes, train/valid data and GPUs", flush=True)
    env = environment()
    for relative, expected in read_json(HERE / "preservation.json").items():
        if sha256(PROJECT / relative) != expected:
            raise ValueError(f"Original source changed: {relative}")
    data_files = {}
    for split, count in [("train", 70), ("valid", 20)]:
        pairs = list_split(BASE["data"], split)
        if len(pairs) != count:
            raise ValueError(f"{split}: expected {count}, found {len(pairs)}")
        for ip, mp in pairs:
            if mp is None:
                raise ValueError(f"Missing mask: {ip}")
            mask = load_mask(mp)
            with Image.open(ip) as im:
                if im.size != (mask.shape[1], mask.shape[0]):
                    raise ValueError(f"Shape mismatch: {ip}")
            for path in [ip, mp]:
                data_files[str(Path(path).relative_to(PROJECT))] = sha256(path)
        print(f"[preflight] {split}={count}; shapes/palette verified", flush=True)
    gpu_text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.free,memory.total", "--format=csv,noheader,nounits"
    ], text=True)
    gpu_lines = {int(line.split(",")[0]): line for line in gpu_text.strip().splitlines()}
    for gpu in GPUS:
        if gpu not in gpu_lines:
            raise ValueError(f"Physical GPU {gpu} missing")
        if int(gpu_lines[gpu].split(",")[-2].strip()) < 15000:
            raise ValueError(f"GPU {gpu} has <15000 MiB free; free this GPU before starting")
        print(f"[preflight] GPU {gpu_lines[gpu]}", flush=True)
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if not WEIGHTS.is_file() or sha256(WEIGHTS) != read_json(WEIGHTS.with_suffix(".json"))["sha256"]:
        raise ValueError("Run run.py --prepare first; pretrained encoder missing or changed")
    # CPU construction and strict loading validate encoder compatibility without optimization.
    from vendor.common import build_model
    torch.set_num_threads(2)
    model = build_model("unet", "efficientnet-b4", pretrained=False)
    model.encoder.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    if tuple(output.shape) != (1, 4, 64, 64) or not np.isfinite(output.numpy()).all():
        raise ValueError("CPU forward validation failed")
    record = dict(environment=env, dataset_sha256=data_files, gpus=gpu_text,
                  encoder_sha256=sha256(WEIGHTS), free_disk_gib=shutil.disk_usage(HERE).free / 2**30,
                  training_started=False)
    print(f"[preflight] PASS; free disk={record['free_disk_gib']:.1f}GiB; no training performed", flush=True)
    return record


def stream_output(proc, logfile, messages, label):
    with Path(logfile).open("x", buffering=1) as stream:
        for line in proc.stdout:
            stream.write(line)
            messages.put(f"{label} {line.rstrip()}")
    proc.stdout.close()


def is_complete(job, result):
    run = Path(job["run"])
    status_path = run / "status.json"
    if not status_path.exists() or read_json(status_path)["state"] != "completed":
        return False
    metric = Path(result) / "metrics" / f"seed_{job['seed']:02d}.json"
    if not (run / "last.pt").is_file() or not metric.is_file():
        return False
    rows = read_json(metric).get("per_image", [])
    names = {Path(r["image"]).stem + ".png" for r in rows}
    if len(rows) != 20 or len(names) != 20:
        return False
    for folder in ["predictions", "valid_mask_comparison"]:
        actual = {p.name for p in (Path(result) / folder / f"seed_{job['seed']:02d}").glob("*.png")}
        if actual != names:
            return False
    return True


def run_queue(jobs, source, session, result, heartbeat=30, gpus=GPUS, command_builder=None):
    """One process per GPU; launch the next seed immediately when a slot frees."""
    messages = queue.Queue()
    pending = deque(jobs)
    active = {}
    completed_status = []
    started = time.monotonic()
    last_heartbeat = 0.0
    final_state = "failed"
    source = Path(source)

    def save_status(state, remaining=None):
        elapsed = time.monotonic() - started
        record = dict(state=state, jobs=jobs, elapsed_seconds=elapsed, eta_seconds=remaining,
                      session=str(session), result=str(result), updated_at=time.time())
        write_json(Path(session) / "suite_status.json", record)
        write_json(Path(result) / "suite.json", record)
        return record

    def drain():
        while True:
            try:
                print(messages.get_nowait(), flush=True)
            except queue.Empty:
                break

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    old_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while pending or active:
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                job = pending.popleft()
                job.update(state="running", gpu=gpu)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=str(job["seed"]),
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           NO_ALBUMENTATIONS_UPDATE="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8")
                cmd = ([sys.executable, "-B", "-u", str(source / "train.py"), "--job", str(job["job_file"])]
                       if command_builder is None else command_builder(job, gpu))
                proc = subprocess.Popen(cmd, env=env, cwd=source, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
                job["pid"] = proc.pid
                reader = threading.Thread(target=stream_output, args=(proc, Path(job["run"]) / "worker.log",
                                          messages, f"[GPU {gpu} seed {job['seed']:02d}]"), daemon=True)
                active[gpu] = (proc, reader, job)
                reader.start()
                print(f"[START] GPU={gpu} seed={job['seed']} pid={proc.pid} run={job['run']}", flush=True)
            drain()
            changed = False
            for gpu, (proc, reader, job) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                reader.join(timeout=5)
                drain()
                job["exit_code"] = code
                job["state"] = "completed" if code == 0 and is_complete(job, result) else "failed"
                if job["state"] == "completed":
                    completed_status.append(read_json(Path(job["run"]) / "status.json"))
                print(f"[{job['state'].upper()}] GPU={gpu} seed={job['seed']} exit_code={code}", flush=True)
                del active[gpu]
                changed = True
            if changed:
                suite_report(result, jobs, "running", time.monotonic() - started)
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat or changed:
                statuses = []
                for gpu, (_, _, job) in active.items():
                    status_file = Path(job["run"]) / "status.json"
                    s = read_json(status_file) if status_file.exists() else dict(state="loading")
                    statuses.append(s)
                    age = time.time() - s.get("updated_at", time.time())
                    print(f"[STATUS] GPU={gpu} seed={job['seed']:02d} phase={s['state']} "
                          f"iter={s.get('iteration', 0)}/{BASE['iters']} "
                          f"seed_ETA={duration(s.get('eta_seconds'))} last_update={age:.0f}s", flush=True)
                remaining = suite_eta(statuses, len(pending), len(gpus), completed_status)
                record = save_status("running", remaining)
                done = sum(j["state"] == "completed" for j in jobs)
                failed = sum(j["state"] == "failed" for j in jobs)
                iterations = done * BASE["iters"] + sum(s.get("iteration", 0) for s in statuses)
                budget = len(jobs) * BASE["iters"]
                timed = len(statuses) and all(s.get("d4_seconds_per_image") for s in statuses)
                quality = "measured+remaining_work" if completed_status or timed else "provisional_includes_validation"
                finish = ((datetime.now(timezone.utc) + timedelta(seconds=remaining)).strftime("%Y-%m-%dT%H:%M:%SZ")
                          if remaining is not None else "measuring")
                print(f"[SUITE] done={done}/{len(jobs)} failed={failed} active={len(active)} "
                      f"queued={len(pending)} elapsed={duration(record['elapsed_seconds'])} "
                      f"iterations={iterations}/{budget} ({100*iterations/budget:.2f}%) "
                      f"total_ETA={duration(remaining)} finish_UTC={finish} estimate={quality}", flush=True)
                last_heartbeat = now
            if pending or active:
                time.sleep(0.5)
        final_state = "completed" if all(j["state"] == "completed" for j in jobs) else "failed"
    except BaseException:
        final_state = "interrupted"
        raise
    finally:
        # Kill each worker's process group, including its DataLoader children.
        for proc, _, job in active.values():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            job["state"] = "interrupted"
        for proc, reader, _ in active.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            reader.join(timeout=2)
        drain()
        for s, old in old_handlers.items():
            signal.signal(s, old)
        record = save_status(final_state, 0 if final_state == "completed" else None)
        report = suite_report(result, jobs, final_state, record["elapsed_seconds"])
        print(f"[SUITE {final_state.upper()}] report={report} "
              f"elapsed={duration(record['elapsed_seconds'])}", flush=True)
    return 0 if final_state == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="Cache ImageNet encoder on CPU; no training")
    mode.add_argument("--preflight", action="store_true", help="Validate readiness; no training")
    mode.add_argument("--dry-run", action="store_true", help="Print execution plan; no GPU work")
    args = parser.parse_args()
    if args.dry_run:
        print(f"seeds={SEEDS}\nphysical GPUs={GPUS}\nmax_parallel=6\nconfig={BASE}\n"
              "result=results/model_results_MMDD_N\ntraining_started=False")
        return 0
    if args.prepare:
        prepare()
        return 0
    if args.preflight:
        record = preflight()
        write_json(HERE / "preflight.json", record)
        return 0
    output = HERE / "outputs"
    output.mkdir(exist_ok=True)
    with (output / "suite.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another seed10 suite is already running")
        record = preflight()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        session = output / stamp
        session.mkdir()
        source = session / "source"
        source.mkdir()
        for p in HERE.iterdir():
            if p.is_file() and p.suffix in {".py", ".sh", ".md", ".json", ".txt"}:
                shutil.copy2(p, source / p.name)
        shutil.copytree(HERE / "vendor", source / "vendor", ignore=shutil.ignore_patterns("__pycache__"))
        record["source_sha256"] = {str(p.relative_to(source)): sha256(p)
                                   for p in source.rglob("*") if p.is_file()}
        write_json(session / "provenance.json", record)
        result = allocate_results(PROJECT / "results")
        (result / "metrics").mkdir()
        jobs = []
        for seed in SEEDS:
            run = session / f"seed_{seed:02d}"
            run.mkdir()
            job_file = run / "job.json"
            write_json(job_file, dict(BASE, seed=seed, run=str(run), result=str(result)))
            jobs.append(dict(seed=seed, state="queued", run=str(run), job_file=str(job_file)))
        suite_report(result, jobs, "starting", 0)
        write_json(output / "latest_run.json", dict(session=str(session), result=str(result)))
        print(f"[SUITE] seeds={SEEDS} GPUs={GPUS} max_parallel=6 result={result}", flush=True)
        print("[SUITE] total_ETA starts after the first iterations; validation proxy is replaced by measurements", flush=True)
        return run_queue(jobs, source, session, result)


if __name__ == "__main__":
    raise SystemExit(main())
