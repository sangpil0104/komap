"""One child per physical GPU, dependency ordering, logs, and failure cleanup."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from tqdm import tqdm


def gpu_list(value):
    try:
        ids = [int(v.strip()) for v in value.split(",")]
    except ValueError as error:
        raise ValueError("GPU IDs must be comma-separated nonnegative integers") from error
    if not ids or len(ids) != len(set(ids)) or any(g < 0 for g in ids):
        raise ValueError("GPU IDs must be distinct nonnegative integers")
    return ids


def validate_tasks(tasks):
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs")
    remaining = {t["id"]: set(t["deps"]) for t in tasks}
    for task in tasks:
        if task["kind"] not in ("cpu", "gpu") or not set(task["deps"]) <= set(ids):
            raise ValueError(f"Invalid task: {task['id']}")
    done = set()
    while remaining:
        ready = [name for name, deps in remaining.items() if deps <= done]
        if not ready:
            raise ValueError("Task dependency cycle")
        for name in ready:
            done.add(name)
            del remaining[name]


def render(values, source, run, workers):
    return [v.format(source=str(source), run=str(run), workers=str(workers)) for v in values]


def write_status(path, status):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(path)


def log_tail(path):
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 4096))
        lines = stream.read().decode("utf-8", errors="replace").replace("\r", "\n").splitlines()
    return next((line.strip() for line in reversed(lines) if line.strip()), "starting")


def terminate_jobs(jobs):
    # A leader may already have exited while loader/pool descendants remain.
    for job in jobs.values():
        try:
            os.killpg(job["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    while any(j["process"].poll() is None for j in jobs.values()) and time.monotonic() < deadline:
        time.sleep(.1)
    for job in jobs.values():
        try:
            os.killpg(job["process"].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        job["process"].wait()
        job["handle"].close()


def run_tasks(tasks, run, source, gpus, gpu_targets, python, base_env, workers=4,
              lock_fd=None, poll_interval=.5, report_interval=15):
    validate_tasks(tasks)
    run = Path(run)
    (run / "logs").mkdir(parents=True, exist_ok=True)
    pending = {t["id"]: t for t in tasks}
    completed, jobs, events = {}, {}, []
    started = time.monotonic()
    last_report = -float("inf")
    event_path = run / "events.jsonl"
    if event_path.exists() or (run / "status.json").exists():
        raise FileExistsError("Refusing to reuse a scheduler output directory")
    progress = tqdm(total=len(tasks), desc="Model349 stages", unit="stage", mininterval=5, dynamic_ncols=True)

    def event(kind, **values):
        record = {"event": kind, "elapsed_seconds": time.monotonic() - started, **values}
        with event_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        events.append(record)

    def status(phase, error=None):
        active = [{"id": name, "kind": job["task"]["kind"], "gpu": job["gpu"],
                   "pid": job["process"].pid, "elapsed_seconds": time.monotonic() - job["started"],
                   "log": str(job["log"]), "last_output": log_tail(job["log"])}
                  for name, job in jobs.items()]
        write_status(run / "status.json", {"phase": phase, "gpus": gpus, "completed": completed,
                     "pending": list(pending), "active": active, "error": error,
                     "elapsed_seconds": time.monotonic() - started})
        return active

    previous_handlers = {}
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, interrupt)
    try:
        while pending or jobs:
            # Reap first, before starting any task that depends on a newly finished job.
            for name, job in list(jobs.items()):
                code = job["process"].poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f"{name} failed (exit {code}); see {job['log']}")
                for expected in render(job["task"].get("produces", []), source, run, workers):
                    if not Path(expected).exists():
                        raise RuntimeError(f"{name} completed without required output: {expected}")
                job["handle"].close()
                completed[name] = {"gpu": job["gpu"], "elapsed_seconds": time.monotonic() - job["started"]}
                event("finished", task=name, gpu=job["gpu"])
                del jobs[name]
                progress.update(1)
                tqdm.write(f"DONE {name} on {'CPU' if job['gpu'] is None else 'GPU ' + str(job['gpu'])}")
            occupied = {job["gpu"] for job in jobs.values() if job["gpu"] is not None}
            free = [g for g in gpus if g not in occupied]
            cpu_busy = any(job["gpu"] is None for job in jobs.values())
            ready = sorted((t for t in pending.values() if set(t["deps"]) <= completed.keys()),
                           key=lambda t: (t["priority"], t["serial_index"]))
            for task in ready:
                if task["kind"] == "gpu":
                    if not free:
                        continue
                    gpu = free.pop(0)
                else:
                    if cpu_busy:
                        continue
                    gpu, cpu_busy = None, True
                for required in render(task.get("requires", []), source, run, workers):
                    if not Path(required).exists():
                        raise FileNotFoundError(f"{task['id']} missing prerequisite: {required}")
                env = base_env.copy()
                env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else gpu_targets[gpu]
                env["MODEL349_TASK_ID"] = task["id"]
                env["MODEL349_PHYSICAL_GPU"] = "cpu" if gpu is None else str(gpu)
                command = [python, "-B", "-u"] + render(task["argv"], source, run, workers)
                log = run / "logs" / f"{task['id']}.log"
                handle = log.open("x")
                try:
                    process = subprocess.Popen(command, cwd=run, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                               start_new_session=True, pass_fds=(() if lock_fd is None else (lock_fd,)))
                except BaseException:
                    handle.close()
                    raise
                jobs[task["id"]] = {"process": process, "handle": handle, "log": log, "gpu": gpu,
                                     "task": task, "started": time.monotonic()}
                del pending[task["id"]]
                event("started", task=task["id"], gpu=gpu, pid=process.pid, command=command,
                      cuda_visible_devices=env["CUDA_VISIBLE_DEVICES"])
                tqdm.write(f"START {task['id']} on {'CPU' if gpu is None else 'GPU ' + str(gpu)} | {log}")
            if pending and not jobs:
                raise RuntimeError("No runnable task remains")
            if time.monotonic() - last_report >= report_interval:
                active = status("running")
                for row in active:
                    tqdm.write(f"[{row['id']} / {'CPU' if row['gpu'] is None else 'GPU ' + str(row['gpu'])}] {row['last_output']}")
                progress.set_postfix(running_gpu=sum(j["gpu"] is not None for j in jobs.values()), pending=len(pending))
                last_report = time.monotonic()
            if jobs:
                time.sleep(poll_interval)
        status("complete")
        return events
    except BaseException as error:
        event("failed_or_interrupted", error=str(error))
        status("failed_or_interrupted", str(error))
        terminate_jobs(jobs)
        jobs.clear()
        status("failed_or_interrupted", str(error))
        raise
    finally:
        progress.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
