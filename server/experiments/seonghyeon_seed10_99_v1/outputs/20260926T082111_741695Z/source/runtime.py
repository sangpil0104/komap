"""Atomic status files and wall-clock estimates for a parallel seed queue."""
import heapq
import json
import os
from pathlib import Path
import statistics
import time


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def duration(seconds):
    if seconds is None:
        return "measuring"
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def seed_estimate(s):
    """Remaining includes queued periodic validations, final D4 and rendering.

    Before validation is measured use a clearly provisional training-rate proxy.
    Each full-image forward is larger than a crop; measured values replace it.
    """
    if s.get("state") == "completed":
        return 0.0, s["elapsed_seconds"]
    rate = s.get("seconds_per_iteration")
    if not rate:
        return None, None
    n, it, every = s["iterations"], s["iteration"], s["eval_every"]
    images = s["valid_count"]
    plain = s.get("plain_seconds_per_image") or rate * 4
    d4 = s.get("d4_seconds_per_image") or plain * 8
    evaluations = n // every + int(n % every != 0)
    pending = evaluations - s.get("plain_evaluations_done", 0)
    remaining = (n - it) * rate + pending * images * plain + images * d4
    if s.get("state") == "valid_plain":
        remaining -= s.get("eval_image", 0) * plain
    if s.get("state") == "valid_d4":
        remaining -= s.get("eval_image", 0) * d4
    if s.get("state") == "reporting":
        remaining = 5.0
    typical = n * rate + evaluations * images * plain + images * d4
    return max(0.0, remaining), typical


def suite_eta(active, pending, slots, completed=()):
    """Greedily schedule pending equal-budget seeds on the next free GPU."""
    estimates = [seed_estimate(s) for s in active]
    typicals = [total for _, total in estimates if total is not None]
    typicals += [s["elapsed_seconds"] for s in completed]
    if not typicals:
        return None
    typical = statistics.median(typicals)
    lanes = [remaining if remaining is not None else typical for remaining, _ in estimates]
    lanes += [0.0] * (slots - len(lanes))
    heapq.heapify(lanes)
    for _ in range(pending):
        heapq.heappush(lanes, heapq.heappop(lanes) + typical)
    return max(lanes, default=0.0)


class Progress:
    def __init__(self, run, config):
        self.path = Path(run) / "status.json"
        self.started = time.monotonic()
        self.state = dict(state="loading", iteration=0, iterations=config["iters"],
                          eval_every=config["eval_every"], valid_count=20,
                          seed=config["seed"], gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                          plain_evaluations_done=0, train_seconds=0.0)
        self.update()

    def update(self, **fields):
        self.state.update(fields)
        self.state["elapsed_seconds"] = time.monotonic() - self.started
        self.state["updated_at"] = time.time()
        remaining, _ = seed_estimate(self.state)
        self.state["eta_seconds"] = remaining
        write_json(self.path, self.state)

    def log(self, message):
        print(f"{message} elapsed={duration(self.state['elapsed_seconds'])} "
              f"seed_ETA={duration(self.state['eta_seconds'])}", flush=True)
