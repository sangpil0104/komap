#!/usr/bin/env python
"""One merged text log plus suite ETA; preserved training code is not modified."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
GPU_IDS = "0,1,2,5,6,7"
WORLD = 6


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def seconds_per_update(rows):
    # Use positive adjacent intervals, so resumed duplicate/rolled-back rows do not skew ETA.
    rates = []
    window = rows[-21:]
    for a, b in zip(window, window[1:]):
        steps = b["update"] - a["update"]
        seconds = b["elapsed_seconds"] - a["elapsed_seconds"]
        if steps > 0 and seconds > 0:
            rates.append(seconds / steps)
    return statistics.median(rates) if rates else None


def duration(seconds):
    seconds = max(0, round(seconds))
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class Task:
    name: str
    scale: int
    updates: int
    initial: int = 0
    run: Path | None = None
    update: int = 0
    phase: str = "대기"
    run_started: float | None = None
    train_started: float | None = None
    eval_started: float | None = None
    finished: float | None = None
    rows: deque = field(default_factory=lambda: deque(maxlen=21))
    offset: int = 0
    observed_rate: float | None = None
    initial_elapsed: float = 0.
    elapsed_training: float = 0.
    previous_evaluations: set = field(default_factory=set)
    evaluation_counts: dict = field(default_factory=dict)
    logged_summaries: set = field(default_factory=set)


class SuiteTracker:
    def __init__(self, run, remaining, timezone_name="Asia/Seoul", postprocess_seconds=180., startup_seconds=30., clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.zone = ZoneInfo(timezone_name)
        self.postprocess_seconds, self.startup_seconds = postprocess_seconds, startup_seconds
        cfg = read_json(run / "config.json")
        latest = read_json(run / "latest.json")
        initial = latest["update"]
        completed = (run / "completed.json").exists()
        first = Task(cfg["experiment"], cfg["scale"], cfg["train"]["updates"], initial,
                     run, cfg["train"]["updates"] if completed else initial)
        first.previous_evaluations = set((run / "evaluations").glob("*"))
        history_path = run / "training.jsonl"
        first.offset = history_path.stat().st_size if history_path.exists() else 0
        current_history = read_rows(history_path)
        previous_history = read_rows(run / "training_before_migration.jsonl")
        migration = read_json(run / "migration.json") if (run / "migration.json").exists() else {}
        if not previous_history and migration.get("source_run"):
            previous_history = read_rows(Path(migration["source_run"]) / "training.jsonl")
        history = [r for r in current_history if r["update"] <= initial]
        if not history:
            history = [r for r in previous_history if r["update"] <= initial]
        for row in history:
            if row["update"] <= initial:
                first.initial_elapsed = row["elapsed_seconds"]
        old_rate = seconds_per_update(history)
        if old_rate is None:
            old_rate = 1.0
            self.initial_basis = "관측 속도 없음: 임시 1초/update 가정, 실제 진행 후 보정"
        else:
            self.initial_basis = "기존 학습의 최근 20개 구간 중앙값 × GPU별 micro-batch round 비율 × 입력 면적 비율"
        # Imported old-world checkpoint versus checkpoints subsequently made by v2.
        imported = (migration.get("imported_update") == initial
                    and Path(latest["checkpoint"]).name.startswith("imported_"))
        source_world = migration.get("source_world_size", WORLD) if imported else WORLD
        batch = cfg["train"]["effective_batch"]
        self.initial_rate = old_rate * math.ceil(batch / WORLD) / math.ceil(batch / source_world)
        self.initial_scale = cfg["scale"]
        self.tasks = [first]
        for name in remaining:
            if name == cfg["experiment"] or name not in ("S0", "S1", "S2"):
                raise ValueError("Remaining experiments must be distinct from the resumed model")
            self.tasks.append(Task(name, int(name[1]) + 1, cfg["train"]["updates"]))
        if len({t.name for t in self.tasks}) != len(self.tasks):
            raise ValueError("Duplicate experiments in suite")
        self.active = None
        self.next_task = 0
        self.total_planned = sum(t.updates - t.initial for t in self.tasks)

    def log(self, tag, message):
        timestamp = datetime.now(self.zone).strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{timestamp}] [{tag}] {message}", flush=True)

    def rate(self, task):
        if task.observed_rate is not None:
            return task.observed_rate, "실측"
        known = [t for t in self.tasks if t.observed_rate is not None]
        if known:
            nearest = min(known, key=lambda t: abs(math.log(task.scale / t.scale)))
            return nearest.observed_rate * (task.scale / nearest.scale) ** 2, f"{nearest.name} 실측+면적 근사"
        return self.initial_rate * (task.scale / self.initial_scale) ** 2, "초기 근사"

    def accept_output(self, line):
        self.log("OUTPUT", line)
        if line.startswith("Run: "):
            if self.next_task >= len(self.tasks):
                raise ValueError("Unexpected extra run in child output")
            task = self.tasks[self.next_task]
            self.next_task += 1
            run = Path(line.removeprefix("Run: ").strip()).resolve()
            cfg = read_json(run / "config.json")
            if cfg["experiment"] != task.name or cfg["scale"] != task.scale:
                raise ValueError("Launched model differs from ETA plan")
            if task.run is not None and task.run.resolve() != run:
                raise ValueError("Unexpected resume run")
            task.run = run
            task.run_started = self.clock()
            task.phase = "준비"
            self.active = task
            self.log("STAGE", f"{task.name} x{task.scale} | 시작 update={task.initial}/{task.updates} | run={run}")
        elif line.startswith("Result report: ") and self.active is not None:
            self.poll()
            task = self.active
            task.phase, task.finished = "완료", self.clock()
            task.update = task.updates
            if task.eval_started is not None:
                self.postprocess_seconds = max(1., task.finished - task.eval_started)
            self.log("STAGE_DONE", f"{task.name} 완료 | 다음 검증/보고서 시간 추정={duration(self.postprocess_seconds)}")
            self.active = None

    def poll(self):
        task = self.active
        if task is None or task.run is None:
            return
        path = task.run / "training.jsonl"
        if path.exists():
            with path.open("rb") as handle:
                handle.seek(task.offset)
                while True:
                    begin = handle.tell()
                    line = handle.readline()
                    if not line or not line.endswith(b"\n"):
                        task.offset = begin
                        break
                    task.offset = handle.tell()
                    row = json.loads(line)
                    if row["update"] <= task.initial:
                        continue
                    task.rows.append(row)
                    task.update = row["update"]
                    task.phase = "학습"
                    if task.train_started is None:
                        task.train_started = self.clock()
                    task.elapsed_training = max(0., row["elapsed_seconds"] - task.initial_elapsed)
                    if len(task.rows) >= 4:
                        task.observed_rate = seconds_per_update(list(task.rows))
                    self.log("TRAIN", f"{task.name} update={task.update}/{task.updates} "
                             f"loss={row['loss']:.5f} ce={row['ce']:.5f} dice={row['dice']:.5f} "
                             f"lr={row['encoder_lr']:.3e} 누적학습={duration(row['elapsed_seconds'])}")
        if (task.run / "completed.json").exists() and task.eval_started is None:
            task.update = task.updates
            task.phase = "검증/보고서"
            task.eval_started = self.clock()
            completed = read_json(task.run / "completed.json")
            task.elapsed_training = max(0., completed["elapsed_seconds"] - task.initial_elapsed)
            self.log("EVALUATION", f"{task.name} 학습 완료. 검증 20장 및 마스크 비교/보고서 생성 중.")
        if task.eval_started is not None:
            for folder in sorted((task.run / "evaluations").glob("*")):
                if not folder.is_dir() or folder in task.previous_evaluations:
                    continue
                count = len(list((folder / "comparisons").glob("*.png")))
                if task.evaluation_counts.get(folder) != count:
                    task.evaluation_counts[folder] = count
                    self.log("VALID", f"{task.name} 검증/비교 이미지={count}/20 | {folder.name}")
                summary = folder / "summary.json"
                if summary.exists() and summary not in task.logged_summaries:
                    self.log("METRICS", f"{task.name} {json.dumps(read_json(summary), ensure_ascii=False)}")
                    task.logged_summaries.add(summary)

    def estimate(self, now=None):
        now = self.clock() if now is None else now
        training_left, overhead_left = 0., 0.
        details = []
        for task in self.tasks:
            rate, basis = self.rate(task)
            left = max(0, task.updates - task.update) * rate if task.phase != "완료" else 0.
            training_left += left
            extra = 0.
            if task.phase != "완료":
                if task.eval_started is not None:
                    extra = max(10., self.postprocess_seconds - (now - task.eval_started))
                else:
                    extra = self.postprocess_seconds
                    if task.phase == "대기":
                        extra += self.startup_seconds
                    elif task.phase == "준비":
                        extra += max(0., self.startup_seconds - (now - task.run_started))
            overhead_left += extra
            details.append((task, left, rate, basis))
        remaining = training_left + overhead_left
        elapsed = max(0., now - self.started)
        return dict(training_left=training_left, remaining=remaining, elapsed=elapsed,
                    expected_total=elapsed + remaining, details=details)

    def progress(self, tag="TOTAL"):
        self.poll()
        estimate = self.estimate()
        done = sum(max(0, t.update - t.initial) for t in self.tasks)
        finish = datetime.now(timezone.utc) + timedelta(seconds=estimate["remaining"])
        self.log(tag, f"전체 학습 update={done}/{self.total_planned} "
                 f"({100 * done / max(1, self.total_planned):.1f}%) | 이번 실행 경과={duration(estimate['elapsed'])} | "
                 f"남은 학습={duration(estimate['training_left'])} | 검증 포함 남은 시간≈{duration(estimate['remaining'])} | "
                 f"이번 실행 총 예상 소요≈{duration(estimate['expected_total'])}")
        self.log("FINISH_ETA", f"예상 전체 완료≈{finish.astimezone(self.zone):%Y-%m-%d %H:%M:%S %Z} "
                 f"({finish:%Y-%m-%d %H:%M:%S UTC}) | 실측 진행에 따라 갱신")
        for task, left, rate, basis in estimate["details"]:
            self.log("MODEL_ETA", f"{task.name} x{task.scale} {task.phase} "
                     f"update={task.update}/{task.updates} | 학습 남음≈{duration(left)} | "
                     f"{rate:.3f}초/update ({basis})")


def pump(stream, events):
    try:
        for line in stream:
            line = line.rstrip("\r\n")
            if line:
                events.put(line)
    finally:
        events.put(None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--remaining", nargs="*", choices=("S0", "S1", "S2"), default=["S1", "S2"])
    parser.add_argument("--interval", type=float, default=60.)
    parser.add_argument("--timezone", default="Asia/Seoul")
    parser.add_argument("--postprocess-seconds", type=float, default=180.)
    parser.add_argument("--startup-seconds", type=float, default=30.)
    parser.add_argument("--plan-only", action="store_true", help="Print estimates without starting any subprocess")
    args = parser.parse_args()
    if args.interval <= 0 or args.postprocess_seconds < 0 or args.startup_seconds < 0:
        parser.error("Interval must be positive; time reserves must be nonnegative")
    run = (args.run or Path(read_json(ROOT / "outputs/migration/ready.json")["run"])).resolve()
    tracker = SuiteTracker(run, args.remaining, args.timezone, args.postprocess_seconds, args.startup_seconds)
    tracker.log("START", f"GPUs={GPU_IDS} | 유효 배치=8 | 로그={os.environ.get('KOMAP_LOG_PATH', 'stdout')} | "
                f"전체 순서={' → '.join(t.name for t in tracker.tasks)}")
    tracker.log("ESTIMATE_BASIS", tracker.initial_basis)
    tracker.log("ESTIMATE_BASIS", f"통신/검증 시간 미측정 초기 근사. 준비 {duration(args.startup_seconds)}, "
                f"검증/보고서 {duration(args.postprocess_seconds)}씩 가정. 새 로그 4개부터 학습 속도 실측 보정.")
    tracker.progress("INITIAL_ESTIMATE")
    if args.plan_only:
        tracker.log("PLAN_ONLY", "예상 출력만 수행. 학습 프로세스를 시작하지 않았습니다.")
        return 0
    command = [sys.executable, "-u", "-B", str(ROOT / "run.py"), "continue-suite", str(run),
               "--gpus", str(WORLD), "--remaining", *args.remaining]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU_IDS, "PYTHONUNBUFFERED": "1",
                   "PYTHONDONTWRITEBYTECODE": "1", "TQDM_DISABLE": "1"}
    # All workers, validation and reports inherit this single stdout/stderr pipe.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1, env=environment)
    tracker.log("PROCESS", f"suite_pid={process.pid} | launcher_pid={os.getpid()}")
    events = queue.Queue()
    reader = threading.Thread(target=pump, args=(process.stdout, events), daemon=True)
    reader.start()
    eof = False
    next_progress = time.monotonic() + args.interval
    try:
        while not eof or process.poll() is None:
            try:
                event = events.get(timeout=1.)
                if event is None:
                    eof = True
                else:
                    tracker.accept_output(event)
            except queue.Empty:
                pass
            tracker.poll()
            if time.monotonic() >= next_progress:
                tracker.progress()
                next_progress = time.monotonic() + args.interval
        code = process.wait()
    except BaseException:
        # Leave an explicit recovery instruction and request a safe stop on the active trainer.
        if tracker.active is not None and tracker.active.run is not None:
            (tracker.active.run / "SAVE_AND_STOP").touch(exist_ok=True)
        tracker.log("LOGGER_ERROR", "로그 관리자 오류: 실행 중 모델에 저장/중지 요청. 아래 traceback을 확인하세요.")
        raise
    tracker.poll()
    finished = sum(t.phase == "완료" for t in tracker.tasks)
    if code == 0 and finished == len(tracker.tasks):
        tracker.log("DONE", f"전체 학습·검증·보고서 완료 | 실제 이번 실행 소요={duration(time.monotonic() - tracker.started)} | "
                    f"완료 시각={datetime.now(tracker.zone):%Y-%m-%d %H:%M:%S %Z}")
    elif code == 0:
        tracker.log("PAUSED", f"저장 후 중지됨. 완료 모델={finished}/{len(tracker.tasks)}; 전체 완료가 아닙니다.")
    else:
        tracker.log("FAILED", f"종료 코드={code}; 실제 경과={duration(time.monotonic() - tracker.started)}. 위 오류와 마지막 checkpoint를 확인하세요.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
