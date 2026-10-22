"""Scheduler-only process fixture: creates text artifacts, never imports ML code."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

mode, spec_path, task_id = sys.argv[1:]
spec = json.loads(Path(spec_path).read_text())
task = next(t for t in spec if t["id"] == task_id)
assert os.environ["MODEL364_TASK_ID"] == task_id
visible = os.environ["CUDA_VISIBLE_DEVICES"]
if task["kind"] == "cpu":
    assert visible == ""
else:
    assert visible == "TEST-GPU-" + os.environ["MODEL364_PHYSICAL_GPU"]
for required in task.get("requires", []):
    assert Path(required).exists(), required
if mode == "fail":
    time.sleep(.3)
    raise SystemExit(7)
if mode == "linger":
    child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    Path(spec_path).with_suffix(".child_pid").write_text(str(child.pid))
    time.sleep(60)
if mode == "missing":
    raise SystemExit(0)
time.sleep(.4)
for value in task.get("produces", []):
    path = Path(value)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic scheduler artifact\n")
    else:
        path.mkdir(parents=True, exist_ok=True)
print(json.dumps({"task": task_id, "visible": visible, "training": False}), flush=True)
