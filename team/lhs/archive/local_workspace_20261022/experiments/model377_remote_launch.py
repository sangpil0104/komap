#!/usr/bin/env python3
"""Start the isolated Model377 grouped-OOF runner in the existing Colab VM."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/content/komap_model1")
RUNNER = ROOT / "run_model377_directional_group_oof.sh"
OUTPUT = ROOT / "outputs/model377_directional_group_oof"


def main() -> None:
    if not RUNNER.is_file():
        raise FileNotFoundError(RUNNER)
    launch_record = OUTPUT / "launch.json"
    if launch_record.exists():
        raise FileExistsError(f"Model377 already has a launch record: {launch_record}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / "runner_background.log"
    with log_path.open("ab", buffering=0) as log_stream:
        process = subprocess.Popen(
            ["bash", str(RUNNER)],
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    launch_record.write_text(json.dumps({
        "model": 377,
        "pid": process.pid,
        "started_unix": time.time(),
        "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        "stdout_log": str(log_path),
    }, indent=2) + "\n")
    print({"launched": True, "model": 377, "pid": process.pid,
           "runner_log": str(log_path)}, flush=True)


if __name__ == "__main__":
    main()
