"""Launch the Drive-backed Model374 runner while keeping the Colab kernel busy."""

from __future__ import annotations

import os
import subprocess
import sys


RUNNER = "/content/run_model374_fold34_drive_a100.sh"


def main() -> int:
    if not os.path.isfile(RUNNER):
        raise SystemExit(f"upload the Drive runner first: {RUNNER}")
    env = os.environ.copy()
    env.setdefault(
        "MODEL374_DRIVE_ROOT",
        "/content/drive/MyDrive/KoMaP/model374_fold34_retry_20261001",
    )
    print(f"Launching {RUNNER}; Drive output: {env['MODEL374_DRIVE_ROOT']}", flush=True)
    process = subprocess.Popen(
        ["bash", RUNNER],
        cwd="/content",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    status = process.wait()
    print(f"Model374 Drive runner exited with status {status}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
