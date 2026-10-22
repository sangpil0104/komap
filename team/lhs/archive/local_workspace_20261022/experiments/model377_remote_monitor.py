#!/usr/bin/env python3
"""Read-only status report for the isolated Model377 Colab run."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


ROOT = Path("/content/komap_model1/outputs/model377_directional_group_oof")
DRIVE_ROOT = Path("/content/drive/MyDrive/KoMaP/model377_directional_group_oof")


def tail(path: Path, count: int = 16) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(errors="replace").splitlines()[-count:]


def main() -> None:
    record = json.loads((ROOT / "launch.json").read_text())
    pid = int(record["pid"])
    alive = True
    process_state = None
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        if stat.is_file():
            fields = stat.read_text().split()
            process_state = fields[2]
            alive = process_state != "Z"
    except ProcessLookupError:
        alive = False

    fold_status = {}
    for fold in range(1, 5):
        run = ROOT / f"fold{fold}"
        log_path = run / "train.log"
        log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        epochs = re.findall(r"epoch\s+(\d+)\s*/\s*\d+\s+train_loss=", log_text)
        checkpoint = run / "last.pt"
        drive_fold = DRIVE_ROOT / f"fold{fold}"
        fold_status[str(fold)] = {
            "latest_logged_epoch": int(epochs[-1]) if epochs else None,
            "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
            "latest_training_log": tail(run / "train.log", 3),
            "drive_checkpoint_snapshots": sorted(
                path.name for path in (drive_fold / "checkpoints").glob("snapshot_epoch_*")
                if path.is_dir()
            ),
            "drive_sync_log_tail": tail(run.parent / f"fold{fold}_drive_sync.log", 3),
            "drive_completion_manifest_exists": (drive_fold / "complete.json").is_file(),
            "oof_summary_exists": (ROOT / f"fold{fold}_oof/valid_summary.json").is_file(),
        }

    print(json.dumps({
        "model": 377,
        "runner_pid": pid,
        "runner_alive": alive,
        "runner_process_state": process_state,
        "folds": fold_status,
        "runner_log_tail": tail(ROOT / "runner_background.log", 18),
        "run_log_tail": tail(ROOT / "run.log", 12),
        "evaluation_exists": (ROOT / "model377_oof_evaluation.json").is_file(),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
