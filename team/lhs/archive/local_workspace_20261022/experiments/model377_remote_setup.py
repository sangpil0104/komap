#!/usr/bin/env python3
"""Create an isolated Model377 source copy on the existing Colab session."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path("/content/komap_model1")
BASE = ROOT / "model376_hrnet_boundary_ce"
DEST = ROOT / "model377_directional"
OUT = ROOT / "outputs/model377_directional_group_oof"
DRIVE_OUT = Path("/content/drive/MyDrive/KoMaP/model377_directional_group_oof")
PAYLOAD = {
    "train.py": ROOT / "model377_payload_train.py",
    "komap_seg/losses.py": ROOT / "model377_payload_losses.py",
    "experiments/model377_drive_checkpoint_sync.py": ROOT / "model377_payload_sync.py",
    "experiments/model377_oof_evaluation.py": ROOT / "model377_payload_eval.py",
    "run_model377_directional_group_oof.sh": ROOT / "model377_payload_runner.sh",
}


def main() -> None:
    if not (BASE / "train.py").is_file() or not (BASE / "evaluate_image_mean.py").is_file():
        raise FileNotFoundError(f"frozen Model376 source is incomplete: {BASE}")
    if DEST.exists() or OUT.exists() or DRIVE_OUT.exists():
        raise FileExistsError("a Model377 destination/output already exists; refusing to overwrite")
    for relative, source in PAYLOAD.items():
        if not source.is_file():
            raise FileNotFoundError(f"uploaded Model377 payload missing: {source}")

    shutil.copytree(BASE, DEST)
    try:
        for relative, source in PAYLOAD.items():
            destination = ROOT / relative if relative.startswith("run_") else DEST / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        subprocess.run([sys.executable, "-m", "py_compile",
                        str(DEST / "train.py"), str(DEST / "komap_seg/losses.py"),
                        str(DEST / "experiments/model377_drive_checkpoint_sync.py"),
                        str(DEST / "experiments/model377_oof_evaluation.py")], check=True)
        env = dict(__import__("os").environ)
        env["PYTHONPATH"] = str(DEST)
        subprocess.run([sys.executable, str(DEST / "train.py"), "--help"],
                       check=True, stdout=subprocess.DEVNULL, env=env)
        sys.path.insert(0, str(DEST))
        import torch
        from komap_seg.losses import directional_unlikelihood_loss
        labels = torch.zeros((1, 3, 3), dtype=torch.long)
        labels[0, 1, 1], labels[0, 1, 2] = 3, 1
        logits = torch.zeros((1, 4, 3, 3), requires_grad=True)
        loss, support = directional_unlikelihood_loss(logits, labels)
        if not torch.isfinite(loss) or int(support) != 1:
            raise RuntimeError("remote Model377 directional-loss smoke test failed")
    except Exception:
        shutil.rmtree(DEST, ignore_errors=True)
        raise

    digests = {
        str(path.relative_to(DEST)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (DEST / "train.py", DEST / "komap_seg/losses.py",
                     DEST / "experiments/model377_drive_checkpoint_sync.py",
                     DEST / "experiments/model377_oof_evaluation.py")
    }
    print({"created": str(DEST), "output": str(OUT), "drive_output": str(DRIVE_OUT),
           "source_sha256": digests, "parser_and_loss_smoke": "passed"}, flush=True)


if __name__ == "__main__":
    main()
