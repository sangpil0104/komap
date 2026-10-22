from __future__ import annotations

import builtins
import os
import time
from contextlib import contextmanager
from pathlib import Path

from alphase.engine import trainer
from alphase.engine.progress import Progress
from alphase.utils import write_json
from auglab.dataset import AugmentedDataset
from auglab.workspace import SOURCE, config, read_json, run_dir, verify_preservation


class TerminalProgress(Progress):
    """Reuse the original stage-aware ETA without generating a web dashboard."""
    def __init__(self, run, cfg, world, elapsed=0):
        self.run, self.cfg = Path(run), cfg
        self.started, self.prior_elapsed = time.monotonic(), elapsed
        self.rates, self.validation_seconds, self.last = {}, 0., {}
        if elapsed and (self.run / "status.json").is_file():
            previous = read_json(self.run / "status.json")
            self.rates = {s["name"]: s["seconds_per_update"] for s in previous.get("stages", []) if s.get("seconds_per_update")}
            self.validation_seconds = previous.get("validation_seconds", 0.)


def terminal_print(*args, **kwargs):
    # The frozen engine's introductory print contains its legacy monitor hint.
    values = [a.split("\nMonitor:", 1)[0] if isinstance(a, str) and "\nMonitor:" in a else a for a in args]
    builtins.print(*values, **kwargs)


@contextmanager
def data_adapter():
    """Process-local dependency injection; no frozen source files are edited."""
    prior = (trainer.TrainDataset, trainer.Progress, getattr(trainer, "print", None))
    trainer.TrainDataset, trainer.Progress, trainer.print = AugmentedDataset, TerminalProgress, terminal_print
    try:
        yield
    finally:
        trainer.TrainDataset, trainer.Progress = prior[:2]
        if prior[2] is None:
            del trainer.print
        else:
            trainer.print = prior[2]


def run_training(args):
    cfg = config(args.experiment, args.seed, args.smoke)
    folder = run_dir(args.experiment, args.seed, args.split, args.smoke)
    verify_preservation(full=False)
    if folder.exists() and any(folder.iterdir()) and not args.resume:
        raise FileExistsError(f"Run exists: {folder}. Use --resume; overwriting is disabled.")
    resume = folder / "last.pt" if args.resume else None
    if resume and not resume.is_file():
        raise FileNotFoundError(resume)
    original = Path.cwd()
    try:
        os.chdir(SOURCE)
        with data_adapter():
            result = trainer.train(cfg, split=args.split, run=folder, resume=resume, stop_after=args.stop_after)
    finally:
        os.chdir(original)
    if int(os.environ.get("RANK", 0)) == 0:
        write_json(folder / "augmentation_provenance.json", dict(
            extension_sha256=cfg["augmentation_source_sha256"],
            source=str(SOURCE), augmentation=cfg["augmentation"],
            extension=cfg["augmentation_extension"], terminal_progress=True, smoke_test=args.smoke))
    return result
