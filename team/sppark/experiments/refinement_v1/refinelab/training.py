import builtins
import os
import time
from contextlib import contextmanager
from pathlib import Path

from alphase.engine import trainer
from alphase.engine.progress import Progress
from alphase.losses import segmentation
from alphase.utils import write_json
from refinelab.dataset import RefineDataset
from refinelab.losses import segmentation_loss
from refinelab.mining import verify_mining
from refinelab.models import build_model
from refinelab.workspace import SOURCE, config, read_json, run_dir, verify


class TerminalProgress(Progress):
    def __init__(self, run, cfg, world, elapsed=0):
        self.run, self.cfg = Path(run), cfg
        self.started, self.prior_elapsed = time.monotonic(), elapsed
        self.rates, self.validation_seconds, self.last = {}, 0., {}
        if elapsed and (self.run / "status.json").is_file():
            prior = read_json(self.run / "status.json")
            self.rates = {s["name"]: s["seconds_per_update"] for s in prior["stages"] if s.get("seconds_per_update")}
            self.validation_seconds = prior.get("validation_seconds", 0.)


def terminal_print(*args, **kwargs):
    builtins.print(*[v.split("\nMonitor:", 1)[0] if isinstance(v, str) and "\nMonitor:" in v else v for v in args], **kwargs)


@contextmanager
def adapter():
    prior = trainer.TrainDataset, trainer.build_model, trainer.Progress, segmentation.segmentation_loss
    prior_print = getattr(trainer, "print", None)
    trainer.TrainDataset, trainer.build_model, trainer.Progress = RefineDataset, build_model, TerminalProgress
    segmentation.segmentation_loss, trainer.print = segmentation_loss, terminal_print
    try:
        yield
    finally:
        trainer.TrainDataset, trainer.build_model, trainer.Progress, segmentation.segmentation_loss = prior
        if prior_print is None:
            del trainer.print
        else:
            trainer.print = prior_print


def train(args):
    cfg = config(args.experiment, args.seed, args.smoke)
    folder = run_dir(args.experiment, args.seed, args.smoke)
    verify(full=False)
    if cfg["refinement"]["hard"]:
        verify_mining(cfg)
    if folder.exists() and any(folder.iterdir()) and not args.resume:
        raise FileExistsError(f"Run exists: {folder}; use --resume")
    resume = folder / "last.pt" if args.resume else None
    if resume and not resume.is_file():
        raise FileNotFoundError(resume)
    previous = Path.cwd()
    try:
        os.chdir(SOURCE)
        with adapter():
            result = trainer.train(cfg, split="official", run=folder, resume=resume, stop_after=args.stop_after)
    finally:
        os.chdir(previous)
    if int(os.environ.get("RANK", 0)) == 0:
        write_json(folder / "refinement_provenance.json", dict(**cfg["refinement"], smoke_test=args.smoke,
                    architecture=cfg["model"]["name"], class_weights=cfg["loss"]["class_weights"], terminal_only=True))
    return result
