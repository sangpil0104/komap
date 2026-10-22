from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf
from alphase.utils import json_hash, sha256

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
BASELINE = ROOT.parent / "accuracy_v1/baseline"
SOURCE = BASELINE / "source"
OUTPUTS = ROOT / "outputs"
PARENT = BASELINE / "run/best.pt"


def read_json(path):
    return json.loads(Path(path).read_text())


def plan():
    return OmegaConf.to_container(OmegaConf.load(ROOT / "configs/plan.yaml"), resolve=True)["experiments"]


def output_path(*parts):
    if OUTPUTS.is_symlink():
        raise ValueError("Output root cannot be a symlink")
    p = OUTPUTS.joinpath(*map(str, parts))
    if not p.resolve().is_relative_to(OUTPUTS.resolve()):
        raise ValueError("Output escapes refinement_v1/outputs")
    current = p
    while current != OUTPUTS:
        if current.is_symlink():
            raise ValueError(f"Output symlink rejected: {current}")
        current = current.parent
    return p


def run_dir(name, seed=2026, smoke=False):
    if name not in plan() or seed < 0:
        raise ValueError("Unknown experiment or invalid seed")
    return output_path("smoke" if smoke else "runs", name, f"seed{seed}", "official")


def source_hash():
    files = [ROOT / "run.py", ROOT / "configs/plan.yaml", *sorted((ROOT / "refinelab").glob("*.py"))]
    return json_hash({str(p.relative_to(ROOT)): sha256(p) for p in files})


def parent_hash():
    records = read_json(ROOT / "preservation.json")["files"]
    return next(r["sha256"] for r in records if r["path"] == str(PARENT.relative_to(PROJECT)))


def verify(full=True):
    records = read_json(ROOT / "preservation.json")["files"]
    if not full:
        records = [r for r in records if r["path"].startswith((
            "experiments/accuracy_v1/baseline/source/", "experiments/accuracy_v1/baseline/data/"))
            or r["path"] == str(PARENT.relative_to(PROJECT))]
    failed = []
    for r in records:
        p = PROJECT / r["path"]
        if not p.is_file() or p.stat().st_size != r["bytes"] or sha256(p) != r["sha256"]:
            failed.append(r["path"])
    if failed:
        raise RuntimeError("Protected files changed (not automatically restored):\n" + "\n".join(failed))
    return dict(unchanged=True, files_checked=len(records), full=full)


def config(name, seed=2026, smoke=False):
    entries = plan()
    if name not in entries or seed < 0:
        raise ValueError("Unknown experiment or invalid seed")
    options = entries[name]
    cfg = OmegaConf.to_container(OmegaConf.load(SOURCE / "configs/cnn_unet.yaml"), resolve=True)
    cfg["experiment"], cfg["seed"] = f"{name}_seed{seed}" + ("_smoke" if smoke else ""), seed
    cfg["data"].update(root=str(PROJECT / "dataset"), manifest=str(BASELINE / "data/manifest.csv"), folds=str(BASELINE / "data/folds.json"))
    cfg["model"] = dict(name="cnn_aspp_v1" if options.get("aspp") else "cnn_unet")
    cfg["refinement"] = dict(parent_checkpoint=str(PARENT), parent_sha256=parent_hash(), source_sha256=source_hash(),
                             hard=bool(options.get("hard")), mining_folder=str(output_path("mining")),
                             description=options["description"])
    if options.get("hard"):
        cfg["sampling"]["mix"] = dict(uniform=.45, eutectic_center=.25, error_or_boundary=.30)
    if options.get("weighted"):
        cfg["loss"]["class_weights"] = [1., 1.15, 1.5, 1.]
    if options.get("overlap"):
        cfg["loss"].update(dice=.75, lovasz=.75)
    cfg["train"].update(ema_decay=.99, eval_every_updates=200)
    budgets = [600, 400, 1000] if options.get("long_tail") else [800, 600, 600]
    for stage, budget, rate, warmup in zip(cfg["train"]["stages"], budgets, [3e-5, 1e-5, 5e-6], [50, 30, 25]):
        stage.update(max_updates=budget, new_lr=rate, warmup_updates=warmup)
    if smoke:
        cfg["train"].update(num_workers=1, eval_every_updates=1, log_every_updates=1, checkpoint_every_updates=1)
        for stage in cfg["train"]["stages"]:
            stage.update(max_updates=1, warmup_updates=1)
    return cfg
