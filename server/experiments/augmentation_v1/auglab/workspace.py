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
NAMES = ("control", "photo_mild", "noise_mild", "scale_mild")
SPLITS = ("official", "all90", "fold0", "fold1", "fold2", "fold3", "fold4")
DEFAULT_EXPERIMENTS = NAMES[1:]


def read_json(path):
    return json.loads(Path(path).read_text())


def output_path(*parts):
    if ROOT.is_symlink() or OUTPUTS.is_symlink():
        raise ValueError("Experiment root/output must not be symlinks")
    path = OUTPUTS.joinpath(*map(str, parts))
    if not path.resolve().is_relative_to(OUTPUTS.resolve()):
        raise ValueError("Output must stay in augmentation_v1/outputs")
    current = path
    while current != OUTPUTS:
        if current.is_symlink():
            raise ValueError(f"Output symlink rejected: {current}")
        current = current.parent
    return path


def run_dir(name, seed=2026, split="official", smoke=False):
    if name not in NAMES or split not in SPLITS or seed < 0:
        raise ValueError("Invalid experiment, split or seed")
    return output_path("smoke" if smoke else "runs", name, f"seed{seed}", split)


def extension_hash():
    files = [ROOT / "run.py", *sorted((ROOT / "auglab").glob("*.py"))]
    return json_hash({str(p.relative_to(ROOT)): sha256(p) for p in files})


def verify_preservation(full=True):
    records = read_json(ROOT / "preservation.json")["files"]
    if not full:
        prefixes = ("experiments/accuracy_v1/baseline/source/", "experiments/accuracy_v1/baseline/data/")
        records = [r for r in records if r["path"].startswith(prefixes)]
    failed = []
    for r in records:
        p = PROJECT / r["path"]
        if not p.is_file() or p.stat().st_size != r["bytes"] or sha256(p) != r["sha256"]:
            failed.append(r["path"])
    if failed:
        raise RuntimeError("Protected files changed; no automatic restoration:\n" + "\n".join(failed))
    return dict(unchanged=True, files_checked=len(records), full=full)


def config(name, seed=2026, smoke=False):
    if name not in NAMES or seed < 0:
        raise ValueError("Invalid experiment or seed")
    cfg = OmegaConf.to_container(OmegaConf.merge(
        OmegaConf.load(SOURCE / "configs/cnn_unet.yaml"),
        OmegaConf.load(ROOT / "configs" / f"{name}.yaml")), resolve=True)
    cfg["experiment"] = f"aug_{name}_seed{seed}" + ("_smoke" if smoke else "")
    cfg["seed"] = seed
    cfg["data"].update(root=str(PROJECT / "dataset"), manifest=str(BASELINE / "data/manifest.csv"),
                       folds=str(BASELINE / "data/folds.json"))
    ext = {"noise_probability": 0., "noise_std": [0., .01],
           "scale_probability": 0., "scale_choices": [.9, 1., 1.1],
           **cfg.get("augmentation_extension", {})}
    for key in ("noise_probability", "scale_probability"):
        if not 0 <= ext[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if not 0 <= ext["noise_std"][0] <= ext["noise_std"][1] <= .1:
        raise ValueError("Noise standard deviation must be in [0, .1]")
    if not ext["scale_choices"] or any(not .5 <= v <= 2 for v in ext["scale_choices"]):
        raise ValueError("Scale choices must be in [.5, 2]")
    cfg["augmentation_extension"] = ext
    cfg["augmentation_source_sha256"] = extension_hash()
    if smoke:
        cfg["train"].update(eval_every_updates=1, checkpoint_every_updates=1, log_every_updates=1, num_workers=1)
        for stage in cfg["train"]["stages"]:
            stage.update(max_updates=1, warmup_updates=1)
    return cfg
