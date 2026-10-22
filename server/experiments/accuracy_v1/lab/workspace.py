from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

from alphase.utils import json_hash, sha256, write_json

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT = WORKSPACE.parents[1]
BASELINE = WORKSPACE / "baseline"
OUTPUTS = WORKSPACE / "outputs"
SOURCE = BASELINE / "source"


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Use letters, digits, underscores and hyphens in names: {value!r}")
    return value


def output_path(*parts):
    """Never allow a symlink or traversal to redirect writes to the old project."""
    if OUTPUTS.is_symlink() or OUTPUTS.resolve() != WORKSPACE.resolve()/"outputs":
        raise ValueError("The experiment outputs directory must not be a symlink")
    path = OUTPUTS.joinpath(*map(str, parts))
    if not path.resolve().is_relative_to(OUTPUTS.resolve()):
        raise ValueError(f"Output escapes the isolated workspace: {path}")
    parent = path
    while parent != OUTPUTS:
        if parent.is_symlink():
            raise ValueError(f"Output symlink is not allowed: {parent}")
        parent = parent.parent
    return path


def verify_records(root, records):
    failures = []
    for item in records:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            failures.append(item["path"])
    if failures:
        raise RuntimeError("Preservation check failed:\n" + "\n".join(failures))
    return len(records)


def verify_preservation(full=True):
    original = read_json(WORKSPACE/"preservation.json")
    snapshot = read_json(WORKSPACE/"baseline_manifest.json")
    if full:
        n_original = verify_records(PROJECT, original["files"])
        n_snapshot = verify_records(WORKSPACE, snapshot["files"])
    else:
        n_original = 0
        n_snapshot = verify_records(WORKSPACE, [r for r in snapshot["files"] if r["path"].startswith("baseline/source/") or r["path"].startswith("baseline/data/")])
    return dict(original_files_checked=n_original, snapshot_files_checked=n_snapshot, unchanged=True)


def plan():
    return yaml.safe_load((WORKSPACE/"configs/plan.yaml").read_text())


def expand_paths(value):
    if isinstance(value, dict):
        return {k: expand_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_paths(v) for v in value]
    if isinstance(value, str):
        for prefix, root in (("@project/", PROJECT), ("@baseline/", BASELINE), ("@workspace/", WORKSPACE)):
            if value.startswith(prefix):
                return str(root/value.removeprefix(prefix))
    return value


def load_experiment(name, overrides=(), pretrained=None):
    from omegaconf import OmegaConf
    entries = plan()["experiments"]
    if name not in entries:
        raise ValueError(f"Unknown experiment {name}; choose from {list(entries)}")
    cfg = OmegaConf.merge(OmegaConf.load(WORKSPACE/entries[name]["config"]), OmegaConf.from_dotlist(list(overrides)))
    cfg = expand_paths(OmegaConf.to_container(cfg, resolve=True))
    if pretrained:
        cfg["model"]["pretrained_checkpoint"] = str(Path(pretrained).expanduser().resolve())
    return cfg


def run_dir(name, split="official", smoke=False):
    safe_name(name)
    if split not in ["official", "all90"]+[f"fold{i}" for i in range(5)]:
        raise ValueError("split must be official, all90, or fold0...fold4")
    return output_path("smoke" if smoke else "runs", name, split)


def checkpoint_for(name, split="official", smoke=False):
    if name == "baseline":
        if split != "official" or smoke:
            raise ValueError("The frozen baseline exists only for the official split")
        checkpoint = BASELINE / "run/best.pt"
        record = next(r for r in read_json(WORKSPACE/"baseline_manifest.json")["files"] if r["path"] == "baseline/run/best.pt")
        verify_records(WORKSPACE, [record])
        return checkpoint
    checkpoint = run_dir(name, split, smoke)/"best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Train {name} first; checkpoint missing: {checkpoint}")
    return checkpoint


def experiment_source_hash():
    files = [WORKSPACE/"run.py", *sorted((WORKSPACE/"lab").glob("*.py"))]
    return json_hash({str(p.relative_to(WORKSPACE)): sha256(p) for p in files})


def atomic_npy(path, array):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".npy.tmp")
    with temp.open("wb") as file:
        np.save(file, array, allow_pickle=False)
    temp.replace(path)


def require_manifest():
    from alphase.data.audit import verify_manifest
    return verify_manifest(BASELINE/"data/manifest.csv")
