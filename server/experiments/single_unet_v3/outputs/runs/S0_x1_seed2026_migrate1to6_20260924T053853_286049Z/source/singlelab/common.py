from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = next(p for p in ROOT.parents if (p / "spec.md").is_file() and (p / "dataset").is_dir())
OUTPUTS = ROOT / "outputs"
CLASSES = ("primary_si", "al3ni", "eutectic_si", "al")
PALETTE = np.array([(76, 178, 76), (25, 76, 153), (204, 204, 204), (153, 127, 76)], np.uint8)


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sources():
    paths = [ROOT / "run.py", ROOT / "run_suite.sh", ROOT / "run_6gpu.sh", ROOT / "configs/base.yaml"]
    paths += sorted((ROOT / "singlelab").glob("*.py"))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def snapshot(run):
    hashes = sources()
    for relative in hashes:
        target = run / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    write_json(run / "source_hashes.json", hashes)


def config(experiment, seed=None, scale=None, split="official"):
    cfg = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    if experiment not in ("S0", "S1", "S2", "S3"):
        raise ValueError("Experiment must be S0, S1, S2, or S3")
    if experiment == "S3" and scale not in (1, 2, 3):
        raise ValueError("S3 requires --scale 1, 2, or 3; no automatic validation-driven choice")
    if experiment != "S3" and scale is not None:
        raise ValueError("Only S3 accepts --scale; S0/S1/S2 use 1/2/3 respectively")
    cfg.update(experiment=experiment, scale=scale if experiment == "S3" else int(experiment[1]) + 1)
    cfg["seed"] = cfg["seed"] if seed is None else seed
    cfg["data"]["split"] = split
    if experiment == "S3":
        cfg["loss"]["boundary_weight"] = 0.2
    return cfg


def records(cfg):
    manifest = PROJECT / cfg["data"]["manifest"]
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("image_path", "mask_path"):
            if row.get(key):
                p = Path(row[key])
                row[key] = str(p if p.is_absolute() else PROJECT / p)
    official = {s: [r for r in rows if r["source_split"] == s] for s in ("train", "valid", "test")}
    if [len(official[s]) for s in ("train", "valid", "test")] != [70, 20, 10]:
        raise ValueError("Expected the preserved official 70/20/10 dataset")
    split = cfg["data"]["split"]
    if split == "official":
        parts = official
    elif re.fullmatch(r"fold[0-4]", split):
        folds = read_json(PROJECT / cfg["data"]["folds"])
        if folds["manifest_sha256"] != digest(manifest):
            raise ValueError("Fold manifest hash mismatch")
        item = folds["folds"][int(split[-1])]
        parts = {"train": [r for r in rows if r["image_id"] in item["train_ids"]],
                 "valid": [r for r in rows if r["image_id"] in item["validation_ids"]],
                 "test": official["test"]}
        if {r["group_id"] for r in parts["train"]} & {r["group_id"] for r in parts["valid"]}:
            raise ValueError("Group leakage in fold")
    else:
        raise ValueError("split must be official or fold0...fold4")
    sets = [{r["image_id"] for r in parts[s]} for s in ("train", "valid", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Image IDs overlap across splits")
    return parts


def dataset_signature(cfg, verify=False):
    parts = records(cfg)
    if verify:
        from tqdm import tqdm
        rows = [r for split in parts.values() for r in split]
        for row in tqdm(rows, desc="Verify dataset", unit="image", dynamic_ncols=True):
            for key, expected in (("image_path", "image_sha256"), ("mask_path", "mask_sha256")):
                if row.get(key) and digest(row[key]) != row[expected]:
                    raise ValueError(f"Dataset changed: {row[key]}")
    return {"manifest_sha256": digest(PROJECT / cfg["data"]["manifest"]),
            "ids": {s: [r["image_id"] for r in rows] for s, rows in parts.items()}}
