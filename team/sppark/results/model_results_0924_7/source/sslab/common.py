from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = next(p for p in ROOT.parents if (p / "spec.md").is_file())
CLASSES = ("primary_si", "al3ni", "eutectic_si", "al")
CLASS_NAMES = ("Primary Si", "Al3Ni", "Eutectic Si", "Al")
PALETTE = np.array([(76, 178, 76), (25, 76, 153), (204, 204, 204), (153, 127, 76)], np.uint8)


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sources():
    files = list(ROOT.glob("*.py")) + list(ROOT.glob("*.sh")) + list((ROOT / "sslab").glob("*.py"))
    files += list((ROOT / "configs").glob("*.yaml"))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def snapshot(path):
    hashes = sources()
    for name in hashes:
        destination = path / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    write_json(path / "source_hashes.json", hashes)


def config(path=None):
    cfg = yaml.safe_load(Path(path or ROOT / "configs/base.yaml").read_text())
    if cfg["model"]["encoder"] != "resnet18" or cfg["scale"] != 1:
        raise ValueError("This experiment fixes ResNet18 and native resolution")
    if cfg["evaluation"]["empty_union"] != "one":
        raise ValueError("Primary metric is fixed to empty=one; alternatives are also saved")
    size = cfg["sampling"]["core"] + 2 * cfg["sampling"]["halo"]
    if size % 32 or size % cfg["ssl"]["mask_block"] or size < 32:
        raise ValueError("Input size must be divisible by 32 and the mask block size")
    if not 0 < cfg["ssl"]["mask_ratio"] < 1 or cfg["sampling"]["core"] <= 0 or cfg["sampling"]["halo"] < 0:
        raise ValueError("Invalid masking/crop parameters")
    for section, key in [("train", "updates"), ("train", "effective_batch"), ("ssl", "updates"),
                         ("train", "log_every"), ("train", "checkpoint_every"), ("control", "max_extra_updates")]:
        if cfg[section][key] < 1:
            raise ValueError(f"{section}.{key} must be positive")
    return cfg


def records(cfg):
    path = PROJECT / cfg["data"]["manifest"]
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    parts = {s: [r for r in rows if r["source_split"] == s] for s in ("train", "valid", "test")}
    if len(rows) != 100 or [len(parts[s]) for s in parts] != [70, 20, 10]:
        raise ValueError("Expected fixed official train70/valid20/test10")
    if len({r["image_id"] for r in rows}) != 100:
        raise ValueError("Repeated image ID")
    if any('/' in r['image_id'] or '\\' in r['image_id'] or r['image_id'] in ('.','..') for r in rows):
        raise ValueError("Unsafe image ID")
    for row in rows:
        for key in ("image_path", "mask_path"):
            if row[key]:
                row[key] = str(PROJECT / row[key])
        if row["source_split"] != "test" and not row["mask_path"]:
            raise ValueError("Train/valid require masks")
    hashes = [{r["image_sha256"] for r in parts[s]} for s in parts]
    if any(hashes[i] & hashes[j] for i in range(3) for j in range(i)):
        raise ValueError("Identical image files occur in different splits")
    return parts


def signature(cfg, verify=False):
    parts = records(cfg)
    if verify:
        from .data import gray, labels
        started = time.monotonic()
        for index, row in enumerate([r for v in parts.values() for r in v]):
            expected = (int(row["height"]), int(row["width"]))
            if gray(row["image_path"]).shape != expected:
                raise ValueError(f"Image size mismatch: {row['image_id']}")
            if row["mask_path"] and labels(row["mask_path"]).shape != expected:
                raise ValueError(f"Mask size mismatch: {row['image_id']}")
            for kind in ("image", "mask"):
                if row[f"{kind}_path"] and digest(row[f"{kind}_path"]) != row[f"{kind}_sha256"]:
                    raise ValueError(f"Data hash mismatch: {row['image_id']} {kind}")
            if (index + 1) % 10 == 0:
                elapsed = time.monotonic() - started
                remaining = elapsed / (index + 1) * (99 - index)
                print(f"[preflight] {index+1}/100 elapsed={elapsed:.1f}s remaining~{remaining:.1f}s", flush=True)
    return {"manifest_sha256": digest(PROJECT / cfg["data"]["manifest"]),
            "ids": {s: [r["image_id"] for r in rows] for s, rows in parts.items()},
            "file_hashes": {r["image_id"]: [r["image_sha256"], r["mask_sha256"]] for rows in parts.values() for r in rows}}


def eta(seconds):
    return (datetime.now().astimezone() + timedelta(seconds=max(0, seconds))).isoformat(timespec="seconds")


def fresh_results(root=None):
    root = Path(root or PROJECT / "results")
    root.mkdir(parents=True, exist_ok=True)
    prefix = "model_results_" + datetime.now().strftime("%m%d") + "_"
    existing = [int(p.name[len(prefix):]) for p in root.iterdir()
                if p.name.startswith(prefix) and p.name[len(prefix):].isdigit()]
    number = max(existing, default=0) + 1
    while True:
        path = root / f"{prefix}{number}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            number += 1
