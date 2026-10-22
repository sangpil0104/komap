from pathlib import Path
import hashlib
import json
import os
import tempfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def ids_hash(ids):
    return digest(sorted(ids))


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(Path(temp))
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def json_write(path, value):
    atomic(
        path,
        lambda p: p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"),
    )


def jsonl_write(path, rows):
    atomic(
        path,
        lambda p: p.write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
        ),
    )


def jsonl_read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def torch_save(path, obj):
    atomic(path, lambda p: torch.save(obj, p))


def numpy_save(path, array):
    def write(p):
        with p.open("wb") as f:
            np.save(f, array, allow_pickle=False)

    atomic(path, write)


def local_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
