"""Shared helpers for the OOF fold x seed design (paths, hashing, labels, metric, fold layout)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
EXH14 = HERE.parent                      # Model349_exh14 bundle root (inference code in src/)
VENDOR = HERE / 'vendor'
CLASS_ORDER = ['Al3Ni', 'Primary Si', 'Al', 'Eutectic Si']          # route/main order
PALETTE = np.array([[25, 76, 153], [76, 178, 76], [153, 127, 76], [204, 204, 204]], np.uint8)
FOLDS = 4


def specimen(name: str) -> str:
    m = re.match(r'(A\d+)_', Path(name).name)
    if not m:
        raise ValueError(f'cannot parse specimen id: {name}')
    return m.group(1)


def stem(path: Path) -> str:
    return path.name.removesuffix('_image.png').removesuffix('_mask.png').removesuffix('.png')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def decode_mask(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert('RGB'))
    out = np.full(rgb.shape[:2], 255, np.uint8)
    for cid, color in enumerate(PALETTE):
        out[np.all(rgb == color, axis=-1)] = cid
    if (out == 255).any():
        raise ValueError(f'unknown mask colour in {path}')
    return out


def image_miou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, list[float]]:
    """Official metric for one image: mean of the four class IoUs; empty union counts as 0."""
    ious = []
    for c in range(4):
        union = np.count_nonzero((pred == c) | (gt == c))
        ious.append(np.count_nonzero((pred == c) & (gt == c)) / union if union else 0.0)
    return float(np.mean(ious)), ious


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def manifest(work: Path) -> dict:
    return read_json(work / 'data_manifest.json')


def fold_of(work: Path) -> dict[str, int]:
    return {row['name']: row['fold'] for row in manifest(work)['labeled']}


def add_inference_paths() -> None:
    """Make the frozen Model349_exh14 inference modules importable."""
    for p in (EXH14 / 'src', EXH14):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def vendor_env() -> dict:
    """Environment for vendored trainers: deterministic runtime first on PYTHONPATH."""
    env = dict(os.environ)
    paths = [HERE / 'runtime', VENDOR / 'experiments', VENDOR / 'experiments' / 'historical_scripts']
    env['PYTHONPATH'] = os.pathsep.join([str(p) for p in paths] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    env.setdefault('PYTHONHASHSEED', '0')
    env['PYTHONUNBUFFERED'] = '1'
    env['NO_ALBUMENTATIONS_UPDATE'] = '1'
    return env
