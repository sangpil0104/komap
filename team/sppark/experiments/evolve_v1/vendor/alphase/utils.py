from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def load_config(path, overrides=()):
    cfg = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_container(cfg, resolve=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment():
    import torchvision
    def commit(folder):
        result = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "code_commit": commit("."),
        "backbone_commit": commit("third_party/dinov3"),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def code_hash():
    root = Path(__file__).parent
    return json_hash({str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))})


def amp_context(device, precision):
    if precision not in ("bf16", "fp32"):
        raise ValueError("Supported precision: bf16, fp32")
    enabled = device.type == "cuda" and precision == "bf16"
    if enabled and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16; use train.precision=fp32")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)
