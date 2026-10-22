import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def utc_stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as f:
            f.write(text)
        return
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require_komap():
    if Path(sys.prefix).name != 'komap':
        raise RuntimeError('Use conda run -n komap --no-capture-output python -B ...')


def environment():
    return dict(python=sys.version, executable=sys.executable, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), packages={
                    k: importlib.metadata.version(k) for k in
                    ['torch', 'torchvision', 'segmentation-models-pytorch', 'timm', 'albumentations',
                     'opencv-python-headless', 'numpy', 'scipy', 'pillow', 'tqdm']})


def immutable_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    try:
        torch.save(value, tmp)
        # Hard-link creation fails atomically if a previous checkpoint exists.
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def capture_rng():
    import random
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    import random
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def reserve_result(results_root):
    day = datetime.now(timezone.utc).strftime('%m%d')
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(1, 100000):
        path = root / f'model_results_{day}_{i}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            pass
    raise RuntimeError('No available result directory')
