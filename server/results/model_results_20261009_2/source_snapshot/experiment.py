"""Isolated fusion805 retraining: paths, hashes, environment, and GPU inventory."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = Path('/Data3/sppark/komap')
GPUS = [2, 3, 4, 5, 6, 7]
VALID = ROOT / 'data/valid'


def inference():
    sys.path.insert(0, str(ROOT / 'frozen'))
    import infer_lite
    return infer_lite


def stamp():
    return datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment(root=ROOT):
    env = dict(os.environ)
    env.update(PYTHONPATH=f'{root}/source/experiments:{root}/source/experiments/historical_scripts',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', NO_ALBUMENTATIONS_UPDATE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               HF_HUB_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return env


def require_komap():
    if Path(sys.prefix) != Path('/home/sppark/miniconda3/envs/komap') or os.environ.get('CONDA_DEFAULT_ENV') != 'komap':
        raise RuntimeError('Run this experiment with ./run.sh (conda komap).')


def inventory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], text=True)
    devices = {}
    for line in output.splitlines():
        index, uuid, name, memory, utilization = [x.strip() for x in line.split(',')]
        devices[int(index)] = dict(uuid=uuid, name=name, used_mib=int(memory), utilization=int(utilization))
    return {g: devices[g] for g in GPUS}
