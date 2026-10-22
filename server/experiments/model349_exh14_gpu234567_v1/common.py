"""Local experiment paths; the GitHub inference recipe remains unchanged."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = ROOT / 'source'
GPUS = [2, 3, 4, 5, 6, 7]
VALID = ROOT / 'data/valid'


def stamp():
    return datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inference():
    sys.path.insert(0, str(SOURCE))
    import infer_lite
    return infer_lite


def gpu_inventory(require_idle=False):
    raw = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    all_gpus = {}
    for line in raw.splitlines():
        index, uuid, name, memory, utilization = [x.strip() for x in line.split(',')]
        all_gpus[int(index)] = dict(index=int(index), uuid=uuid, name=name,
                                   memory_mib=int(memory), utilization=int(utilization))
    selected = [all_gpus[i] for i in GPUS]
    if require_idle:
        busy = [g for g in selected if g['memory_mib'] > 64 or g['utilization']]
        if busy:
            raise RuntimeError(f'GPU 2~7 must be idle before evaluation: {busy}')
    return selected


def require_komap():
    expected = Path('/home/sppark/miniconda3/envs/komap')
    if Path(sys.prefix) != expected or os.environ.get('CONDA_DEFAULT_ENV') != 'komap':
        raise RuntimeError('Use ./run.sh so this experiment runs in conda komap.')
