"""Isolated orchestration; old experiments and their recipes remain immutable."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = Path('/Data3/sppark/komap')
ROOT = REPO/'experiments/model349_r70_priority_gpu4567_v1'
SEED = REPO/'experiments/model349_lite20_seed100_gpu4567_v1'
SEED_RUN = SEED/'outputs/runs/seed100_20261011_170427_031422'
R70 = REPO/'experiments/model349_lite20_r70_retrain_gpu4567_v1'
PREVIEW = REPO/'reports/lite20_seed100_R_preview_20261011_184826_304477/metrics.json'
GPUS = [4, 5, 6, 7]
IDS = [f'336_{i:02d}' for i in range(8)]

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def stamp():
    return datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')

def require_komap():
    expected = Path('/home/sppark/miniconda3/envs/komap').resolve()
    if Path(sys.prefix).resolve() != expected:
        raise RuntimeError('Use ./run.sh: conda komap is required.')

def environment():
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               NO_ALBUMENTATIONS_UPDATE='1', HF_HUB_OFFLINE='1')
    return env

def records(run):
    return {p.stem: read(p) for p in (Path(run)/'completed').glob('*.json')}

def verify_files(files, root=None):
    from tqdm import tqdm
    for key, value in tqdm(files.items(), desc='Verify preserved files', unit='file', mininterval=5):
        path = Path(key) if root is None else Path(root)/key
        if not path.is_file() or sha(path) != value:
            raise ValueError(f'Missing or changed artifact: {path}')

def files_under(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}

def new_attempt(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    for n in range(1, 100000):
        dest = root/f'attempt_{n}'
        try:
            dest.mkdir(); return dest
        except FileExistsError:
            pass
    raise RuntimeError('Too many attempts')

def inventory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    devices = {}
    for line in raw.splitlines():
        index, uuid, memory, utilization = [s.strip() for s in line.split(',')]
        if int(index) in GPUS:
            devices[int(index)] = dict(uuid=uuid, memory=int(memory), utilization=int(utilization))
    if set(devices) != set(GPUS):
        raise RuntimeError('Physical GPUs 4,5,6,7 are required.')
    return devices

def tail(path):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2); stream.seek(max(0, stream.tell()-2500))
            lines = stream.read().decode(errors='replace').replace('\r', '\n').splitlines()
        return next((s for s in reversed(lines) if s.strip()), 'starting')
    except FileNotFoundError:
        return 'starting'

def process(pid):
    try:
        directory = Path('/proc')/str(pid)
        fields = (directory/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]),
                    start=fields[19], args=(directory/'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0'))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None

def alive(identity):
    current = process(identity['pid'])
    return current is not None and current['start'] == identity['start'] and current['state'] not in ['Z', 'X']

def lock_busy(path):
    import fcntl
    path = Path(path)
    if not path.exists(): return False
    with path.open('r') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False
