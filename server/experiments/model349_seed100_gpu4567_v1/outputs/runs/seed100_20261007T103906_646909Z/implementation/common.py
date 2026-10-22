"""Paths, immutable plan, and metrics for the isolated 100-training seed search."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/Data3/sppark/komap')
BASE = REPO / 'experiments/model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z'
ORIGINAL_BUNDLE = REPO / 'results/model349_inference_bundle_20261007_1'
EXPECTED_SCORE = 0.8006879890006227
PALETTE = [(25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204)]
CLASS_NAMES = ['Al3Ni', 'Primary Si', 'Al', 'Eutectic Si']
COMPONENT_DIRS = {'123': 'model123_ema', '269': 'model269_fullstats_1x_extended',
                  '322': 'model322_team_b4_group_oof', '326': 'model326_hybrid_fullstats_group_oof',
                  '329': 'model329_fullstats_width4000_uniform_group_oof',
                  '336': 'model336_hrnet_best', '347': 'model347_hrnet_inverse_ce_finetune'}

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def key(name):
    return Path(name).stem.removesuffix('_image').removesuffix('_mask')

def class_ious(pred, target):
    import numpy as np
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError('Expected equally shaped H,W masks')
    if not (np.isin(pred, range(4)).all() and np.isin(target, range(4)).all()):
        raise ValueError('Labels must be 0..3')
    intersection = np.array([np.count_nonzero((pred == c) & (target == c)) for c in range(4)])
    union = np.array([np.count_nonzero((pred == c) | (target == c)) for c in range(4)])
    return np.divide(intersection, union, out=np.zeros(4, dtype=np.float64), where=union > 0)

def make_plan():
    tasks = []
    def add(model, seed, priority, seconds, **extra):
        tasks.append(dict(id=f'm{model}_s{seed}', model=str(model), seed=seed, kind='train',
                          deps=[], priority=priority, expected_seconds=seconds, **extra))
    for seed in range(3407, 3423): add(269, seed, 10, 330)
    for seed in range(3423, 3431): add(347, seed, 11, 560)
    for seed in range(3431, 3443): add(336, seed, 20, 2820)
    for seed in range(3443, 3451):
        add(347, seed, 25, 560, parent_rank=(seed - 3443) // 4)
        tasks[-1]['deps'] = ['select_parents']
    for model, start, priority in [(329, 3451, 30), (326, 3459, 31)]:
        for offset in range(8):
            add(model, start + offset, priority, 3120, fold=offset % 4 + 1,
                group=f'm{model}_set{offset // 4 + 1}')
    for seed in range(3467, 3507): add(123, seed, 12, 210)
    tasks.append(dict(id='select_parents', kind='select', model='336',
                      deps=[f'm336_s{s}' for s in range(3431, 3443)], priority=0,
                      expected_seconds=5))
    for model, start in [(329, 3451), (326, 3459)]:
        for group in range(2):
            tasks.append(dict(id=f'm{model}_set{group+1}', kind='ensemble', model=str(model),
                              deps=[f'm{model}_s{s}' for s in range(start+group*4, start+group*4+4)],
                              priority=1, expected_seconds=160))
    return dict(training_count=100, seed_start=3407, seed_end=3506,
                gpus=[4,5,6,7], tasks=tasks)

def validate_plan(plan):
    training = [t for t in plan['tasks'] if t['kind'] == 'train']
    assert len(training) == 100
    assert sorted(t['seed'] for t in training) == list(range(3407,3507))
    assert sum(t['model'] == '123' for t in training) == 40
    ids = [t['id'] for t in plan['tasks']]
    assert len(ids) == len(set(ids))
    done = set()
    while len(done) < len(ids):
        ready = {t['id'] for t in plan['tasks'] if set(t['deps']) <= done} - done
        assert ready, 'Dependency cycle or unknown dependency'
        done |= ready
    for task in plan['tasks']:
        if task['kind'] == 'ensemble':
            folds = [t['fold'] for t in training if t.get('group') == task['id']]
            assert sorted(folds) == [1,2,3,4]
    return plan

def environment():
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', NO_ALBUMENTATIONS_UPDATE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               HF_HUB_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONPATH=f'{ROOT}/source/experiments:{ROOT}/source/experiments/historical_scripts')
    return env
