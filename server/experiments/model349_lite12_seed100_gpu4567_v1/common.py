"""Isolated lite12 seed search: immutable recipes, paths, and image-first metric."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/Data3/sppark/komap')
BASE = REPO / 'experiments/model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z'
ORIGINAL_BUNDLE = REPO / 'results/model349_inference_bundle_20261007_1'
PALETTE = [(25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204)]
CLASS_NAMES = ['Al3Ni', 'Primary Si', 'Al', 'Eutectic Si']
HIGH = ['109', 'restart51', 'restart59', 'restart61']
MODELS = ['269', '336', *HIGH, '326', '329']
COMPONENT_DIRS = {
    '109': 'model109_aspp_native', 'restart51': 'restart51_aspp_768_finetune',
    'restart59': 'restart59_aspp768_seed456', 'restart61': 'restart61_pairwise_margin',
    '269': 'model269_fullstats_1x_extended', '336': 'model336_hrnet_best',
    '326': 'model326_hybrid_fullstats_group_oof', '329': 'model329_fullstats_width4000_uniform_group_oof',
}
FOLDS = {'326': [1, 3], '329': [1, 2, 3, 4]}

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()

def weight_relative(model, fold=None):
    folder = Path('weights') / COMPONENT_DIRS[model]
    return folder / f'fold{fold}' / 'last.pt' if fold else folder / ('last.pt' if model == '336' else 'best.pt')

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
    # Every completed experiment is scored through exactly the same lite12 route.
    tasks = [dict(id=f'baseline_{c}', kind='baseline', model=c, deps=[], priority=0,
                  expected_seconds=300 if c == 'base' else 120)
             for c in ['326', '329', 'base', '336', '269']]
    tasks.append(dict(id='baseline_score', kind='select', model='baseline', priority=0,
                      deps=[t['id'] for t in tasks], expected_seconds=20))
    seed = 3407
    for model, count, seconds in [('269',16,450), ('336',12,3000), ('109',16,300),
                                  ('restart51',16,600), ('restart59',8,550), ('restart61',8,650)]:
        for _ in range(count):
            tasks.append(dict(id=f'm{model}_s{seed}', model=model, seed=seed, kind='train',
                              deps=['baseline_score'], priority=20 if model == '336' else 10,
                              expected_seconds=seconds))
            seed += 1
    for model in ['326', '329']:
        for group in range(1, 5):
            deps = []
            for fold in FOLDS[model]:
                identifier = f'm{model}_s{seed}'
                tasks.append(dict(id=identifier, model=model, seed=seed, kind='train',
                                  deps=['baseline_score'], priority=30, expected_seconds=3120,
                                  fold=fold, group=f'm{model}_set{group}'))
                deps.append(identifier); seed += 1
            tasks.append(dict(id=f'm{model}_set{group}', kind='ensemble', model=model,
                              deps=deps, priority=1, expected_seconds=200))
    return dict(training_count=100, candidate_count=84, seed_start=3407, seed_end=3506,
                gpus=[4,5,6,7], metric='image_mean_four_classes_empty0', tasks=tasks)

def validate_plan(plan):
    if plan != make_plan(): raise ValueError('Plan differs from the fixed 100-training lite12 plan')
    training = [t for t in plan['tasks'] if t['kind'] == 'train']
    assert len(training) == 100
    assert sorted(t['seed'] for t in training) == list(range(3407,3507))
    ids = [t['id'] for t in plan['tasks']]; assert len(ids) == len(set(ids))
    done = set()
    while len(done) < len(ids):
        ready = {t['id'] for t in plan['tasks'] if set(t['deps']) <= done} - done
        assert ready, 'Dependency cycle or unknown dependency'
        done |= ready
    for task in plan['tasks']:
        if task['kind'] == 'ensemble':
            assert sorted(t['fold'] for t in training if t.get('group') == task['id']) == FOLDS[task['model']]
    return plan

def environment():
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', NO_ALBUMENTATIONS_UPDATE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               HF_HUB_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               PYTHONPATH=f'{ROOT}/source/experiments:{ROOT}/source/experiments/historical_scripts')
    return env

def fp16_checkpoint(source, target):
    """Create a new inference-only checkpoint; training source stays untouched."""
    import torch
    source, target = Path(source), Path(target)
    if target.exists(): raise FileExistsError(target)
    payload = torch.load(source, map_location='cpu', weights_only=False)
    def half(state):
        return {k: v.half() if torch.is_floating_point(v) else v for k,v in state.items()}
    if 'state_dict' in payload:
        output = {'state_dict': half(payload['state_dict']), 'model_config': payload['model_config']}
    elif 'model' in payload:
        output = {'model': half(payload['model'])}
    else:
        output = half(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream: torch.save(output, stream)
    return dict(source=str(source), source_sha256=sha(source), path=str(target),
                sha256=sha(target), bytes=target.stat().st_size)
