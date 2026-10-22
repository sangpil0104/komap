from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from omegaconf import OmegaConf
from alphase.utils import sha256, json_hash
from evolvelab.catalog import catalog

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
HUNDRED = ROOT.parent / 'hundred_v1'
BASE_DATA = ROOT.parent / 'accuracy_v1/baseline'
OUTPUTS = ROOT / 'outputs'
HARD_MAPS = ROOT / 'data/hard_maps'

# The 10 image-mIoU top-10 hundred_v1 parents (read-only: never written by this suite).
PARENTS = {
    'm011_seed2028_parent': HUNDRED / 'outputs/runs/m011_seed2028_parent/best_trained.pt',
    'm013_fullframe_parent': HUNDRED / 'outputs/runs/m013_fullframe_parent/best_trained.pt',
    'm063_aux_02': HUNDRED / 'outputs/runs/m063_aux_02/best_trained.pt',
    'm085_local_contrast': HUNDRED / 'outputs/runs/m085_local_contrast/best_trained.pt',
    'm009_head_only': HUNDRED / 'outputs/runs/m009_head_only/best_trained.pt',
    'm012_photo_parent': HUNDRED / 'outputs/runs/m012_photo_parent/best_trained.pt',
    'm008_decoder_only': HUNDRED / 'outputs/runs/m008_decoder_only/best_trained.pt',
    'm006_ema999': HUNDRED / 'outputs/runs/m006_ema999/best_trained.pt',
    'm062_aux_01': HUNDRED / 'outputs/runs/m062_aux_01/best_trained.pt',
    'm016_aspp_w64_staged': HUNDRED / 'outputs/runs/m016_aspp_w64_staged/best_trained.pt',
}


def read_json(path):
    return json.loads(Path(path).read_text())


def output_path(*parts):
    path = OUTPUTS.joinpath(*map(str, parts))
    if not path.resolve().is_relative_to(OUTPUTS.resolve()):
        raise ValueError('Output escapes this experiment workspace')
    for p in (path, *path.parents):
        if p == ROOT:
            break
        if p.is_symlink():
            raise ValueError(f'Output symlink: {p}')
    return path


def run_dir(name, smoke=False):
    if name not in catalog():
        raise ValueError('Unknown experiment')
    return output_path('smoke_runs' if smoke else 'runs', name)


@lru_cache(None)
def protected():
    path = ROOT / 'preservation.json'
    if not path.is_file():
        return {}
    return {r['path']: r for r in read_json(path)['files']}


@lru_cache(None)
def source_hash():
    paths = [ROOT / 'run.py', ROOT / 'configs/base.yaml', *sorted((ROOT / 'evolvelab').glob('*.py')),
             *sorted((ROOT / 'vendor/alphase').rglob('*.py'))]
    return json_hash({str(p.relative_to(ROOT)): sha256(p) for p in paths})


def verify():
    rows = list(protected().values())
    changed = [r['path'] for r in rows if not (PROJECT / r['path']).is_file()
               or (PROJECT / r['path']).stat().st_size != r['bytes'] or sha256(PROJECT / r['path']) != r['sha256']]
    if changed:
        raise RuntimeError('Preserved files changed: ' + str(changed))
    return dict(unchanged=True, files_checked=len(rows))


def config(name, smoke=False):
    item = deepcopy(catalog()[name])
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / 'configs/base.yaml'), resolve=True)
    cfg['experiment'] = name + ('_smoke' if smoke else '')
    cfg['seed'] = 2026
    cfg['data'].update(root=str(PROJECT / 'dataset'), manifest=str(BASE_DATA / 'data/manifest.csv'), folds=str(BASE_DATA / 'data/folds.json'))
    cfg['model'] = dict(name='evolve_cnn', techniques={})
    cfg['model'].update(item.get('model', {}))
    parent = item['parent']
    parent_path = PARENTS[parent]
    parent_key = str(parent_path.relative_to(PROJECT))
    cfg['hundred'] = dict(
        parent=parent, parent_checkpoint=str(parent_path),
        parent_sha256=protected().get(parent_key, {}).get('sha256', sha256(parent_path)),
        source_sha256=source_hash(), mining_folder=str(HARD_MAPS / parent),
        group=item['group'], description=item['description'], smoke=smoke)
    cfg['augmentation'].update(item.get('augmentation', {}))
    cfg['loss'].update(item.get('loss', {}))
    cfg['sampling'].update(item.get('sampling', {}))
    cfg['train'].update(item.get('train', {}))
    cfg['train']['stages'] = item['stages']
    for i, s in enumerate(cfg['train']['stages']):
        s['name'] = f's{i}_' + ('fullframe' if s['mode'] == 'full_frame' else 'patch')
        s.setdefault('policy', 'all')
    if smoke:
        cfg['train'].update(eval_every_updates=1, checkpoint_every_updates=1, log_every_updates=1, num_workers=1)
        for s in cfg['train']['stages']:
            s.update(max_updates=2, warmup_updates=1)
    return cfg
