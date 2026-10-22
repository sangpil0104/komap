from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from omegaconf import OmegaConf
from alphase.utils import sha256, json_hash
from hundredlab.catalog import catalog

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
BASE = ROOT.parent / 'accuracy_v1/baseline'
OUTPUTS = ROOT / 'outputs'
PARENTS = {
 'baseline': BASE / 'run/best.pt',
 'seed2028': ROOT.parent / 'accuracy_v1/outputs/runs/seed2028/official/best.pt',
 'fullframe': ROOT.parent / 'accuracy_v1/outputs/runs/fullframe/official/best.pt',
 'photo': ROOT.parent / 'augmentation_v1/outputs/runs/photo_mild/seed2026/official/best.pt',
}

def read_json(path):
    return json.loads(Path(path).read_text())

def output_path(*parts):
    path = OUTPUTS.joinpath(*map(str,parts))
    if not path.resolve().is_relative_to(OUTPUTS.resolve()):
        raise ValueError('Output escapes this experiment workspace')
    for p in (path,*path.parents):
        if p == ROOT: break
        if p.is_symlink(): raise ValueError(f'Output symlink: {p}')
    return path

def run_dir(name, smoke=False):
    if name not in catalog(): raise ValueError('Unknown experiment')
    return output_path('smoke_runs' if smoke else 'runs',name)

@lru_cache(None)
def protected():
    return {r['path']:r for r in read_json(ROOT/'preservation.json')['files']}

@lru_cache(None)
def source_hash():
    paths=[ROOT/'run.py',ROOT/'configs/base.yaml',*sorted((ROOT/'hundredlab').glob('*.py')),
           *sorted((ROOT/'vendor/alphase').rglob('*.py'))]
    return json_hash({str(p.relative_to(ROOT)):sha256(p) for p in paths})

def verify(full=False):
    rows=protected().values()
    if not full:
        required={str(p.relative_to(PROJECT)) for p in PARENTS.values()}
        rows=[r for r in rows if r['path'] in required or r['path'].startswith('experiments/accuracy_v1/baseline/data/')
              or r['path'].startswith('experiments/refinement_v1/outputs/mining/')]
    rows=list(rows)
    changed=[r['path'] for r in rows if not (PROJECT/r['path']).is_file()
             or (PROJECT/r['path']).stat().st_size!=r['bytes'] or sha256(PROJECT/r['path'])!=r['sha256']]
    if changed: raise RuntimeError('Preserved files changed: '+str(changed))
    return dict(unchanged=True,files_checked=len(rows),full=full)

def config(name,smoke=False):
    item=deepcopy(catalog()[name])
    cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'configs/base.yaml'),resolve=True)
    cfg['experiment']=name+('_smoke' if smoke else '')
    cfg['seed']=2026
    cfg['data'].update(root=str(PROJECT/'dataset'),manifest=str(BASE/'data/manifest.csv'),folds=str(BASE/'data/folds.json'))
    cfg['model']=dict(name='hundred_cnn',architecture='warm',adapter='none',variant=1,**{})
    cfg['model'].update(item.get('model',{}))
    parent=item.get('parent','baseline')
    parent_path=PARENTS[parent] if parent else None
    cfg['hundred']=dict(parent=parent,parent_checkpoint=str(parent_path) if parent_path else None,
        parent_sha256=protected()[str(parent_path.relative_to(PROJECT))]['sha256'] if parent_path else None,
        source_sha256=source_hash(),hard=item.get('hard',False),
        mining_folder=str(ROOT.parent/'refinement_v1/outputs/mining'),group=item['group'],
        description=item['description'],smoke=smoke)
    cfg['augmentation'].update(photometric_probability=0.)
    cfg['augmentation'].update(item.get('augmentation',{}))
    cfg['loss'].update(item.get('loss',{}))
    cfg['sampling'].update(item.get('sampling',{}))
    cfg['train'].update(ema_decay=.99,eval_every_updates=200,checkpoint_every_updates=100)
    cfg['train'].update(item.get('train',{}))
    cfg['train']['stages']=item.get('stages',[dict(max_updates=600,new_lr=2e-6,mode='full_frame',warmup_updates=25)])
    for i,s in enumerate(cfg['train']['stages']):
        s['name']=f's{i}_'+('fullframe' if s['mode']=='full_frame' else 'patch')
        s.setdefault('policy','all')
    if smoke:
        cfg['train'].update(eval_every_updates=1,checkpoint_every_updates=1,log_every_updates=1,num_workers=1)
        for s in cfg['train']['stages']: s.update(max_updates=2,warmup_updates=1)
    return cfg
