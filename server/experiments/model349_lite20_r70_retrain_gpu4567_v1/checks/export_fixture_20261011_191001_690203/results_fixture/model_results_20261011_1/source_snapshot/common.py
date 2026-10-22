"""Pinned lite20 control and the three full-train70 B4 experiments."""
from pathlib import Path
import json,os,fcntl
from experiment import ROOT,REPO,GPUS,environment,inventory,require_komap,stamp,sha,read_json

BASE=REPO/'experiments/model349_lite20_b4ws_retrain_gpu234567_v1/outputs/lite20_b4ws_train_20261011_130221_265591'
MODEL=BASE/'retrained_model'
GROUP_COUNTS={'322_full70':1,'326_full70':1,'329_full70':1}
B4_DIRS={'322':'model322_team_b4_group_oof','326':'model326_hybrid_fullstats_group_oof',
         '329':'model329_fullstats_width4000_uniform_group_oof'}
EXTERNAL_LOCKS=[REPO/f'experiments/{name}/outputs/search.lock' for name in
    ['model349_lite20_seed100_gpu4567_v1','model349_seed100_gpu4567_v1','model349_lite12_seed100_gpu4567_v1']]

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def records(run):return {p.stem:read_json(p) for p in (run/'completed').glob('*.json')}

def artifacts(paths):
    result={}
    for root in map(Path,paths):
        for p in sorted(root.rglob('*')) if root.is_dir() else [root]:
            if p.is_file():result[str(p)]=sha(p)
    return result

def new_attempt(run,identifier):
    job=run/'jobs'/identifier;job.mkdir(parents=True,exist_ok=True)
    for i in range(1,100000):
        p=job/f'attempt_{i}'
        try:p.mkdir();return p
        except FileExistsError:pass
    raise RuntimeError('Too many attempts')

def blocking_experiments():
    blocked=[]
    for path in EXTERNAL_LOCKS:
        if not path.exists():continue
        with path.open('r') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:blocked.append(str(path))
            else:fcntl.flock(f,fcntl.LOCK_UN)
    return blocked

def make_plan():
    tasks=[]
    for c in B4_DIRS:
        tasks.append(dict(id=f'train_{c}',kind='train',group=c,deps=[],priority=0,seed=0,expected_seconds=4100))
    tasks.append(dict(id='package',kind='cpu',group='package',deps=[t['id'] for t in tasks],priority=0,expected_seconds=120))
    for variant in ['lite20_r70','lite20']:
        for split in ['valid','test']:
            tasks.append(dict(id=f'{variant}_{split}',kind='gpu',group='inference',variant=variant,split=split,
                              deps=['package'],priority=1,expected_seconds=1000 if split=='valid' else 600))
    tasks.append(dict(id='export',kind='cpu',group='export',deps=[t['id'] for t in tasks if t['group']=='inference'],priority=0,expected_seconds=120))
    return dict(model='Model349_lite20_r70',training_count=3,stage_count=len(tasks),seed_index=0,seed=0,
                iterations=21000,gpus=GPUS,baseline=str(BASE),baseline_result='model_results_20261011_1',
                deployed_weights=17,control_weights=20,selection='fixed last checkpoint; no valid/test selection',tasks=tasks)

def validate_plan(plan):
    expected=make_plan()
    if plan!=expected:raise ValueError('Plan differs from the pinned S=0 r70 paired experiment')
    seen=set()
    for t in plan['tasks']:
        assert t['id'] not in seen and set(t['deps'])<=seen;seen.add(t['id'])
    return plan

def command_for(component,output):
    gray=component=='322';width=4000 if component=='329' else 0
    code=ROOT/'source/experiments'/('model322_team_b4' if gray else 'model326_hybrid_fullstats')/'KoMaP/train.py'
    import sys
    return [sys.executable,'-B','-u',str(code),'--data',str(ROOT/'train_data'),'--train_splits','train','--out',str(output),
            '--arch','unet','--encoder','efficientnet-b4','--norm','imagenet' if gray else 'image_global',
            '--input','gray3' if gray else 'zcdf3','--crop','1536','--scale','1' if width else '3','--target_width',str(width),
            '--sample','uniform' if width else 'area','--bs','2','--iters','21000','--lr','3e-4','--wd','1e-2',
            '--dice_w','1','--cldice_w','0','--seed','0','--workers','4','--no_valid_eval','--deterministic_workers']
