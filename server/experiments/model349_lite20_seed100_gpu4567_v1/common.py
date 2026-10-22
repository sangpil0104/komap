"""Fixed lite20 baseline and reproducible 100-training search design."""
from pathlib import Path
import json, os, hashlib
from experiment import ROOT, REPO, GPUS, sha, read_json, environment, inventory, require_komap, stamp

BASE = REPO/'experiments/model349_lite20_b4ws_retrain_gpu234567_v1/outputs/lite20_b4ws_train_20261011_130221_265591'
MODEL = BASE/'retrained_model'
EXPECTED_SCORE = 0.8025584552762022
GROUP_COUNTS = {'336':12,'347':12,'269':8,'restart51':12,'restart61':4,'326':8,'322':4,'E_b4':24,'E_hr':8,'F':8}
R_DIRS = {'336':'model336_hrnet_best','347':'model347_hrnet_inverse_ce_finetune',
          '269':'model269_fullstats_1x_extended','restart51':'restart51_aspp_768_finetune',
          'restart61':'restart61_pairwise_margin','326':'model326_hybrid_fullstats_group_oof',
          '322':'model322_team_b4_group_oof'}
HR_SLOTS = [(1,2),(1,4),(2,1),(2,4)]

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def weight_key(group, fold=None, slot=None):
    if group in R_DIRS:
        return 'R/weights/'+R_DIRS[group]+(f'/fold{fold}/last.pt' if fold else '/'+('last.pt' if group in ['336','347'] else 'best.pt'))
    if group=='E_b4':return f'E/model329_ws/fold{fold}/last.pt'
    if group=='E_hr':return f'E/model392/seed{slot}/fold{fold}/last.pt'
    if group=='F':return 'E/refiner/epoch8.pt'
    raise ValueError(group)

def baseline_weights():
    return {r['namespace']+'/'+r['path']:str((MODEL/'Model349_exh14' if r['namespace']=='R' else MODEL/'assets')/r['path'])
            for r in read_json(MODEL/'lite20/selected_assets_lite20.json')['checkpoints']}

def artifacts(paths):
    result={}
    for root in paths:
        root=Path(root)
        for p in sorted(root.rglob('*')) if root.is_dir() else [root]:
            if p.is_file():result[str(p)]=sha(p)
    return result

def records(run):return {p.stem:read_json(p) for p in (run/'completed').glob('*.json')}

def new_attempt(run, identifier):
    root=run/'jobs'/identifier;root.mkdir(parents=True,exist_ok=True)
    for n in range(1,100000):
        p=root/f'attempt_{n}'
        try:p.mkdir();return p
        except FileExistsError:continue
    raise RuntimeError('Too many attempts')

def make_plan():
    tasks=[];next_seed=3407
    def task(identifier,kind,group,deps=(),seconds=60,**kwargs):
        row=dict(id=identifier,kind=kind,group=group,deps=list(deps),expected_seconds=seconds,
                 priority=0 if group in ['336','parents','baseline'] else 1,**kwargs)
        tasks.append(row);return identifier
    def train(group,index,recipe,deps=('baseline',),seconds=600,**kwargs):
        nonlocal next_seed
        identifier=f'{group}_{index:02d}';seed=next_seed;next_seed+=1
        task(identifier,'train',group,deps,seconds,seed=seed,recipe=recipe,**kwargs);return identifier
    task('baseline','cpu','baseline',seconds=120)
    parents=[train('336',i,'train_model336',seconds=2600) for i in range(12)]
    task('parents','cpu','parents',parents,seconds=1)
    for i in range(12):
        train('347',i,'train_model347',deps=['baseline'] if i<4 else ['parents'],seconds=500,
              parent_rank=None if i<4 else (i-4)//4)
    for group,n,sec in [('269',8,430),('restart51',12,400),('restart61',4,360)]:
        for i in range(n):train(group,i,'train_'+(group if group.startswith('restart') else 'model'+group),seconds=sec)
    for group,folds,sets in [('326',[1,2],4),('322',[3],4),('E_b4',[1,2,3,4],6)]:
        for s in range(sets):
            members=[]
            for f in folds:
                recipe=f'train_b4ws_fold{f}' if group=='E_b4' else f'train_model{group}_fold{f}'
                members.append(train(group,s*len(folds)+len(members),recipe,seconds=450 if group=='E_b4' else 3100,fold=f,set=s))
            task(f'{group}_set{s}','group',group,members,seconds=180 if group=='E_b4' else 100,set=s)
    for s in range(2):
        members=[]
        for slot,fold in HR_SLOTS:
            members.append(train('E_hr',s*4+len(members),f'train_expert_hr{slot}_fold{fold}',seconds=2000,fold=fold,slot=slot,set=s))
        task(f'E_hr_set{s}','group','E_hr',members,seconds=550,set=s)
    re_deps=[t['id'] for t in tasks if t['kind'] in ['train','group']]
    task('select_RE','cpu','select_RE',re_deps,seconds=2400)
    task('features','gpu','features',['select_RE'],seconds=700)
    fs=[train('F',i,'train_refiner',deps=['features'],seconds=110) for i in range(8)]
    task('select_final','cpu','select_final',fs,seconds=2400)
    task('export','gpu','export',['select_final'],seconds=1200)
    plan=dict(model='Model349_lite20_b4ws',training_count=100,gpus=GPUS,seeds=list(range(3407,3507)),
              baseline=str(BASE),baseline_image_miou=EXPECTED_SCORE,group_counts=GROUP_COUNTS,
              deployed_checkpoints=20,coordinate_rounds=2,top_per_group=2,tasks=tasks)
    validate_plan(plan);return plan

def validate_plan(plan):
    train=[t for t in plan['tasks'] if t['kind']=='train'];ids={t['id'] for t in plan['tasks']}
    assert len(ids)==len(plan['tasks']) and len(train)==100 and plan['gpus']==GPUS
    assert sorted(t['seed'] for t in train)==list(range(3407,3507))
    assert {g:sum(t['group']==g for t in train) for g in GROUP_COUNTS}==GROUP_COUNTS
    visited=set()
    for t in plan['tasks']:
        assert set(t['deps'])<=visited,(t['id'],t['deps']);visited.add(t['id'])
    assert sum(t['kind']=='train' for t in plan['tasks'][:next(i for i,t in enumerate(plan['tasks']) if t['id']=='select_RE')])==92
    return plan
