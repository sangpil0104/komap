"""Compile the upstream S=0 recipe into the local six-GPU dependency scheduler."""
from pathlib import Path
import copy, json, re, shlex

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'model349_fusion805_retrain_gpu234567_v1'
old = json.loads((OLD/'plan.json').read_text())
tasks = [copy.deepcopy(t) for t in old['tasks'] if not t['id'].startswith('train_model329_')]
by_id = {t['id']: t for t in tasks}
release_slots = []
for task in old['tasks']:
    if task['family'] != 'release':
        continue
    fold = task['id'][-1]
    parent = 'train_expert_b4_fold'+fold if task['id'].startswith('train_model329_') else task['id']
    release_slots.append(dict(task=parent, path='weights/'+task['checkpoint'].removeprefix('{run}/outputs/')))

# Original R shell commands are parsed as text, never executed during setup.
text = (ROOT/'compact_source/retrain/retrain_r.sh').read_text()
stages = {}
for line in text.splitlines():
    if not line.startswith('stage '):
        continue
    line = re.sub(r'\$\(sd (\d+)\)', r'\1', line)
    values = shlex.split(line)
    name, checkpoint, *argv = values[1:]
    def local(s):
        for a,b in {'$W':'{run}/outputs', '$H':'{source}/experiments/historical_scripts',
                    '$TR':'{source}', '$DATA':'{source}/../data'}.items():
            s=s.replace(a,b)
        return s
    stages[name] = (local(checkpoint), list(map(local,argv)))

parents = {'model6':None, 'model62':'model6', 'model97':'model62', 'model98':'model97',
           'model194':'model109', 'model195':'model194', 'model239':'model195',
           'model242':'model239', 'model257':'model242'}
times = {'model6':600,'model62':600,'model97':1500,'model98':450,'model194':1800,
         'model195':800,'model239':550,'model242':550,'model257':850}
for i,(name,parent) in enumerate(parents.items()):
    checkpoint,argv=stages[name]
    argv=[a.replace('image_mean_miou_exclude','image_mean_miou') for a in argv]
    required=[argv[argv.index('--checkpoint')+1]] if '--checkpoint' in argv else []
    task=dict(id='train_'+name,kind='gpu',role='train',family='ancestor',priority=0,
              serial_index=i,deps=['train_'+parent] if parent else [],argv=argv,
              checkpoint=checkpoint,requires=required,produces=[checkpoint],
              seed=int(argv[argv.index('--seed')+1]),expected_seconds=times[name])
    tasks.append(task);by_id[task['id']]=task
for child,parent,folder in [('train_model109','model98','model98_hard_negative'),
                            ('train_model269','model257','model257_full_image_stats_1x')]:
    t=by_id[child]; t['deps']=['train_'+parent]
    t['argv']=[s.replace('{source}/../initial_weights/'+folder,'{run}/outputs/'+folder) for s in t['argv']]
    t['requires']=[by_id['train_'+parent]['checkpoint']]

for fold in range(1,5):
    base=by_id[f'train_expert_b4_fold{fold}']
    task=copy.deepcopy(base)
    task.update(id=f'train_b4ws_fold{fold}',family='warmstart',seed=819349,
                deps=[base['id']],priority=0,expected_seconds=420,
                requires=[base['checkpoint']],serial_index=200+fold)
    task['argv']=[s.replace('model397_reproduction_bundle/model329/train.py','b4ws/train_bnfreeze.py')
                  .replace('{run}/expert/model329/', '{run}/expert/model329_ws/') for s in base['argv']]
    for flag,value in [('--iters','2000'),('--lr','3e-5'),('--seed','819349')]:
        task['argv'][task['argv'].index(flag)+1]=value
    task['argv'] += ['--init_checkpoint',base['checkpoint']]
    task['checkpoint']=base['checkpoint'].replace('/model329/','/model329_ws/')
    task['produces']=[s.replace('/model329/','/model329_ws/') for s in base['produces']]
    task['produces'].append(task['checkpoint'].removesuffix('last.pt')+'bn_freeze_check.json')
    tasks.append(task);by_id[task['id']]=task

# F uses all 12 original (pre-warmstart) OOF branches; valid uses only lite23 branches.
for family,folds in [('b4',[1,2,3,4]),('hr1',[2,4]),('hr2',[1,4])]:
    t=by_id['valid_'+family]
    t['argv'] += ['--fold-ids',','.join(map(str,folds))]
    prefix='train_b4ws' if family=='b4' else 'train_expert_'+family
    t['deps']=[f'{prefix}_fold{fold}' for fold in folds]
    if family=='b4': t['argv'] += ['--b4-dir','model329_ws']

# Keep long independent jobs available while the ancestor chain progresses.
for i,t in enumerate(tasks):
    t['serial_index']=i
    if t['role']=='train' and t['family']=='expert':t['priority']=1
    if t['family']=='release' and 'fold' in t['id']:t['priority']=1
assert sum(t['role']=='train' for t in tasks)==37 and len(tasks)==53
plan=dict(training_count=37,stage_count=53,deployed_checkpoint_count=23,recipe_seed=0,
          gpus=[2,3,4,5,6,7],release_slots=release_slots,tasks=tasks,
          primary_stage='stage2',expert_hr_folds={'1':[2,4],'2':[1,4]},
          b4_warmstart=dict(iterations=2000,lr=3e-5,seed=819349,bn_running_stats_frozen=True),
          shared_r_e_b4_folds=[1,3,4],refiner_oof_uses_warmstart=False)
(ROOT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print('37 training jobs / 53 scheduled stages / 23 deployed weights; S=0')
