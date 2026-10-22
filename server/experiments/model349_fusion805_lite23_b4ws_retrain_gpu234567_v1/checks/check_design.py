"""Parse real CLI commands without executing trainer bodies or touching GPUs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import ROOT,read_json,write_json,environment
from scheduler import render,validate_tasks
plan=read_json(ROOT/'plan.json');validate_tasks(plan['tasks'])
tasks={t['id']:t for t in plan['tasks']}
assert len(tasks)==53 and sum(t['role']=='train' for t in tasks.values())==37
assert len(plan['release_slots'])==14 and plan['deployed_checkpoint_count']==23
assert len(tasks['build_refiner_features']['deps'])==12
assert all('b4ws' not in d for d in tasks['build_refiner_features']['deps'])
assert tasks['train_model109']['deps']==['train_model98']
assert tasks['train_model269']['deps']==['train_model257']
assert tasks['valid_hr1']['deps']==['train_expert_hr1_fold2','train_expert_hr1_fold4']
assert tasks['valid_hr2']['deps']==['train_expert_hr2_fold1','train_expert_hr2_fold4']
env=environment();env['CUDA_VISIBLE_DEVICES']=''
out=ROOT/'checks/parsed_arguments';out.mkdir(exist_ok=True)
def parse(t):
    argv=render(t['argv'],ROOT/'source',ROOT/'checks/never_executed',4)
    path=out/f"{t['id']}.json"
    r=subprocess.run([sys.executable,'-B',str(ROOT/'checks/parse_only.py'),argv[0],str(path),*argv[1:]],
                     env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/f"{t['id']}.log").write_text(r.stdout)
    if r.returncode:raise RuntimeError(f"{t['id']}: {r.stdout}")
    a=read_json(path)
    if t['family']=='warmstart':
        assert a['iters']==2000 and a['lr']==3e-5 and a['seed']==819349
        assert a['deterministic_workers'] and a['no_valid_eval'] and a['init_checkpoint'].endswith('/last.pt')
    if t['family']=='expert':
        if 'b4' in t['id']:
            assert a['iters']==16000 and a['target_width']==4000 and a['seed']==0
            assert a['no_valid_eval'] and a['deterministic_workers']
        else:
            assert a['epochs']==70 and a['normalize_width'] and a['val_every']==0
    assert 'initial_weights' not in str(t)
    return t['id']
with ThreadPoolExecutor(max_workers=4) as pool:completed=list(pool.map(parse,tasks.values()))
assert not (ROOT/'checks/never_executed').exists()
write_json(ROOT/'checks/design.json',dict(ok=True,commands_parsed=len(completed),training_count=37,
    training_started=False,dependencies_verified=True,warmstart_settings_verified=True))
print('PASS: 53 real commands parse, all ancestors and original OOF dependencies verified.')
