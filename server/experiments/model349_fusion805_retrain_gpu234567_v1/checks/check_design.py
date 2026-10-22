"""Parse each scheduled command and audit fixed training settings, without training."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import ROOT,read_json,write_json,environment
from scheduler import render,validate_tasks
plan=read_json(ROOT/'plan.json');validate_tasks(plan['tasks'])
tasks={t['id']:t for t in plan['tasks']}
assert len(tasks)==43 and sum(t['role']=='train' for t in tasks.values())==27
assert len([t for t in tasks.values() if t['family']=='release'])==14
assert len([t for t in tasks.values() if t['family']=='expert'])==12
assert tasks['train_refiner']['deps']==['build_refiner_features']
assert len(tasks['build_refiner_features']['deps'])==12
for t in tasks.values():
    if t['id'].startswith('oof_'):assert len(t['deps'])==1 and t['deps'][0].startswith('train_expert_')
    if t['id'].startswith('valid_'):assert len(t['deps'])==4
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
    if t['family']=='expert':
        if 'b4' in t['id']:
            assert a['iters']==16000 and a['target_width']==4000 and a['seed']==0
            assert a['no_valid_eval'] and a['deterministic_workers'] and a['input']=='zcdf3'
        else:
            assert a['epochs']==70 and a['normalize_width'] and a['val_every']==0 and a['rng_mode']=='legacy'
            assert a['batch_size']==4 and a['accumulation_steps']==2 and a['patch_size']==512
    return t['id']
with ThreadPoolExecutor(max_workers=4) as pool:completed=list(pool.map(parse,tasks.values()))
assert not (ROOT/'checks/never_executed').exists()
write_json(ROOT/'checks/design.json',dict(ok=True,commands_parsed=len(completed),training_count=27,
    training_started=False,dependencies_verified=True,expert_settings_verified=True))
print('PASS: 43 real commands parse; 27 training jobs, OOF dependencies and expert settings verified.')
