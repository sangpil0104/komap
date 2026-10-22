"""Resume fixed trials after the preserved baseline, then paired repeats and test parity."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NO_ALBUMENTATIONS_UPDATE='1')
import fcntl,importlib.util,json,shutil,subprocess,time
from pathlib import Path
import yaml
ROOT=Path('/workspace/model349_compress/audit_20261008'); PYTHON='/workspace/venv-komap/bin/python'
INPUT=Path('/workspace/komap_model1/data/valid/images'); GT=Path('/workspace/komap_model1/data/valid/masks')
lock=(ROOT/'gpu7.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(p,obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def load(p): return json.loads(p.read_text())
def idle():
    assert subprocess.run(['pgrep','-f','[i]nfer_lite.py|[i]nfer_audited.py'],stdout=subprocess.DEVNULL).returncode!=0, 'Existing inference; refusing overlap'
    for _ in range(2):
        raw=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        mem,util=map(int,raw.strip().split(','))
        if mem>64 or util: raise RuntimeError(f'GPU7 occupied: {raw}')
        time.sleep(2)
def make(key,source,steps=None):
    src=ROOT/'bundles'/source; dst=ROOT/'bundles'/key
    if dst.exists():
        assert (dst/'infer_lite.py').read_bytes()==(ROOT/'infer_cpu2.py').read_bytes()
        assert (dst/'weights').resolve()==(src/'weights').resolve()
        assert load(dst/'checkpoint_manifest.json')==load(src/'checkpoint_manifest.json')
        return dst
    dst.mkdir(exist_ok=False)
    shutil.copy2(ROOT/'infer_cpu2.py',dst/'infer_lite.py')
    shutil.copytree(src/'src',dst/'src'); shutil.copytree(src/'configs',dst/'configs')
    (dst/'weights').symlink_to((src/'weights').resolve(),target_is_directory=True)
    shutil.copy2(src/'checkpoint_manifest.json',dst/'checkpoint_manifest.json')
    if steps is not None:
        cfg=yaml.safe_load((dst/'configs/model349_lite.yaml').read_text()); cfg['inference']['tent']['steps']=steps
        (dst/'configs/model349_lite.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    return dst
def run(key,bundle=None,test=False):
    existing=ROOT/'runs'/key
    if (existing/'completion.json').is_file():
        print('PRESERVED',key,load(existing/'completion.json'),flush=True)
        return existing
    idle(); bundle=ROOT/'bundles'/(bundle or key); out=ROOT/'runs'/key
    args=[PYTHON,'-B','-u',str(bundle/'infer_lite.py'),'--input',str(INPUT if not test else INPUT.parent.parent/'test/images'),'--output',str(out),'--gpus','7','--keep-probabilities']
    if not test: args+=['--gt',str(GT)]
    start=time.time(); telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True)
    with (ROOT/f'{key}.log').open('w') as f: subprocess.run(args,cwd=bundle,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(out/'run_audit.json',{'started_unix':start,'finished_unix':time.time(),'command':args,'bundle':str(bundle),'gpu_snapshot_start':telemetry})
    print('FINISHED',key,load(out/'completion.json'),flush=True)
    return out
# Preserve the positively identified in-flight FP16 TENT5 inference.
transition=load(ROOT/'probe_transition.json')
while not (ROOT/'runs'/transition['active_run']/'completion.json').is_file():
    proc=Path(f"/proc/{transition['active_inference_pid']}/cmdline")
    if not proc.exists() or transition['active_inference_code'] not in proc.read_bytes().replace(b'\0',b' ').decode():
        raise RuntimeError('In-flight inference ended without completion')
    time.sleep(10)
keys=[]
for source in ['lite15','fp16_t5','fp16_t3','fp16_12','fp16_10']:
    key='cpu2_'+source; make(key,source); run(key); keys.append(key)
spec=importlib.util.spec_from_file_location('analysis',ROOT/'analyze_runs.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
refpath=ROOT/'runs/lite15'; ref=mod.score(refpath,GT)
def quality(key):
    path=ROOT/'runs'/key; metrics=mod.score(path,GT)
    return mod.pair(refpath,path,ref,metrics)
if quality('cpu2_fp16_t3')['quality_pass'] and quality('cpu2_fp16_12')['quality_pass']:
    key='cpu2_fp16_12_t3'; make(key,'fp16_12',steps=3); run(key); keys.append(key)
reference_bytes=load(ROOT/'bundles/lite15/checkpoint_manifest.json')['bytes']
eligible=[key for key in keys if key!='cpu2_lite15' and quality(key)['quality_pass'] and load(ROOT/'bundles'/key/'checkpoint_manifest.json')['bytes']<reference_bytes]
write(ROOT/'initial_controlled_results.json',{'quality':{key:quality(key) for key in keys},'eligible':eligible})
subprocess.run([PYTHON,'-B',str(ROOT/'analyze_runs.py'),'--root',str(ROOT),'--gt',str(GT)],check=True)
write(ROOT/'session_complete.json',{'status':'fixed_probes_completed','eligible_quality_storage_only':eligible,'timing_adoption_not_claimed':'Single-run probe times; final repeated speed decision uses conservative 15-model batched stage'})
