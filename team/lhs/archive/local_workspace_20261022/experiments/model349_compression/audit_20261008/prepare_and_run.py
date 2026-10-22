"""Fixed, sequential GPU7 raw-image audit. Original bundles are read-only."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NO_ALBUMENTATIONS_UPDATE='1')
import fcntl, hashlib, importlib.util, json, shutil, subprocess, sys, time
from pathlib import Path
import yaml
BASE=Path('/workspace/model349_compress')
ROOT=BASE/'audit_20261008'
PYTHON='/workspace/venv-komap/bin/python'
GT=Path('/workspace/komap_model1/data/valid/masks')
INPUT=Path('/workspace/komap_model1/data/valid/images')

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def idle():
    for _ in range(2):
        raw=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        mem,util=map(int,raw.strip().split(','))
        if mem>64 or util: raise RuntimeError(f'GPU7 occupied: {raw.strip()}')
        time.sleep(2)

lock=(ROOT/'gpu7.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if subprocess.run(['pgrep','-f','[i]nfer_lite.py|[i]nfer_audited.py'], stdout=subprocess.DEVNULL).returncode == 0:
    raise RuntimeError('Existing inference in this container; refusing overlap')

bundles={}
for key,source in [('lite15', 'model349_lite_rep_free_15weights_v2'), ('fp16_t5','lite14_fp16_storage'), ('fp16_t3','lite14_fp16_storage'), ('fp16_12','lite12_fp16_fold326_13'), ('fp16_10','lite14_fp16_storage')]:
    src=BASE/source; dst=ROOT/'bundles'/key
    dst.mkdir(parents=True,exist_ok=False)
    shutil.copy2(ROOT/'infer_audited.py', dst/'infer_lite.py')
    shutil.copytree(src/'src', dst/'src')
    shutil.copytree(src/'configs', dst/'configs')
    config=yaml.safe_load((src/'configs/model349_lite.yaml').read_text())
    config['validation_image_miou']=None
    config['inference']['ema_pruned']=key!='lite15'
    if key=='fp16_t3': config['inference']['tent']['steps']=3
    if key=='fp16_10':
        config['route']['base']['primary_gamma']=0.0
        config['route']['base']['al3ni_gamma']=0.0
        config['route']['gamma329'][0]=0.8
        config['inference']['active_components']=['329','base','336','269']
        shutil.copytree(src/'weights', dst/'weights', ignore=lambda folder,names: {'model326_hybrid_fullstats_group_oof'} if Path(folder)==src/'weights' else set())
    else: (dst/'weights').symlink_to(src/'weights',target_is_directory=True)
    (dst/'configs/model349_lite.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    cps=sorted((dst/'weights').rglob('*.pt'))
    write(dst/'checkpoint_manifest.json',{'source':str(src),'count':len(cps),'bytes':sum(p.stat().st_size for p in cps),'checkpoints':[{'path':str(p.relative_to(dst)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in cps]})
    bundles[key]=dst
# Recover only final fusion; do not assign an invented full-run time.
old=BASE/'runs/lite14_fp16_valid20'; recovered=ROOT/'runs/fp16_t5_recovered'
recovered.mkdir(parents=True)
shutil.copy2(old/'input_manifest.json',recovered/'input_manifest.json')
(recovered/'probabilities').symlink_to(old/'probabilities',target_is_directory=True)
spec=importlib.util.spec_from_file_location('recovery',bundles['fp16_t5']/'infer_lite.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
for c in module.COMPONENTS:
    value=json.loads((old/'probabilities'/c/'complete.json').read_text())
    assert value['images']==20
    extension='pkl' if c=='base' else 'npy'
    assert len(list((old/'probabilities'/c).glob('*.'+extension)))==20
start=time.monotonic(); scores=module.assemble(recovered,GT)
write(recovered/'completion.json',{'images':20,'elapsed_seconds':None,'fusion_recovery_seconds':time.monotonic()-start,'recovered_from':str(old),'timing_valid':False,'mean_image_miou':sum(r['image_miou'] for r in scores)/20})
write(ROOT/'recovery.json',{'source':str(old),'component_completion':[json.loads((old/'probabilities'/c/'complete.json').read_text()) for c in module.COMPONENTS],'full_time_unknown':True})
print('RECOVERED',flush=True)
for key in bundles:
    idle()
    out=ROOT/'runs'/key
    with (ROOT/f'{key}.log').open('w') as f:
        subprocess.run([PYTHON,'-B','-u',str(bundles[key]/'infer_lite.py'),'--input',str(INPUT),'--gt',str(GT),'--output',str(out),'--gpus','7','--keep-probabilities'],cwd=bundles[key],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('FINISHED',key,json.loads((out/'completion.json').read_text()),flush=True)
write(ROOT/'initial_complete.json',{'variants':list(bundles)})
