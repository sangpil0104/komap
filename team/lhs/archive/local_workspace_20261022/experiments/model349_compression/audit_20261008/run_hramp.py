"""Conservative 15-model FP16 + D4 batching, same-GPU alternating raw-image repeats."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NO_ALBUMENTATIONS_UPDATE='1')
import fcntl,hashlib,importlib.util,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import yaml
ROOT=Path('/workspace/model349_compress/audit_20261008'); PYTHON='/workspace/venv-komap/bin/python'; BASE=ROOT.parent
INPUT=Path('/workspace/komap_model1/data/valid/images'); GT=Path('/workspace/komap_model1/data/valid/masks')
def load(p): return json.loads(p.read_text())
def write(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
while not (ROOT/'batched_complete.json').exists():
    if subprocess.run(['pgrep','-f','[r]un_batched.py'],stdout=subprocess.DEVNULL).returncode!=0: raise RuntimeError('Controlled queue ended without completion')
    time.sleep(10)
previous=load(ROOT/'batched_summary.json')
if previous.get('exploratory_adoption_pass') or not previous.get('reference_times'):
    write(ROOT/'hramp_complete.json',{'status':'skipped','reason':'D4 stage passed adoption criteria or failed quality; conditional HRAMP not needed/eligible'})
    raise SystemExit(0)
lock=(ROOT/'gpu7.lock').open('w')
# The previous coordinator has written its completion but may still be exiting.
for attempt in range(10):
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); break
    except BlockingIOError: time.sleep(2)
else: raise RuntimeError('Previous experiment lock remains held')
def idle():
    assert subprocess.run(['pgrep','-f','[i]nfer_lite.py|[i]nfer_audited.py'],stdout=subprocess.DEVNULL).returncode!=0
    for _ in range(2):
        raw=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        mem,util=map(int,raw.strip().split(',')); assert mem<=64 and util==0,raw
        time.sleep(2)
monitor_log=(ROOT/'hramp_monitor.log').open('w')
subprocess.Popen([PYTHON,'-B',str(ROOT/'monitor_hramp.py')],stdout=monitor_log,stderr=subprocess.STDOUT,start_new_session=True)
monitor_log.close()
# Same verified FP16 weights; only the new HR forward autocast is enabled.
source=ROOT/'bundles/fp16_15_batched'; dst=ROOT/'bundles/fp16_15_batched_hramp'; dst.mkdir(exist_ok=False)
for file in source.iterdir():
    if file.is_file() and file.name!='infer_lite.py': shutil.copy2(file,dst/file.name)
if (source/'provenance').is_dir(): shutil.copytree(source/'provenance',dst/'provenance')
shutil.copytree(source/'src',dst/'src'); shutil.copytree(source/'configs',dst/'configs')
(dst/'weights').symlink_to(source/'weights',target_is_directory=True)
shutil.copy2(ROOT/'infer_batched_hramp.py',dst/'infer_lite.py')
cfg=yaml.safe_load((dst/'configs/model349_lite.yaml').read_text()); cfg['inference']['hrnet']['autocast']='float16'; cfg['inference']['hrnet']['batch_size']=8
(dst/'configs/model349_lite.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
cps=sorted((dst/'weights').rglob('*.pt')); assert len(cps)==15
write(dst/'checkpoint_manifest.json',{'count':15,'bytes':sum(p.stat().st_size for p in cps),'checkpoints':[{'path':str(p.relative_to(dst)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in cps]})
spec=importlib.util.spec_from_file_location('analysis',ROOT/'analyze_runs.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
refpath=ROOT/'runs/lite15'; reference=mod.score(refpath,GT)
def run(key,bundle,test=False):
    idle(); out=ROOT/'runs'/key; folder=ROOT/'bundles'/bundle
    args=[PYTHON,'-B','-u',str(folder/'infer_lite.py'),'--input',str(INPUT if not test else INPUT.parent.parent/'test/images'),'--output',str(out),'--gpus','7','--keep-probabilities']
    if not test: args+=['--gt',str(GT)]
    started=time.time()
    with (ROOT/f'{key}.log').open('w') as f: subprocess.run(args,cwd=folder,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(out/'run_audit.json',{'started_unix':started,'finished_unix':time.time(),'command':args,'bundle':str(folder)})
    print('FINISHED',key,load(out/'completion.json'),flush=True)
    return out
refs=[]; candidates=[]; quality={}; phase_quality={}
for i in range(1,4):
    a=f'hramp_control{i}'; run(a,'lite15'); refs.append(a)
    b='fp16_15_batched_hramp' if i==1 else f'fp16_15_batched_hramp_repeat{i}'
    path=run(b,'fp16_15_batched_hramp'); candidates.append(b)
    quality[b]=mod.pair(refpath,path,reference,mod.score(path,GT))
    phase_quality[b]=mod.pair(ROOT/'runs'/a,path,mod.score(ROOT/'runs'/a,GT),mod.score(path,GT))
    if not quality[b]['quality_pass'] or not phase_quality[b]['quality_pass']:
        write(ROOT/'hramp_summary.json',{'status':'quality_rejected','quality':quality,'candidates':candidates,'references':refs})
        write(ROOT/'hramp_complete.json',{'status':'quality_rejected'}); raise SystemExit(0)
a_times=[load(ROOT/'runs'/k/'completion.json')['elapsed_seconds'] for k in refs]
b_times=[load(ROOT/'runs'/k/'completion.json')['elapsed_seconds'] for k in candidates]
summary={'references':refs,'candidates':candidates,'reference_times':a_times,'candidate_times':b_times,'reference_median':float(np.median(a_times)),'candidate_median':float(np.median(b_times)),'median_time_reduction_percent':100*(1-float(np.median(b_times))/float(np.median(a_times))),'quality':quality,'paired_phase_quality':phase_quality,'checkpoint_bytes':load(dst/'checkpoint_manifest.json')['bytes'],'reference_checkpoint_bytes':load(ROOT/'bundles/lite15/checkpoint_manifest.json')['bytes']}
summary['exploratory_adoption_pass']=summary['median_time_reduction_percent']>=10 and summary['checkpoint_bytes']<summary['reference_checkpoint_bytes']
write(ROOT/'hramp_summary.json',summary)
path=run('fp16_15_batched_hramp_test10','fp16_15_batched_hramp',test=True); old=BASE/'runs/lite_rep_free_bundle_v2_test10'
names=load(path/'input_manifest.json')['images']; old_names=load(old/'input_manifest.json')['images']
assert len(names)==10 and {r['name']:r['sha256'] for r in names}=={r['name']:r['sha256'] for r in old_names}
from PIL import Image
changed=total=0; rows=[]
for r in names:
    name=r['name']; a=np.asarray(Image.open(old/'predicted_labels'/f'{name}.png')); b=np.asarray(Image.open(path/'predicted_labels'/f'{name}.png'))
    assert a.shape==b.shape; n=int(np.count_nonzero(a!=b)); changed+=n; total+=a.size; rows.append({'image':name,'changed_pixels':n,'total_pixels':a.size})
write(ROOT/'hramp_test_parity.json',{'images':10,'changed_pixels':changed,'total_pixels':total,'agreement':1-changed/total,'images_detail':rows,'ground_truth_available':False,'score_claimed':False})
subprocess.run([PYTHON,'-B',str(ROOT/'analyze_runs.py'),'--root',str(ROOT),'--gt',str(GT)],check=True)
write(ROOT/'hramp_complete.json',{'status':'completed','exploratory_adoption_pass':summary['exploratory_adoption_pass']})
