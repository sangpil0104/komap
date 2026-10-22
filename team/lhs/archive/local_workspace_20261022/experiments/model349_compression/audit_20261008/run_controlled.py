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
    src=ROOT/'bundles'/source; dst=ROOT/'bundles'/key; dst.mkdir(exist_ok=False)
    shutil.copy2(ROOT/'infer_cpu2.py',dst/'infer_lite.py')
    shutil.copytree(src/'src',dst/'src'); shutil.copytree(src/'configs',dst/'configs')
    (dst/'weights').symlink_to((src/'weights').resolve(),target_is_directory=True)
    shutil.copy2(src/'checkpoint_manifest.json',dst/'checkpoint_manifest.json')
    if steps is not None:
        cfg=yaml.safe_load((dst/'configs/model349_lite.yaml').read_text()); cfg['inference']['tent']['steps']=steps
        (dst/'configs/model349_lite.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    return dst
def run(key,bundle=None,test=False):
    idle(); bundle=ROOT/'bundles'/(bundle or key); out=ROOT/'runs'/key
    args=[PYTHON,'-B','-u',str(bundle/'infer_lite.py'),'--input',str(INPUT if not test else INPUT.parent.parent/'test/images'),'--output',str(out),'--gpus','7','--keep-probabilities']
    if not test: args+=['--gt',str(GT)]
    start=time.time(); telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True)
    with (ROOT/f'{key}.log').open('w') as f: subprocess.run(args,cwd=bundle,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(out/'run_audit.json',{'started_unix':start,'finished_unix':time.time(),'command':args,'bundle':str(bundle),'gpu_snapshot_start':telemetry})
    print('FINISHED',key,load(out/'completion.json'),flush=True)
    return out
# Wait only for the current, positively identified baseline. No arbitrary historical PID.
while not (ROOT/'runs/lite15/completion.json').is_file():
    pid=2086945; proc=Path(f'/proc/{pid}/cmdline')
    if not proc.exists() or 'audit_20261008/bundles/lite15/infer_lite.py' not in proc.read_bytes().replace(b'\0',b' ').decode():
        raise RuntimeError('Baseline ended without completion; inspect its log')
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
if not eligible:
    write(ROOT/'session_complete.json',{'status':'no_quality_eligible_candidate'}); raise SystemExit(0)
selected=min(eligible,key=lambda k:load(ROOT/'runs'/k/'completion.json')['elapsed_seconds'])
write(ROOT/'selection.json',{'selected':selected,'reason':'Fastest initial full run among prespecified quality/storage passing candidates; timing repeats remain necessary'})
for key,bundle in [(selected+'_repeat2',selected),('lite15_repeat2','lite15'),('lite15_repeat3','lite15'),(selected+'_repeat3',selected)]: run(key,bundle)
import numpy as np
ref_times=[load(ROOT/'runs'/key/'completion.json')['elapsed_seconds'] for key in ['lite15','lite15_repeat2','lite15_repeat3']]
can_times=[load(ROOT/'runs'/key/'completion.json')['elapsed_seconds'] for key in [selected,selected+'_repeat2',selected+'_repeat3']]
quality_repeats={key:quality(key) for key in [selected,selected+'_repeat2',selected+'_repeat3']}
summary={'selected':selected,'reference_times':ref_times,'candidate_times':can_times,'reference_median':float(np.median(ref_times)),'candidate_median':float(np.median(can_times)),'median_time_reduction_percent':100*(1-float(np.median(can_times))/float(np.median(ref_times))),'quality_repeats':quality_repeats,'storage_ratio':load(ROOT/'bundles'/selected/'checkpoint_manifest.json')['bytes']/reference_bytes}
summary['exploratory_adoption_pass']=all(r['quality_pass'] for r in quality_repeats.values()) and summary['median_time_reduction_percent']>=10 and summary['storage_ratio']<1
write(ROOT/'timing_summary.json',summary)
# test has no labels: only preserve masks and compare with the historical lite15.
if all(r['quality_pass'] for r in quality_repeats.values()):
    path=run(selected+'_test10',selected,test=True); ref_test=ROOT.parent/'runs/lite_rep_free_bundle_v2_test10'
    names=load(path/'input_manifest.json')['images']; old=load(ref_test/'input_manifest.json')['images']
    assert {r['name']:r['sha256'] for r in names}=={r['name']:r['sha256'] for r in old} and len(names)==10
    changed=total=0; rows=[]
    from PIL import Image
    for row in names:
        name=row['name']; a=np.asarray(Image.open(ref_test/'predicted_labels'/f'{name}.png')); b=np.asarray(Image.open(path/'predicted_labels'/f'{name}.png'))
        assert a.shape==b.shape; n=int(np.count_nonzero(a!=b)); changed+=n; total+=a.size; rows.append({'image':name,'changed_pixels':n,'total_pixels':a.size})
    write(ROOT/'test_parity.json',{'reference':str(ref_test),'candidate':str(path),'images':10,'changed_pixels':changed,'total_pixels':total,'agreement':1-changed/total,'images_detail':rows,'ground_truth_available':False,'score_claimed':False})
subprocess.run([PYTHON,'-B',str(ROOT/'analyze_runs.py'),'--root',str(ROOT),'--gt',str(GT)],check=True)
write(ROOT/'session_complete.json',{'status':'completed','selected':selected,'exploratory_adoption_pass':summary['exploratory_adoption_pass']})
