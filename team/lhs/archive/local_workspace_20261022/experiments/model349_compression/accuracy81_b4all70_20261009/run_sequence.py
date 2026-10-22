"""Memory-gated B4 all70 continuation, fixed three folds, then fresh valid20."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import fcntl, hashlib, json, signal, subprocess, time, traceback, shutil
import yaml
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
SOURCE = PARENT / 'Model349_exh14'
PY = '/workspace/venv-komap/bin/python'
GPU = '3'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def idle():
    while True:
        clear = True
        for _ in range(3):
            mem, util = map(int, subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(','))
            apps = subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            if mem > 64 or util or apps:
                clear = False
                break
            time.sleep(2)
        if clear:
            return
        write(ROOT / 'queue_status.json', {'state': 'waiting_gpu3_idle', 'memory_mib': mem, 'util': util, 'compute_pids': apps, 'time': time.time()})
        time.sleep(30)

def run(args, stage, cwd=ROOT):
    idle()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, NO_ALBUMENTATIONS_UPDATE='1')
    with (ROOT / (stage + '.log')).open('x') as log:
        p = subprocess.Popen([PY, '-B', '-u', *map(str, args)], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(ROOT / 'active_child.json', {'pid': p.pid, 'stage': stage, 'physical_gpu': GPU, 'time': time.time()})
        while p.poll() is None:
            time.sleep(5)
            apps = subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
            if len(apps) > 1 and p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=20)
                raise RuntimeError('Foreign GPU3 compute appeared; stopped our created process group only')
        assert p.returncode == 0, (stage, p.returncode)

def main():
    assert not (ROOT / 'completion.json').exists()
    assert subprocess.run(['ps','-p','2193241'],capture_output=True).returncode != 0
    assert shutil.disk_usage(ROOT).free > 10 * 1024**3
    locks=[]
    for p in sorted(set(PARENT.glob('*/gpu3.lock')) | {ROOT/'gpu3.lock'}):
        f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    lock=json.loads((ROOT/'preflight_source_lock.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in lock.items())
    frozen=json.loads((PARENT/'accuracy81_20261008/frozen_before.json').read_text())
    assert all(sha(SOURCE/p)==h for p,h in frozen.items())
    write(ROOT/'parent_frozen.json',frozen)
    with (ROOT/'data_audit.log').open('x') as log:
        subprocess.run([PY,'-B',str(ROOT/'audit_data.py')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
    assert json.loads((ROOT/'data_audit.json').read_text())['passed']
    run([ROOT/'train.py','--fold','1','--smoke'],'smoke_memory')
    smoke=json.loads((ROOT/'smoke_memory.json').read_text());assert smoke['passed'] and smoke['batch']==2 and smoke['crop']==1536
    finals={};receipts={}
    for fold in [1,3,4]:
        run([ROOT/'train.py','--fold',str(fold)],f'fold{fold}_train')
        out=ROOT/'training'/f'fold{fold}';receipt=json.loads((out/'completion.json').read_text())
        assert receipt['finished'] and receipt['updates']==400 and receipt['train_images_seen']==70
        assert receipt['parent_unchanged'] and receipt['bn_buffers_unchanged'] and receipt['all_parameters_finite']
        assert not receipt['valid_read'] and not receipt['test_read']
        assert sha(out/'last.pt')==receipt['checkpoint_sha256']
        finals[f'weights/model329_fullstats_width4000_uniform_group_oof/fold{fold}/last.pt']=out/'last.pt';receipts[str(fold)]=receipt
        assert all(sha(SOURCE/p)==h for p,h in frozen.items())
    assert len({v['sample_sequence_sha256'] for v in receipts.values()})==1
    manifest=json.loads((PARENT/'exh14_base_search_20261008/bundles/control14/checkpoint_manifest.json').read_text())
    bundle=ROOT/'bundle';shutil.copytree(SOURCE,bundle,ignore=shutil.ignore_patterns('weights','__pycache__','.git','results','search'))
    rows=[]
    for row in manifest['checkpoints']:
        row=dict(row);dest=bundle/row['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        target=finals.get(row['path'],SOURCE/row['path']);dest.symlink_to(target)
        if row['path'] in finals:row.update(sha256=sha(target),bytes=target.stat().st_size)
        args=(SOURCE/row['path']).parent/'args.json'
        if args.exists():shutil.copy2(args,dest.parent/'args.json')
        rows.append(row)
    assert len(rows)==14 and sum(x['path'] in finals for x in rows)==3
    write(bundle/'checkpoint_manifest.json',dict(manifest,checkpoints=rows))
    cfgpath=bundle/'configs/model349_lite.yaml';cfg=yaml.safe_load(cfgpath.read_text());cfg['validation_image_miou']=None;cfgpath.write_text(yaml.safe_dump(cfg,sort_keys=False))
    output=ROOT/'runs/b4all70_valid20';output.parent.mkdir(exist_ok=False)
    run([bundle/'infer_lite.py','--input','/workspace/komap_model1/data/valid/images','--output',output,'--gpus',GPU,'--keep-probabilities','--gt','/workspace/komap_model1/data/valid/masks'],'b4all70_valid',bundle)
    assert all(sha(SOURCE/p)==h for p,h in frozen.items())
    write(ROOT/'completion.json',{'finished':True,'parent_source_weights_unchanged':True,'whole_fresh_pipeline':True,'miou_percent':100*json.loads((output/'metrics.json').read_text())['image_miou'],'checkpoint_count':14,'checkpoint_bytes':sum(r['bytes'] for r in rows),'folds':[1,3,4],'training_receipts':receipts,'same_training_sample_sequence':True,'independent_generalization_verified':False,'test_used':False,'time':time.time()})

if __name__=='__main__':
    try:main()
    except BaseException:
        write(ROOT/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
