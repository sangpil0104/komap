"""Memory-gated, train-only HR336 context experiment, then fresh valid20."""
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
    assert shutil.disk_usage(ROOT).free > 10 * 1024**3
    locks = []
    for p in sorted(set(PARENT.glob('*/gpu3.lock')) | {ROOT / 'gpu3.lock'}):
        f = p.open('a')
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(f)
    lock = json.loads((ROOT / 'preflight_source_lock.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in lock.items())
    frozen = json.loads((PARENT / 'accuracy81_20261008/frozen_before.json').read_text())
    assert all(sha(SOURCE / p) == h for p, h in frozen.items())
    write(ROOT / 'parent_frozen.json', frozen)
    run([ROOT / 'smoke_memory.py'], 'smoke_memory')
    smoke = json.loads((ROOT / 'smoke_memory.json').read_text())
    assert smoke['passed'] and smoke['batch'] == 2 and smoke['patch'] == 1024
    run([ROOT / 'train.py', '--arm', 'context'], 'context_train')
    receipt = json.loads((ROOT / 'training/context/completion.json').read_text())
    final = Path(receipt['final_checkpoint'])
    assert receipt['epochs'] == 4 and receipt['optimizer_updates'] == 420
    assert not receipt['valid_read'] and not receipt['test_read']
    assert sha(final) == receipt['final_sha256']
    manifest = json.loads((PARENT / 'exh14_base_search_20261008/bundles/control14/checkpoint_manifest.json').read_text())
    bundle = ROOT / 'bundle'
    shutil.copytree(SOURCE, bundle, ignore=shutil.ignore_patterns('weights', '__pycache__', '.git', 'results', 'search'))
    rows = []
    for row in manifest['checkpoints']:
        row = dict(row)
        dest = bundle / row['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        is_hr = row['path'] == 'weights/model336_hrnet_best/last.pt'
        target = final if is_hr else SOURCE / row['path']
        dest.symlink_to(target)
        if is_hr:
            row.update(sha256=sha(final), bytes=final.stat().st_size)
        args = (SOURCE / row['path']).parent / 'args.json'
        if args.exists():
            shutil.copy2(args, dest.parent / 'args.json')
        rows.append(row)
    write(bundle / 'checkpoint_manifest.json', dict(manifest, checkpoints=rows))
    config = bundle / 'configs/model349_lite.yaml'
    cfg = yaml.safe_load(config.read_text())
    cfg['validation_image_miou'] = None
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    output = ROOT / 'runs/context_valid20'
    output.parent.mkdir(exist_ok=False)
    run([bundle / 'infer_lite.py', '--input', '/workspace/komap_model1/data/valid/images', '--output', output, '--gpus', GPU, '--keep-probabilities', '--gt', '/workspace/komap_model1/data/valid/masks'], 'context_valid', bundle)
    assert all(sha(SOURCE / p) == h for p, h in frozen.items())
    write(ROOT / 'completion.json', {'finished': True, 'parent_source_weights_unchanged': True, 'whole_fresh_pipeline': True, 'miou_percent': 100 * json.loads((output / 'metrics.json').read_text())['image_miou'], 'checkpoint_count': len(rows), 'checkpoint_bytes': sum(r['bytes'] for r in rows), 'final_checkpoint_sha256': sha(final), 'independent_generalization_verified': False, 'test_used': False, 'time': time.time()})

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write(ROOT / 'failure.json', {'traceback': traceback.format_exc(), 'time': time.time()})
        raise
