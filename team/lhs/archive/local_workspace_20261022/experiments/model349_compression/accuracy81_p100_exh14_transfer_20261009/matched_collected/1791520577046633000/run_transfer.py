"""Prepare a P100-weight copy of exh14 and run one fresh valid20 when GPU3 is idle."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import fcntl
import hashlib
import json
import shutil
import signal
import subprocess
import time
import traceback
import yaml

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
SOURCE = PARENT / 'Model349_exh14'
P100 = Path('/workspace/komap_model1/outputs')
PY = '/workspace/venv-komap/bin/python'
GPU = '3'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def gpu_state():
    mem, util = map(int, subprocess.check_output(
        ['nvidia-smi', '-i', GPU, '--query-gpu=memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True).strip().split(','))
    apps = subprocess.check_output(
        ['nvidia-smi', '-i', GPU, '--query-compute-apps=pid',
         '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    return mem, util, apps


def wait_idle():
    deadline = time.monotonic() + 10800
    while time.monotonic() < deadline:
        clear = True
        for _ in range(3):
            mem, util, apps = gpu_state()
            if mem > 64 or util or apps:
                clear = False
                break
            time.sleep(2)
        if clear:
            return
        write(ROOT / 'queue_status.json', {'state': 'waiting_gpu3_idle', 'memory_mib': mem,
              'utilization_percent': util, 'compute_pids': apps, 'time': time.time()})
        time.sleep(30)
    raise TimeoutError('GPU3 did not become idle within the fixed 3-hour wait')


def main():
    assert not (ROOT / 'completion.json').exists()
    assert not (ROOT / 'failure.json').exists()
    lock = json.loads((ROOT / 'source_lock.json').read_text())
    assert all(sha(ROOT / name) == digest for name, digest in lock.items())
    assert shutil.disk_usage(ROOT).free > 10 * 1024**3
    held = []
    for path in sorted(set(PARENT.glob('*/gpu3.lock')) | {ROOT / 'gpu3.lock'}):
        stream = path.open('a')
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held.append(stream)
    source_manifest = json.loads((SOURCE / 'checkpoint_manifest.json').read_text())
    assert source_manifest['count'] == 14
    bundle = ROOT / 'bundle'
    assert not bundle.exists()
    shutil.copytree(SOURCE, bundle, ignore=shutil.ignore_patterns(
        'weights', 'results', 'search', '__pycache__'))
    rows = []
    for row in source_manifest['checkpoints']:
        original = SOURCE / row['path']
        assert original.stat().st_size == row['bytes'] and sha(original) == row['sha256']
        candidate = P100 / Path(*Path(row['path']).parts[1:])
        assert candidate.is_file(), candidate
        dest = bundle / row['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(candidate)
        args = original.parent / 'args.json'
        if args.exists():
            shutil.copy2(args, dest.parent / 'args.json')
        rows.append(dict(row, sha256=sha(candidate), bytes=candidate.stat().st_size,
                         source_path=str(candidate)))
    write(bundle / 'checkpoint_manifest.json', {'source': 'P100 outputs with fixed exh14 inference method',
          'count': 14, 'bytes': sum(x['bytes'] for x in rows), 'checkpoints': rows})
    write(bundle / 'recipe_manifest.json', {'kind': 'P100-weight exh14 transfer candidate',
          'source_release_bundle': str(SOURCE), 'source_release_manifest_sha256': sha(SOURCE / 'checkpoint_manifest.json'),
          'source_inference_sha256': sha(SOURCE / 'infer_lite.py'),
          'p100_weight_paths': {x['path']: x['source_path'] for x in rows},
          'independent_generalization_verified': False})
    cfg_path = bundle / 'configs/model349_lite.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg['inference']['b4_fold_ids'] == {'322': [3, 4], '326': [1, 2], '329': [1, 3, 4]}
    cfg['validation_image_miou'] = None
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    write(ROOT / 'prepared.json', {'source_inference_sha256': sha(SOURCE / 'infer_lite.py'),
          'transfer_inference_sha256': sha(bundle / 'infer_lite.py'),
          'source_config_route': yaml.safe_load((SOURCE / 'configs/model349_lite.yaml').read_text())['route'],
          'candidate_config_route': cfg['route'], 'source_manifest_sha256': sha(SOURCE / 'checkpoint_manifest.json'),
          'p100_checkpoint_count': len(rows), 'p100_checkpoint_bytes': sum(x['bytes'] for x in rows),
          'p100_checkpoint_sha256': {x['path']: x['sha256'] for x in rows}, 'time': time.time()})
    assert sha(bundle / 'infer_lite.py') == sha(SOURCE / 'infer_lite.py')
    wait_idle()
    output = ROOT / 'runs/p100_exh14_valid20'
    output.parent.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, NO_ALBUMENTATIONS_UPDATE='1')
    with (ROOT / 'p100_exh14_valid.log').open('x') as log:
        child = subprocess.Popen([PY, '-B', '-u', str(bundle / 'infer_lite.py'),
            '--input', '/workspace/komap_model1/data/valid/images', '--output', str(output),
            '--gpus', GPU, '--keep-probabilities', '--gt', '/workspace/komap_model1/data/valid/masks'],
            cwd=bundle, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(ROOT / 'active_child.json', {'pid': child.pid, 'stage': 'p100_exh14_valid', 'time': time.time()})
        while child.poll() is None:
            time.sleep(5)
            if len(gpu_state()[2]) > 1 and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=20)
                raise RuntimeError('Foreign GPU3 compute appeared; stopped only our child')
        assert child.returncode == 0, child.returncode
    done = json.loads((output / 'completion.json').read_text())
    metrics = json.loads((output / 'metrics.json').read_text())
    assert done['images'] == 20 and metrics['count'] == 20
    write(ROOT / 'completion.json', {'finished': True, 'fresh_valid20': True,
          'miou_percent_reported': 100 * metrics['image_miou'],
          'elapsed_seconds': done['elapsed_seconds'], 'weights': 14,
          'p100_checkpoint_bytes': sum(x['bytes'] for x in rows),
          'original_release_unchanged': True, 'p100_outputs_unchanged': True,
          'independent_png_scoring_pending': True, 'test_used': False, 'time': time.time()})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write(ROOT / 'failure.json', {'traceback': traceback.format_exc(), 'time': time.time()})
        raise
