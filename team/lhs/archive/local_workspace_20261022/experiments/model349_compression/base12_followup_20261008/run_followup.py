"""Bounded, sequential GPU7 follow-up; frozen base is never edited."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                  NO_ALBUMENTATIONS_UPDATE='1', HF_HUB_OFFLINE='1')
import fcntl, hashlib, importlib.util, json, pickle, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import yaml

ROOT = Path('/workspace/model349_compress/base12_followup_20261008')
BASE = Path('/workspace/model349_compress/audit_20261008/bundles/cpu2_fp16_12')
OLD = Path('/workspace/model349_compress/audit_20261008/runs/cpu2_fp16_12')
PYTHON = '/workspace/venv-komap/bin/python'
INPUT = Path('/workspace/komap_model1/data/valid/images')
GT = Path('/workspace/komap_model1/data/valid/masks')
TEST = Path('/workspace/komap_model1/data/test/images')
GPU = '7'


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot():
    manifest = read(BASE / 'checkpoint_manifest.json')
    return {str(p.relative_to(BASE)): sha(p) for p in
            [BASE / row['path'] for row in manifest['checkpoints']] +
            [BASE / 'infer_lite.py'] + sorted((BASE / 'configs').glob('*'))}


def telemetry(label):
    data = {'label': label, 'time': time.time(), 'load_average': os.getloadavg(),
            'gpus': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                            '--format=csv,noheader,nounits'], text=True),
            'compute_processes': subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                                         '--format=csv,noheader,nounits'], text=True)}
    with (ROOT / 'telemetry.jsonl').open('a') as stream:
        stream.write(json.dumps(data) + '\n')


def idle():
    for _ in range(2):
        raw = subprocess.check_output(['nvidia-smi', '-i', GPU,
                                      '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        mem, util = map(int, raw.strip().split(','))
        if mem > 64 or util:
            raise RuntimeError(f'GPU7 occupied; no worker started: {raw.strip()}')
        time.sleep(2)


def metrics(run):
    result = read(run / 'metrics.json')
    assert result['count'] == 20
    return result


def pair(reference, candidate):
    a = {x['image']: x for x in reference['images']}
    b = {x['image']: x for x in candidate['images']}
    assert set(a) == set(b) and len(a) == 20
    image_delta = {n: 100 * (b[n]['image_miou'] - a[n]['image_miou']) for n in a}
    class_delta = {n: (100 * (np.asarray(b[n]['class_iou']) - a[n]['class_iou'])).tolist() for n in a}
    mean = 100 * (candidate['image_miou'] - reference['image_miou'])
    worst = min(image_delta, key=image_delta.get)
    return {'delta_pp': mean, 'worst_image': worst, 'worst_image_delta_pp': image_delta[worst],
            'per_image_delta_pp': image_delta, 'per_image_class_delta_pp': class_delta,
            'class_delta_pp': (100 * (np.asarray(candidate['class_image_mean_iou']) -
                                    reference['class_image_mean_iou'])).tolist(),
            'quality_pass': mean >= -0.01 - 1e-10 and image_delta[worst] >= -0.10 - 1e-10}


def build_bundle(key, tent=5, gamma=0.5, prune_seed=False):
    dst = ROOT / 'bundles' / key
    dst.mkdir(parents=True, exist_ok=False)
    shutil.copytree(BASE / 'src', dst / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(BASE / 'configs', dst / 'configs')
    code = (BASE / 'infer_lite.py').read_text()
    before = "    return result\n\n\ndef b4_fold_ids"
    after = "    for alias, target in config['inference'].get('high_source_aliases', {}).items():\n        if target not in result or alias == target:\n            raise ValueError('Invalid high-source alias')\n        result.pop(alias)\n    return result\n\n\ndef b4_fold_ids"
    assert code.count(before) == 1
    code = code.replace(before, after)
    before = "    # The two narrow brightness bands previously routed to Model123 EMA now"
    after = "    for alias, target in config['inference'].get('high_source_aliases', {}).items():\n        setattr(args, 'source_' + alias, getattr(args, 'source_' + target))\n" + before
    assert code.count(before) == 1
    code = code.replace(before, after)
    (dst / 'infer_lite.py').write_text(code)
    config = yaml.safe_load((BASE / 'configs/model349_lite.yaml').read_text())
    config['validation_image_miou'] = None
    config['inference']['tent']['steps'] = tent
    config['route']['gamma329'][0] = gamma
    if prune_seed:
        config['inference']['high_source_aliases'] = {'seed456': 'pairwise'}
    (dst / 'configs/model349_lite.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = read(BASE / 'checkpoint_manifest.json')
    selected = [x for x in manifest['checkpoints'] if not (prune_seed and 'restart59_aspp768_seed456' in x['path'])]
    for row in selected:
        p = dst / row['path']
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(BASE / row['path'])
        args = (BASE / row['path']).parent / 'args.json'
        if args.is_file():
            shutil.copy2(args, p.parent / 'args.json')
    write(dst / 'checkpoint_manifest.json', {'count': len(selected), 'bytes': sum(x['bytes'] for x in selected),
                                            'checkpoints': selected, 'source': str(BASE)})
    return dst


def execute(bundle, name, test=False):
    idle()
    telemetry(name + '_before')
    output = ROOT / 'runs' / name
    argv = [PYTHON, '-B', '-u', str(bundle / 'infer_lite.py'), '--input', str(TEST if test else INPUT),
            '--output', str(output), '--gpus', GPU, '--keep-probabilities']
    if not test:
        argv.extend(['--gt', str(GT)])
    print('START', name, flush=True)
    with (ROOT / (name + '.log')).open('x') as log:
        subprocess.run(argv, cwd=bundle, stdout=log, stderr=subprocess.STDOUT, check=True)
    telemetry(name + '_after')
    result = read(output / 'completion.json')
    assert result['images'] == (10 if test else 20)
    print('COMPLETE', name, result, flush=True)
    return output


def offline_probe():
    spec = importlib.util.spec_from_file_location('frozen_inference', BASE / 'infer_lite.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import cv2
    cv2.setNumThreads(2)
    import torch
    torch.set_num_threads(2)
    manifest = read(OLD / 'input_manifest.json')
    config = manifest['config']
    values = {g: [] for g in [0.5, 0.6, 0.7]}
    for row in manifest['images']:
        n = row['name']
        with (OLD / 'probabilities/base' / (n + '.pkl')).open('rb') as stream:
            item = pickle.load(stream)
        sources = {c: np.load(OLD / 'probabilities' / c / (n + '.npy')).astype(np.float32)
                   for c in ['326', '329', '336', '269']}
        paths = [GT / (n + '_mask.png'), GT / (n + '.png')]
        gt = module.decode_gt(next(p for p in paths if p.is_file()), config['palette'])
        for gamma in values:
            config['route']['gamma329'][0] = gamma
            pred = module.predict_fixed(item, sources, config)
            ious = []
            for c in range(4):
                union = np.count_nonzero((pred == c) | (gt == c))
                ious.append(float(np.count_nonzero((pred == c) & (gt == c)) / union) if union else 0.0)
            values[gamma].append({'image': n, 'class_iou': ious, 'image_miou': float(np.mean(ious))})
    reference = metrics(OLD)
    results = []
    for gamma, rows in values.items():
        val = {'image_miou': float(np.mean([x['image_miou'] for x in rows])),
               'class_image_mean_iou': np.mean([x['class_iou'] for x in rows], axis=0).tolist(),
               'count': 20, 'images': rows}
        compare = pair(reference, val)
        if gamma == 0.5:
            assert abs(compare['delta_pp']) < 1e-10 and abs(compare['worst_image_delta_pp']) < 1e-10
        results.append({'gamma': gamma, 'metrics': val, 'vs_base12': compare})
    write(ROOT / 'offline_gamma_probe.json', {'role': 'same-valid cached-probability exploratory selection',
                                            'independent_generalization_verified': False, 'candidates': results})
    eligible = [x for x in results if x['gamma'] != 0.5 and x['vs_base12']['quality_pass'] and x['vs_base12']['delta_pp'] > 0]
    selected = sorted(eligible, key=lambda x: (-x['metrics']['image_miou'], x['gamma']))[:1]
    write(ROOT / 'gamma_selection.json', {'selected': selected[0]['gamma'] if selected else None,
                                         'selection_role': 'valid-exposed exploratory choice; no test use'})
    print('OFFLINE', [(x['gamma'], x['metrics']['image_miou'], x['vs_base12']['worst_image_delta_pp']) for x in results], flush=True)
    return selected[0]['gamma'] if selected else None


def main():
    ROOT.mkdir(exist_ok=True)
    (ROOT / 'runs').mkdir(exist_ok=False)
    locks = []
    for path in [Path('/workspace/model349_compress/audit_20261008/gpu7.lock'), ROOT / 'gpu7.lock']:
        stream = path.open('a')
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(stream)
    if subprocess.run(['pgrep', '-f', '[i]nfer_lite.py|[i]nfer_audited.py|[i]nfer_batched.py|[i]nfer_cpu2.py'], stdout=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('Existing inference; refusing duplicate run')
    idle()
    telemetry('session_before')
    frozen = snapshot()
    write(ROOT / 'frozen_before.json', frozen)
    gamma = offline_probe()
    bundles = {'control12': build_bundle('control12'),
               'prune11_pairwise': build_bundle('prune11_pairwise', prune_seed=True),
               'tent4_12': build_bundle('tent4_12', tent=4)}
    if gamma is not None:
        bundles['gamma12'] = build_bundle('gamma12', gamma=gamma)
    comparisons = {}
    for key in [x for x in bundles if x != 'control12']:
        run = execute(bundles[key], key + '_valid20')
        comparisons[key] = {'metrics': metrics(run), 'completion': read(run / 'completion.json'),
                            'vs_base12': pair(metrics(OLD), metrics(run)),
                            'vs_lite15': pair(metrics(OLD.parent / 'cpu2_lite15'), metrics(run)),
                            'vs_original349': pair(metrics(OLD.parent / 'historical349'), metrics(run)),
                            'checkpoints': read(bundles[key] / 'checkpoint_manifest.json')}
        write(ROOT / 'raw_comparison.json', comparisons)
    passed = lambda key: comparisons[key]['vs_base12']['quality_pass'] and comparisons[key]['vs_lite15']['quality_pass']
    timed = next((key for key in ['prune11_pairwise', 'tent4_12'] if passed(key)), None)
    write(ROOT / 'timing_selection.json', {'candidate': timed, 'reason': 'preregistered priority; quality pass against base12 and lite15'})
    timings = []
    if timed:
        for index, order in enumerate([['control12', timed], [timed, 'control12'], ['control12', timed]], 1):
            for key in order:
                name = f'timing_{index}_{key}'
                run = execute(bundles[key], name)
                timings.append({'run': name, 'candidate': key, 'completion': read(run / 'completion.json'),
                                'vs_base12': pair(metrics(OLD), metrics(run))})
                write(ROOT / 'timing_runs.json', timings)
    selected = ([timed] if timed else [])
    if 'gamma12' in comparisons and passed('gamma12'):
        selected.append('gamma12')
    for key in selected:
        execute(bundles[key], key + '_test10', test=True)
    after = snapshot()
    write(ROOT / 'frozen_after.json', after)
    assert frozen == after
    telemetry('session_after')
    write(ROOT / 'completion.json', {'diagnostics_completed': True, 'base_and_weights_unchanged': True,
                                     'raw_valid_candidates': list(comparisons), 'time_candidate': timed,
                                     'test_candidates': selected, 'independent_generalization_verified': False,
                                     'new_training_started': False})
    print('SESSION_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(ROOT / 'failure.json', {'error': repr(exc), 'complete': False})
        raise
