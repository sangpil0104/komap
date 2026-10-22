"""One preregistered high-resolution TTA change, queued after input reuse."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import fcntl, importlib.util, subprocess, time
from pathlib import Path
PREVIOUS = Path('/workspace/model349_compress/base12_inputcache_20261008')
UTILITIES = Path('/workspace/model349_compress/base12_followup_20261008')
ROOT = Path('/workspace/model349_compress/base12_highflip4_20261008')
spec = importlib.util.spec_from_file_location('utilities', UTILITIES / 'run_followup.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)
u.ROOT = ROOT

def main():
    while not any((PREVIOUS / name).exists() for name in ['completion.json', 'failure.json']):
        time.sleep(5)
    locks = []
    for path in [Path('/workspace/model349_compress/audit_20261008/gpu7.lock'), UTILITIES / 'gpu7.lock', PREVIOUS / 'gpu7.lock', ROOT / 'gpu7.lock']:
        stream = path.open('a')
        for attempt in range(12):
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if attempt == 11: raise
                time.sleep(5)
        locks.append(stream)
    if subprocess.run(['pgrep', '-f', '[i]nfer_lite.py|[i]nfer_audited.py|[i]nfer_batched.py|[i]nfer_cpu2.py'], stdout=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('Existing inference; refusing overlap')
    u.idle()
    (ROOT / 'runs').mkdir(exist_ok=False)
    frozen = u.snapshot()
    u.write(ROOT / 'frozen_before.json', frozen)
    control = u.build_bundle('control12')
    candidate = u.build_bundle('highflip4_12')
    p = candidate / 'src/advanced_inference.py'
    code = p.read_text()
    before = '    if mode == "d4":'
    assert code.count(before) == 1
    p.write_text(code.replace(before, '    if mode == "flip4":\n        return [(0, False), (0, True), (2, False), (2, True)]\n' + before))
    p = candidate / 'infer_lite.py'
    code = p.read_text()
    before = "device, True, 1, 'd4', 'gaussian') for im in [raw, style]"
    assert code.count(before) == 1
    p.write_text(code.replace(before, "device, True, 1, 'flip4', 'gaussian') for im in [raw, style]"))
    source_hashes = {str(p.relative_to(candidate)): u.sha(p) for p in candidate.rglob('*') if p.is_file() and p.suffix in ['.py', '.yaml', '.json']}
    u.write(ROOT / 'candidate_source_hashes.json', source_hashes)
    monitor_code = (UTILITIES / 'monitor.py').read_text().replace(str(UTILITIES), str(ROOT))
    (ROOT / 'monitor.py').write_text(monitor_code)
    subprocess.Popen([u.PYTHON, '-B', str(ROOT / 'monitor.py')], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    run = u.execute(candidate, 'highflip4_12_valid20')
    comparison = {'metrics': u.metrics(run), 'completion': u.read(run / 'completion.json'), 'vs_base12': u.pair(u.metrics(u.OLD), u.metrics(run)), 'vs_lite15': u.pair(u.metrics(u.OLD.parent / 'cpu2_lite15'), u.metrics(run)), 'checkpoints': u.read(candidate / 'checkpoint_manifest.json')}
    u.write(ROOT / 'raw_comparison.json', {'highflip4_12': comparison})
    passed = comparison['vs_base12']['quality_pass'] and comparison['vs_lite15']['quality_pass']
    timings = []
    if passed:
        for index, order in enumerate([['control12', 'highflip4_12'], ['highflip4_12', 'control12'], ['control12', 'highflip4_12']], 1):
            for key in order:
                run = u.execute(control if key == 'control12' else candidate, f'timing_{index}_{key}')
                timings.append({'run': run.name, 'candidate': key, 'completion': u.read(run / 'completion.json'), 'vs_base12': u.pair(u.metrics(u.OLD), u.metrics(run)), 'vs_lite15': u.pair(u.metrics(u.OLD.parent / 'cpu2_lite15'), u.metrics(run))})
                u.write(ROOT / 'timing_runs.json', timings)
        u.execute(candidate, 'highflip4_12_test10', test=True)
    after = u.snapshot()
    u.write(ROOT / 'frozen_after.json', after)
    assert frozen == after
    u.write(ROOT / 'completion.json', {'diagnostics_completed': True, 'base_and_weights_unchanged': True, 'raw_valid_candidates': ['highflip4_12'], 'time_candidate': 'highflip4_12' if passed else None, 'test_candidates': ['highflip4_12'] if passed else [], 'independent_generalization_verified': False, 'new_training_started': False})
    print('HIGHFLIP_SESSION_COMPLETE', flush=True)

if __name__ == '__main__':
    try: main()
    except Exception as exc:
        u.write(ROOT / 'failure.json', {'error': repr(exc), 'complete': False})
        raise
