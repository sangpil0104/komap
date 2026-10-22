"""Separate implementation-only follow-up, never concurrent with the preceding run."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                  MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import fcntl, importlib.util, json, subprocess, time
from pathlib import Path
PREVIOUS = Path('/workspace/model349_compress/base12_followup_20261008')
ROOT = Path('/workspace/model349_compress/base12_inputcache_20261008')
spec = importlib.util.spec_from_file_location('utilities', PREVIOUS / 'run_followup.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)
u.ROOT = ROOT


def main():
    while not any((PREVIOUS / name).exists() for name in ['completion.json', 'failure.json']):
        time.sleep(5)
    locks = []
    for path in [Path('/workspace/model349_compress/audit_20261008/gpu7.lock'), PREVIOUS / 'gpu7.lock', ROOT / 'gpu7.lock']:
        stream = path.open('a')
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(stream)
    if subprocess.run(['pgrep', '-f', '[i]nfer_lite.py|[i]nfer_audited.py|[i]nfer_batched.py|[i]nfer_cpu2.py'], stdout=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('Existing inference; refusing overlap')
    u.idle()
    (ROOT / 'runs').mkdir(exist_ok=False)
    frozen = u.snapshot()
    u.write(ROOT / 'frozen_before.json', frozen)
    control = u.build_bundle('control12')
    cache = u.build_bundle('cache12')
    p = cache / 'src/b4_zcdf.py'
    code = p.read_text()
    code = code.replace('    acc = 0\n    for mem in members:', '    acc = 0\n    prepared_inputs = {}\n    for mem in members:')
    old = '''        g = resize_gray(gray, scale)
        h, w = g.shape
        x = torch.from_numpy(make_input(g, norm, input_mode, stats))[None].to(device)
        x = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")'''
    new = '''        import json
        key = (float(scale), norm, input_mode, json.dumps(stats, sort_keys=True))
        if key not in prepared_inputs:
            g = resize_gray(gray, scale)
            h, w = g.shape
            x = torch.from_numpy(make_input(g, norm, input_mode, stats))[None].to(device)
            x = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
            prepared_inputs[key] = (x, h, w)
        x, h, w = prepared_inputs[key]'''
    assert code.count(old) == 1
    p.write_text(code.replace(old, new))
    monitor_code = (PREVIOUS / 'monitor.py').read_text().replace(str(PREVIOUS), str(ROOT))
    (ROOT / 'monitor.py').write_text(monitor_code)
    subprocess.Popen([u.PYTHON, '-B', str(ROOT / 'monitor.py')], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='7')
    u.telemetry('probe_before')
    with (ROOT / 'probe.log').open('x') as f:
        subprocess.run([u.PYTHON, '-B', '-u', str(ROOT / 'probe.py')], env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    u.telemetry('probe_after')
    run = u.execute(cache, 'cache12_valid20')
    comparison = {'metrics': u.metrics(run), 'completion': u.read(run / 'completion.json'),
                  'vs_base12': u.pair(u.metrics(u.OLD), u.metrics(run)),
                  'vs_lite15': u.pair(u.metrics(u.OLD.parent / 'cpu2_lite15'), u.metrics(run)),
                  'checkpoints': u.read(cache / 'checkpoint_manifest.json')}
    u.write(ROOT / 'raw_comparison.json', {'cache12': comparison})
    passed = comparison['vs_base12']['quality_pass'] and comparison['vs_lite15']['quality_pass']
    timings = []
    if passed:
        for index, order in enumerate([['control12', 'cache12'], ['cache12', 'control12'], ['control12', 'cache12']], 1):
            for key in order:
                run = u.execute(control if key == 'control12' else cache, f'timing_{index}_{key}')
                timings.append({'run': run.name, 'candidate': key, 'completion': u.read(run / 'completion.json'),
                                'vs_base12': u.pair(u.metrics(u.OLD), u.metrics(run))})
                u.write(ROOT / 'timing_runs.json', timings)
        u.execute(cache, 'cache12_test10', test=True)
    after = u.snapshot()
    u.write(ROOT / 'frozen_after.json', after)
    assert frozen == after
    u.write(ROOT / 'completion.json', {'diagnostics_completed': True, 'base_and_weights_unchanged': True,
                                      'raw_valid_candidates': ['cache12'], 'time_candidate': 'cache12' if passed else None,
                                      'test_candidates': ['cache12'] if passed else [],
                                      'independent_generalization_verified': False, 'new_training_started': False})
    print('CACHE_SESSION_COMPLETE', flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        u.write(ROOT / 'failure.json', {'error': repr(exc), 'complete': False})
        raise
