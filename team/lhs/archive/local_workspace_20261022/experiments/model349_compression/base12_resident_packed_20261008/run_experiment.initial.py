"""Fixed additional candidates, all sequential under shared GPU7 locks."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import fcntl, hashlib, importlib.util, json, shutil, subprocess, zipfile
from pathlib import Path
import torch
ROOT = Path('/workspace/model349_compress/base12_resident_packed_20261008')
UTIL = Path('/workspace/model349_compress/base12_followup_20261008')
spec = importlib.util.spec_from_file_location('utilities', UTIL / 'run_followup.py')
u = importlib.util.module_from_spec(spec); spec.loader.exec_module(u); u.ROOT = ROOT
torch.set_num_threads(2)

def update_manifest(bundle):
    manifest = u.read(bundle / 'checkpoint_manifest.json')
    for row in manifest['checkpoints']:
        p = bundle / row['path']; row['bytes'] = p.stat().st_size; row['sha256'] = u.sha(p)
    manifest['bytes'] = sum(r['bytes'] for r in manifest['checkpoints'])
    u.write(bundle / 'checkpoint_manifest.json', manifest)

def soup(bundle):
    src = u.BASE / 'weights'
    names = ['restart59_aspp768_seed456', 'restart61_pairwise_margin']
    values = [torch.load(src / n / 'best.pt', map_location='cpu', weights_only=False)['model'] for n in names]
    assert values[0].keys() == values[1].keys()
    merged = {}
    for k, a in values[0].items():
        b = values[1][k]; assert a.shape == b.shape and a.dtype == b.dtype
        merged[k] = ((a.float() + b.float()) * 0.5).to(a.dtype) if a.is_floating_point() else torch.maximum(a, b)
    p = bundle / 'weights/restart61_pairwise_margin/best.pt'
    assert p.is_symlink(); p.unlink()
    provenance = {'synthetic_checkpoint': True, 'training_started': False, 'ratio': [0.5, 0.5], 'integer_buffers': 'elementwise maximum', 'sources': [{'path': str(src / n / 'best.pt'), 'sha256': u.sha(src / n / 'best.pt')} for n in names]}
    torch.save({'model': merged, 'provenance': provenance}, p)
    u.write(bundle / 'soup_provenance.json', provenance)
    update_manifest(bundle)

def resident(bundle):
    p = bundle / 'src/tent_style_search.py'; code = p.read_text()
    before = '''def load_aspp(path, device):
    model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
    model.load_state_dict(torch.load(path, map_location="cpu")["model"]); model.eval()
    return model'''
    after = '''_pristine_states = {}
_resident_model = None
_resident_device = None
_restores = 0

def load_aspp(path, device):
    global _resident_model, _resident_device, _restores
    key = str(path)
    if key not in _pristine_states:
        _pristine_states[key] = torch.load(path, map_location="cpu", weights_only=False)["model"]
    if _resident_model is None or _resident_device != str(device):
        _resident_model = build_model("aspp", pretrained=False, checkpoint=None).to(device)
        _resident_device = str(device)
    # Restore every parameter and buffer before this image's fresh optimizer.
    _resident_model.load_state_dict(_pristine_states[key], strict=True)
    for parameter in _resident_model.parameters():
        parameter.grad = None
        parameter.requires_grad_(True)
    _resident_model.eval()
    _restores += 1
    print(f"RESIDENT source={key} restore={_restores} cpu_sources={len(_pristine_states)}", flush=True)
    return _resident_model'''
    assert code.count(before) == 1; p.write_text(code.replace(before, after))
    p = bundle / 'infer_lite.py'; code = p.read_text()
    before = '        gc.collect()\n        torch.cuda.empty_cache()'
    assert code.count(before) == 1
    p.write_text(code.replace(before, '        # Keep the pristine-restored model and CUDA allocator across images.'))

def hramp(bundle):
    p = bundle / 'infer_lite.py'; code = p.read_text()
    before = "        model = load_model(weights / HR_DIRS[component] / 'last.pt', device)\n"
    after = before + '''        dtype_seen = [False]
        def record_dtype(module, inputs, output):
            if not dtype_seen[0]:
                print(f'HRNET parameter_dtype={module.weight.dtype} output_dtype={output.dtype} batch={output.shape[0]}', flush=True)
                dtype_seen[0] = True
        next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d)).register_forward_hook(record_dtype)
'''
    assert code.count(before) == 1; code = code.replace(before, after)
    before = '''            value = predict_probs_tta(model, im, patch_size=h['patch'], overlap=h['overlap'],
                                      batch_size=h['batch_size'], tta=False, input_scale=h['input_scale'],
                                      device=device, tta_mode=h['tta']).numpy()'''
    after = '''            with torch.autocast(device_type='cuda', dtype=torch.float16):
                value = predict_probs_tta(model, im, patch_size=h['patch'], overlap=h['overlap'],
                                          batch_size=h['batch_size'], tta=False, input_scale=h['input_scale'],
                                          device=device, tta_mode=h['tta']).numpy()'''
    assert code.count(before) == 1; p.write_text(code.replace(before, after))
    import yaml
    p = bundle / 'configs/model349_lite.yaml'; config = yaml.safe_load(p.read_text())
    config['inference']['hrnet']['batch_size'] = 8
    p.write_text(yaml.safe_dump(config, sort_keys=False))

def deflate(bundle):
    rows = []
    for row in u.read(bundle / 'checkpoint_manifest.json')['checkpoints']:
        p = bundle / row['path']; original = p.resolve(); temp = p.parent / 'packed.tmp'
        hashes = {}; packed_hashes = {}
        with zipfile.ZipFile(original) as src, zipfile.ZipFile(temp, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as dst:
            for name in src.namelist():
                data = src.read(name); hashes[name] = hashlib.sha256(data).hexdigest(); dst.writestr(name, data)
        with zipfile.ZipFile(temp) as archive:
            for name in archive.namelist(): packed_hashes[name] = hashlib.sha256(archive.read(name)).hexdigest()
        assert hashes == packed_hashes
        a = torch.load(original, map_location='cpu', weights_only=False)['model']
        b = torch.load(temp, map_location='cpu', weights_only=False)['model']
        assert a.keys() == b.keys()
        assert all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]) for k in a)
        rows.append({'path': row['path'], 'old_bytes': original.stat().st_size, 'new_bytes': temp.stat().st_size, 'zip_entries_exact': True, 'all_model_tensors_exact': True, 'tensor_count': len(a), 'source_sha256': u.sha(original), 'packed_sha256': u.sha(temp)})
        if p.exists(): p.unlink()
        temp.rename(p)
    u.write(bundle / 'lossless_storage_verification.json', rows); update_manifest(bundle)

def code_hashes(bundle):
    u.write(bundle / 'source_hashes.json', {str(p.relative_to(bundle)): u.sha(p) for p in bundle.rglob('*') if p.is_file() and p.suffix in ['.py', '.yaml']})

def main():
    locks = []
    paths = ['/workspace/model349_compress/audit_20261008/gpu7.lock'] + [f'/workspace/model349_compress/{n}/gpu7.lock' for n in ['base12_followup_20261008', 'base12_inputcache_20261008', 'base12_highflip4_20261008', ROOT.name]]
    for path in paths:
        stream = open(path, 'a'); fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(stream)
    if subprocess.run(['pgrep', '-f', '[i]nfer_lite.py|[i]nfer_audited.py|[i]nfer_batched.py|[i]nfer_cpu2.py'], stdout=subprocess.DEVNULL).returncode == 0: raise RuntimeError('Existing inference; refusing overlap')
    u.idle(); (ROOT / 'runs').mkdir(exist_ok=False)
    frozen = u.snapshot(); u.write(ROOT / 'frozen_before.json', frozen)
    control = u.build_bundle('control12')
    bundles = {'soup11': u.build_bundle('soup11', prune_seed=True), 'resident12': u.build_bundle('resident12'), 'hramp12': u.build_bundle('hramp12')}
    soup(bundles['soup11']); resident(bundles['resident12']); hramp(bundles['hramp12'])
    for bundle in bundles.values(): code_hashes(bundle)
    monitor = (UTIL / 'monitor.py').read_text().replace(str(UTIL), str(ROOT)); (ROOT / 'monitor.py').write_text(monitor)
    subprocess.Popen([u.PYTHON, '-B', str(ROOT / 'monitor.py')], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    with (ROOT / 'probe.log').open('x') as log:
        subprocess.run([u.PYTHON, '-B', '-u', str(ROOT / 'probe.py')], env=dict(os.environ, CUDA_VISIBLE_DEVICES='7'), stdout=log, stderr=subprocess.STDOUT, check=True)
    comparisons = {}
    def evaluate(key):
        run = u.execute(bundles[key], key + '_valid20')
        comparisons[key] = {'metrics': u.metrics(run), 'completion': u.read(run / 'completion.json'), 'vs_base12': u.pair(u.metrics(u.OLD), u.metrics(run)), 'vs_lite15': u.pair(u.metrics(u.OLD.parent / 'cpu2_lite15'), u.metrics(run)), 'checkpoints': u.read(bundles[key] / 'checkpoint_manifest.json')}
        u.write(ROOT / 'raw_comparison.json', comparisons)
    def passed(key): return key in comparisons and comparisons[key]['vs_base12']['quality_pass'] and comparisons[key]['vs_lite15']['quality_pass']
    for key in ['soup11', 'resident12', 'hramp12']: evaluate(key)
    if passed('resident12') and passed('hramp12'):
        key = 'packed_fast11' if passed('soup11') else 'packed_fast12'
        bundles[key] = u.build_bundle(key, prune_seed=key.endswith('11'))
        if key.endswith('11'):
            p = bundles[key] / 'weights/restart61_pairwise_margin/best.pt'; p.unlink(); shutil.copy2(bundles['soup11'] / p.relative_to(bundles[key]), p)
            shutil.copy2(bundles['soup11'] / 'soup_provenance.json', bundles[key] / 'soup_provenance.json')
        resident(bundles[key]); hramp(bundles[key]); deflate(bundles[key]); code_hashes(bundles[key]); evaluate(key)
    selected = next((k for k in ['packed_fast11', 'packed_fast12', 'soup11', 'resident12', 'hramp12'] if passed(k)), None)
    u.write(ROOT / 'timing_selection.json', {'candidate': selected, 'selection': 'preregistered priority; no ratio/fold search'})
    timings = []
    if selected:
        for index, order in enumerate([['control12', selected], [selected, 'control12'], ['control12', selected]], 1):
            for key in order:
                run = u.execute(control if key == 'control12' else bundles[key], f'timing_{index}_{key}')
                timings.append({'run': run.name, 'candidate': key, 'completion': u.read(run / 'completion.json'), 'vs_base12': u.pair(u.metrics(u.OLD), u.metrics(run)), 'vs_lite15': u.pair(u.metrics(u.OLD.parent / 'cpu2_lite15'), u.metrics(run))})
                u.write(ROOT / 'timing_runs.json', timings)
        u.execute(bundles[selected], selected + '_test10', test=True)
    after = u.snapshot(); u.write(ROOT / 'frozen_after.json', after); assert frozen == after
    u.write(ROOT / 'completion.json', {'diagnostics_completed': True, 'base_and_weights_unchanged': True, 'raw_valid_candidates': list(comparisons), 'time_candidate': selected, 'test_candidates': [selected] if selected else [], 'independent_generalization_verified': False, 'new_training_started': False})
    print('SESSION_COMPLETE', flush=True)

if __name__ == '__main__':
    try: main()
    except Exception as exc:
        u.write(ROOT / 'failure.json', {'error': repr(exc), 'complete': False}); raise
