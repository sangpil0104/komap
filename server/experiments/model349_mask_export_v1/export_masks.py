"""Read-only Model349 checkpoint inference and valid/test RGB mask packaging.

Uses the completed 20260930 run's source snapshot and frozen route settings.
All new artifacts are isolated in this experiment. No training is launched.
The inherited per-image TENT step is part of the original inference recipe.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path('/Data3/sppark/komap')
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'experiments/model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z'
SOURCE = RUN / 'source_snapshot'
OUTPUTS = RUN / 'outputs'
VALID_RESULT = ROOT / 'results/model_results_0930_1'
SENTINEL = 'A117_step4_sample2'
PALETTE = np.array([(25, 76, 153), (76, 178, 76), (153, 127, 76), (204, 204, 204)], dtype=np.uint8)
TEAM_TO_MAIN = [1, 2, 0, 3]
B4_DIRS = {'322': 'model322_team_b4_group_oof',
           '326': 'model326_hybrid_fullstats_group_oof',
           '329': 'model329_fullstats_width4000_uniform_group_oof'}
HR_DIRS = {'336': 'model336_hrnet_best', '347': 'model347_hrnet_inverse_ce_finetune'}

sys.path[:0] = [str(SOURCE / 'experiments'), str(SOURCE / 'experiments/historical_scripts')]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def canonical(name):
    return Path(name).stem.removesuffix('_image').removesuffix('_mask')


def archive_paths():
    paths = {'269': OUTPUTS / 'model269_fullstats_1x_extended/valid_probs.npz'}
    paths.update({k: OUTPUTS / v / 'valid_ensemble/probabilities.npz' for k, v in B4_DIRS.items()})
    paths.update({k: OUTPUTS / v / 'valid_eval/valid_probabilities.npz' for k, v in HR_DIRS.items()})
    return paths


def load_archive(path, team=False):
    with np.load(path) as data:
        return {canonical(k): data[k].astype(np.float32)[TEAM_TO_MAIN if team else slice(None)]
                for k in data.files}


def settings():
    base = json.loads((OUTPUTS / 'model339_joint_gamma_route_search/results.json').read_text())['best_image_empty_zero']
    r341 = json.loads((OUTPUTS / 'model341_model329_joint_route_search/results.json').read_text())['best_image_empty_zero']
    r346 = json.loads((OUTPUTS / 'model346_model336_into_341_route/results.json').read_text())
    r349 = json.loads((OUTPUTS / 'model349_model347_into_346_route/results.json').read_text())
    assert r341['base_model'] == 339
    return base, tuple(r341['gamma329']), tuple(r346['raw_best_gamma336']), tuple(r349['raw_best_gamma347'])


def predict_fixed(item, sources):
    import model346_model336_into_341_route as final
    route = final.model341.route
    base, g329, g336, g347 = settings()
    row = {'item': item, 'base': route.undo_model181_primary_blend(item),
           'sources': {**sources, '342': sources['347']}}
    return final._candidate_prediction(row, g347, base, g329,
                                      route.bilateral_weights(item['z']), g336, (0., 0., 0., 0.))


def prepare(work):
    """Verify frozen route reproduces all saved masks before new inference."""
    work.mkdir(parents=True, exist_ok=False)
    (work / 'logs').mkdir()
    (work / 'components').mkdir()
    (work / 'checks').mkdir()
    test_images = sorted((ROOT / 'dataset/test/images').glob('*_image.png'))
    assert len(test_images) == 10
    manifest = [{'split': 'valid_check', 'name': SENTINEL,
                 'path': str(RUN / f'data/valid/images/{SENTINEL}_image.png')}]
    manifest += [{'split': 'test', 'name': canonical(p.name), 'path': str(p)} for p in test_images]
    for x in manifest:
        p = Path(x['path'])
        x['sha256'] = digest(p)
        with Image.open(p) as im:
            x['size'] = list(im.size)
    write_json(work / 'images.json', manifest)
    print('Checking cached validation parity (20 images)', flush=True)
    with (OUTPUTS / 'model183_metric_calibration/entries.pkl').open('rb') as stream:
        entries = pickle.load(stream)
    names = sorted(canonical(p.name) for p in (RUN / 'data/valid/images').glob('*_image.png'))
    assert len(entries) == len(names) == 20
    archives = {k: load_archive(p, k != '269') for k, p in archive_paths().items()}
    checks = []
    for name, item in tqdm(list(zip(names, entries)), desc='Frozen Model349 parity', unit='image'):
        pred = predict_fixed(item, {k: a[name] for k, a in archives.items()})
        old = np.asarray(Image.open(VALID_RESULT / 'predicted_labels' / f'{name}.png'))
        different = int(np.count_nonzero(pred != old))
        assert different == 0, (name, different)
        rgb = np.asarray(Image.open(VALID_RESULT / 'predicted_masks' / f'{name}_mask.png'))
        assert np.array_equal(PALETTE[old], rgb)
        checks.append({'image': name, 'different_pixels': different})
    # Reuse the original training-derived priors verbatim, with no test labels.
    write_json(work / 'priors.json', {'z': entries[0]['prior_z_stats'], 'q': entries[0]['prior_q_stats']})
    sentinel_item = entries[names.index(SENTINEL)]
    with (work / 'checks/base_reference.pkl').open('wb') as f:
        pickle.dump({k: sentinel_item[k] for k in ['mixed', 'extra', 'z', 'q']}, f)
    for k, archive in archives.items():
        np.save(work / 'checks' / f'{k}_reference.npy', archive[SENTINEL])
    write_json(work / 'checks/cached_valid_parity.json', {'images': checks, 'all_pixel_identical': True})
    del entries, archives, sentinel_item
    gc.collect()
    checkpoints = [OUTPUTS / d / f'fold{i}/last.pt' for d in B4_DIRS.values() for i in range(1, 5)]
    checkpoints += [OUTPUTS / d / 'last.pt' for d in HR_DIRS.values()]
    checkpoints += [OUTPUTS / d / 'best.pt' for d in [
        'model97_native_patch', 'model98_hard_negative', 'model109_aspp_native',
        'restart61_pairwise_margin', 'restart51_aspp_768_finetune',
        'restart59_aspp768_seed456', 'model123_ema', 'model181_raw_zcdf_aspp',
        'model269_fullstats_1x_extended']]
    source_paths = sorted(SOURCE.rglob('*.py'))
    write_json(work / 'preservation.json', [
        {'path': str(p), 'sha256': digest(p), 'size': p.stat().st_size,
         'mtime_ns': p.stat().st_mtime_ns}
        for p in tqdm(checkpoints + source_paths, desc='Record source/checkpoint hashes', unit='file')])
    write_json(work / 'route.json', {'base': settings()[0], 'gamma329': settings()[1],
                                   'gamma336': settings()[2], 'gamma347': settings()[3]})
    shutil.copy2(__file__, work / 'export_masks_snapshot.py')


def probability_check(work, component, value):
    ref = np.load(work / 'checks' / f'{component}_reference.npy')
    delta = np.abs(value.astype(np.float32) - ref)
    check = {'component': component, 'max_abs': float(delta.max()),
             'mean_abs': float(delta.mean()),
             'argmax_different_fraction': float(np.mean(value.argmax(0) != ref.argmax(0)))}
    write_json(work / 'checks' / f'{component}_reinference.json', check)
    print(json.dumps(check), flush=True)
    # Allow FP16 GPU-kernel rounding, but reject materially different inference.
    assert check['mean_abs'] < 2e-4 and check['max_abs'] < .02, check


def worker(work, component):
    import torch
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    images = json.loads((work / 'images.json').read_text())
    out = work / 'components' / component
    out.mkdir(exist_ok=False)
    print(f'component={component} gpu={os.environ.get("CUDA_VISIBLE_DEVICES")} device={torch.cuda.get_device_name(0)}', flush=True)

    if component in B4_DIRS:
        folder = 'model322_team_b4' if component == '322' else 'model326_hybrid_fullstats'
        sys.path.insert(0, str(SOURCE / 'experiments' / folder / 'KoMaP'))
        from common import load_member, load_gray, predict_probs
        members = [load_member(str(OUTPUTS / B4_DIRS[component] / f'fold{i}'), str(device)) for i in range(1, 5)]
        for row in tqdm(images, desc=f'B4 Model{component}', unit='image'):
            value = predict_probs(members, load_gray(row['path']), tta=True, device=str(device)).cpu().numpy()
            # Preserve the original NPZ FP16 quantization before route fusion.
            value = value.astype(np.float16)[TEAM_TO_MAIN]
            np.save(out / f"{row['name']}.npy", value)
            if row['split'] == 'valid_check':
                probability_check(work, component, value)
            print(f"DONE {component} {row['name']}", flush=True)
    elif component in HR_DIRS:
        sys.path.insert(0, str(SOURCE / 'experiments/model336_hrnet_best'))
        from komap_seg.checkpoint import load_model
        from komap_seg.dataset import build_eval_transform, load_image_rgb
        from komap_seg.inference import predict_probs_tta
        model = load_model(OUTPUTS / HR_DIRS[component] / 'last.pt', device)
        transform = build_eval_transform()
        for row in tqdm(images, desc=f'HRNet Model{component}', unit='image'):
            im = transform(image=load_image_rgb(Path(row['path'])))['image'].float()
            value = predict_probs_tta(model, im, patch_size=512, overlap=170, batch_size=2,
                                      tta=True, input_scale=2., device=device, tta_mode='flip4').numpy()
            value = value.astype(np.float16)[TEAM_TO_MAIN]
            np.save(out / f"{row['name']}.npy", value)
            if row['split'] == 'valid_check':
                probability_check(work, component, value)
            print(f"DONE {component} {row['name']}", flush=True)
    elif component == '269':
        from model253_highres_scale_finetune import Stride1NativeStemASPP, predict_scaled
        model = Stride1NativeStemASPP().to(device)
        model.load_state_dict(torch.load(OUTPUTS / 'model269_fullstats_1x_extended/best.pt', map_location='cpu', weights_only=False)['model'])
        model.eval()
        for row in tqdm(images, desc='Native Model269', unit='image'):
            value = predict_scaled(model, Path(row['path']), 1024, 576, 1., device, True, 'd4').astype(np.float16)
            np.save(out / f"{row['name']}.npy", value)
            if row['split'] == 'valid_check':
                probability_check(work, component, value)
            print(f"DONE {component} {row['name']}", flush=True)
    elif component == 'base':
        base_worker(work, images, out, device)
    else:
        raise ValueError(component)
    write_json(out / 'complete.json', {'component': component, 'image_count': len(images)})


def base_worker(work, images, out, device):
    import torch
    import calibrate_al
    import model183_metric_calibration as cache
    from model187_schedule_branch_blend import undo_model181_primary_blend
    from highres1024_search import choose_source
    from hard_source_search import route_params
    from prior_representation_search import percentile_image
    from style_tta_search import predict_array, transform_image
    from tent_style_search import adapt_model
    from types import SimpleNamespace

    priors = json.loads((work / 'priors.json').read_text())
    stats = []
    for image_path in sorted((RUN / 'data/train/images').glob('*_image.png')):
        gray = np.asarray(Image.open(image_path).convert('L'), dtype=np.float32)
        stats.append((float(gray.mean()), float(gray.std())))
    train_mean = float(np.mean([x[0] for x in stats]))
    train_std = float(np.mean([x[1] for x in stats]))
    models = [calibrate_al.load_model(str(OUTPUTS / d / 'best.pt'), device)
              for d in ['model97_native_patch', 'model98_hard_negative']]
    models.append(cache.load_aspp(OUTPUTS / 'model109_aspp_native/best.pt', device))
    pair = cache.load_aspp(OUTPUTS / 'restart61_pairwise_margin/best.pt', device)
    rep = cache.load_rep(OUTPUTS / 'model181_raw_zcdf_aspp/best.pt', device)
    candidates = {'restart51': 'restart51_aspp_768_finetune', 'model109': 'model109_aspp_native',
                  'seed456': 'restart59_aspp768_seed456', 'pairwise': 'restart61_pairwise_margin',
                  'ema': 'model123_ema'}
    args = SimpleNamespace(**{f'source_{k}': str(OUTPUTS / v / 'best.pt') for k, v in candidates.items()})
    for row in tqdm(images, desc='Original Model349 base', unit='image'):
        path = Path(row['path'])
        raw = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
        gray = raw[..., 0].astype(np.float32)
        mean, std = float(gray.mean()), float(gray.std())
        branches = []
        for model, patch, stride in zip(models, (640, 512, 512), (512, 384, 384)):
            p = calibrate_al.predict_probabilities(model, path, patch, stride, device, True, 2, True)
            branches.append(p.astype(np.float16).astype(np.float32))
        pair_value = cache.predict_rgb(pair, path, 768, 576, device, True, 1, 'd4', 'gaussian')
        checkpoint = choose_source(mean, args)
        adapted = adapt_model(checkpoint, raw, device, True, 5, 1e-5)
        style = transform_image(raw, 'standardize', train_mean, train_std)
        high768 = tuple(predict_array(adapted, im, 768, 576, device, True, 1, 'd4', 'gaussian') for im in [raw, style])
        high1024 = tuple(predict_array(adapted, im, 1024, 768, device, True, 1, 'd4', 'gaussian') for im in [raw, style])
        high = tuple(.8 * high768[i] + .2 * high1024[i] for i in (0, 1))
        item = {'branches': branches, 'pair': pair_value.astype(np.float32),
                'extra': cache.predict_rep(rep, path, 512, 384, device, True, 2),
                'z': (gray - mean) / max(std, 1e-5), 'q': percentile_image(gray),
                'prior_z_stats': priors['z'], 'prior_q_stats': priors['q'],
                'params': route_params(mean, std)}
        item['mixed'] = cache.make_mixed(item, high, gamma=.40)
        if row['split'] == 'valid_check':
            with (work / 'checks/base_reference.pkl').open('rb') as f:
                ref = pickle.load(f)
            a, b = undo_model181_primary_blend(item), undo_model181_primary_blend(ref)
            delta = np.abs(a-b)
            check = {'component': 'base', 'max_abs': float(delta.max()), 'mean_abs': float(delta.mean())}
            write_json(work / 'checks/base_reinference.json', check)
            print(json.dumps(check), flush=True)
            assert check['mean_abs'] < 2e-4 and check['max_abs'] < .02, check
        # Keep only fields consumed by the frozen route; no masks/GT for test.
        result = {k: item[k] for k in ['mixed', 'extra', 'z', 'q', 'prior_z_stats', 'prior_q_stats', 'params']}
        with (out / f"{row['name']}.pkl").open('wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"DONE base {row['name']} source={checkpoint}", flush=True)
        del adapted, high768, high1024, high, item, result
        gc.collect()
        torch.cuda.empty_cache()


def assemble(work):
    images = json.loads((work / 'images.json').read_text())
    bundle = work / 'bundle'
    (bundle / 'valid').mkdir(parents=True, exist_ok=False)
    (bundle / 'test').mkdir()
    records = []
    for row in tqdm(images, desc='Frozen route and masks', unit='image'):
        name = row['name']
        with (work / 'components/base' / f'{name}.pkl').open('rb') as f:
            item = pickle.load(f)
        sources = {c: np.load(work / 'components' / c / f'{name}.npy').astype(np.float32)
                   for c in ['269', '322', '326', '329', '336', '347']}
        pred = predict_fixed(item, sources).astype(np.uint8)
        assert pred.shape == (row['size'][1], row['size'][0])
        if row['split'] == 'valid_check':
            old = np.asarray(Image.open(VALID_RESULT / 'predicted_labels' / f'{name}.png'))
            different = int(np.count_nonzero(pred != old))
            check = {'image': name, 'different_pixels': different,
                     'different_fraction': different / old.size,
                     'note': 'Original saved valid masks are packaged verbatim.'}
            write_json(work / 'checks/full_reinference.json', check)
            assert check['different_fraction'] < .001, check
        else:
            target = bundle / 'test' / f'{name}_mask.png'
            Image.fromarray(PALETTE[pred]).save(target)
            records.append({'split': 'test', 'file': target.relative_to(bundle).as_posix(),
                            'sha256': digest(target), 'size': row['size']})
    for src in tqdm(sorted((VALID_RESULT / 'predicted_masks').glob('*_mask.png')), desc='Copy valid masks', unit='image'):
        dst = bundle / 'valid' / src.name
        shutil.copy2(src, dst)
        assert digest(src) == digest(dst)
        with Image.open(dst) as im:
            size = list(im.size)
        records.append({'split': 'valid', 'file': dst.relative_to(bundle).as_posix(),
                        'sha256': digest(dst), 'size': size})
    assert len(list((bundle / 'valid').glob('*.png'))) == 20
    assert len(list((bundle / 'test').glob('*.png'))) == 10
    palette_set = {tuple(x) for x in PALETTE.tolist()}
    for r in tqdm(records, desc='Validate RGB masks', unit='image'):
        with Image.open(bundle / r['file']) as im:
            assert im.mode == 'RGB' and list(im.size) == r['size']
            colors = {color for _, color in im.getcolors(maxcolors=5) or []}
            assert colors and colors <= palette_set
    for item in tqdm(json.loads((work / 'preservation.json').read_text()), desc='Verify source preservation', unit='file'):
        p = Path(item['path'])
        assert digest(p) == item['sha256'], p
    manifest = {'model': 'Model349', 'original_run': str(RUN),
                'valid_image_miou': .8006879890006227,
                'valid_count': 20, 'test_count': 10,
                'valid_masks': 'Byte-identical copy of results/model_results_0930_1/predicted_masks',
                'test_masks': 'New inference with original checkpoints, source snapshot and fixed selected route',
                'test_labels_used': False, 'route_search_performed': False,
                'inference_tent': 'Original base branch adapts BN per input image, 5 entropy steps at lr=1e-5',
                'class_order': ['Al3Ni', 'Primary Si', 'Al', 'Eutectic Si'],
                'palette': PALETTE.tolist(), 'files': records,
                'checks': {'cached_valid_parity': json.loads((work / 'checks/cached_valid_parity.json').read_text()),
                           'reinference': json.loads((work / 'checks/full_reinference.json').read_text())}}
    write_json(bundle / 'manifest.json', manifest)
    (bundle / 'README.txt').write_text(
        'Model349 predicted masks\n\nvalid/: 20 saved validation RGB masks (image mIoU 80.068799%).\n'
        'test/: 10 newly inferred test RGB masks using the same trained Model349 and frozen route.\n'
        'Images retain original dimensions and *_mask.png filenames.\n'
        'Colors: Al3Ni (25,76,153); Primary Si (76,178,76); Al (153,127,76); Eutectic Si (204,204,204).\n'
        'No comparison panels or ground-truth masks are included. Test metrics are unavailable.\n'
        'See manifest.json for provenance, SHA-256 checksums and verification.\n', encoding='utf-8')
    zip_path = ROOT / 'results' / f'model349_valid_test_predicted_masks_{work.name}.zip'
    members = sorted(p for p in bundle.rglob('*') if p.is_file())
    with zipfile.ZipFile(zip_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in tqdm(members, desc='Create ZIP', unit='file'):
            z.write(p, p.relative_to(bundle).as_posix())
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        for r in records:
            assert hashlib.sha256(z.read(r['file'])).hexdigest() == r['sha256']
    result = {'zip': str(zip_path), 'bytes': zip_path.stat().st_size, 'sha256': digest(zip_path),
              'valid': 20, 'test': 10, 'integrity_verified': True}
    write_json(work / 'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def run_all(work, gpus):
    prepare(work)
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    devices = {}
    for line in gpu_rows.splitlines():
        index, uuid, memory = [v.strip() for v in line.split(',')]
        if index in gpus:
            assert int(memory) < 1000, f'GPU {index} is busy: {memory} MiB'
            devices[index] = uuid
    assert set(devices) == set(gpus)
    pending = ['322', '326', '329', 'base', '336', '347', '269']
    running = {}
    last_print = 0.
    try:
        with tqdm(total=len(pending), desc='Model349 inference components', unit='component') as bar:
            while pending or running:
                for index, uuid in devices.items():
                    if not pending or index in running:
                        continue
                    component = pending.pop(0)
                    log = (work / 'logs' / f'{component}.log').open('w')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONDONTWRITEBYTECODE='1',
                               PYTHONUNBUFFERED='1', NO_ALBUMENTATIONS_UPDATE='1',
                               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                    proc = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                                             'worker', '--work', str(work), '--component', component],
                                            env=env, stdout=log, stderr=subprocess.STDOUT)
                    running[index] = (proc, log, component)
                    print(f'START component={component} GPU={index} PID={proc.pid}', flush=True)
                for index, (proc, log, component) in list(running.items()):
                    rc = proc.poll()
                    if rc is not None:
                        log.close()
                        del running[index]
                        if rc:
                            raise RuntimeError(f'Component {component} failed ({rc}); see {work}/logs/{component}.log')
                        bar.update(1)
                        print(f'COMPLETE component={component} GPU={index}', flush=True)
                if time.monotonic() - last_print >= 20:
                    last_print = time.monotonic()
                    for index, (_, _, component) in running.items():
                        tail = (work / 'logs' / f'{component}.log').read_text(errors='replace')[-2500:].replace('\r', '\n').splitlines()
                        latest = [line for line in tail if line.strip()]
                        print(f'GPU {index} {component}: {latest[-1] if latest else "loading"}', flush=True)
                time.sleep(1)
    finally:
        for proc, log, _ in running.values():
            proc.terminate()
            proc.wait(timeout=30)
            log.close()
    assemble(work)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['all', 'worker', 'assemble'])
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--component')
    parser.add_argument('--gpus', default='1,2,3,6,7')
    args = parser.parse_args()
    if args.action == 'all':
        run_all(args.work, args.gpus.split(','))
    elif args.action == 'worker':
        worker(args.work, args.component)
    else:
        assemble(args.work)
