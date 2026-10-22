"""Portable frozen Model349 inference. No training or route search entrypoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from types import SimpleNamespace

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
TEAM_TO_MAIN = [1, 2, 0, 3]
B4_DIRS = {'322': 'model322_team_b4_group_oof', '326': 'model326_hybrid_fullstats_group_oof',
           '329': 'model329_fullstats_width4000_uniform_group_oof'}
HR_DIRS = {'336': 'model336_hrnet_best', '347': 'model347_hrnet_inverse_ce_finetune'}
COMPONENTS = ['326', '329', 'base', '336', '269']  # lite: 322 and 347 removed
HIGH_CANDIDATES = {'restart51': 'restart51_aspp_768_finetune', 'model109': 'model109_aspp_native',
                   'seed456': 'restart59_aspp768_seed456', 'pairwise': 'restart61_pairwise_margin'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config['model'] != 'Model349':
        raise ValueError('This executable is for Model349 only')
    return config


def active_components(config):
    return config['inference'].get('active_components', COMPONENTS)


def high_candidates(config):
    result = dict(HIGH_CANDIDATES)
    if not config['inference'].get('ema_pruned', True):
        result['ema'] = 'model123_ema'
    for alias, target in config['inference'].get('high_source_aliases', {}).items():
        if target not in result or alias == target:
            raise ValueError('Invalid high-source alias')
        result.pop(alias)
    return result


def b4_fold_ids(config, component):
    count = config['inference']['b4_folds']
    ids = config['inference'].get('b4_fold_ids', {}).get(component, list(range(1, count + 1)))
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(i, int) or i < 1 or i > count for i in ids):
        raise ValueError(f'Invalid B4 fold selection for {component}: {ids}')
    return ids


def predict_fixed(item, sources, config):
    import model335_exhaustive_b4_route_search as route
    settings = config['route']
    # The historical auxiliary Primary-Si blend is undone here. In the
    # representation-free path the same pre-blend probability is stored directly.
    base = item['preaux'] if 'preaux' in item else route.undo_model181_primary_blend(item)
    sources = dict(sources)
    for key in ('primary', 'al3ni', 'eutectic'):
        if not settings['base'][key + '_gamma']:
            sources.setdefault(settings['base'][key + '_source'], base)
    row = {'item': item, 'base': base, 'sources': sources}
    mixed = route.make_mixed(row, settings['base'])
    for component, key in [('329', 'gamma329'), ('336', 'gamma336'), ('347', 'gamma347')]:
        for class_id, gamma in enumerate(settings[key]):
            if gamma:
                mixed = route.replace_classes(mixed, sources[component], (class_id,), gamma)
    return route.predict(item, mixed, settings['base'], route.bilateral_weights(item['z']), exact=True)


def worker(work, component):
    import torch
    manifest = json.loads((work / 'input_manifest.json').read_text())
    config = manifest['config']
    torch.set_num_threads(config['inference']['threads'])
    import cv2
    cv2.setNumThreads(2)
    print(f'CPU torch_threads={torch.get_num_threads()} opencv_threads={cv2.getNumThreads()}', flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the original Model349 inference recipe')
    device = torch.device('cuda:0')
    weights = ROOT / config['weights_dir']
    images = manifest['images']
    out = work / 'probabilities' / component
    out.mkdir(exist_ok=False)
    print(f'component={component} GPU={os.environ.get("CUDA_VISIBLE_DEVICES")} device={torch.cuda.get_device_name(0)}', flush=True)
    started = time.monotonic()

    if component in B4_DIRS:
        if component == '322':
            from b4_gray import load_member, load_gray, predict_probs
        else:
            from b4_zcdf import load_member, load_gray, predict_probs
        members = [load_member(str(weights / B4_DIRS[component] / f'fold{i}'), str(device))
                   for i in b4_fold_ids(config, component)]
        for row in tqdm(images, desc=f'B4 Model{component}', unit='image'):
            value = predict_probs(members, load_gray(row['path']), tta=False, device=str(device)).cpu().numpy()
            # Match the original cached probability quantization and channel order.
            np.save(out / f"{row['name']}.npy", value.astype(np.float16)[TEAM_TO_MAIN])
            print(f"DONE {component} {row['name']}", flush=True)
    elif component in HR_DIRS:
        from komap_seg.checkpoint import load_model
        from komap_seg.dataset import build_eval_transform, load_image_rgb
        from komap_seg.inference import predict_probs_tta
        model = load_model(weights / HR_DIRS[component] / 'last.pt', device)
        dtype_seen = [False]
        def record_dtype(module, inputs, output):
            if not dtype_seen[0]:
                print(f'HRNET parameter_dtype={module.weight.dtype} output_dtype={output.dtype} batch={output.shape[0]}', flush=True)
                dtype_seen[0] = True
        next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d)).register_forward_hook(record_dtype)
        transform = build_eval_transform()
        h = config['inference']['hrnet']
        for row in tqdm(images, desc=f'HRNet Model{component}', unit='image'):
            im = transform(image=load_image_rgb(Path(row['path'])))['image'].float()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                value = predict_probs_tta(model, im, patch_size=h['patch'], overlap=h['overlap'],
                                          batch_size=h['batch_size'], tta=False, input_scale=h['input_scale'],
                                          device=device, tta_mode=h['tta']).numpy()
            np.save(out / f"{row['name']}.npy", value.astype(np.float16)[TEAM_TO_MAIN])
            print(f"DONE {component} {row['name']}", flush=True)
    elif component == '269':
        from model253_highres_scale_finetune import Stride1NativeStemASPP, predict_scaled
        model = Stride1NativeStemASPP().to(device)
        model.load_state_dict(torch.load(weights / 'model269_fullstats_1x_extended/best.pt',
                                        map_location='cpu', weights_only=False)['model'])
        model.eval()
        n = config['inference']['native']
        for row in tqdm(images, desc='Native Model269', unit='image'):
            value = predict_scaled(model, Path(row['path']), n['patch'], n['stride'], n['scale'], device, True, 'none')
            np.save(out / f"{row['name']}.npy", value.astype(np.float16))
            print(f"DONE {component} {row['name']}", flush=True)
    elif component == 'base':
        base_worker(images, out, device, config, weights)
    else:
        raise ValueError(component)
    write_json(out / 'complete.json', {'component': component, 'images': len(images),
                                      'elapsed_seconds': time.monotonic() - started})


def base_worker(images, out, device, config, weights):
    import torch
    from highres1024_search import choose_source
    from hard_source_search import route_params
    from prior_representation_search import percentile_image
    from style_tta_search import predict_array, transform_image
    from tent_style_search import adapt_model

    calibration = json.loads((ROOT / config['calibration']).read_text())
    priors = calibration['priors']
    train_mean, train_std = calibration['train_brightness_mean'], calibration['train_brightness_std']
    args = SimpleNamespace(**{f'source_{k}': str(weights / v / 'best.pt')
                             for k, v in high_candidates(config).items()})
    for alias, target in config['inference'].get('high_source_aliases', {}).items():
        setattr(args, 'source_' + alias, getattr(args, 'source_' + target))
    # The two narrow brightness bands previously routed to Model123 EMA now
    # reuse restart51; no Model123 checkpoint or forward is required.
    if config['inference'].get('ema_pruned', True):
        args.source_ema = args.source_restart51
    tent = config['inference']['tent']
    for row in tqdm(images, desc='Model349 base', unit='image'):
        path = Path(row['path'])
        raw = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
        gray = raw[..., 0].astype(np.float32)
        mean, std = float(gray.mean()), float(gray.std())
        checkpoint = choose_source(mean, args)
        # Original per-image entropy adaptation; no labels, no saved weight changes.
        adapted = adapt_model(checkpoint, raw, device, True, tent['steps'], tent['lr'])
        style = transform_image(raw, 'standardize', train_mean, train_std)
        high768 = tuple(predict_array(adapted, im, 768, 576, device, True, 1, 'd4', 'gaussian') for im in [raw, style])
        high1024 = None  # lite: 1024 tiles skipped
        high = high768
        params = dict(route_params(mean, std), pair_weight=0.0)
        preaux = ((1.0 - params['style_weight']) * high[0]
                  + params['style_weight'] * high[1]).astype(np.float32)
        preaux /= np.maximum(preaux.sum(axis=0, keepdims=True), 1e-7)
        result = {'preaux': preaux, 'z': (gray - mean) / max(std, 1e-5),
                  'q': percentile_image(gray), 'prior_z_stats': priors['z'],
                  'prior_q_stats': priors['q'], 'params': params}
        with (out / f"{row['name']}.pkl").open('wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"DONE base {row['name']} source={Path(checkpoint).parent.name}", flush=True)
        del adapted, high768, high1024, high, preaux, result
        # Keep the pristine-restored model and CUDA allocator across images.


def decode_gt(path, palette):
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 2 and np.isin(arr, np.arange(4)).all():
        return arr.astype(np.uint8)
    with Image.open(path) as im:
        rgb = np.array(im.convert('RGB'))
    labels = np.full(rgb.shape[:2], 255, np.uint8)
    for cid, color in enumerate(palette):
        labels[np.all(rgb == color, axis=-1)] = cid
    if np.any(labels == 255):
        raise ValueError(f'Unknown ground-truth mask colors: {path}')
    return labels


def assemble(work, gt_dir=None):
    manifest = json.loads((work / 'input_manifest.json').read_text())
    config = manifest['config']
    palette = np.array(config['palette'], dtype=np.uint8)
    (work / 'predicted_labels').mkdir()
    (work / 'predicted_masks').mkdir()
    scores = []
    for row in tqdm(manifest['images'], desc='Model349 fusion / masks', unit='image'):
        name = row['name']
        with (work / 'probabilities/base' / f'{name}.pkl').open('rb') as f:
            item = pickle.load(f)
        sources = {c: np.load(work / 'probabilities' / c / f'{name}.npy').astype(np.float32)
                   for c in active_components(config) if c != 'base'}
        pred = predict_fixed(item, sources, config).astype(np.uint8)
        assert pred.shape == (row['size'][1], row['size'][0])
        Image.fromarray(pred).save(work / 'predicted_labels' / f'{name}.png')
        Image.fromarray(palette[pred]).save(work / 'predicted_masks' / f'{name}_mask.png')
        if gt_dir is not None:
            candidates = [gt_dir / f'{name}_mask.png', gt_dir / f'{name}.png']
            target = next((p for p in candidates if p.is_file()), None)
            if target is None:
                raise FileNotFoundError(f'Missing ground-truth mask: {name}')
            gt = decode_gt(target, palette)
            if gt.shape != pred.shape:
                raise ValueError(f'Ground-truth shape differs: {name}')
            ious = []
            for cid in range(4):
                intersection = np.count_nonzero((pred == cid) & (gt == cid))
                union = np.count_nonzero((pred == cid) | (gt == cid))
                ious.append(float(intersection / union) if union else 0.0)
            scores.append({'image': name, 'class_iou': ious, 'image_miou': float(np.mean(ious))})
    if scores:
        write_json(work / 'metrics.json', {'metric': 'mean of per-image four-class mIoU; empty union = 0',
                    'class_order': config['class_order'], 'count': len(scores),
                    'image_miou': float(np.mean([r['image_miou'] for r in scores])),
                    'class_image_mean_iou': np.mean([r['class_iou'] for r in scores], axis=0).tolist(),
                    'images': scores})
    return scores


def run(args):
    config = load_config(args.config)
    p = args.input.resolve()
    files = [p] if p.is_file() else sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'])
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError(f'No input images: {p}')
    rows = []
    names = set()
    for path in files:
        name = path.stem.removesuffix('_image')
        if name in names:
            raise ValueError(f'Duplicate image name: {name}')
        names.add(name)
        with Image.open(path) as im:
            size = list(im.size)
        rows.append({'name': name, 'path': str(path), 'size': size, 'sha256': sha(path)})
    required = ([f'weights/{B4_DIRS[c]}/fold{i}/last.pt' for c in ('326', '329')
                 for i in b4_fold_ids(config, c) if c in active_components(config)]
                + ['weights/model336_hrnet_best/last.pt',
                   'weights/model269_fullstats_1x_extended/best.pt']
                + [f'weights/{folder}/best.pt' for folder in high_candidates(config).values()])
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing model weights: {missing}')
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    (work / 'logs').mkdir()
    (work / 'probabilities').mkdir()
    write_json(work / 'input_manifest.json', {'model': 'Model349', 'config': config, 'images': rows,
                 'calibration_sha256': sha(ROOT / config['calibration']),
                 'required_weights': required})
    gpus = [x.strip() for x in args.gpus.split(',') if x.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError('--gpus must contain one or more distinct GPU indices/UUIDs')
    pending = list(active_components(config))
    running = {}
    started = time.monotonic()
    last_print = started
    try:
        with tqdm(total=len(active_components(config)), desc='Model349 components', unit='component') as bar:
            while pending or running:
                for gpu in gpus:
                    if gpu in running or not pending:
                        continue
                    time.sleep(1)
                    usage = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
                    memory, utilization = map(int, usage.strip().split(','))
                    if memory > 64 or utilization:
                        raise RuntimeError(f'GPU {gpu} occupied before component: {usage.strip()}')
                    component = pending.pop(0)
                    log = (work / 'logs' / f'{component}.log').open('w')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1',
                               PYTHONUNBUFFERED='1', NO_ALBUMENTATIONS_UPDATE='1',
                               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                    process = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                                                '--_worker', component, '--output', str(work)],
                                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    running[gpu] = (process, log, component)
                    print(f'START component={component} GPU={gpu} log={work}/logs/{component}.log', flush=True)
                for gpu, (process, log, component) in list(running.items()):
                    code = process.poll()
                    if code is not None:
                        log.close()
                        del running[gpu]
                        if code:
                            tail = (work / 'logs' / f'{component}.log').read_text(errors='replace')[-6000:]
                            raise RuntimeError(f'Component {component} failed (exit {code}):\n{tail}')
                        bar.update(1)
                        print(f'COMPLETE component={component} GPU={gpu}', flush=True)
                if time.monotonic() - last_print >= 20:
                    last_print = time.monotonic()
                    for gpu, (_, _, component) in running.items():
                        lines = (work / 'logs' / f'{component}.log').read_text(errors='replace')[-2000:].replace('\r', '\n').splitlines()
                        lines = [x for x in lines if x.strip()]
                        print(f'GPU {gpu} {component}: {lines[-1] if lines else "loading weights"}', flush=True)
                time.sleep(.5)
    finally:
        for process, log, _ in running.values():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
    import cv2
    cv2.setNumThreads(2)
    fusion_started = time.monotonic()
    scores = assemble(work, args.gt.resolve() if args.gt else None)
    write_json(work / 'completion.json', {'model': 'Model349', 'images': len(rows),
               'elapsed_seconds': time.monotonic() - started, 'training_started': False,
               'tent_steps': config['inference']['tent']['steps'],
               'gpus': gpus, 'opencv_threads': 2, 'fusion_seconds': time.monotonic() - fusion_started,
               'inference_sha256': sha(Path(__file__)),
               'tent': 'Per-image BN adaptation; checkpoint files never modified',
               'mean_image_miou': float(np.mean([x['image_miou'] for x in scores])) if scores else None})
    if not args.keep_probabilities:
        shutil.rmtree(work / 'probabilities')
    print(f'FINISHED: {work}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='One image or a directory of images (not recursive)')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'), help='A NEW output directory')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/model349_lite.yaml')
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0], help='Physical CUDA GPU indices/UUIDs, e.g. 0 or 0,1')
    parser.add_argument('--gt', type=Path, help='Optional ground-truth mask folder; enables image mIoU evaluation only')
    parser.add_argument('--limit', type=int, help='Process first N images for a smoke run')
    parser.add_argument('--keep-probabilities', action='store_true', help='Keep intermediate probabilities for auditing')
    parser.add_argument('--check', action='store_true', help='Check imports/CUDA and report the frozen configuration')
    parser.add_argument('--_worker', choices=COMPONENTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(args.output, args._worker)
    elif args.check:
        import torch
        import segmentation_models_pytorch
        import b4_gray, b4_zcdf
        import model335_exhaustive_b4_route_search
        from komap_seg.checkpoint import load_model
        print(json.dumps({'model': load_config(args.config)['model'], 'python': sys.version,
                          'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                          'cuda_available': torch.cuda.is_available(), 'gpu_count': torch.cuda.device_count(),
                          'bundle_root': str(ROOT)}, indent=2))
        if not torch.cuda.is_available():
            raise SystemExit('CUDA unavailable; inference requires a CUDA GPU')
    else:
        if args.input is None:
            parser.error('--input is required unless --check is used')
        if args.limit is not None and args.limit < 1:
            parser.error('--limit must be positive')
        run(args)


if __name__ == '__main__':
    main()
