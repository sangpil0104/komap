"""Exact fixed-route evaluation; never trains or changes the source checkpoints."""
from __future__ import annotations
import gc
import json
import pickle
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import yaml
from common import (ROOT, BASE, EXPECTED_SCORE, PALETTE, CLASS_NAMES,
                    COMPONENT_DIRS, key, read_json, write_json, class_ious, sha)

sys.path.insert(0, str(ROOT / 'frozen'))
import infer as frozen

def baseline_cache():
    return ROOT / 'cache'

def archive_to_npy(archive, out, team_order=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    expected = set(read_json(baseline_cache() / 'manifest.json')['names'])
    found = set()
    with np.load(archive) as values:
        for name in values.files:
            canonical = key(name)
            if canonical in found: raise ValueError('Duplicate probability key')
            found.add(canonical)
            value = values[name]
            if value.ndim != 3 or value.shape[0] != 4 or not np.isfinite(value).all():
                raise ValueError(f'Invalid probability array: {archive}: {name}')
            if team_order: value = value[[1,2,0,3]]
            # Preserve the stored quantization; do not add another fp16 rounding.
            np.save(out / f'{canonical}.npy', value)
    if found != expected: raise ValueError(f'Probability image set differs: {archive}')

def prepare_cache():
    cache = baseline_cache()
    if (cache / 'complete.json').exists():
        return read_json(cache / 'complete.json')
    if cache.exists():
        raise RuntimeError(f'Incomplete cache preserved at {cache}; inspect before retrying')
    cache.mkdir()
    images = sorted((BASE / 'data/valid/images').glob('*_image.png'))
    names = [key(p.name) for p in images]
    assert len(names) == 20
    ema_names = []
    with (BASE / 'outputs/model183_metric_calibration/entries.pkl').open('rb') as stream:
        entries = pickle.load(stream)
    assert len(entries) == 20
    (cache / 'base').mkdir(); (cache / 'ema_inputs').mkdir()
    palette = np.array(PALETTE, np.uint8)
    for name, path, item in tqdm(list(zip(names, images, entries)), desc='Prepare frozen route', unit='image'):
        target = frozen.decode_gt(BASE / 'data/valid/masks' / f'{name}_mask.png', palette)
        assert np.array_equal(item['target'], target), name
        selected = {k: item[k] for k in ['mixed','extra','z','q','prior_z_stats','prior_q_stats','params','target']}
        with (cache / 'base' / f'{name}.pkl').open('wb') as f: pickle.dump(selected, f)
        gray = np.array(Image.open(path).convert('RGB'))[...,0]
        mean = float(gray.mean())
        if 161.3 <= mean < 162.0 or 155.2 <= mean < 155.7:
            ema_names.append(name)
            with (cache / 'ema_inputs' / f'{name}.pkl').open('wb') as f:
                pickle.dump({k: item[k] for k in ['branches','pair','extra','params']}, f)
    del entries; gc.collect()
    write_json(cache / 'manifest.json', dict(names=names, ema_names=ema_names,
               source_run=str(BASE), reference_image_miou=EXPECTED_SCORE))
    for component in tqdm(['269','322','326','329','336','347'], desc='Convert fixed probabilities'):
        parent = BASE / 'outputs' / COMPONENT_DIRS[component]
        archive = (parent / 'valid_probs.npz' if component == '269' else
                   parent / 'valid_ensemble/probabilities.npz' if component in ['322','326','329'] else
                   parent / 'valid_eval/valid_probabilities.npz')
        archive_to_npy(archive, cache / component, component != '269')
    result = evaluate({}, cache / 'baseline', comparisons=True)
    if abs(result['image_miou'] - EXPECTED_SCORE) > 1e-12:
        raise AssertionError(f'Baseline mismatch: {result["image_miou"]} != {EXPECTED_SCORE}')
    # Bind the prepared cache to every byte read by later experiments.
    manifest = {str(p.relative_to(cache)): sha(p) for p in sorted(cache.rglob('*')) if p.is_file()}
    write_json(cache / 'complete.json', dict(image_miou=result['image_miou'], files=manifest))
    return read_json(cache / 'complete.json')

def comparison(name, prediction, target, score, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    palette = np.array(PALETTE, np.uint8)
    original = Image.open(BASE / 'data/valid/images' / f'{name}_image.png').convert('RGB')
    errors = np.full((*target.shape, 3), 255, np.uint8)
    wrong = prediction != target; errors[wrong] = palette[target[wrong]]
    tiles = [original, Image.fromarray(errors), Image.fromarray(palette[prediction]), Image.fromarray(palette[target])]
    w, h = 640, 480
    canvas = Image.new('RGB', (w*2, h*2+105), 'white')
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 20) if Path(font_path).exists() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((12,8), f'{name} | image mIoU: {100*score:.6f}% | 100%-image mIoU: {100*(1-score):.6f}%', fill='black', font=font)
    labels = ['Original image', 'Errors: white = correct; color = ground truth', 'Predicted mask', 'Ground-truth mask']
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        x, y = (i%2)*w, 45+(i//2)*(h+25)
        draw.text((x+8,y), label, fill='black',font=font)
        tile.thumbnail((w,h), Image.Resampling.LANCZOS if i==0 else Image.Resampling.NEAREST)
        canvas.paste(tile,(x+(w-tile.width)//2,y+25+(h-tile.height)//2))
    canvas.save(directory / f'{name}.png')

def evaluate(selection, out=None, comparisons=False, quiet=False):
    """selection maps component -> npy directory; '123' contains changed base mixed arrays."""
    cache = baseline_cache()
    manifest = read_json(cache / 'manifest.json')
    config = yaml.safe_load((ROOT / 'frozen/configs/model349.yaml').read_text())
    rows = []
    if out:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        (out / 'predicted_labels').mkdir(exist_ok=True)
    for name in tqdm(manifest['names'], desc='Model349 image mIoU', unit='image', mininterval=10, disable=quiet):
        with (cache / 'base' / f'{name}.pkl').open('rb') as f: item = pickle.load(f)
        if '123' in selection and (Path(selection['123']) / f'{name}.npy').exists():
            item['mixed'] = np.load(Path(selection['123']) / f'{name}.npy')
        sources = {c: np.load(Path(selection.get(c, cache/c)) / f'{name}.npy').astype(np.float32)
                   for c in ['269','322','326','329','336','347']}
        pred = frozen.predict_fixed(item, sources, config)
        ious = class_ious(pred, item['target']); score = float(ious.mean())
        rows.append(dict(image=name, image_miou=score, class_iou=ious.tolist()))
        if out:
            Image.fromarray(pred).save(out / 'predicted_labels' / f'{name}.png')
            if comparisons: comparison(name, pred, item['target'], score, out/'valid_mask_comparison')
    result = dict(image_miou=float(np.mean([r['image_miou'] for r in rows])),
                  metric='image_mean_four_classes_empty0', image_count=len(rows),
                  class_order=CLASS_NAMES, images=rows,
                  selection={k:str(v) for k,v in selection.items()})
    if out:
        write_json(out/'metrics.json', result)
        (out/'report.md').write_text(f'# Model349 candidate\n\nImage mIoU: **{result["image_miou"]*100:.6f}%**\n\n'
                    'Validation 20 images; four classes per image; empty union = 0.\n'
                    'Comparisons: original / errors / prediction / ground truth.\n'
                    'Correct pixels are white; incorrect pixels use the ground-truth color.\n')
    return result

def ema_probabilities(checkpoint, out):
    """Recompute only the two images routed through Model123, retaining fixed branches."""
    import torch
    from model183_metric_calibration import make_mixed
    from tent_style_search import adapt_model
    from style_tta_search import predict_array, transform_image
    torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)
    cache = baseline_cache(); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    calibration = read_json(ROOT/'frozen/configs/calibration.json')
    for name in tqdm(read_json(cache/'manifest.json')['ema_names'], desc='EMA routed images'):
        raw = np.array(Image.open(BASE/'data/valid/images'/f'{name}_image.png').convert('RGB'))
        device = torch.device('cuda:0')
        model = adapt_model(str(checkpoint), raw, device, True, 5, 1e-5)
        style = transform_image(raw, 'standardize', calibration['train_brightness_mean'], calibration['train_brightness_std'])
        high768 = [predict_array(model, x, 768, 576, device, True, 1, 'd4', 'gaussian') for x in [raw, style]]
        high1024 = [predict_array(model, x, 1024, 768, device, True, 1, 'd4', 'gaussian') for x in [raw, style]]
        with (cache/'ema_inputs'/f'{name}.pkl').open('rb') as f: item=pickle.load(f)
        high = [.8*high768[i]+.2*high1024[i] for i in (0,1)]
        np.save(out/f'{name}.npy', make_mixed(item, high, gamma=.4))
        del model, high, high768, high1024, item
        gc.collect(); torch.cuda.empty_cache()
