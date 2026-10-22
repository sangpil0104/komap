"""Evaluate candidates with the unmodified GitHub lite12 inference functions."""
from __future__ import annotations
import copy
import pickle
import random
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import yaml
from common import ROOT, BASE, HIGH, COMPONENT_DIRS, PALETTE, CLASS_NAMES, read_json, write_json, class_ious, sha

sys.path.insert(0, str(ROOT/'frozen'))
import infer_lite as frozen

def samples():
    return read_json(ROOT/'data_manifest.json')['images']

def image_family(row):
    return row['high_family']

def make_weight_view(directory, replacements):
    """Point at immutable FP16 files; never modify the source model directory."""
    directory.mkdir()
    for row in read_json(ROOT/'local_weights.json')['checkpoints']:
        relative = row['relative']
        source = Path(replacements.get(relative, ROOT/relative)).resolve()
        target = directory/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
        if '/fold' in relative:
            (target.parent/'args.json').symlink_to((ROOT/'frozen'/Path(relative).parent/'args.json').resolve())
    return directory/'weights'

def predict(component, work, replacements=None, family=None):
    import cv2
    import torch
    torch.set_num_threads(2); cv2.setNumThreads(2)
    # Inference/TENT randomness is fixed independently of each training seed.
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    work.mkdir()
    config = copy.deepcopy(frozen.load_config(ROOT/'frozen/configs/model349_lite.yaml'))
    config['weights_dir'] = str(make_weight_view(work/'weight_view', replacements or {}))
    rows = [r for r in samples() if family is None or image_family(r) == family]
    if not rows: raise ValueError(f'No validation images routed through {family}')
    write_json(work/'input_manifest.json', dict(config=config, images=rows))
    (work/'probabilities').mkdir()
    frozen.worker(work, component)
    return work/'probabilities'/component

def baseline_selection(run):
    return {c: read_json(run/'completed'/f'baseline_{c}.json')['probabilities']
            for c in ['326','329','base','336','269']}

def comparison(row, prediction, target, score, directory):
    directory.mkdir(parents=True, exist_ok=True)
    palette = np.array(PALETTE, np.uint8)
    original = Image.open(row['path']).convert('RGB')
    errors = np.full((*target.shape,3),255,np.uint8)
    wrong = prediction != target; errors[wrong] = palette[target[wrong]]
    tiles = [original,Image.fromarray(errors),Image.fromarray(palette[prediction]),Image.fromarray(palette[target])]
    w,h = 720,540
    canvas = Image.new('RGB',(2*w,2*h+110),'white')
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path),19) if font_path.exists() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((12,8),f'{row["name"]} | image mIoU: {score*100:.6f}% | 100%-image mIoU: {(1-score)*100:.6f}%',fill='black',font=font)
    labels = ['Original image','Errors: white = correct; color = ground truth','Predicted mask','Ground-truth mask']
    for i,(tile,label) in enumerate(zip(tiles,labels)):
        x,y = (i%2)*w,45+(i//2)*(h+30)
        draw.text((x+8,y),label,fill='black',font=font)
        tile.thumbnail((w,h),Image.Resampling.LANCZOS if i==0 else Image.Resampling.NEAREST)
        canvas.paste(tile,(x+(w-tile.width)//2,y+25+(h-tile.height)//2))
    canvas.save(directory/f'{row["name"]}.png')

def evaluate(run, selection, out=None, comparisons=False, quiet=False):
    import cv2
    cv2.setNumThreads(2)
    baseline = baseline_selection(run)
    config = frozen.load_config(ROOT/'frozen/configs/model349_lite.yaml')
    rows = []
    if out:
        out = Path(out); out.mkdir(parents=True,exist_ok=True)
        (out/'predicted_labels').mkdir(exist_ok=True)
    for row in tqdm(samples(),desc='Lite12 image mIoU',unit='image',mininterval=5,disable=quiet):
        name = row['name']
        base = Path(selection.get(row['high_family'],baseline['base']))/f'{name}.pkl'
        with base.open('rb') as stream: item = pickle.load(stream)
        sources = {}
        for component in ['269','326','329','336']:
            value = np.load(Path(selection.get(component,baseline[component]))/f'{name}.npy').astype(np.float32)
            if value.shape != (4,row['size'][1],row['size'][0]) or not np.isfinite(value).all():
                raise ValueError(f'Invalid probabilities: {component}/{name}')
            sources[component] = value
        pred = frozen.predict_fixed(item,sources,config).astype(np.uint8)
        target = frozen.decode_gt(row['gt'],np.array(PALETTE,np.uint8))
        ious = class_ious(pred,target); score = float(ious.mean())
        rows.append(dict(image=name,image_miou=score,class_iou=ious.tolist()))
        if out:
            Image.fromarray(pred).save(out/'predicted_labels'/f'{name}.png')
            if comparisons: comparison(row,pred,target,score,out/'valid_mask_comparison')
    if len(rows) != 20: raise ValueError('Exactly 20 validation images required')
    result = dict(image_miou=float(np.mean([r['image_miou'] for r in rows])),image_count=20,
                  metric='image_mean_four_classes_empty0',class_order=CLASS_NAMES,images=rows,
                  selection={k:str(v) for k,v in selection.items()})
    if out:
        write_json(out/'metrics.json',result)
        (out/'report.md').write_text(f'# Model349 lite12 candidate\n\nImage mIoU: **{100*result["image_miou"]:.6f}%**\n\n'
            'Validation 20 images, four classes per image, empty union = 0.\n'
            'Fixed lite12 route, FP16 stored checkpoints. This is a validation-selected development result.\n')
    return result

def artifact_hashes(paths):
    result = {}
    for root in paths:
        root = Path(root)
        for path in sorted(root.rglob('*')) if root.is_dir() else [root]:
            if path.is_file(): result[str(path)] = sha(path)
    return result
