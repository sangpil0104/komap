"""Read-only training readiness checks; never train or run inference."""
import ast
import importlib.metadata
import os
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

from experiment import ROOT, GPUS, read_json, sha, inventory, require_komap
from scheduler import validate_tasks, render


def dataset_check(root=ROOT):
    palette = np.asarray([[25,76,153], [76,178,76], [153,127,76], [204,204,204]], np.uint8)
    counts = {}
    for split, expected in [('train',70), ('valid',20)]:
        images = sorted((root / 'data' / split / 'images').glob('*_image.png'))
        masks = sorted((root / 'data' / split / 'masks').glob('*_mask.png'))
        if len(images) != expected or len(masks) != expected:
            raise ValueError(f'Expected {expected} {split} images and masks.')
        for path in tqdm(images, desc=f'Check {split} masks', unit='image'):
            with Image.open(path) as im:
                size = im.size
            target = path.parent.parent / 'masks' / path.name.replace('_image.png','_mask.png')
            with Image.open(target) as im:
                rgb = np.asarray(im.convert('RGB'))
                if im.size != size:
                    raise ValueError(f'Shape mismatch: {path}')
            if not np.any(np.all(rgb[...,None,:] == palette, axis=-1), axis=-1).all():
                raise ValueError(f'Unknown ground truth color: {target}')
        counts[split] = len(images)
    all_train = {p.name for p in (root / 'data/train/images').glob('*.png')}
    fold_counts, held_out = {}, []
    for fold in range(1,5):
        train = {p.name for p in (root / f'folds/train_fold{fold}/images').glob('*.png')}
        valid = {p.name for p in (root / f'folds/oof_fold{fold}/images').glob('*.png')}
        if train & valid or train | valid != all_train:
            raise ValueError(f'Fold {fold} is not a disjoint partition of train70.')
        groups = lambda names: {re.match(r'(A\d+)_', name).group(1) for name in names}
        if groups(train) & groups(valid):
            raise ValueError(f'Specimen leakage within fold {fold}.')
        for path in (root / 'folds').glob(f'*_fold{fold}/*/*.png'):
            if not path.is_file() or not path.resolve().is_relative_to((root / 'data/train').resolve()):
                raise ValueError(f'Invalid local fold link: {path}')
        fold_counts[str(fold)] = dict(train=len(train), heldout=len(valid))
        held_out.extend(valid)
    if len(held_out) != 70 or set(held_out) != all_train:
        raise ValueError('Each train image must be held out exactly once.')
    return dict(counts=counts, folds=fold_counts)


def check():
    require_komap()
    plan = read_json(ROOT / 'plan.json')
    validate_tasks(plan['tasks'])
    if plan['training_count'] != 14 or len(plan['tasks']) != 14 or plan['gpus'] != GPUS:
        raise ValueError('Expected fourteen tasks on GPUs 2~7.')
    frozen = read_json(ROOT / 'frozen/checkpoint_manifest.json')
    actual = {'weights/' + t['checkpoint'].removeprefix('{run}/outputs/') for t in plan['tasks']}
    if actual != {r['path'] for r in frozen['checkpoints']}:
        raise ValueError('Training outputs must match all fourteen exh14 checkpoints.')
    for name, digest in tqdm(read_json(ROOT / 'input_manifest.json').items(), desc='Verify data / ancestor weights', unit='file'):
        if sha(ROOT / name) != digest:
            raise ValueError(f'Input changed: {name}')
    for name, digest in read_json(ROOT / 'code_manifest.json').items():
        path = ROOT / name
        if sha(path) != digest:
            raise ValueError(f'Source/config changed: {name}')
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
    for name, digest in read_json(ROOT / 'frozen/recipe_manifest.json')['files_sha256'].items():
        if sha(ROOT / 'frozen' / name) != digest:
            raise ValueError(f'Original inference recipe changed: {name}')
    links = {str(p.relative_to(ROOT)):str(p.readlink()) for p in (ROOT/'folds').rglob('*.png') if p.is_symlink()}
    if links != read_json(ROOT/'fold_manifest.json'):
        raise ValueError('The recorded specimen fold membership changed.')
    devices = inventory()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices[g]['uuid'] for g in GPUS)
    import torch
    import segmentation_models_pytorch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 6:
        raise RuntimeError('The komap environment must see all six requested GPUs.')
    return dict(ok=True, training_started=False, inference_started=False, training_count=14,
                gpus=devices, python=sys.executable, data=dataset_check(),
                versions={p:importlib.metadata.version(p) for p in
                          ['torch','torchvision','segmentation-models-pytorch','timm','numpy','albumentations','tqdm']})
