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


def expert_check():
    config=read_json(ROOT/'fusion_source/model397_reproduction_bundle/config/model397_group_folds.json')
    train_names={p.stem.removesuffix('_image') for p in (ROOT/'data/train/images').glob('*.png')}
    group=lambda name:re.match(r'A\d+',name).group()
    counts={}
    for row in config['folds']:
        fold=row['fold'];held_groups=set(row['heldout_groups'])
        held={n for n in train_names if group(n) in held_groups}
        trained=train_names-held
        counts[str(fold)]=dict(train=len(trained),heldout=len(held))
        def names(directory):return {p.stem.removesuffix('_image') for p in directory.glob('*.png')}
        if names(ROOT/f'expert_splits/model329_cv/train_fold{fold}/images')!=trained or names(ROOT/f'expert_splits/model329_cv/oof_fold{fold}/images')!=held:
            raise ValueError('Expert B4 fold mismatch.')
        for seed in [1,2]:
            base=ROOT/f'expert_splits/model392_cv/seed{seed}_fold{fold}'
            if names(base/'train/images')!=trained or names(base/'valid/images')!=held:raise ValueError('Expert HRNet fold mismatch.')
    for path in (ROOT/'expert_splits').rglob('*.png'):
        if not path.is_file() or not path.resolve().is_relative_to((ROOT/'data/train').resolve()):
            raise ValueError(f'Expert fold points outside train70: {path}')
    return dict(images=70,specimen_groups=59,folds=counts,valid_test_in_training=False)

