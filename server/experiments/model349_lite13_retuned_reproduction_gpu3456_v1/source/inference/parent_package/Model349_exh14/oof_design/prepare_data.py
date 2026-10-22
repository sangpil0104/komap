"""Step 1: freeze the labeled pool, specimen-group folds, inner selection splits and per-fold statistics.

    python prepare_data.py --train DIR [--valid DIR --confirm-valid-training-allowed] --work WORK

* All images of one specimen (A<id>_...) go to the same fold (4 folds, deterministic greedy balance).
* Inside every fold's training part a separate specimen-grouped *selection* split is held out; the
  vendored trainers pick checkpoints on it, never on the fold's OOF images.
* Brightness statistics / class intensity priors (calibration.json) are fitted per fold on that
  fold's training images only; work/all/calibration.json (all labeled images) is for test inference.
* Writes data_manifest.json with the SHA-256 of every input file and the frozen assignments.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from common import FOLDS, PALETTE, read_json, sha256, specimen, write_json


def pairs(root: Path):
    out = []
    for img in sorted((root / 'images').glob('*_image.png')):
        mask = root / 'masks' / img.name.replace('_image.png', '_mask.png')
        if not mask.is_file():
            raise FileNotFoundError(mask)
        out.append((img, mask))
    if not out:
        raise FileNotFoundError(f'no *_image.png under {root}/images')
    return out


def link(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        os.symlink(src.resolve(), dst)


def percentile_image(gray):
    flat = gray.reshape(-1)
    order = np.argsort(flat, kind='stable')
    result = np.empty(flat.shape, dtype=np.float32)
    result[order] = np.linspace(0.0, 1.0, len(flat), dtype=np.float32)
    return result.reshape(gray.shape)


def fit_calibration(items):
    """Same estimator as historical prior_representation_search.py, restricted to `items`."""
    means, stds = [], []
    zv, qv = [[] for _ in range(4)], [[] for _ in range(4)]
    for img, mask in items:
        g = np.asarray(Image.open(img).convert('L'), dtype=np.float32)
        means.append(float(g.mean())); stds.append(float(g.std()))
        z = (g - g.mean()) / max(float(g.std()), 1e-5)
        q = percentile_image(g)
        rgb = np.asarray(Image.open(mask).convert('RGB'))
        ids = np.full(rgb.shape[:2], -1, np.int8)
        for cid, color in enumerate(PALETTE):
            ids[np.all(rgb == color, axis=2)] = cid
        for arr, store in ((z, zv), (q, qv)):
            rng = np.random.default_rng(sum(img.name.encode('utf-8')))
            for cid in range(4):
                cur = arr[ids == cid]
                if len(cur):
                    store[cid].append(cur[rng.choice(len(cur), min(len(cur), 10000), replace=False)])
    pri = {}
    for key, store, floor in (('z', zv, .15), ('q', qv, .02)):
        pri[key] = [[float(np.concatenate(v).mean()), max(float(np.concatenate(v).std()), floor)] if v else [0.0, 1.0]
                    for v in store]
    return {'priors': pri, 'train_brightness_mean': float(np.mean(means)),
            'train_brightness_std': float(np.mean(stds)), 'source_train_image_count': len(items)}


def class_weights(items):
    """Inverse pixel frequency in the trainers' channel order [Al, Al3Ni, Primary Si, Eutectic Si], mean 1."""
    counts = np.zeros(4)
    for _, mask in items:
        rgb = np.asarray(Image.open(mask).convert('RGB'))
        for team, main in enumerate((2, 0, 1, 3)):
            counts[team] += np.count_nonzero(np.all(rgb == PALETTE[main], axis=2))
    inv = 1.0 / np.maximum(counts / counts.sum(), 1e-6)
    return (inv / inv.mean()).round(8).tolist()


def greedy_folds(groups: dict[str, list]):
    order = sorted(groups, key=lambda g: (-len(groups[g]), int(g[1:])))
    load, assign = [0] * FOLDS, {}
    for g in order:
        k = min(range(FOLDS), key=lambda i: (load[i], i))
        assign[g] = k + 1
        load[k] += len(groups[g])
    return assign


def inner_split(train_groups: dict[str, list], fold: int, frac: float):
    need = math.ceil(frac * sum(len(v) for v in train_groups.values()))
    order = sorted(train_groups, key=lambda g: hashlib.sha1(f'{g}-{fold}'.encode()).hexdigest())
    chosen, n = [], 0
    for g in order:
        if n >= need:
            break
        chosen.append(g); n += len(train_groups[g])
    return set(chosen)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', type=Path, required=True, help='dir with images/ and masks/ (official train)')
    ap.add_argument('--valid', type=Path, help='dir with images/ and masks/ (official valid) to merge')
    ap.add_argument('--confirm-valid-training-allowed', action='store_true',
                    help='required with --valid: you confirmed the competition rules allow training on valid labels')
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--inner-frac', type=float, default=0.15)
    ap.add_argument('--fold-manifest', type=Path, help='Model303 progress.json: use its specimen folds (held-out groups) instead of '
                    'the greedy rule, so the published B4/E fold models are valid OOF predictors for the same folds')
    a = ap.parse_args()
    if a.valid and not a.confirm_valid_training_allowed:
        ap.error('--valid merges the official validation labels into training. Confirm the rules allow it and '
                 'pass --confirm-valid-training-allowed, or omit --valid to use train only.')
    if (a.work / 'data_manifest.json').exists():
        ap.error(f'{a.work} already prepared; use a new --work directory')

    labeled = [('train', i, m) for i, m in pairs(a.train)] + ([('valid', i, m) for i, m in pairs(a.valid)] if a.valid else [])
    names = [i.name for _, i, _ in labeled]
    if len(names) != len(set(names)):
        raise ValueError('duplicate image names across splits')
    groups = defaultdict(list)
    for src, i, m in labeled:
        groups[specimen(i.name)].append((src, i, m))
    if a.fold_manifest:
        assign = {}
        for rec in read_json(a.fold_manifest)['completed_folds']:
            for g in rec['heldout_groups']:
                assert g not in assign, g
                assign[g] = int(rec['fold'])
        missing = sorted(set(groups) - set(assign))
        if missing:
            raise ValueError(f'specimens missing from the fold manifest: {missing}')
    else:
        assign = greedy_folds(groups)

    rows = []
    for src, i, m in labeled:
        rows.append({'name': i.name.removesuffix('_image.png'), 'source_split': src, 'specimen': specimen(i.name),
                     'fold': assign[specimen(i.name)], 'image_sha256': sha256(i), 'mask_sha256': sha256(m)})
    folds = {}
    for k in range(1, FOLDS + 1):
        oof = [(i, m) for s, i, m in labeled if assign[specimen(i.name)] == k]
        train_groups = {g: v for g, v in groups.items() if assign[g] != k}
        inner = inner_split(train_groups, k, a.inner_frac)
        fit = [(i, m) for g, v in train_groups.items() if g not in inner for _, i, m in v]
        sel = [(i, m) for g in sorted(inner) for _, i, m in train_groups[g]]
        full = fit + sel
        base = a.work / 'folds' / f'fold{k}'
        for i, m in fit:
            link(i, base / 'sel/train/images'); link(m, base / 'sel/train/masks')
        for i, m in sel:
            link(i, base / 'sel/valid/images'); link(m, base / 'sel/valid/masks')
        for i, m in full:
            link(i, base / 'full/train/images'); link(m, base / 'full/train/masks')
            link(i, base / 'b4cv/train_all/images'); link(m, base / 'b4cv/train_all/masks')
        for i, _ in oof:                      # OOF images only: their masks are never linked here
            link(i, base / 'oof/images')
        for d in ('sel/test/images', 'full/valid/images', 'full/valid/masks', 'full/test/images'):
            (base / d).mkdir(parents=True, exist_ok=True)
        write_json(base / 'calibration.json', fit_calibration(full))
        write_json(base / 'class_weights.json', {'order': ['Al', 'Al3Ni', 'Primary Si', 'Eutectic Si'],
                                                 'weights': class_weights(full)})
        folds[k] = {'oof_images': len(oof), 'train_images': len(full), 'fit_images': len(fit),
                    'selection_images': len(sel), 'selection_specimens': sorted(inner),
                    'oof_specimens': sorted(g for g in groups if assign[g] == k)}
    allp = [(i, m) for _, i, m in labeled]
    for i, m in allp:
        link(i, a.work / 'all/train/images'); link(m, a.work / 'all/train/masks')
    write_json(a.work / 'all/calibration.json', fit_calibration(allp))
    write_json(a.work / 'all/class_weights.json', {'weights': class_weights(allp)})
    write_json(a.work / 'data_manifest.json', {
        'valid_merged': bool(a.valid), 'confirm_valid_training_allowed': bool(a.confirm_valid_training_allowed),
        'images': len(labeled), 'specimens': len(groups), 'folds': folds, 'inner_frac': a.inner_frac,
        'fold_rule': (f'held-out groups of {a.fold_manifest.name}' if a.fold_manifest else
                      'specimens sorted by (-image count, id); each to the fold with fewest images (ties: lowest fold)'),
        'labeled': rows})
    print({k: (v['train_images'], v['oof_images']) for k, v in folds.items()}, 'specimens', len(groups))


if __name__ == '__main__':
    main()
