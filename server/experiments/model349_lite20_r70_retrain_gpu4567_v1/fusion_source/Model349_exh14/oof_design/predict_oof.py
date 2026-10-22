"""Step 3: OOF probabilities of one fold, one file per (seed, image). Never reads masks.

    python predict_oof.py --work WORK --fold K [--seeds 0,1,...] [--device cuda]

For every trained seed of fold K and every OOF image of that fold, writes
work/oof/foldK/seedS/<name>.npz with float16 member probabilities (main class order
[Al3Ni, Primary Si, Al, Eutectic Si]) and work/oof/meta/<name>.npz/.json with GT-free image
quantities. Priors and the brightness-standardization reference come from the fold's own
calibration.json (fitted on that fold's training images only).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import members as M
from common import read_json, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seeds', default='', help='default: every runs/foldK/seed* with all stages finished')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    cal = read_json(a.work / 'folds' / f'fold{a.fold}' / 'calibration.json')
    runs_root = a.work / 'runs' / f'fold{a.fold}'
    seeds = [int(s) for s in a.seeds.split(',')] if a.seeds else sorted(
        int(p.name[4:]) for p in runs_root.glob('seed*') if (p / 'hrnet347' / 'STAGE_DONE.json').exists())
    images = sorted((a.work / 'folds' / f'fold{a.fold}' / 'oof' / 'images').glob('*_image.png'))
    meta_dir = a.work / 'oof' / 'meta'
    for img in images:
        name = img.name.removesuffix('_image.png')
        if not (meta_dir / f'{name}.json').exists():
            m = M.image_meta(img, cal)
            meta_dir.mkdir(parents=True, exist_ok=True)
            np.savez(meta_dir / f'{name}.npz', z=m.pop('z'), q=m.pop('q'))
            write_json(meta_dir / f'{name}.json', dict(m, fold=a.fold))
    for seed in seeds:
        paths = M.member_paths(runs_root / f'seed{seed}')
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(f'fold{a.fold} seed{seed} missing members: {missing}')
        out = a.work / 'oof' / f'fold{a.fold}' / f'seed{seed}'
        out.mkdir(parents=True, exist_ok=True)
        for img in images:
            name = img.name.removesuffix('_image.png')
            if (out / f'{name}.npz').exists():
                continue
            params = read_json(meta_dir / f'{name}.json')['params']
            arr = {k: M.b4(paths[k], img, a.device) for k in M.B4_STAGES}
            arr.update({k: M.hrnet(paths[k], img, a.device) for k in M.HR_STAGES})
            arr['n269'] = M.native269(paths['n269'], img, a.device)
            arr.update({k: M.high_preaux(paths[k], img, cal, params, a.device) for k in paths if k.startswith('high_')})
            np.savez(out / f'{name}.tmp.npz', **{k: v.astype(np.float16) for k, v in arr.items()})
            (out / f'{name}.tmp.npz').rename(out / f'{name}.npz')
            print(f'fold{a.fold} seed{seed} {name}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
