"""OOF probabilities of the expert E for one fold, one file per (seed, image). Never reads masks.

    python predict_oof_expert.py --work WORK --fold K [--device cuda]

Uses seeds that finished both the R run (b4_329) and train_expert.py (hrnet392).
Writes work/oof_expert/foldK/seedS/<name>.npy (float16, main class order).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from expert_common import expert


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    seeds = sorted(int(p.name[4:]) for p in (a.work / 'runs_expert' / f'fold{a.fold}').glob('seed*')
                   if (p / 'hrnet392' / 'STAGE_DONE.json').exists()
                   and (a.work / 'runs' / f'fold{a.fold}' / p.name / 'b4_329' / 'STAGE_DONE.json').exists())
    images = sorted((a.work / 'folds' / f'fold{a.fold}' / 'oof' / 'images').glob('*_image.png'))
    for s in seeds:
        out = a.work / 'oof_expert' / f'fold{a.fold}' / f'seed{s}'
        out.mkdir(parents=True, exist_ok=True)
        for img in images:
            name = img.name.removesuffix('_image.png')
            if (out / f'{name}.npy').exists():
                continue
            p = expert(a.work / 'runs' / f'fold{a.fold}' / f'seed{s}' / 'b4_329',
                       a.work / 'runs_expert' / f'fold{a.fold}' / f'seed{s}' / 'hrnet392' / 'last.pt', img, a.device)
            np.save(out / f'{name}.tmp.npy', p.astype(np.float16))
            (out / f'{name}.tmp.npy').rename(out / f'{name}.npy')
            print(f'fold{a.fold} seed{s} {name}', flush=True)
    print('DONE seeds', seeds, flush=True)


if __name__ == '__main__':
    main()
