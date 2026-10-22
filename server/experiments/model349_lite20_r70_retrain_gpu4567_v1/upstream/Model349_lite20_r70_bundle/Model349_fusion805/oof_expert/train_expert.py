"""Train the E-only member (width-normalized HRNet392) for one (fold, seed) on the fold's training images.

    python train_expert.py --work WORK --fold K --seed S [--smoke]

Uses folds/foldK/full (the fold's OOF images are absent). Seed S trains HRNet392 with seed S+1
(S=0,1 reproduce the historical seeds 1,2). --rng-mode per_item makes crops/augmentation a function of
(seed, epoch, sample index). B4-329 is not trained here: the R run's b4_329 checkpoint is reused.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from expert_common import MODEL392, OOF_DESIGN  # noqa: F401  (also sets sys.path)
from common import vendor_env, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--python', default=sys.executable)
    a = ap.parse_args()
    data = a.work / 'folds' / f'fold{a.fold}' / 'full'
    out = a.work / 'runs_expert' / f'fold{a.fold}' / f'seed{a.seed}' / 'hrnet392'
    if (out / 'STAGE_DONE.json').exists():
        print('skip (done)'); return
    if out.exists() and any(out.iterdir()):
        sys.exit(f'{out} exists without STAGE_DONE.json; keep it and use a new --work')
    out.mkdir(parents=True, exist_ok=True)
    epochs, spi = ('1', '1') if a.smoke else ('70', '12')
    cmd = [a.python, '-u', str(MODEL392 / 'train.py'), '--data-root', str(data), '--output-dir', str(out),
           '--architecture', 'UnetPlusPlus', '--encoder-name', 'tu-hrnet_w48', '--encoder-weights', 'imagenet',
           '--patch-size', '512', '--input-scale', '2.0', '--normalize-width', '--samples-per-image', spi,
           '--batch-size', '4', '--accumulation-steps', '2', '--num-workers', '4', '--epochs', epochs, '--lr', '3e-4',
           '--encoder-lr-mult', '1.0', '--weight-decay', '1e-4', '--boundary-head-weight', '1.0',
           '--seed', str(a.seed + 1), '--rng-mode', 'per_item', '--val-every', '0']
    env = vendor_env()
    env['PYTHONPATH'] = str(MODEL392) + ':' + env['PYTHONPATH']
    t0 = time.time()
    with (out / 'stage.log').open('w') as log:
        code = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(MODEL392))
    if code or not (out / 'last.pt').is_file():
        sys.exit(f'HRNet392 training failed (exit {code}); see {out / "stage.log"}')
    write_json(out / 'STAGE_DONE.json', {'command': cmd, 'seconds': time.time() - t0, 'smoke': a.smoke})
    print('DONE', out)


if __name__ == '__main__':
    main()
