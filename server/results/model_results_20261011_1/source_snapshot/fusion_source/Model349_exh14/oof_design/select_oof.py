"""Step 4: choose the fusion configuration on OOF and report the three kinds of score.

    python select_oof.py --work WORK [--workers 32] [--passes 3] [--allow-partial]

1. Seed-averaged OOF probabilities are cached per image (work/oof/avg/<name>.npz).
2. Selection rule (fixed): start from the exh14 configuration and run coordinate ascent over the fixed
   grid in fusion.GRID / fusion.ORDER; a coordinate moves only if the mean image-wise mIoU of the
   selection images improves by > 1e-6; ties keep the current value; at most --passes passes.
3. Scores written to work/select/report.json:
   * dev_oof          : configuration selected on all OOF images, scored on the same images
                          (development score of the final configuration)
   * outer_estimate     : for each outer fold j the rule is re-run on the other folds' OOF images and
                          the result scored on fold j; pooled over all images
                          (performance estimate that includes the selection step)
   * per_seed           : the final configuration applied to each seed's own OOF predictions;
                          mean and SD of the per-seed OOF scores (retraining reproducibility),
                          distinct from the seed-averaged ensemble score (dev_oof)
Masks are read only here, for scoring; predictions never use them.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

import fusion as F
from common import FOLDS, decode_mask, image_miou, manifest, read_json, write_json

WORK: Path = None
_cache = {}


def load(name: str, seed=None):
    tag = (name, seed)
    if tag in _cache:
        return _cache[tag]
    _cache.clear()
    meta = read_json(WORK / 'oof' / 'meta' / f'{name}.json')
    zq = np.load(WORK / 'oof' / 'meta' / f'{name}.npz')
    src = WORK / 'oof' / 'avg' / f'{name}.npz' if seed is None else WORK / 'oof' / f'fold{meta["fold"]}' / f'seed{seed}' / f'{name}.npz'
    arr = {k: v for k, v in np.load(src).items()}
    arr['z'], arr['q'] = zq['z'], zq['q']
    gt = decode_mask(WORK / 'all' / 'train' / 'masks' / f'{name}_mask.png')
    w = F.route.bilateral_weights(arr['z'])
    _cache[tag] = (arr, meta, gt, w)
    return _cache[tag]


def score_task(args):
    name, cfgs, seed = args
    arr, meta, gt, w = load(name, seed)
    return name, [image_miou(F.predict(arr, meta, c, w), gt)[0] for c in cfgs]


def label_task(args):
    name, cfg, seed, out = args
    arr, meta, gt, w = load(name, seed)
    pred = F.predict(arr, meta, cfg, w)
    Image.fromarray(pred).save(Path(out) / f'{name}.png')
    return name, image_miou(pred, gt)


class Scorer:
    def __init__(self, names, pool):
        self.names, self.pool, self.table = names, pool, {}

    def ensure(self, cfgs, subset):
        need = {}
        for c in cfgs:
            k = F.key(c)
            row = self.table.setdefault(k, {})
            for n in subset:
                if n not in row:
                    need.setdefault(n, []).append(c)
        tasks = [(n, cs, None) for n, cs in need.items()]
        for n, vals in self.pool.imap_unordered(score_task, tasks):
            for c, v in zip(need[n], vals):
                self.table[F.key(c)][n] = v

    def mean(self, cfg, subset):
        row = self.table[F.key(cfg)]
        return float(np.mean([row[n] for n in subset]))


def select(scorer, subset, passes, log):
    cur = json.loads(F.key(F.DEFAULT))
    scorer.ensure([cur], subset)
    best = scorer.mean(cur, subset)
    for p in range(passes):
        moved = False
        for coord in F.ORDER:
            cands = F.neighbours(cur, coord)
            scorer.ensure(cands, subset)
            top = max(cands, key=lambda c: scorer.mean(c, subset))   # max keeps the first of equal scores
            if scorer.mean(top, subset) > best + 1e-6:
                cur, best, moved = top, scorer.mean(top, subset), True
        log(f'  pass {p + 1}: {best * 100:.4f} on {len(subset)} images')
        if not moved:
            break
    return cur, best


def avg_cache(work, names, fold_of, allow_partial):
    seeds_by_fold = {}
    for k in range(1, FOLDS + 1):
        seeds_by_fold[k] = sorted(int(p.name[4:]) for p in (work / 'oof' / f'fold{k}').glob('seed*'))
    out = work / 'oof' / 'avg'
    out.mkdir(parents=True, exist_ok=True)
    kept = []
    for n in names:
        k = fold_of[n]
        files = [work / 'oof' / f'fold{k}' / f'seed{s}' / f'{n}.npz' for s in seeds_by_fold[k]]
        files = [f for f in files if f.exists()]
        if not files:
            if allow_partial:
                continue
            raise FileNotFoundError(f'no OOF prediction for {n} (fold {k})')
        kept.append(n)
        dst = out / f'{n}.npz'
        if dst.exists() and read_json(out / f'{n}.json')['seeds'] == len(files):
            continue
        acc = None
        for f in files:
            d = np.load(f)
            acc = {k2: d[k2].astype(np.float32) for k2 in d.files} if acc is None else {k2: acc[k2] + d[k2] for k2 in acc}
        np.savez(dst, **{k2: (v / len(files)).astype(np.float16) for k2, v in acc.items()})
        write_json(out / f'{n}.json', {'seeds': len(files)})
    common = sorted(set.intersection(*[set(v) for v in seeds_by_fold.values() if v])) if any(seeds_by_fold.values()) else []
    return kept, seeds_by_fold, common


def main():
    global WORK
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--allow-partial', action='store_true', help='smoke runs: use only images that have OOF predictions')
    a = ap.parse_args()
    WORK = a.work
    man = manifest(a.work)
    fold_of = {r['name']: r['fold'] for r in man['labeled']}
    names, seeds_by_fold, common_seeds = avg_cache(a.work, sorted(fold_of), fold_of, a.allow_partial)
    out = a.work / 'select'
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / 'select.log').open('a')

    def log(msg):
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()

    folds_present = sorted({fold_of[n] for n in names})
    log(f'OOF images {len(names)}; folds {folds_present}; seeds per fold {seeds_by_fold}')
    with Pool(a.workers, maxtasksperchild=200) as pool:
        sc = Scorer(names, pool)
        log('dev selection on all OOF images')
        final, dev = select(sc, names, a.passes, log)
        outer = {}
        held = {}
        for j in folds_present:
            inner = [n for n in names if fold_of[n] != j]
            test = [n for n in names if fold_of[n] == j]
            if not inner:
                log(f'outer fold {j}: no other fold available (partial run), skipped')
                continue
            log(f'outer fold {j}: select on {len(inner)} images, evaluate {len(test)}')
            cfg_j, _ = select(sc, inner, a.passes, log)
            sc.ensure([cfg_j], test)
            row = sc.table[F.key(cfg_j)]
            held.update({n: row[n] for n in test})
            outer[j] = {'config': cfg_j, 'heldout_miou': sc.mean(cfg_j, test), 'images': len(test)}
        # labels for the final and outer configurations (seed-averaged), consumed by the fusion805 stage
        for tag, cfg in [('final', final)] + [(f'outer{j}', v['config']) for j, v in outer.items()]:
            d = out / 'labels' / tag
            d.mkdir(parents=True, exist_ok=True)
            write_json(d / 'config.json', cfg)
            list(pool.imap_unordered(label_task, [(n, cfg, None, str(d)) for n in names]))
        per_seed = {}
        for s in common_seeds:
            d = out / 'labels' / f'final_seed{s}'
            d.mkdir(parents=True, exist_ok=True)
            res = dict(pool.imap_unordered(label_task, [(n, final, s, str(d)) for n in names]))
            per_seed[s] = float(np.mean([res[n][0] for n in names]))
    ps = np.array(list(per_seed.values())) * 100 if per_seed else np.array([])
    report = {
        'metric': 'mean over images of the 4-class image mIoU; empty union = 0',
        'images': len(names), 'complete': len(names) == len(fold_of), 'valid_merged': man['valid_merged'],
        'dev_oof': dev * 100,
        'default_exh14_config_oof': sc.mean(json.loads(F.key(F.DEFAULT)), names) * 100,
        'outer_estimate': float(np.mean(list(held.values())) * 100) if held else None,
        'outer_folds': {str(j): dict(v, heldout_miou=v['heldout_miou'] * 100) for j, v in outer.items()},
        'per_seed': {'seeds': list(per_seed), 'scores': ps.round(4).tolist(),
                     'mean': float(ps.mean()) if len(ps) else None,
                     'sd': float(ps.std(ddof=1)) if len(ps) > 1 else None},
        'final_config': final, 'search': {'grid': F.GRID, 'order': F.ORDER, 'passes': a.passes},
        'per_image_dev': {n: sc.table[F.key(final)][n] * 100 for n in names},
    }
    write_json(out / 'report.json', report)
    write_json(out / 'final_config.json', final)
    log(json.dumps({k: report[k] for k in ('images', 'dev_oof', 'default_exh14_config_oof', 'outer_estimate')}))
    log(f"per-seed OOF mean {report['per_seed']['mean']} sd {report['per_seed']['sd']}")


if __name__ == '__main__':
    main()
