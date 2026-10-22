"""Offline scoring of further reductions on top of the team's lite14 bundle (run from lite14 bundle root).

Inputs: probe/ (B4 per-fold views, HRNet, 269, base q) and probe_high/ (pre-aux prob per high-res candidate).
All variants keep lite14's fusion + post-processing code (predict_fixed with 'preaux').
"""
import copy, itertools, json, os, re, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path.cwd(); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT))
import yaml
from infer_lite import predict_fixed, decode_gt

PROBE, HIGH, GT, OUT = map(Path, sys.argv[1:5])
CONFIG = yaml.safe_load((ROOT / 'configs/model349_lite.yaml').read_text())
PALETTE = np.array(CONFIG['palette'], dtype=np.uint8)
NAMES = sorted(p.stem for p in HIGH.glob('*.npz'))
CANDS = ['restart51', 'model109', 'seed456', 'pairwise', 'ema']
LITE14_MAP = {'restart51': 'restart51', 'model109': 'model109', 'seed456': 'seed456', 'pairwise': 'pairwise', 'ema': 'restart51'}
BASE = dict(route=LITE14_MAP, folds326=(1, 2, 3, 4), folds329=(1, 2, 3, 4), drop=(), al3ni329=None)


def load(name):
    z = {c: dict(np.load(PROBE / c / f'{name}.npz')) for c in ('326', '329', '336', '269')}
    z['high'] = dict(np.load(HIGH / f'{name}.npz'))
    z['meta'] = json.loads((HIGH / f'{name}.json').read_text())
    z['q'] = np.load(PROBE / 'base' / f'{name}.npz')['q']
    return z


def build(z, spec):
    meta = z['meta']
    src = spec['route'][meta['routed']]
    item = {'preaux': z['high'][src].astype(np.float32), 'z': z['high']['z'], 'q': z['q'],
            'prior_z_stats': meta['prior_z_stats'], 'prior_q_stats': meta['prior_q_stats'], 'params': meta['params']}
    sources = {}
    for c in ('326', '329'):
        folds = spec[f'folds{c}']
        sources[c] = np.mean([z[c][f'f{i}_v1'].astype(np.float32) for i in folds], axis=0) if folds else None
    sources['336'] = z['336']['v1'].astype(np.float32)
    sources['269'] = z['269']['v1'].astype(np.float32)
    cfg = copy.deepcopy(CONFIG); r = cfg['route']
    if '326' in spec['drop'] or not spec['folds326']:
        r['base']['primary_gamma'] = 0.0; r['base']['al3ni_gamma'] = 0.0
    if '269' in spec['drop']:
        r['base']['eutectic_gamma'] = 0.0
    if '336' in spec['drop']:
        r['gamma336'] = [0.0] * 4
    if spec.get('al3ni329') is not None:
        r['gamma329'][0] = spec['al3ni329']
    pf = spec.get('preaux_from')
    if pf:
        # single-model baseline: that model's probability goes straight into the same post-processing
        if pf == 'high':
            pass
        elif pf.startswith('329f') or pf.startswith('326f'):
            item['preaux'] = z[pf[:3]][f'f{pf[4:]}_v1'].astype(np.float32)
        else:
            item['preaux'] = sources[pf]
        r['base']['primary_gamma'] = r['base']['al3ni_gamma'] = r['base']['eutectic_gamma'] = r['base']['al_gamma'] = 0.0
        for g in ('gamma329', 'gamma336', 'gamma347'):
            r[g] = [0.0] * 4
    sources = {k: v for k, v in sources.items() if v is not None}
    return item, sources, cfg


def score(args):
    name, specs = args
    z = load(name); gt = decode_gt(GT / f'{name}_mask.png', PALETTE)
    res = {}
    for key, spec in specs.items():
        item, sources, cfg = build(z, spec)
        pred = predict_fixed(item, sources, cfg).astype(np.uint8)
        ious = []
        for cid in range(4):
            inter = np.count_nonzero((pred == cid) & (gt == cid)); uni = np.count_nonzero((pred == cid) | (gt == cid))
            ious.append(inter / uni if uni else 0.0)
        res[key] = ious
    return name, res


def variants():
    v = {'lite14': BASE}
    keep_bands = ['model109', 'seed456', 'pairwise']
    # routing: keep a subset of the non-restart51 bands, send the rest to restart51
    for r in range(0, 3):
        for kept in itertools.combinations(keep_bands, r):
            m = {k: (k if k in kept else 'restart51') for k in CANDS}
            v['route_keep=' + ('+'.join(kept) or 'none')] = dict(BASE, route=m)
    for x in ['model109', 'seed456', 'pairwise']:
        v[f'route_all={x}'] = dict(BASE, route={k: x for k in CANDS})
    v['drop269'] = dict(BASE, drop=('269',))
    v['drop326'] = dict(BASE, drop=('326',))
    for g in (0.6, 0.7, 0.8, 1.0):
        v[f'drop326,al3ni329={g}'] = dict(BASE, drop=('326',), al3ni329=g)
    for k in (1, 2, 3):
        for fs in itertools.combinations((1, 2, 3, 4), k):
            v[f'folds326={fs}'] = dict(BASE, folds326=fs)
            v[f'folds329={fs}'] = dict(BASE, folds329=fs)
    return v


if __name__ == '__main__':
    specs = variants()
    if len(sys.argv) > 5:
        for k, s in json.loads(sys.argv[5]).items():
            sp = dict(BASE)
            for kk, vv in s.items():
                sp[kk] = tuple(vv) if kk in ('drop', 'folds326', 'folds329') else vv
            if 'route_keep' in s:
                sp['route'] = {c: (c if c in s['route_keep'] else 'restart51') for c in CANDS}
            specs[k] = sp
        if os.environ.get('ONLY_EXTRA'):
            specs = {k: specs[k] for k in ['lite14'] + list(json.loads(sys.argv[5]))}
    with Pool(int(os.environ.get('NPROC', '12'))) as pool:
        rows = dict(pool.map(score, [(n, specs) for n in NAMES]))
    summary = {}
    for key in specs:
        cls = np.array([rows[n][key] for n in NAMES])
        summary[key] = {'miou': float(cls.mean(1).mean() * 100), 'class': (cls.mean(0) * 100).round(3).tolist(),
                        'per_image': (cls.mean(1) * 100).round(4).tolist()}
    OUT.write_text(json.dumps({'names': NAMES, 'routed': {n: json.loads((HIGH / f'{n}.json').read_text())['routed'] for n in NAMES},
                               'summary': summary}, indent=1))
    for key, s in sorted(summary.items(), key=lambda kv: -kv[1]['miou']):
        print(f"{s['miou']:.4f}  {key}")
