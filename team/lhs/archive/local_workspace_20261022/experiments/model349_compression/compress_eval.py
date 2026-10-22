"""Score compressed Model349 variants from compress_probe.py dumps (CPU only).

Every variant reuses the frozen fusion + post-processing (predict_fixed) unchanged;
only the probability inputs change. Run from the bundle root.
"""
import copy, itertools, json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
import yaml
import model183_metric_calibration as cache
from infer import predict_fixed, decode_gt

PROBE = Path(sys.argv[1])
GT = Path(sys.argv[2])
OUT = Path(sys.argv[3])
CONFIG = yaml.safe_load((ROOT / 'configs/model349.yaml').read_text())
PALETTE = np.array(CONFIG['palette'], dtype=np.uint8)
NAMES = sorted(p.stem for p in (PROBE / 'base').glob('*.npz'))

FULL = dict(folds=4, b4v=8, hrv=4, n269='v8', tent=True, use1024=True, style=True, pair=True, branches=True, drop=())


def b4_source(z, folds, v):
    return np.mean([z[f'f{i}_v{v}'].astype(np.float32) for i in range(1, folds + 1)], axis=0)


def build(name, spec, cache_z):
    z = cache_z
    sources = {}
    for c in ('322', '326', '329'):
        if spec.get('soup'):
            sources[c] = z[c + 'soup'][f"v{spec['b4v']}"].astype(np.float32)
        else:
            sources[c] = b4_source(z[c], spec['folds'], spec['b4v'])
    for c in ('336', '347'):
        sources[c] = z[c][f"v{spec['hrv']}"].astype(np.float32)
    sources['269'] = z['269'][spec['n269']].astype(np.float32)
    b, meta = z['base'], z['base_meta']
    params = dict(meta['params'])
    tag = 'tent' if spec['tent'] else 'notent'
    def hi(size, kind):
        return b[f'high{size}_{kind}_{tag}'].astype(np.float32)
    if spec['use1024']:
        high = tuple(.8 * hi(768, k) + .2 * hi(1024, k) for k in ('raw', 'style'))
    else:
        high = tuple(hi(768, k) for k in ('raw', 'style'))
    if not spec['style']:
        params['style_weight'] = 0.0
    if not spec['pair']:
        params['pair_weight'] = 0.0
    branches = [b[f'branch{i}'].astype(np.float32) for i in range(3)]
    if not spec['branches']:
        branches = [np.zeros_like(x) for x in branches]
    like = {'branches': branches, 'pair': b['pair'].astype(np.float32), 'params': params, 'extra': b['extra'].astype(np.float32)}
    item = {'mixed': cache.make_mixed(like, high, gamma=.40), 'extra': like['extra'], 'z': b['z'], 'q': b['q'],
            'prior_z_stats': meta['prior_z_stats'], 'prior_q_stats': meta['prior_q_stats'], 'params': params}
    cfg = copy.deepcopy(CONFIG)
    r = cfg['route']
    for c in spec['drop']:
        if c in ('329', '336', '347'):
            r[f'gamma{c}'] = [0.0] * 4
        elif c == '326':
            r['base']['primary_gamma'] = 0.0; r['base']['al3ni_gamma'] = 0.0
        elif c == '322':
            r['base']['al_gamma'] = 0.0
        elif c == '269':
            r['base']['eutectic_gamma'] = 0.0
    return item, sources, cfg


def load(name):
    z = {}
    for c in ('322', '326', '329', '336', '347', '269', 'base', '322soup', '326soup', '329soup'):
        if (PROBE / c / f'{name}.npz').exists():
            z[c] = dict(np.load(PROBE / c / f'{name}.npz'))
    z['base_meta'] = json.loads((PROBE / 'base' / f'{name}.json').read_text())
    return z


def score_image(args):
    name, specs = args
    z = load(name)
    gt = decode_gt(GT / f'{name}_mask.png', PALETTE)
    res = {}
    for key, spec in specs.items():
        item, sources, cfg = build(name, spec, z)
        pred = predict_fixed(item, sources, cfg).astype(np.uint8)
        ious = []
        for cid in range(4):
            inter = np.count_nonzero((pred == cid) & (gt == cid)); uni = np.count_nonzero((pred == cid) | (gt == cid))
            ious.append(inter / uni if uni else 0.0)
        res[key] = ious
    return name, res


def variants():
    v = {'full': FULL}
    for k, vals in (('folds', (1, 2, 3)), ('b4v', (1, 2, 4)), ('hrv', (1, 2))):
        for x in vals:
            v[f'{k}={x}'] = dict(FULL, **{k: x})
    v['269=v1'] = dict(FULL, n269='v1')
    for x in (1, 2, 4, 8):
        v[f'soup,b4v={x}'] = dict(FULL, soup=True, b4v=x)
    for k in ('tent', 'use1024', 'style', 'pair', 'branches'):
        v[f'no_{k}'] = dict(FULL, **{k: False})
    comps = ('322', '326', '329', '336', '347', '269')
    for r in range(1, len(comps) + 1):
        for d in itertools.combinations(comps, r):
            v['drop=' + '+'.join(d)] = dict(FULL, drop=d)
    return v


if __name__ == '__main__':
    specs = variants()
    extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    if os.environ.get('ONLY_EXTRA'):
        specs = {'full': FULL}
    for k, s in extra.items():
        specs[k] = dict(FULL, **{kk: (tuple(vv) if kk == 'drop' else vv) for kk, vv in s.items()})
    with Pool(int(os.environ.get('NPROC', '12'))) as pool:
        rows = pool.map(score_image, [(n, specs) for n in NAMES])
    per = {n: r for n, r in rows}
    summary = {}
    for key in specs:
        cls = np.array([per[n][key] for n in NAMES])
        summary[key] = {'miou': float(cls.mean(1).mean() * 100), 'class': (cls.mean(0) * 100).round(3).tolist(),
                        'per_image': (cls.mean(1) * 100).round(4).tolist()}
    OUT.write_text(json.dumps({'names': NAMES, 'specs': {k: {kk: list(vv) if isinstance(vv, tuple) else vv for kk, vv in s.items()} for k, s in specs.items()},
                               'summary': summary}, indent=1))
    for key, s in sorted(summary.items(), key=lambda kv: -kv[1]['miou'])[:15]:
        print(f"{s['miou']:.4f}  {key}")
    print('full', summary['full']['miou'])
