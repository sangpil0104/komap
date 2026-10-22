"""Exhaustive component-subset search for Model349 (CPU, offline, run from the lite14 bundle root).

A combination is a tuple
  (m322, m326, m329, h336, h347, n269, branches, pair, route)
  m3xx     : 4-bit mask of B4 folds kept (0 = component removed)
  h336/h347/n269 : 1 = keep
  branches : 3-bit mask of the base branches (model97, model98, model109)
  pair     : 1 = keep the separate pairwise prediction
  route    : 4-bit mask over high-res bands [model109, seed456, pairwise, ema];
             bit=1 keeps that band's own checkpoint, 0 sends it to restart51
Fusion gammas are the original Model349 ones; a removed component gets gamma 0.
Single-view TTA (lite) probabilities, TENT-adapted 768 high-res tiles.
Writes per-combination, per-image class IoU arrays.
"""
import itertools, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path.cwd(); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT))
import model335_exhaustive_b4_route_search as route
from highres1024_search import MATRIX
from PIL import Image


def decode_gt(path, palette):
    rgb = np.array(Image.open(path).convert('RGB'))
    lab = np.full(rgb.shape[:2], 255, np.uint8)
    for cid, color in enumerate(palette):
        lab[np.all(rgb == color, axis=-1)] = cid
    assert not np.any(lab == 255), path
    return lab

HIGH, GT = Path(os.environ['SLIM']), Path(os.environ['GT'])
PALETTE = np.array([[25, 76, 153], [76, 178, 76], [153, 127, 76], [204, 204, 204]], np.uint8)
NAMES = sorted(p.stem for p in HIGH.glob('*.npz'))
BANDS = ['model109', 'seed456', 'pairwise', 'ema']
BASE_SET = dict(primary_source='326', primary_gamma=1.0, al3ni_source='326', al3ni_gamma=0.7, al_source='322', al_gamma=0.6,
                eutectic_source='269', eutectic_gamma=0.7, prior_scale=1.25, beta_scale=0.8, al_delta=0.0, eut_delta=0.15,
                gaussian_blend=0.5, eut_edge_blend=0.25, cleanup_threshold=128)
G329, G336, G347 = [0.5, 1.0, 0.0, 0.0], [0.0, 0.25, 0.75, 0.75], [0.0, 0.0, 0.25, 0.0]

_cache = {}


def load(name):
    if name in _cache:
        return _cache[name]
    _cache.clear()
    d = np.load(HIGH / f'{name}.npz')
    z = {}
    for c in ('322', '326', '329'):
        z[c] = [d[f'{c}_f{i}'].astype(np.float32) for i in range(1, 5)]
    for c in ('336', '347', '269'):
        z[c] = d[c].astype(np.float32)
    z['branches'] = [d[f'branch{i}'].astype(np.float32) for i in range(3)]
    z['pair'] = d['pair'].astype(np.float32)
    z['q'] = d['q'].astype(np.float32)
    z['high'] = {k: d['high_' + k].astype(np.float32) for k in ['restart51'] + BANDS}
    z['z'] = d['z']
    z['meta'] = json.loads((HIGH / f'{name}.json').read_text())
    z['weights'] = route.bilateral_weights(d['z'])
    z['gt'] = decode_gt(GT / f'{name}_mask.png', PALETTE)
    z['zero'] = np.zeros_like(z['336'])
    _cache[name] = z
    return z


def predict(z, combo):
    m322, m326, m329, h336, h347, n269, br, pair, rmask = combo
    meta = z['meta']; params = dict(meta['params'])
    routed = meta['routed']
    src = routed if routed == 'restart51' else (routed if rmask >> BANDS.index(routed) & 1 else 'restart51')
    high = z['high'][src]
    hw = params['high_weight']; pw = params['pair_weight'] if pair else 0.0
    if br == 0 and not pair:
        base = high.copy()
    else:
        base = hw * high + (pw * z['pair'] if pair else 0.0)
        for m in range(3):
            if br >> m & 1:
                base = base + (MATRIX[:, m] * (1.0 - hw - pw))[:, None, None] * z['branches'][m]
        base = base / np.maximum(base.sum(0, keepdims=True), 1e-7)
    params['pair_weight'] = pw
    item = {'z': z['z'], 'q': z['q'], 'prior_z_stats': meta['prior_z_stats'], 'prior_q_stats': meta['prior_q_stats'], 'params': params}
    s = dict(BASE_SET)
    sources = {}
    for c, m in (('322', m322), ('326', m326), ('329', m329)):
        folds = [z[c][i] for i in range(4) if m >> i & 1]
        sources[c] = np.mean(folds, axis=0) if folds else z['zero']
    sources['336'], sources['347'], sources['269'] = z['336'], z['347'], z['269']
    if not m326: s['primary_gamma'] = 0.0; s['al3ni_gamma'] = 0.0
    if not m322: s['al_gamma'] = 0.0
    if not n269: s['eutectic_gamma'] = 0.0
    gam = {'329': G329 if m329 else [0.0] * 4, '336': G336 if h336 else [0.0] * 4, '347': G347 if h347 else [0.0] * 4}
    mixed = route.make_mixed({'item': item, 'base': base, 'sources': sources}, s)
    for c in ('329', '336', '347'):
        for cid, g in enumerate(gam[c]):
            if g:
                mixed = route.replace_classes(mixed, sources[c], (cid,), g)
    return route.predict(item, mixed, s, z['weights'], exact=True)


def score_task(args):
    name, combos = args
    z = load(name); gt = z['gt']
    out = np.zeros((len(combos), 4), np.float32)
    for k, cb in enumerate(combos):
        pred = predict(z, cb)
        for cid in range(4):
            inter = np.count_nonzero((pred == cid) & (gt == cid)); uni = np.count_nonzero((pred == cid) | (gt == cid))
            out[k, cid] = inter / uni if uni else 0.0
    return name, out


def run(combos, tag, nproc, chunk=128):
    combos = [tuple(int(v) for v in c) for c in combos]
    tasks = [(n, combos[i:i + chunk]) for n in NAMES for i in range(0, len(combos), chunk)]
    res = {n: [] for n in NAMES}
    t0 = time.time()
    with Pool(nproc, maxtasksperchild=8) as pool:
        for k, (n, arr) in enumerate(pool.imap(score_task, tasks)):  # ordered: name-major, chunk order kept
            res[n].append(arr)
            if k % 20 == 0:
                print(f'{tag} {k + 1}/{len(tasks)} tasks {time.time() - t0:.0f}s', flush=True)
    iou = np.stack([np.concatenate(res[n]) for n in NAMES], axis=1)  # (combos, images, 4)
    np.savez(f'{os.environ["OUT"]}/{tag}.npz', combos=np.array(combos), iou=iou, names=np.array(NAMES))
    m = iou.mean(2).mean(1) * 100
    order = np.argsort(-m)
    print(tag, 'done', f'{time.time() - t0:.0f}s', 'best', [(combos[i], round(float(m[i]), 4)) for i in order[:5]], flush=True)
    return combos, m


if __name__ == '__main__':
    stage, nproc = sys.argv[1], int(sys.argv[2])
    LITE14 = (0, 15, 15, 1, 0, 1, 0, 0, 0b0111)
    FULL = (15, 15, 15, 1, 1, 1, 7, 1, 0b1111)
    if stage == 'smoke':
        run([LITE14, FULL], 'smoke', nproc)
    elif stage == 'A':
        run([(a, b, c, 1, 0, 1, 0, 0, 0b0111) for a in range(16) for b in range(16) for c in range(16)], 'stageA', nproc)
    elif stage == 'B':
        run([(15, 15, 15, h6, h7, n9, br, p, r) for h6 in (0, 1) for h7 in (0, 1) for n9 in (0, 1)
             for br in range(8) for p in (0, 1) for r in range(16)], 'stageB', nproc)
    elif stage == 'C':
        A = np.load(f'{os.environ["OUT"]}/stageA.npz'); B = np.load(f'{os.environ["OUT"]}/stageB.npz')
        ta = A['combos'][np.argsort(-A['iou'].mean(2).mean(1))[:50]]
        tb = B['combos'][np.argsort(-B['iou'].mean(2).mean(1))[:40]]
        run([tuple(a[:3]) + tuple(b[3:]) for a in ta for b in tb], 'stageC', nproc)
