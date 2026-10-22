"""lite18h, E pruning: B4ws 2-of-4 folds and/or HR392 one fold per seed (E weights 0.5 / 0.25 / 0.25 kept), valid20 final (F unchanged).
All combinations are reported; picking the best one on valid is a selection, so the mean over combinations is also given."""
import json, itertools, importlib.util
from pathlib import Path
from multiprocessing import Pool
import numpy as np, torch
from PIL import Image
K = Path.home() / 'komap'; A = K / 'add'; PAR = A / 'parent0'; T2M = [1, 2, 0, 3]; pe = A / 'pe/valid'
B4S = [c for c in itertools.combinations([1, 2, 3, 4], 2)]; HRS = [(f1, f2) for f1 in (2, 4) for f2 in (1, 4)]
CANDS = {'published': ((1, 2, 3, 4), None)}
for b in B4S: CANDS[f'b4{b}'] = (b, None)
for h in HRS: CANDS[f'hr{h}'] = ((1, 2, 3, 4), h)
for b in B4S:
    for h in HRS: CANDS[f'b4{b}+hr{h}'] = (b, h)
def P(d, n): return np.load(d / f'{n}.npy').astype(np.float32)[T2M]
def task(n):
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('fu', PAR / 'infer_fusion.py'); fu = importlib.util.module_from_spec(spec); spec.loader.exec_module(fu)
    m = fu.Residual().eval(); m.load_state_dict(torch.load(K / 'weights/all/assets/refiner/epoch8.pt', map_location='cpu', weights_only=False)['state_dict'])
    man = json.loads((A / 'base_valid/input_manifest.json').read_text()); pal = np.asarray(man['config']['palette'], np.uint8)
    b4 = {k: P(pe / f'model329_ws_f{k}', n) for k in range(1, 5)}; hr = {(s, k): P(pe / f'hr392_s{s}_f{k}', n) for s, k in ((1, 2), (1, 4), (2, 1), (2, 4))}
    ref = .5 * np.mean(list(b4.values()), 0) + .25 * (hr[1, 2] + hr[1, 4]) / 2 + .25 * (hr[2, 1] + hr[2, 4]) / 2
    e0 = np.load(A / f'e_valid/expert/{n}.npy').astype(np.float32); parent = np.asarray(Image.open(K / f'l18h/R_valid/predicted_labels/{n}.png'))
    image = next(p for p in (K / 'data/valid/images').iterdir() if p.stem.removesuffix('_image') == n)
    gray = torch.from_numpy(fu.prepare_gray(np.asarray(Image.open(image).convert('L'), dtype=np.float32) / 255.)[None, None])
    rgb = np.asarray(Image.open(K / f'data/valid/masks/{n}_mask.png').convert('RGB')); gt = np.full(rgb.shape[:2], 255, np.uint8)
    for c, col in enumerate(pal): gt[np.all(rgb == col, -1)] = c
    out = {}
    for tag, (bf, hf) in CANDS.items():
        x_b4 = np.mean([b4[k] for k in bf], 0)
        x_hr = (.25 * (hr[1, 2] + hr[1, 4]) / 2 + .25 * (hr[2, 1] + hr[2, 4]) / 2) if hf is None else (.25 * hr[1, hf[0]] + .25 * hr[2, hf[1]])
        e = e0 + (.5 * x_b4 + x_hr - ref)
        with torch.no_grad(): v, _ = m(torch.from_numpy(np.log(np.maximum(e, 1e-7))[None]), gray)
        ex = fu.labels(e); r = fu.labels(v.numpy()[0]); pred = np.where(r == 2, r, np.where(np.isin(ex, [0, 2]), ex, parent))
        out[tag] = float(np.mean([((pred == c) & (gt == c)).sum() / u if (u := ((pred == c) | (gt == c)).sum()) else 0. for c in range(4)])) * 100
    return n, out
if __name__ == '__main__':
    names = [r['name'] for r in json.loads((A / 'base_valid/input_manifest.json').read_text())['images']]
    with Pool(4) as p: t = dict(p.map(task, names))
    (K / "codex_team_seed_review_20261012/per_image.json").write_text(json.dumps(t, indent=2))
    res = {}
    for tag in CANDS:
        d = [t[n][tag] - t[n]['published'] for n in names]; res[tag] = (np.mean([t[n][tag] for n in names]), np.mean(d), min(d), sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d))
        print(f'{tag:24s} {res[tag][0]:.4f}  vs lite18h {res[tag][1]:+.4f}  worst {res[tag][2]:+.3f}  better/worse {res[tag][3]}/{res[tag][4]}')
    for grp, keys in (('B4 2-of-4 (6)', [f'b4{b}' for b in B4S]), ('HR 1 per seed (4)', [f'hr{h}' for h in HRS]), ('both (24)', [f'b4{b}+hr{h}' for b in B4S for h in HRS])):
        v = [res[k][1] for k in keys]; print(f'{grp:18s} mean over combos {np.mean(v):+.4f}  min {min(v):+.4f}  max {max(v):+.4f}  ({sum(x>=0 for x in v)}/{len(v)} >= 0)')

    (K / "codex_team_seed_review_20261012/summary.json").write_text(json.dumps({k: [float(x) for x in v] for k,v in res.items()}, indent=2))
    (K / "codex_team_seed_review_20261012/completion.json").write_text(json.dumps({"cpu_only":True,"candidates":len(res),"images":len(names),"release_replaced":False}))
