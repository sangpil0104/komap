"""lite13 valid20 search: (1) R/E/F ratio retune (greedy, with specimen split-half honesty check),
(2) member replacement and (3) member addition in R/E, from stored probabilities. No test data used.
usage: search13.py retune {full|A|B}  |  search13.py cands WORKER NWORKERS
"""
import sys, json, pickle, importlib.util, re, itertools
from pathlib import Path
import numpy as np, torch, yaml
from PIL import Image
K = Path.home() / 'komap'
L = K / 'lite14_N3_R326plain3456_HRtrain70seed5_20261013'
X = L / 'parent_package/Model349_exh14'
OUT = K / 'b4tta/search13'; OUT.mkdir(exist_ok=True)
OLD = K / 'l18h/e2e_valid/release'
PR = K / 'b4tta/probs'
FR = K / 'codex_lite16_N3_L2_fresh_valid_20261013/fresh_probabilities'
S5 = K / 'codex_HR392_train70_seed345_lite15_20261013/evaluation/seed5/probabilities'
T2M = [1, 2, 0, 3]
rt = lambda p: p.astype(np.float16).astype(np.float32)
torch.set_num_threads(2)
sys.path[:0] = [str(X), str(X / 'src')]
import infer_lite
import model335_exhaustive_b4_route_search as route
sp = importlib.util.spec_from_file_location('fu', L / 'parent_package/infer_fusion.py')
fu = importlib.util.module_from_spec(sp); sp.loader.exec_module(fu)
ref = fu.Residual().eval()
ref.load_state_dict(torch.load(L / 'assets/refiner/epoch8.pt', map_location='cpu', weights_only=False)['state_dict'])
CFG = yaml.safe_load((K / 'lite13_R322share326_20261014/selected_R_lite13.yaml').read_text())
pal = np.array(CFG['palette'], np.uint8)


def iou(p, g):
    return np.array([((p == c) & (g == c)).sum() / u if (u := ((p == c) | (g == c)).sum()) else 0. for c in range(4)])


def epf(f, v, n):
    return np.load(PR / 'E' / f'f{f}' / v / f'{n}.npy').astype(np.float32)


names = sorted(p.name.removesuffix('_image.png') for p in (K / 'data/valid/images').glob('*_image.png'))
recs = {}
for n in names:
    src = {c: np.load(OLD / f'probabilities/{c}/{n}.npy').astype(np.float32) for c in ('269', '329', '347')}
    src['326'] = rt(np.load(K / f'codex_r326_ema_seed4_20261012/probabilities/set0/fold2/plain/{n}.npy'))
    with (OLD / f'probabilities/base/{n}.pkl').open('rb') as f:
        item = pickle.load(f)
    rgb = np.asarray(Image.open(K / f'data/valid/masks/{n}_mask.png').convert('RGB'))
    gt = np.full(rgb.shape[:2], 255, np.uint8)
    for c, col in enumerate(pal):
        gt[np.all(rgb == col, -1)] = c
    gray = torch.from_numpy(fu.prepare_gray(np.asarray(Image.open(K / f'data/valid/images/{n}_image.png').convert('L'), np.float32) / 255)[None, None])
    rec = dict(src=src, item=item, gt=gt, gray=gray, e14=np.load(L / f'fresh_valid/expert/{n}.npy').astype(np.float32))
    rec['wsf1'] = rt(epf(1, 'd4', n))[T2M]; rec['wsf3'] = rt(epf(3, 'd4', n))[T2M]
    rec['bold'] = rt((epf(1, 'd4', n) + epf(3, 'd4', n)) / 2)[T2M]
    rec['N3'] = rt(np.load(FR / f'N3/{n}.npy').astype(np.float32))[T2M]
    rec['s1f4'] = np.load(FR / f'hr_s1_f4/{n}.npy').astype(np.float32)[T2M]
    rec['s5'] = np.load(S5 / f'{n}.npy').astype(np.float32)[T2M]
    rec['g1'] = rt(rec['N3'] / 2 + rec['s1f4'] / 2); rec['g2'] = rt(rec['s5'])
    with torch.no_grad():
        v, _ = ref(torch.from_numpy(np.log(np.maximum(rec['e14'], 1e-7))[None]), gray)
    rec['fl14'] = fu.labels(v.numpy()[0])   # F labels for the unchanged E (cache)
    recs[n] = rec


def predict_r(rec, cfg, sources=None, extra=()):
    """infer_lite.predict_fixed plus optional extra class-wise replace sources [(prob, gammas)]."""
    s = dict(rec['src'] if sources is None else sources)
    for slot, donor in cfg['inference'].get('hr_alias', {}).items():
        s[slot] = s[donor]
    base = rec['item']['preaux'] if 'preaux' in rec['item'] else route.undo_model181_primary_blend(rec['item'])
    st = cfg['route']
    for key in ('primary', 'al3ni', 'eutectic'):
        if not st['base'][key + '_gamma']:
            s.setdefault(st['base'][key + '_source'], base)
    mixed = route.make_mixed({'item': rec['item'], 'base': base, 'sources': s}, st['base'])
    for comp, key in (('329', 'gamma329'), ('336', 'gamma336'), ('347', 'gamma347')):
        for c, g in enumerate(st[key]):
            if g:
                mixed = route.replace_classes(mixed, s[comp], (c,), g)
    for prob, gam in extra:
        for c, g in enumerate(gam):
            if g:
                mixed = route.replace_classes(mixed, prob, (c,), g)
    return route.predict(rec['item'], mixed, st['base'], route.bilateral_weights(rec['item']['z']), exact=True)


def finalize(rec, r, e=None, eclasses=(0, 2), use_f=True):
    fl = None
    if e is None:
        e = rec['e14']
        if use_f:
            fl = rec['fl14']
    elif use_f:
        with torch.no_grad():
            v, _ = ref(torch.from_numpy(np.log(np.maximum(e, 1e-7))[None]), rec['gray'])
        fl = fu.labels(v.numpy()[0])
    ex = fu.labels(e)
    p = np.where(np.isin(ex, eclasses), ex, r)
    if fl is not None:
        p = np.where(fl == 2, 2, p)
    return p.astype(np.uint8)


for n, rec in recs.items():
    rec['r13'] = predict_r(rec, CFG)
    rec['base'] = finalize(rec, rec['r13'])
    assert np.array_equal(rec['base'], np.asarray(Image.open(K / f'b4tta/offline13/valid/{n}.png'))), n
    rec['bci'] = iou(rec['base'], rec['gt'])
BASE = 100 * np.mean([r['bci'].mean() for r in recs.values()])
print(f'lite13 offline reproduced exact20: {BASE:.6f}', flush=True)


def score(preds, subset=None):
    ds = {n: 100 * float((iou(p, recs[n]['gt']) - recs[n]['bci']).mean()) for n, p in preds.items()}
    use = [n for n in ds if subset is None or n in subset]
    v = np.array([ds[n] for n in use]); w = min(use, key=ds.get)
    return dict(delta=float(v.mean()), worst=w, worst_d=ds[w], better=int((v > 0).sum()), worse=int((v < 0).sum()),
                passed=bool(v.mean() >= 0 and v.min() >= -.1), per_image=ds)


def show(tag, res):
    print(f"{tag:46s} {BASE + res['delta']:.4f} d={res['delta']:+.4f} worst {res['worst'][:12]} {res['worst_d']:+.3f} "
          f"b/w {res['better']}/{res['worse']} {'PASS' if res['passed'] else ''}", flush=True)


# ---------------------------------------------------------------- (1) ratio retune
R_PARAMS = {'al_gamma': [.4, .5, .6, .7, .8, 1.], 'al3ni_gamma': [.5, .6, .7, .8, .9, 1.],
            'eutectic_gamma': [.5, .6, .7, .8, .9, 1.], 'primary_gamma': [.6, .8, 1.],
            'prior_scale': [1., 1.125, 1.25, 1.375, 1.5], 'beta_scale': [.6, .7, .8, .9, 1.],
            'al_delta': [-.1, -.05, 0., .05, .1], 'eut_delta': [0., .075, .15, .225, .3],
            'gaussian_blend': [.25, .375, .5, .625, .75], 'eut_edge_blend': [0., .125, .25, .375, .5]}
E_PARAMS = {'wb4': [.4, .45, .5, .55, .6], 'g1share': [.3, .4, .5, .6, .7],
            'eclasses': [(0, 2), (0,), (2,), (0, 2, 3), (0, 1, 2)], 'use_f': [True, False]}


def state_eval(st, subset):
    cfg = json.loads(json.dumps(CFG))
    for k, v in st['R'].items():
        if k.startswith('gamma3'):
            comp, c = k.split(':'); cfg['route'][comp][int(c)] = v
        else:
            cfg['route']['base'][k] = v
    E = st['E']; preds = {}
    for n in (subset or names):
        rec = recs[n]
        r = rec['r13'] if not st['R'] else predict_r(rec, cfg)
        e = None
        if not (E['wb4'] == .5 and E['g1share'] == .5):
            hr = 1 - E['wb4']
            e = rec['e14'] + (E['wb4'] - .5) * rec['bold'] + (hr * E['g1share'] - .25) * rec['g1'] \
                + (hr * (1 - E['g1share']) - .25) * rec['g2']
        preds[n] = finalize(rec, r, e, E['eclasses'], E['use_f'])
    return preds


def retune(split):
    specs = sorted({re.match(r'A\d+', n).group() for n in names}, key=lambda s: int(s[1:]))
    half = {'A': set(specs[0::2]), 'B': set(specs[1::2])}
    subset = None if split == 'full' else [n for n in names if re.match(r'A\d+', n).group() in half[split]]
    other = None if split == 'full' else [n for n in names if n not in subset]
    st = {'R': {}, 'E': {'wb4': .5, 'g1share': .5, 'eclasses': (0, 2), 'use_f': True}}
    cur = score(state_eval(st, subset), subset)
    log = []
    gparams = {f'{comp}:{c}': [0., .25, .5, .75, 1.] for comp in ('gamma329', 'gamma336', 'gamma347') for c in range(4)}
    for rnd in range(2):
        changed = False
        for name, vals in list(R_PARAMS.items()) + list(gparams.items()) + list(E_PARAMS.items()):
            isE = name in E_PARAMS
            if isE:
                now = st['E'][name]
            elif name.startswith('gamma3'):
                comp, c = name.split(':'); now = st['R'].get(name, CFG['route'][comp][int(c)])
            else:
                now = st['R'].get(name, CFG['route']['base'][name])
            best = None
            for v in vals:
                if v == now:
                    continue
                trial = {'R': dict(st['R']), 'E': dict(st['E'])}
                (trial['E'] if isE else trial['R'])[name] = v
                res = score(state_eval(trial, subset), subset)
                if best is None or res['delta'] > best[1]['delta']:
                    best = (v, res, trial)
            if best and best[1]['delta'] > cur['delta'] + .002 and best[1]['worst_d'] >= -.1:
                st, cur = best[2], best[1]; changed = True
                log.append({'round': rnd, 'param': name, 'value': best[0], 'delta': cur['delta']})
                print(f'[{split}] r{rnd} {name}={best[0]} -> d={cur["delta"]:+.4f} worst {cur["worst_d"]:+.3f}', flush=True)
        if not changed:
            break
    out = {'split': split, 'state': {'R': st['R'], 'E': {**st['E'], 'eclasses': list(st['E']['eclasses'])}},
           'in_sample': {k: v for k, v in cur.items()}, 'log': log}
    if other:
        held = score(state_eval(st, other), other)
        out['held_out'] = held
        print(f'[{split}] in-sample d={cur["delta"]:+.4f} | held-out half d={held["delta"]:+.4f} worst {held["worst_d"]:+.3f}', flush=True)
    else:
        show('[full] retuned', cur)
    json.dump(out, open(OUT / f'retune_{split}.json', 'w'), indent=1, default=str)
    print(f'RETUNE_{split}_DONE', flush=True)


# ---------------------------------------------------------------- (2)(3) candidates
def candidates():
    c = {}
    c['R336orig'] = K / 'add/base_valid/probabilities/336'
    for s in range(3407, 3415):
        c[f'p336_{s}'] = K / f'add/p336/seed{s}/valid'
    for s, f in itertools.product((1, 2), (1, 2, 3, 4)):
        c[f'hr392_s{s}f{f}'] = K / f'add/pe/valid/hr392_s{s}_f{f}'
    for f in (1, 2, 3, 4):
        c[f'b4_329_f{f}'] = K / f'add/pe/valid/model329_f{f}'
        c[f'b4_329ws_f{f}'] = K / f'add/pe/valid/model329_ws_f{f}'
        for m in ('322', '326', '329'):
            c[f'R{m}_f{f}'] = K / f'add/pr/valid/{m}_f{f}'
    for s in (1, 2):
        c[f'hr70_s{s}'] = K / f'codex_HR392_train70_N3_20261013/evaluation/seed{s}/probabilities'
    for s in (3, 4):
        c[f'hr70_s{s}'] = K / f'codex_HR392_train70_seed345_lite15_20261013/evaluation/seed{s}/probabilities'
    for s in (361, 362, 363, 364):
        c[f'L2_{s}'] = K / f'codex_L2_seed4_20261013/evaluation/seed{s}/probabilities'
    for s in (415, 416, 417):
        c[f'N3_{s}'] = K / f'codex_N3_seed4_20261013/evaluation/seed{s}/probabilities'
    c['L2_fresh'] = FR / 'L2'
    for d in sorted((K / 'eseed/pe').iterdir()):
        if d.is_dir() and not d.name.startswith(('hr5_s', 'hr6_s')) or d.name in ('hr6_s1_f2', 'hr6_s2_f1'):
            c[f'es_{d.name}'] = d
    for st in range(4):
        c[f'r326set{st}f2'] = K / f'codex_r326_ema_seed4_20261012/probabilities/set{st}/fold2/plain'
    for m in ('322', '326', '329'):
        c[f'r70_{m}'] = K / f'codex_r70_swa_20261012/train70/R_valid/probabilities/{m}'
    for d in sorted((K / 'b4tta/p100c').iterdir()):
        c[f'p100_{d.name}'] = d
    return {k: v for k, v in c.items() if (v / f'{names[0]}.npy').exists() or (v / f'{names[0]}.npz').exists()}


def load_cand(path, n):
    p = path / f'{n}.npy'
    if p.exists():
        return np.load(p).astype(np.float32)
    z = np.load(path / f'{n}.npz'); return z[z.files[0]].astype(np.float32)


def cand_rules(key, path):
    probs = {n: load_cand(path, n) for n in names}
    n0 = names[0]
    agree = [(probs[n0][o].argmax(0) == recs[n0]['base']).mean() for o in ([0, 1, 2, 3], T2M)]
    order = [0, 1, 2, 3] if agree[0] >= agree[1] else T2M
    C = {n: rt(probs[n][order]) for n in names}
    res = {'_order': order, '_agree': max(agree)}

    def e_rule(fn):
        return score({n: finalize(recs[n], recs[n]['r13'], fn(recs[n], C[n])) for n in names})

    def r_rule(cfg=CFG, srcfn=None, extra=None):
        return score({n: finalize(recs[n], predict_r(recs[n], cfg, srcfn(recs[n], C[n]) if srcfn else None,
                                                     extra(C[n]) if extra else ())) for n in names})
    # E replacement (each -> 0 new weights net: one member out, candidate in)
    res['E_rep_N3'] = e_rule(lambda r, c: r['e14'] + .25 * (rt(c / 2 + r['s1f4'] / 2) - r['g1']))
    res['E_rep_s1f4'] = e_rule(lambda r, c: r['e14'] + .25 * (rt(r['N3'] / 2 + c / 2) - r['g1']))
    res['E_rep_s5'] = e_rule(lambda r, c: r['e14'] + .25 * (c - r['g2']))
    res['E_rep_wsf1'] = e_rule(lambda r, c: r['e14'] + .5 * (rt((c + r['wsf3']) / 2) - r['bold']))
    res['E_rep_wsf3'] = e_rule(lambda r, c: r['e14'] + .5 * (rt((r['wsf1'] + c) / 2) - r['bold']))
    for w in (.05, .1, .2):   # E addition (+1 weight)
        res[f'E_add_{w}'] = e_rule(lambda r, c, w=w: (1 - w) * r['e14'] + w * c)
    # R replacement
    res['R_rep_347'] = r_rule(srcfn=lambda r, c: {**r['src'], '347': c})
    cfg336 = json.loads(json.dumps(CFG)); cfg336['inference']['hr_alias'] = {'322': '326'}
    res['R_slot336'] = r_rule(cfg336, srcfn=lambda r, c: {**r['src'], '336': c})          # +1 weight
    res['R_rep_269'] = r_rule(srcfn=lambda r, c: {**r['src'], '269': c})
    cfg322 = json.loads(json.dumps(CFG)); cfg322['inference']['hr_alias'] = {'336': '347'}
    res['R_slot322'] = r_rule(cfg322, srcfn=lambda r, c: {**r['src'], '322': c})          # +1 weight
    res['R_rep_326'] = r_rule(srcfn=lambda r, c: {**r['src'], '326': c})
    # R addition: candidate as a class-wise replace source (+1 weight)
    for cl in range(4):
        for g in (.25, .5):
            gam = [0., 0., 0., 0.]; gam[cl] = g
            res[f'R_add_c{cl}_{g}'] = r_rule(extra=lambda c, gam=gam: [(c, gam)])
    return res


def run_cands(worker, nworkers):
    cs = candidates(); keys = sorted(cs)[worker::nworkers]
    print(f'worker {worker}: {len(keys)} of {len(cs)} candidates', flush=True)
    for k in keys:
        f = OUT / f'cand_{k}.json'
        if f.exists():
            continue
        try:
            res = cand_rules(k, cs[k])
        except Exception as ex:
            print('FAIL', k, repr(ex), flush=True); continue
        json.dump(res, open(f, 'w'))
        best = max((v for kk, v in res.items() if not kk.startswith('_')), key=lambda v: v['delta'])
        bk = [kk for kk, v in res.items() if v is best][0]
        print(f"{k:26s} agree {res['_agree']:.3f} best {bk:14s} d={best['delta']:+.4f} worst {best['worst_d']:+.3f} "
              f"passes {sum(1 for kk, v in res.items() if not kk.startswith('_') and v['passed'])}", flush=True)
    print(f'CANDS_{worker}_DONE', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'retune':
        retune(sys.argv[2])
    else:
        run_cands(int(sys.argv[2]), int(sys.argv[3]))
