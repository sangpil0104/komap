"""Analyze exhaustive subset search results: raw ranking, weight frontier, and nested
leave-one-specimen-out selection (how much a 'pick the best subset on valid' procedure
gains on specimens it did not see)."""
import json, re, sys
import numpy as np

files = sys.argv[1:]
combos, iou, names = [], [], None
seen = set()
for f in files:
    d = np.load(f)
    names = list(d['names'])
    for c, v in zip(d['combos'], d['iou']):
        t = tuple(int(x) for x in c)
        if t in seen:
            continue
        seen.add(t); combos.append(t); iou.append(v)
iou = np.array(iou)                      # (n, images, 4)
img = iou.mean(2) * 100                  # (n, images)
mean = img.mean(1)
BANDS = ['model109', 'seed456', 'pairwise', 'ema']


def weights(c):
    m322, m326, m329, h336, h347, n269, br, pair, rmask = c
    w = set()
    for tag, m in (('322', m322), ('326', m326), ('329', m329)):
        w |= {f'{tag}f{i+1}' for i in range(4) if m >> i & 1}
    if h336: w.add('336')
    if h347: w.add('347')
    if n269: w.add('269')
    for i, b in enumerate(('model97', 'model98', 'model109')):
        if br >> i & 1: w.add(b)
    if pair: w.add('pairwise')
    w.add('restart51')
    for i, b in enumerate(BANDS):
        if rmask >> i & 1: w.add({'ema': 'model123'}.get(b, b))
    return w


nw = np.array([len(weights(c)) for c in combos])
key = {c: i for i, c in enumerate(combos)}
LITE14 = (0, 15, 15, 1, 0, 1, 0, 0, 7)
FULL = (15, 15, 15, 1, 1, 1, 7, 1, 15)
ref = key[LITE14]
grp = np.array([re.match(r'(A\d+)_', n).group(1) for n in names])
groups = sorted(set(grp))

print(f'{len(combos)} combos scored; lite14 {mean[ref]:.4f} (w={nw[ref]})' + (f'; full {mean[key[FULL]]:.4f} (w={nw[key[FULL]]})' if FULL in key else ''))
order = np.argsort(-mean)
print('top 10 raw:')
for i in order[:10]:
    print(f'  {mean[i]:.4f} w={nw[i]:2d} {combos[i]} worst_vs_lite14={np.min(img[i]-img[ref]):+.2f}')
print('raw best per weight budget:')
frontier = []
for k in range(nw.min(), nw.max() + 1):
    idx = np.where(nw <= k)[0]
    if not len(idx):
        continue
    b = idx[np.argmax(mean[idx])]
    frontier.append((k, float(mean[b]), combos[b]))
    print(f'  <= {k:2d} weights: {mean[b]:.4f} {combos[b]}')

# nested leave-one-specimen-out: choose on other specimens, score on the held-out one
rng = np.random.default_rng(0)


def nested(cands):
    held = np.zeros(len(names)); picks = []
    for g in groups:
        tr, te = grp != g, grp == g
        b = cands[np.argmax(img[cands][:, tr].mean(1))]
        held[te] = img[b, te]; picks.append(combos[b])
    d = held - img[ref]
    gi = np.array([groups.index(x) for x in grp])
    bs = [d[np.concatenate([np.where(gi == x)[0] for x in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(4000)]
    return held.mean(), d.mean(), np.percentile(bs, [2.5, 97.5]), len(set(picks))


print('nested leave-one-specimen-out (selection repeated without the held-out specimen):')
allc = np.arange(len(combos))
for label, cands in [('all combos', allc)] + [(f'<= {k} weights', np.where(nw <= k)[0]) for k in (6, 8, 10, 12, 14)]:
    if not len(cands):
        continue
    m, d, ci, npick = nested(cands)
    print(f'  {label:14s}: held-out mean {m:.4f}, vs lite14 {d:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}], distinct picks {npick}')
json.dump({'n': len(combos), 'frontier': frontier, 'top': [(combos[i], float(mean[i]), int(nw[i])) for i in order[:30]]},
          open('exh_summary.json', 'w'), indent=1)
