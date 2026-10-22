"""Summary of the R329 fold seed search: lite19a valid20 per seed vs 'orig' (released 329 folds), test10 mask diff."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
out = Path(sys.argv[1]) / 'seeds'
def load(d):
    m = d / 'valid_metrics.json'
    if not m.exists(): return None
    j = json.loads(m.read_text()); return j['image_miou'] * 100, {x['image']: x['image_miou'] * 100 for x in j['images']}
base = load(out / 'orig'); t0 = out / 'orig/lite19a/test/fusion/predicted_labels'
seeds = sorted((d for d in out.glob('s*') if load(d)), key=lambda d: int(d.name[1:])); res = {}
print('seed   valid20(%)   vs_orig(%p)  improved/worse  worst_image(delta)   test10_diff(%)')
if base: print(f"orig   {base[0]:.6f}")
for d in seeds:
    score, per = load(d); r = {'valid20': score}; line = f"{d.name[1:]:6s} {score:.6f}"
    if base:
        dl = {i: per[i] - base[1][i] for i in base[1]}; w = min(dl, key=dl.get)
        r.update(delta=score - base[0], improved=sum(v > 0 for v in dl.values()), worse=sum(v < 0 for v in dl.values()), worst=w, worst_delta=dl[w])
        line += f"   {r['delta']:+.4f}      {r['improved']:2d}/{r['worse']:2d}        {w} ({dl[w]:+.3f})"
    t = d / 'lite19a/test/fusion/predicted_labels'
    if t.exists() and t0.exists():
        c = n = 0
        for p in sorted(t.glob('*.png')):
            x, y = np.asarray(Image.open(p)), np.asarray(Image.open(t0 / p.name)); c += int((x != y).sum()); n += x.size
        r['test_diff_pct'] = 100 * c / n; line += f"   {r['test_diff_pct']:.3f}"
    res[d.name] = r; print(line)
v = [r['valid20'] for r in res.values()]
if v:
    best = max(res, key=lambda k: res[k]['valid20'])
    print(f"\nn={len(v)}  mean {np.mean(v):.4f}%  std {np.std(v, ddof=1) if len(v) > 1 else 0:.4f}  min {min(v):.4f}  max {max(v):.4f} ({best[1:]})"
          + (f"  | orig {base[0]:.4f}%  seeds above orig {sum(x > base[0] for x in v)}/{len(v)}" if base else ''))
failed = sorted(d.name for d in out.glob('s*') if (d / 'failed').exists())
if failed: print('failed seeds:', ' '.join(failed))
(out.parent / 'summary.json').write_text(json.dumps({'orig': base[0] if base else None, 'seeds': res, 'failed': failed}, indent=1) + '\n')
