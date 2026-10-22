"""Paired lite20_r70 vs lite20 comparison written by run_r70.sh:  python r70/compare_r70.py OUT"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
out = Path(sys.argv[1]); res = {}
def per_image(v):
    m = out / v / 'valid_metrics.json'
    if not m.exists(): return None
    d = json.loads(m.read_text()); return d['image_miou'] * 100, {x['image']: x['image_miou'] * 100 for x in d['images']}
a, b = per_image('lite20_r70'), per_image('lite20')
if a and b:
    delta = {k: a[1][k] - b[1][k] for k in b[1]}; worst = min(delta, key=delta.get)
    # specimen-grouped bootstrap of the mean delta (images of one A-number move together)
    groups = {}
    for k, v in delta.items(): groups.setdefault(k.split('_')[0], []).append(v)
    keys = list(groups); rng = np.random.default_rng(81795); boots = []
    for _ in range(20000):
        pick = rng.integers(0, len(keys), len(keys)); vals = [x for i in pick for x in groups[keys[i]]]; boots.append(np.mean(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res['valid20'] = {'lite20_r70': a[0], 'lite20': b[0], 'delta': a[0] - b[0], 'bootstrap95': [lo, hi],
                      'improved': sum(v > 0 for v in delta.values()), 'worse': sum(v < 0 for v in delta.values()),
                      'worst_image': worst, 'worst_delta': delta[worst], 'per_image_delta': delta}
    print(f"valid20 lite20_r70 {a[0]:.6f}%  lite20 {b[0]:.6f}%  delta {a[0]-b[0]:+.4f}%p  specimen bootstrap95 [{lo:+.3f}, {hi:+.3f}]")
    print(f"improved {res['valid20']['improved']} / worse {res['valid20']['worse']}, worst {worst} {delta[worst]:+.3f}%p")
else:
    print('valid metrics missing for one of the variants')
ta, tb = out / 'lite20_r70/test/fusion/predicted_labels', out / 'lite20/test/fusion/predicted_labels'
if ta.exists() and tb.exists():
    c = n = 0
    for p in sorted(ta.glob('*.png')):
        x, y = np.asarray(Image.open(p)), np.asarray(Image.open(tb / p.name)); c += int((x != y).sum()); n += x.size
    res['test_changed_pixels'] = [c, n]; print(f'test10 masks: {c}/{n} pixels differ ({100*c/n:.3f}%)')
(out / 'summary.json').write_text(json.dumps(res, indent=1) + '\n')
