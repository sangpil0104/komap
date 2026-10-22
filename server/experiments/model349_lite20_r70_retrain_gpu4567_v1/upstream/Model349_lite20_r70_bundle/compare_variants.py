"""Summarise verify_variants.sh output: valid20 scores and test-mask differences."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
out = Path(sys.argv[1]); ref = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
expected = {'lite21': 80.5287696, 'lite20': 80.5304, 'lite19': 80.5199}
def labels(d): return {p.stem: np.asarray(Image.open(p)) for p in sorted(Path(d).glob('*.png'))}
def diff(a, b):
    assert a.keys() == b.keys(), 'image sets differ'
    changed = sum(int((a[k] != b[k]).sum()) for k in a); total = sum(a[k].size for k in a)
    return changed, total, {k: int((a[k] != b[k]).sum()) for k in a if (a[k] != b[k]).any()}
res = {}
for v in expected:
    m = out / v / 'valid_metrics.json'
    if m.exists():
        d = json.loads(m.read_text())
        res[v] = {'valid20': d['image_miou'] * 100, 'expected': expected[v],
                  'per_image': {x['image']: x['image_miou'] * 100 for x in d['images']}}
        print(f"{v}: valid20 {res[v]['valid20']:.6f}% (expected ~{expected[v]}%)")
    else:
        print(f'{v}: no valid metrics')
tests = {v: labels(out / v / 'test/fusion/predicted_labels') for v in expected if (out / v / 'test/fusion/predicted_labels').exists()}
if ref is not None:
    r = labels(ref)
    for v, t in tests.items():
        c, n, per = diff(t, r); res.setdefault(v, {})['test_vs_lite23'] = {'changed_pixels': c, 'total': n, 'per_image': per}
        print(f'{v}: test vs lite23 {c}/{n} pixels changed ({100 * c / n:.4f}%), images changed {len(per)}')
names = list(tests)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        c, n, per = diff(tests[names[i]], tests[names[j]])
        print(f'test {names[i]} vs {names[j]}: {c}/{n} pixels ({100 * c / n:.4f}%)')
(out / 'summary.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
