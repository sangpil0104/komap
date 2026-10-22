"""Compare test label maps of candidates against the Model349 reference (no GT on test)."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

W = Path(sys.argv[1])
REF = W / 'model349_full_test10/predicted_labels'
C = ['Al3Ni', 'PrimarySi', 'Al', 'EutSi']
cands = {'Model349 오프라인 재조합(1-view)': W / 'combos/full_offline', 'lite14': W / 'combos/lite14',
         'base12 (원본 재추론, 내 실행)': W / 'base12_e2e_test10/predicted_labels',
         'base12 (팀원 실행)': Path('/workspace/model349_compress/generalization_20261008/runs/cpu2_fp16_12_test10/predicted_labels'),
         'base12 오프라인': W / 'combos/base12', '80.2%판 오프라인': W / 'combos/top802'}
names = sorted(p.stem for p in REF.glob('*.png'))
load = lambda d, n: np.array(Image.open(d / f'{n}.png'))
ref = {n: load(REF, n) for n in names}
out = {}
print('ref class area %:', [round(float(np.mean([np.mean(ref[n] == c) for n in names])) * 100, 3) for c in range(4)])
for tag, d in cands.items():
    if not d.exists():
        print('missing', tag); continue
    agree, miou, diff, area, rows = [], [], 0, np.zeros(4), []
    tot = 0
    for n in names:
        p = load(d, n); r = ref[n]
        agree.append(np.mean(p == r)); diff += int(np.sum(p != r)); tot += p.size
        ious = []
        for c in range(4):
            u = np.sum((p == c) | (r == c)); ious.append(np.sum((p == c) & (r == c)) / u if u else 1.0)
        miou.append(np.mean(ious)); area += [np.mean(p == c) - np.mean(r == c) for c in range(4)]
        rows.append((n, round(float(agree[-1]) * 100, 3), [round(float(x) * 100, 2) for x in ious]))
    out[tag] = dict(agree=float(np.mean(agree) * 100), diff_px=diff, total_px=tot, miou_vs_ref=float(np.mean(miou) * 100),
                    worst=min(rows, key=lambda x: x[1]), area_delta=(area / len(names) * 100).round(3).tolist(), rows=rows)
    print(f"{tag:28s} agree {out[tag]['agree']:.3f}%  diffpx {diff:>8d}/{tot}  mIoU-vs-349 {out[tag]['miou_vs_ref']:.2f}  "
          f"worst {out[tag]['worst'][0]} {out[tag]['worst'][1]}%  areaΔ(pp) {out[tag]['area_delta']}")
(W / 'compare.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
for tag in ('base12 (원본 재추론, 내 실행)', '80.2%판 오프라인'):
    if tag in out:
        print(tag); [print('  ', r) for r in out[tag]['rows']]
