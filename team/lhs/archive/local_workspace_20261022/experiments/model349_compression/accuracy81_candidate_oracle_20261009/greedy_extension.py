"""Screen one extra class-priority rule on the fixed 80.4746% mask."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INVENTORY = json.loads((HERE / 'audit_result.json').read_text())
NAMES = INVENTORY['names']
RECORDS = INVENTORY['records']
BASE = HERE / 'fusion_candidate_valid20'
GT = Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad/gt/valid/masks')
manifest = json.loads((BASE / 'input_manifest.json').read_text())
palette = np.asarray(manifest['config']['palette'], dtype=np.uint8)


def target(name):
    p = GT / f'{name}_mask.png'
    raw = np.asarray(Image.open(p))
    if raw.ndim == 2 and np.all(raw < 4):
        return raw
    rgb = np.asarray(Image.open(p).convert('RGB'))
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for k, color in enumerate(palette):
        out[np.all(rgb == color, axis=-1)] = k
    assert np.all(out < 4)
    return out


def score(hist, subset=0, gate='candidate'):
    cm = np.zeros((4, 4), dtype=np.int64)
    for old in range(4):
        for new in range(4):
            selected = new if gate == 'candidate' else old
            pred = new if ((subset >> selected) & 1) else old
            cm[:, pred] += hist[old, new]
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    return float(np.divide(inter, union, out=np.zeros(4, dtype=float), where=union > 0).mean())


truth = {name: target(name) for name in NAMES}
base = {name: np.asarray(Image.open(BASE / 'predicted_labels' / f'{name}.png'), dtype=np.uint8) for name in NAMES}
rules = [(gate, subset) for gate in ('candidate', 'current') for subset in range(1, 16)]
scores = np.empty((len(RECORDS), len(rules), len(NAMES)), dtype=np.float64)
base_scores = np.empty(len(NAMES), dtype=np.float64)
for i, row in enumerate(RECORDS):
    run = ROOT / row['run']
    for j, name in enumerate(NAMES):
        candidate = np.asarray(Image.open(run / 'predicted_labels' / f'{name}.png'), dtype=np.uint8)
        old, gt = base[name], truth[name]
        assert candidate.shape == old.shape == gt.shape
        hist = np.bincount((old.astype(np.int64) * 16 + candidate.astype(np.int64) * 4 + gt).ravel(),
                           minlength=64).reshape(4, 4, 4)
        base_scores[j] = score(hist)
        for k, (gate, subset) in enumerate(rules):
            scores[i, k, j] = score(hist, subset, gate)

full = scores.mean(axis=2)
best_flat = int(full.argmax())
best_i, best_k = np.unravel_index(best_flat, full.shape)
best = float(full[best_i, best_k])
baseline = float(base_scores.mean())
groups = [name.split('_')[0] for name in NAMES]
cv = []
for j, group in enumerate(groups):
    train = np.asarray([g != group for g in groups])
    means = scores[:, :, train].mean(axis=2)
    if means.max() <= base_scores[train].mean():
        cv.append({'image': NAMES[j], 'run': None, 'class': None, 'score': float(base_scores[j])})
    else:
        k = int(means.argmax())
        i, c = np.unravel_index(k, full.shape)
        cv.append({'image': NAMES[j], 'run': RECORDS[i]['run'],
                   'gate': rules[c][0], 'subset': rules[c][1],
                   'score': float(scores[i, c, j])})
cv_score = float(np.mean([x['score'] for x in cv]))
result = {
    'baseline_percent': baseline * 100,
    'best_run': RECORDS[best_i]['run'],
    'best_gate': rules[best_k][0],
    'best_subset': rules[best_k][1],
    'best_percent': best * 100,
    'best_delta_pp': (best - baseline) * 100,
    'fixed_stage1_specimen_leave_one_out_percent': cv_score * 100,
    'fixed_stage1_specimen_leave_one_out_delta_pp': (cv_score - baseline) * 100,
    'cv_choices': cv,
    'per_run_best': [{'run': row['run'], 'best_percent': float(full[i].max()*100),
                      'gate': rules[int(full[i].argmax())][0],
                      'subset': rules[int(full[i].argmax())][1]}
                     for i,row in enumerate(RECORDS)],
    'stage1_chosen_on_repeated_valid': True,
    'independent_generalization_verified': False,
    'test_used': False,
}
(HERE / 'greedy_extension_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('baseline_percent','best_run','best_gate','best_subset','best_percent',
      'best_delta_pp','fixed_stage1_specimen_leave_one_out_percent',
      'fixed_stage1_specimen_leave_one_out_delta_pp')}, ensure_ascii=False, indent=2))
