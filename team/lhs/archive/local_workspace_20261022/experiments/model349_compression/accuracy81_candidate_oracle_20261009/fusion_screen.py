"""CPU-only class-priority mask fusion screen across completed valid20 runs."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GT = Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad/gt/valid/masks')
inventory = json.loads((HERE / 'audit_result.json').read_text())
records = inventory['records']
names = inventory['names']
base_run = ROOT / inventory['baseline_run']
manifest = json.loads((base_run / 'input_manifest.json').read_text())
palette = np.asarray(manifest['config']['palette'], dtype=np.uint8)


def label(path):
    arr = np.asarray(Image.open(path))
    assert arr.ndim == 2 and np.all(arr < 4)
    return arr.astype(np.uint8)


def gt(name):
    p = GT / f'{name}_mask.png'
    arr = np.asarray(Image.open(p))
    if arr.ndim == 2 and np.all(arr < 4):
        return arr.astype(np.uint8)
    rgb = np.asarray(Image.open(p).convert('RGB'))
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for k, color in enumerate(palette):
        out[np.all(rgb == color, axis=-1)] = k
    assert np.all(out < 4)
    return out


def score(hist, subset, gate):
    cm = np.zeros((4, 4), dtype=np.int64)
    for r in range(4):
        for c in range(4):
            chosen = c if ((subset >> (c if gate == 'candidate' else r)) & 1) else r
            cm[:, chosen] += hist[r, c]
    inter = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - inter
    return float(np.divide(inter, union, out=np.zeros(4, dtype=float), where=union > 0).mean())


targets = {name: gt(name) for name in names}
baseline = {name: label(base_run / 'predicted_labels' / f'{name}.png') for name in names}
assert all(baseline[name].shape == targets[name].shape for name in names)
configurations = [(gate, subset) for gate in ('candidate', 'release') for subset in range(1, 16)]
matrix = np.empty((len(records), len(configurations), len(names)), dtype=np.float64)
for i, row in enumerate(records):
    run = ROOT / row['run']
    for j, name in enumerate(names):
        candidate = label(run / 'predicted_labels' / f'{name}.png')
        release, truth = baseline[name], targets[name]
        assert candidate.shape == release.shape
        hist = np.bincount((release.astype(np.int64) * 16 +
                            candidate.astype(np.int64) * 4 + truth).ravel(),
                           minlength=64).reshape(4, 4, 4)
        for k, (gate, subset) in enumerate(configurations):
            matrix[i, k, j] = score(hist, subset, gate)
    if (i + 1) % 10 == 0:
        print('scored', i + 1, '/', len(records), flush=True)

baseline_score = inventory['baseline_percent'] / 100
full_means = matrix.mean(axis=2)
best_flat = int(full_means.argmax())
best_i, best_k = np.unravel_index(best_flat, full_means.shape)
groups = [name.split('_')[0] for name in names]
cv = []
for j, group in enumerate(groups):
    train = np.asarray([g != group for g in groups])
    flat = int(matrix[:, :, train].mean(axis=2).argmax())
    i, k = np.unravel_index(flat, full_means.shape)
    cv.append({'image': names[j], 'selected_run': records[i]['run'],
               'gate': configurations[k][0], 'subset': configurations[k][1],
               'score': float(matrix[i, k, j])})
cv_score = float(np.mean([v['score'] for v in cv]))
best_row = records[best_i]
report = {
    'metric': inventory['metric'],
    'run_count': len(records),
    'rule_count_per_run': len(configurations),
    'baseline_percent': baseline_score * 100,
    'best_full_run': best_row['run'],
    'best_full_gate': configurations[best_k][0],
    'best_full_subset': configurations[best_k][1],
    'best_full_percent': float(full_means[best_i, best_k] * 100),
    'best_full_delta_pp': float((full_means[best_i, best_k] - baseline_score) * 100),
    'specimen_leave_one_out_percent': cv_score * 100,
    'specimen_leave_one_out_delta_pp': (cv_score - baseline_score) * 100,
    'cv_choices': cv,
    'per_run_best': [{'run': row['run'], 'best_percent': float(full_means[i].max() * 100),
                      'gate': configurations[int(full_means[i].argmax())][0],
                      'subset': configurations[int(full_means[i].argmax())][1]}
                     for i, row in enumerate(records)],
    'selection_uses_repeated_valid_labels': True,
    'independent_generalization_verified': False,
    'test_used': False,
}
(HERE / 'fusion_screen_result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('best_full_run', 'best_full_gate', 'best_full_subset',
      'best_full_percent', 'best_full_delta_pp', 'specimen_leave_one_out_percent',
      'specimen_leave_one_out_delta_pp')}, ensure_ascii=False, indent=2))
