"""Audit all completed valid20 masks and image-selection headroom; CPU only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GT = Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad/gt/valid/masks')
BASE = ROOT / 'exh14_base_search_20261008/exh14_valid20'
spec = importlib.util.spec_from_file_location('audit_runs', ROOT / 'audit_20261008/analyze_runs.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
base = audit.score(BASE, GT)
names = sorted(base['images'])
base_inputs = base['input_sha256']


def signature(run):
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode())
        h.update((run / 'predicted_labels' / f'{name}.png').read_bytes())
    return h.hexdigest()


records = []
skips = []
seen = {}
for labels_dir in sorted(ROOT.glob('**/predicted_labels')):
    run = labels_dir.parent
    if len(list(labels_dir.glob('*.png'))) != 20:
        continue
    try:
        scored = audit.score(run, GT)
        assert scored['input_sha256'] == base_inputs
        assert sorted(scored['images']) == names
        key = signature(run)
        if key in seen:
            seen[key]['duplicate_paths'].append(str(run.relative_to(ROOT)))
            continue
        row = {'run': str(run.relative_to(ROOT)), 'signature': key,
               'miou_percent': 100 * scored['miou'],
               'image_scores': [scored['images'][name]['image_miou'] for name in names],
               'class_scores': [scored['images'][name]['class_iou'] for name in names],
               'duplicate_paths': []}
        records.append(row)
        seen[key] = row
    except Exception as exc:
        skips.append({'run': str(run.relative_to(ROOT)), 'reason': repr(exc)})

assert any(row['run'] == str(BASE.relative_to(ROOT)) for row in records)
scores = np.asarray([row['image_scores'] for row in records], dtype=float)
class_scores = np.asarray([row['class_scores'] for row in records], dtype=float)
base_index = next(i for i, row in enumerate(records) if row['run'] == str(BASE.relative_to(ROOT)))
best_global = int(scores.mean(axis=1).argmax())
oracle_index = scores.argmax(axis=0)
oracle_score = float(scores.max(axis=0).mean())
baseline = float(scores[base_index].mean())
best_pair = None
for candidate in range(len(records)):
    if candidate == base_index:
        continue
    value = float(np.maximum(scores[base_index], scores[candidate]).mean())
    if best_pair is None or value > best_pair['oracle_percent'] / 100:
        best_pair = {'run': records[candidate]['run'], 'oracle_percent': value * 100,
                     'oracle_delta_pp': (value - baseline) * 100,
                     'candidate_percent': float(scores[candidate].mean() * 100)}

groups = [name.split('_')[0] for name in names]
cv_choice = []
for i, group in enumerate(groups):
    train = np.asarray([g != group for g in groups])
    selected = int(scores[:, train].mean(axis=1).argmax())
    cv_choice.append(selected)
cv_score = float(np.mean([scores[model, i] for i, model in enumerate(cv_choice)]))

summary = {
    'metric': '20-image mean of image-wise four-class IoU; empty union=0',
    'input_sha_exact': True,
    'candidate_runs_found': len(records) + sum(len(r['duplicate_paths']) for r in records) + len(skips),
    'unique_candidate_masks_scored': len(records),
    'duplicate_mask_runs': sum(len(r['duplicate_paths']) for r in records),
    'skips': skips,
    'baseline_run': records[base_index]['run'],
    'baseline_percent': baseline * 100,
    'best_global_run': records[best_global]['run'],
    'best_global_percent': float(scores[best_global].mean() * 100),
    'best_pair_image_oracle': best_pair,
    'all_candidate_image_oracle_percent': oracle_score * 100,
    'all_candidate_image_oracle_delta_pp': (oracle_score - baseline) * 100,
    'oracle_selected_runs': {name: records[int(index)]['run'] for name, index in zip(names, oracle_index)},
    'specimen_leave_one_out_global_selection_percent': cv_score * 100,
    'specimen_leave_one_out_global_selection_delta_pp': (cv_score - baseline) * 100,
    'per_class_image_oracle_upper_bound_percent': float(class_scores.max(axis=0).mean() * 100),
    'oracle_uses_ground_truth_for_selection': True,
    'independent_generalization_verified': False,
    'test_used': False,
    'names': names,
    'records': records,
}
(HERE / 'audit_result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: summary[key] for key in ('unique_candidate_masks_scored', 'skips',
      'baseline_percent', 'best_global_run', 'best_global_percent', 'best_pair_image_oracle',
      'all_candidate_image_oracle_percent', 'specimen_leave_one_out_global_selection_percent',
      'per_class_image_oracle_upper_bound_percent')}, indent=2, ensure_ascii=False))
