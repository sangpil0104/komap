"""CPU screen: replace release labels only where both P100 routes agree."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRANSFER = ROOT / 'accuracy81_p100_exh14_transfer_20261009'
RELEASE = ROOT / 'exh14_base_search_20261008/exh14_valid20/predicted_labels'
P100_EXH14 = next((TRANSFER / 'collected').iterdir()) / 'runs/p100_exh14_valid20/predicted_labels'
P100_OLD = next((TRANSFER / 'matched_collected').iterdir()) / 'runs/p100_original_matched_valid20/predicted_labels'
GT = Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad/gt/valid/masks')
manifest = json.loads((RELEASE.parent / 'input_manifest.json').read_text())
palette = np.asarray(manifest['config']['palette'], dtype=np.uint8)


def target_for(name):
    path = GT / f'{name}_mask.png'
    raw = np.asarray(Image.open(path))
    if raw.ndim == 2 and np.isin(raw, range(4)).all():
        return raw.astype(np.uint8)
    rgb = np.asarray(Image.open(path).convert('RGB'))
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for index, color in enumerate(palette):
        out[np.all(rgb == color, axis=-1)] = index
    assert np.all(out < 4)
    return out


def score(pred, target):
    matrix = np.bincount((target.astype(np.int64) * 4 + pred).ravel(), minlength=16).reshape(4, 4)
    inter = np.diag(matrix)
    union = matrix.sum(0) + matrix.sum(1) - inter
    return float(np.divide(inter, union, out=np.zeros(4, dtype=float), where=union > 0).mean())


rows = []
for item in manifest['images']:
    name = item['name']
    release, transfer, original = [np.asarray(Image.open(folder / f'{name}.png'), dtype=np.uint8)
                                   for folder in (RELEASE, P100_EXH14, P100_OLD)]
    target = target_for(name)
    assert release.shape == transfer.shape == original.shape == target.shape
    assert all(np.all(arr < 4) for arr in (release, transfer, original))
    scores = []
    agree = transfer == original
    for subset in range(16):
        use = agree & (((1 << transfer) & subset) != 0)
        pred = np.where(use, transfer, release)
        scores.append(score(pred, target))
    rows.append({'name': name, 'specimen': name.split('_')[0], 'scores': scores,
                 'release': score(release, target), 'transfer': score(transfer, target),
                 'original': score(original, target),
                 'consensus_pixels': int(np.count_nonzero(agree & (transfer != release))),
                 'total_pixels': int(release.size)})

score_matrix = np.asarray([row['scores'] for row in rows])
assert np.allclose(score_matrix[:, 0], [row['release'] for row in rows], atol=1e-12)
full = score_matrix.mean(axis=0)
best = int(np.argmax(full))
holdout = []
for specimen in sorted({row['specimen'] for row in rows}):
    test = np.asarray([row['specimen'] == specimen for row in rows])
    train = ~test
    chosen = int(np.argmax(score_matrix[train].mean(axis=0)))
    holdout += [(i, chosen) for i in np.flatnonzero(test)]
holdout.sort()
oof = float(np.mean([score_matrix[i, chosen] for i, chosen in holdout]))
baseline = float(full[0])
result = {
    'metric': 'image-wise four-class mIoU, empty union=0',
    'candidate': 'release mask except P100 transfer and original agree on a subset of predicted classes',
    'candidate_count': 16,
    'class_order': manifest['config']['class_order'],
    'release_percent': baseline * 100,
    'p100_transfer_percent': float(np.mean([row['transfer'] for row in rows])) * 100,
    'p100_original_percent': float(np.mean([row['original'] for row in rows])) * 100,
    'best_full_subset': best,
    'best_full_percent': float(full[best] * 100),
    'best_full_delta_pp': float((full[best] - baseline) * 100),
    'specimen_leave_one_out_percent': oof * 100,
    'specimen_leave_one_out_delta_pp': (oof - baseline) * 100,
    'screen_gate_pp': 0.5,
    'screen_pass': (oof - baseline) * 100 >= 0.5,
    'per_subset_percent': (full * 100).tolist(),
    'holdout_subsets': {rows[i]['name']: chosen for i, chosen in holdout},
    'rows': rows,
    'independent_generalization_verified': False,
    'test_used': False,
}
(HERE / 'screen_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: result[key] for key in ('release_percent', 'best_full_subset',
     'best_full_percent', 'best_full_delta_pp', 'specimen_leave_one_out_percent',
     'specimen_leave_one_out_delta_pp', 'screen_pass')}, indent=2))
