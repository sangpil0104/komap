"""Check the frozen class-0/2 override on three independent base masks."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
P100 = ROOT / 'accuracy81_p100_exh14_transfer_20261009'
RELEASE = ROOT / 'exh14_base_search_20261008/exh14_valid20'
BASES = {
    'release': RELEASE,
    'p100_transfer': next((P100 / 'collected').iterdir()) / 'runs/p100_exh14_valid20',
    'p100_original_fresh': next((P100 / 'matched_collected').iterdir()) / 'runs/p100_original_matched_valid20',
}
OTHER = ROOT / json.loads((HERE / 'fusion_screen_result.json').read_text())['best_full_run']
GT = Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad/gt/valid/masks')
manifest = json.loads((RELEASE / 'input_manifest.json').read_text())
palette = np.asarray(manifest['config']['palette'], dtype=np.uint8)


def target(name):
    p = GT / f'{name}_mask.png'
    raw = np.asarray(Image.open(p))
    if raw.ndim == 2 and np.all(raw < 4):
        return raw
    rgb = np.asarray(Image.open(p).convert('RGB'))
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for i, color in enumerate(palette):
        out[np.all(rgb == color, axis=-1)] = i
    assert np.all(out < 4)
    return out


def iou(pred, truth):
    vals = []
    for k in range(4):
        a, b = pred == k, truth == k
        union = np.count_nonzero(a | b)
        vals.append(np.count_nonzero(a & b) / union if union else 0.0)
    return float(np.mean(vals))


names = [v['name'] for v in manifest['images']]
for base in BASES.values():
    other_manifest = json.loads((base / 'input_manifest.json').read_text())
    assert {x['name']:x['sha256'] for x in manifest['images']} == {x['name']:x['sha256'] for x in other_manifest['images']}
rows = []
for name in names:
    truth = target(name)
    expert = np.asarray(Image.open(OTHER / 'predicted_labels' / f'{name}.png'), dtype=np.uint8)
    use = np.isin(expert, [0, 2])
    item = {'image': name}
    for key, run in BASES.items():
        base = np.asarray(Image.open(run / 'predicted_labels' / f'{name}.png'), dtype=np.uint8)
        assert base.shape == expert.shape == truth.shape
        item[key + '_base'] = iou(base, truth)
        item[key + '_fusion'] = iou(np.where(use, expert, base), truth)
    rows.append(item)
result = {'rule':'Model401 labels 0/2 override base', 'candidate_fixed_before_crossweight_check':True,
          'scores': {key:{'base_percent':100*np.mean([r[key+'_base'] for r in rows]),
                          'fusion_percent':100*np.mean([r[key+'_fusion'] for r in rows]),
                          'delta_pp':100*np.mean([r[key+'_fusion']-r[key+'_base'] for r in rows])}
                     for key in BASES}, 'rows':rows,
          'repeated_valid':True,'independent_generalization_verified':False,'test_used':False}
(HERE / 'fixed_rule_crossweight_result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result['scores'],indent=2))
