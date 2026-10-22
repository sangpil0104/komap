"""Independent PNG/GT scoring for the materialized fusion candidate."""
from pathlib import Path
import importlib.util
import json
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = HERE / 'fusion_stage2_candidate_valid20'
BASE = ROOT / 'exh14_base_search_20261008/exh14_valid20'
GT = Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad/gt/valid/masks')
manifest = json.loads((RUN / 'input_manifest.json').read_text())
palette = np.asarray(manifest['config']['palette'], dtype=np.uint8)
rows = []
for item in manifest['images']:
    name = item['name']
    pred = np.asarray(Image.open(RUN / 'predicted_labels' / f'{name}.png'))
    raw = np.asarray(Image.open(GT / f'{name}_mask.png'))
    if raw.ndim == 2 and np.all(raw < 4):
        target = raw
    else:
        rgb = np.asarray(Image.open(GT / f'{name}_mask.png').convert('RGB'))
        target = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        for k, color in enumerate(palette):
            target[np.all(rgb == color, axis=-1)] = k
    assert pred.shape == target.shape and np.all(pred < 4) and np.all(target < 4)
    ious = []
    for k in range(4):
        p, t = pred == k, target == k
        union = np.count_nonzero(p | t)
        ious.append(float(np.count_nonzero(p & t) / union) if union else 0.0)
    rows.append({'image': name, 'class_iou': ious, 'image_miou': float(np.mean(ious))})
assert len(rows) == 20
mean = float(np.mean([r['image_miou'] for r in rows]))
metrics = {'image_miou': mean, 'images': rows}
(RUN / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

spec = importlib.util.spec_from_file_location('audit_runs', ROOT / 'audit_20261008/analyze_runs.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
a = audit.score(BASE, GT)
b = audit.score(RUN, GT)
assert abs(b['miou'] - mean) < 1e-12
pair = audit.pair(BASE, RUN, a, b)
screen = json.loads((HERE / 'greedy_extension_result.json').read_text())
assert abs(mean * 100 - screen['best_percent']) < 1e-10
report = {'metric': 'mean of 20 image-wise four-class IoUs; empty union=0',
          'baseline_percent': a['miou'] * 100, 'candidate_percent': b['miou'] * 100,
          'vs_baseline': pair, 'independent_png_gt_scoring': True,
          'input_sha_exact': a['input_sha256'] == b['input_sha256'],
          'selection_used_repeated_valid_labels': True,
          'independent_generalization_verified': False, 'test_used': False}
(HERE / 'fusion_stage2_candidate_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'candidate_percent': b['miou'] * 100,
                  'delta_pp': pair['delta_pp'],
                  'worst_image': pair['worst_image'],
                  'worst_delta_pp': pair['worst_image_delta_pp']}, indent=2))
