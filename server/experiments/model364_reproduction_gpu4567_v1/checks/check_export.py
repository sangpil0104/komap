"""Synthetic validation of fixed gates, image metric and the 20-image export."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for rel in ['adapters', 'source/experiments', 'source/experiments/historical_scripts']:
    sys.path.insert(0, str(ROOT / rel))
import export_model364_fixed as fixed
import local_result_export as export
from local_image_metrics import class_ious

params = json.loads((ROOT / 'source/config/model364_recipe.json').read_text())['fixed_route_parameters']
spatial = fixed.upstream.model360.spatial
# Capture the blended probabilities before the original spatial route is run.
original_parts, original_prediction = spatial.spatial_parts, spatial.prediction
spatial.spatial_parts = lambda row, *args: row['sources']['336'].copy()
spatial.prediction = lambda parts, *args: parts
p336 = np.full((4, 2, 2), .25, dtype=np.float32)
p351 = p336.copy(); p351[:, 0, 0] = [.85, .05, .05, .05]
p361 = p336.copy(); p361[:, 0, 0] = [.97, .01, .01, .01]
p361[:, 1, 1] = [.01, .97, .01, .01]
row = {'sources': {'336': p336}}
try:
    mixed = fixed.fixed_prediction(row, p336, p351, p361, {}, [], params)
    expected = p351.copy()
    expected[:, 0, 0] = .75 * p351[:, 0, 0] + .25 * p361[:, 0, 0]
    expected[:, 1, 1] = .75 * p351[:, 1, 1] + .25 * p361[:, 1, 1]
    np.testing.assert_allclose(mixed, expected, rtol=1e-6, atol=1e-7)
    assert row['sources']['336'] is p336
finally:
    spatial.spatial_parts, spatial.prediction = original_parts, original_prediction

parent = ROOT / 'outputs/checks'
parent.mkdir(parents=True, exist_ok=True)
run = Path(tempfile.mkdtemp(prefix='synthetic_export_', dir=parent))
images = run / 'data/valid/images'; images.mkdir(parents=True)
os.environ['KOMAP_RESULTS_ROOT'] = str(run / 'synthetic_results')
rows, names, preds = [], [], {}
for i in range(20):
    name = f'A{i:02d}_fixture'
    target = np.tile(np.arange(4, dtype=np.uint8), (4, 1))
    pred = target.copy()
    if i == 0:
        pred[0, 1] = 2
    elif i == 1:
        target[:] = 2; pred[:] = 2
    item = {'item': {'target': target}}
    rows.append(item); names.append(name); preds[id(item)] = pred
    Image.fromarray(np.full((4, 4, 3), i, dtype=np.uint8)).save(images / f'{name}_image.png')
scores = [class_ious(preds[id(row)], row['item']['target']).mean() for row in rows]
assert scores[1] == .25
result = {'raw_best_valid_image_mean_miou_empty0': float(np.mean(scores))}
out = export.export_final(rows, names, lambda row: preds[id(row)], run, result)
record = json.loads((out / 'metrics.json').read_text())
assert record['model'].startswith('Model364')
assert record['image_miou'] == float(np.mean(scores))
assert len(list((out / 'valid_mask_comparison').glob('*.png'))) == 20
assert (out / f'{out.name}.md').is_file()
with Image.open(out / 'valid_mask_comparison/A00_fixture.png') as im:
    assert im.getpixel((4 + 1, 76)) == tuple(export.PALETTE[1])
    assert im.getpixel((4, 76)) == (255, 255, 255)
    assert im.getpixel((1, 4 + 2 * 76)) == tuple(export.PALETTE[2])
    assert im.getpixel((4 + 1, 4 + 2 * 76)) == tuple(export.PALETTE[1])
assert export.reserve_result(out.parent) != out
report = {'training_started': False, 'fixed_entropy_gates': 'passed',
          'empty_union_zero': 'passed', 'image_average': 'passed',
          'comparison_layout_and_ground_truth_error_colors': 'passed',
          '20_images_and_markdown': 'passed', 'no_overwrite': 'passed',
          'synthetic_directory': str(run)}
(ROOT / 'reports/export_validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
