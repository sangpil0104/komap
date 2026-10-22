"""Independently score collected probability-fusion PNGs; no training."""
from pathlib import Path
import importlib.util
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GT = Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad/gt/valid/masks')
spec = importlib.util.spec_from_file_location('audit', ROOT / 'audit_20261008/analyze_runs.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
result = json.loads((HERE / 'probability_fusion_result.json').read_text())
runs = HERE / 'probability_fusion_runs'
base = runs / 'config0'
candidate = runs / f"config{result['best_index']}"
a, b = audit.score(base, GT), audit.score(candidate, GT)
assert abs(b['miou'] * 100 - result['best_percent']) < 1e-10
assert abs(a['miou'] * 100 - result['parent_percent']) < 1e-10
canonical = ROOT / 'exh14_base_search_20261008/exh14_valid20'
c = audit.score(canonical, GT)
report = {
    'metric': '20 image-wise four-class IoUs averaged; empty union=0',
    'best_gamma': result['best_gamma'],
    'candidate_percent': b['miou'] * 100,
    'parent_percent': a['miou'] * 100,
    'canonical_percent': c['miou'] * 100,
    'vs_parent': audit.pair(base, candidate, a, b),
    'vs_canonical': audit.pair(canonical, candidate, c, b),
    'specimen_leave_one_out_percent': result['specimen_leave_one_out_percent'],
    'independent_png_gt_scoring': True,
    'selection_used_repeated_valid_labels': True,
    'independent_generalization_verified': False,
    'fresh_raw_image_reproduction': False,
    'test_used': False,
}
(HERE / 'probability_fusion_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['best_gamma', 'candidate_percent', 'vs_parent']}, indent=2))
