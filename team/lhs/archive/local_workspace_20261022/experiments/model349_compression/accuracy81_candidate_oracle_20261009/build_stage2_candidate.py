"""Build the selected hard-mask fusion without reading validation labels."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
report = json.loads((HERE / 'greedy_extension_result.json').read_text())
assert report['best_gate'] == 'candidate' and report['best_subset'] == 4
base = HERE / 'fusion_candidate_valid20'
other = ROOT / report['best_run']
out = HERE / 'fusion_stage2_candidate_valid20'
assert not out.exists()
out.mkdir()
(out / 'predicted_labels').mkdir()
(out / 'predicted_masks').mkdir()
manifest = json.loads((base / 'input_manifest.json').read_text())
other_manifest = json.loads((other / 'input_manifest.json').read_text())
assert {v['name']: v['sha256'] for v in manifest['images']} == {v['name']: v['sha256'] for v in other_manifest['images']}
palette = np.asarray(manifest['config']['palette'], dtype=np.uint8)
shutil.copy2(base / 'input_manifest.json', out / 'input_manifest.json')
rows = []
for item in manifest['images']:
    name = item['name']
    release = np.asarray(Image.open(base / 'predicted_labels' / f'{name}.png'), dtype=np.uint8)
    candidate = np.asarray(Image.open(other / 'predicted_labels' / f'{name}.png'), dtype=np.uint8)
    assert release.shape == candidate.shape and np.all(release < 4) and np.all(candidate < 4)
    use = ((1 << candidate) & 4) != 0
    pred = np.where(use, candidate, release).astype(np.uint8)
    Image.fromarray(pred).save(out / 'predicted_labels' / f'{name}.png')
    Image.fromarray(palette[pred]).save(out / 'predicted_masks' / f'{name}_mask.png')
    rows.append({'name': name, 'changed_pixels': int(np.count_nonzero(pred != release)),
                 'total_pixels': int(pred.size)})

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

(HERE / 'fusion_stage2_candidate_build.json').write_text(json.dumps({
    'base_run': str(base), 'other_run': str(other), 'rule': 'candidate label 2 overrides stage1; otherwise stage1',
    'images': rows, 'input_manifest_sha256': sha(out / 'input_manifest.json'),
    'gt_read_during_build': False, 'test_used': False,
}, ensure_ascii=False, indent=2) + '\n')
print(out)
