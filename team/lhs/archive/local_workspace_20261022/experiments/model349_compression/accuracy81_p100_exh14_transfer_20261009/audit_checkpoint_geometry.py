"""Read-only CPU geometry audit of corresponding release and P100 weights."""
from pathlib import Path
import hashlib
import json
import math
import torch

torch.set_num_threads(2)
ROOT = Path('/workspace/model349_compress')
RELEASE = ROOT / 'Model349_exh14/weights'
P100 = Path('/workspace/komap_model1/outputs')
MANIFEST = ROOT / 'accuracy81_p100_exh14_transfer_20261009/bundle/checkpoint_manifest.json'
OUT = ROOT / 'accuracy81_p100_exh14_transfer_20261009/checkpoint_geometry_audit.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if 'model' in obj and isinstance(obj['model'], dict):
        return obj['model']
    if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        return obj['state_dict']
    return obj


manifest = json.loads(MANIFEST.read_text())
rows = []
for entry in manifest['checkpoints']:
    rel = Path(entry['path']).relative_to('weights')
    release, p100 = RELEASE / rel, P100 / rel
    assert p100.resolve() == Path(entry['source_path']).resolve()
    assert sha(p100) == entry['sha256']
    a, b = state(release), state(p100)
    assert list(a) == list(b)
    ref_square = diff_square = dot = 0.0
    tensor_count = changed = exact = 0
    for key, x in a.items():
        y = b[key]
        assert x.shape == y.shape and x.dtype == y.dtype
        if not torch.is_tensor(x) or not torch.is_floating_point(x):
            continue
        xf, yf = x.float(), y.float()
        assert torch.isfinite(xf).all() and torch.isfinite(yf).all()
        ref_square += float((xf * xf).sum())
        diff_square += float(((xf - yf) ** 2).sum())
        dot += float((xf * yf).sum())
        tensor_count += 1
        if torch.equal(x, y):
            exact += 1
        else:
            changed += 1
    row = {'path': str(rel), 'release_sha256': sha(release), 'p100_sha256': entry['sha256'],
           'relative_l2': math.sqrt(diff_square / ref_square), 'dot_over_reference_norm2': dot / ref_square,
           'floating_tensors': tensor_count, 'changed_floating_tensors': changed,
           'exact_floating_tensors': exact}
    rows.append(row)
    print(row['path'], row['relative_l2'], flush=True)
    del a, b

OUT.write_text(json.dumps({'rows': rows, 'count': len(rows), 'cpu_only': True,
                           'release_and_p100_weights_unchanged': True}, indent=2) + '\n')
