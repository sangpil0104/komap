"""GPU output equality of identical image inputs shared across fixed B4 folds."""
import gc, importlib.util, json, os, sys, time
from pathlib import Path
import torch
import cv2
torch.set_num_threads(2)
cv2.setNumThreads(2)
R = Path('/workspace/model349_compress/base12_inputcache_20261008')
B = R / 'bundles'
sys.path.insert(0, str(B / 'control12/src'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


old = load('original_b4', B / 'control12/src/b4_zcdf.py')
new = load('cached_b4', B / 'cache12/src/b4_zcdf.py')
image = next(iter(sorted(Path('/workspace/komap_model1/data/valid/images').glob('*.png'))))
rows = []
for component, folds in [('326', [1, 3]), ('329', [1, 2, 3, 4])]:
    folder = f'model{component}_' + ('hybrid_fullstats_group_oof' if component == '326' else 'fullstats_width4000_uniform_group_oof')
    members = [old.load_member(str(B / 'control12/weights' / folder / f'fold{x}'), 'cuda:0') for x in folds]
    gray = old.load_gray(image)
    calls = {'old': 0, 'new': 0}
    f_old, f_new = old.make_input, new.make_input
    def a(*args, **kwargs):
        calls['old'] += 1
        return f_old(*args, **kwargs)
    def b(*args, **kwargs):
        calls['new'] += 1
        return f_new(*args, **kwargs)
    old.make_input, new.make_input = a, b
    torch.cuda.synchronize()
    start = time.monotonic()
    expected = old.predict_probs(members, gray, tta=False, device='cuda:0')
    torch.cuda.synchronize()
    t_old = time.monotonic() - start
    start = time.monotonic()
    actual = new.predict_probs(members, gray, tta=False, device='cuda:0')
    torch.cuda.synchronize()
    t_new = time.monotonic() - start
    maximum = float((expected - actual).abs().max())
    equal = torch.equal(expected, actual)
    rows.append({'component': component, 'image': image.name, 'folds': folds,
                 'exact_tensor_equal': equal, 'max_absolute_error': maximum,
                 'preprocessing_calls': calls, 'single_probe_seconds': [t_old, t_new]})
    (R / 'probe_result.json').write_text(json.dumps(rows, indent=2) + '\n')
    assert equal and calls == {'old': len(folds), 'new': 1}
    old.make_input, new.make_input = f_old, f_new
    del members, expected, actual
    gc.collect()
    torch.cuda.empty_cache()
print(json.dumps(rows, indent=2))
