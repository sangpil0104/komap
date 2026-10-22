"""Local verification only; reference caches and data stay outside the ZIP."""
import gc
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path('/Data3/sppark/komap')
BUNDLE = ROOT / 'results/model349_inference_bundle_20261007_1'
RUN = ROOT / 'experiments/model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z'
EXPORT = ROOT / 'experiments/model349_mask_export_v1/outputs/20261007_1'
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('portable_model349', BUNDLE / 'infer.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
config = runtime.load_config(BUNDLE / 'configs/model349.yaml')


def canonical(name):
    return Path(name).stem.removesuffix('_image').removesuffix('_mask')


with (RUN / 'outputs/model183_metric_calibration/entries.pkl').open('rb') as f:
    entries = pickle.load(f)
paths = {'269': RUN / 'outputs/model269_fullstats_1x_extended/valid_probs.npz'}
paths.update({k: RUN / 'outputs' / d / 'valid_ensemble/probabilities.npz' for k, d in runtime.B4_DIRS.items()})
paths.update({k: RUN / 'outputs' / d / 'valid_eval/valid_probabilities.npz' for k, d in runtime.HR_DIRS.items()})
archives = {}
for c, path in tqdm(list(paths.items()), desc='Load reference probabilities', unit='archive'):
    with np.load(path) as a:
        archives[c] = {canonical(k): a[k].astype(np.float32)[runtime.TEAM_TO_MAIN if c != '269' else slice(None)] for k in a.files}
names = sorted(canonical(p.name) for p in (RUN / 'data/valid/images').glob('*_image.png'))
rows = []
for name, item in tqdm(list(zip(names, entries)), desc='Portable route vs original valid', unit='image'):
    pred = runtime.predict_fixed(item, {c: a[name] for c, a in archives.items()}, config)
    reference = np.array(Image.open(ROOT / 'results/model_results_0930_1/predicted_labels' / f'{name}.png'))
    different = int(np.count_nonzero(pred != reference))
    assert different == 0, (name, different)
    rows.append({'image': name, 'different_pixels': different})
del archives, entries
gc.collect()
test_rows = []
for p in tqdm(sorted((EXPORT / 'components/base').glob('*.pkl')), desc='Portable route vs exported masks', unit='image'):
    name = p.stem
    with p.open('rb') as f:
        item = pickle.load(f)
    sources = {c: np.load(EXPORT / 'components' / c / f'{name}.npy').astype(np.float32) for c in runtime.COMPONENTS if c != 'base'}
    pred = runtime.predict_fixed(item, sources, config)
    rgb = np.array(config['palette'], np.uint8)[pred]
    split = 'valid' if name == 'A117_step4_sample2' else 'test'
    reference = np.array(Image.open(EXPORT / 'bundle' / split / f'{name}_mask.png'))
    different = int(np.count_nonzero(np.any(rgb != reference, axis=-1)))
    assert different == 0, (name, different)
    test_rows.append({'image': name, 'split': split, 'different_pixels': different})
result = {'cached_validation_route': {'image_count': len(rows), 'all_pixel_identical': True, 'images': rows},
          'exported_mask_route': {'image_count': len(test_rows), 'all_pixel_identical': True, 'images': test_rows}}
runtime.write_json(HERE / 'checks/cached_parity.json', result)
print(json.dumps({k: {'image_count': v['image_count'], 'all_pixel_identical': True} for k, v in result.items()}, indent=2))
