"""CPU screen of a texture/probability correction on Model392 train OOF."""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.path.insert(0, str(Path(__file__).resolve().parent / 'deps'))
import cv2
import lightgbm as lgb
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = Path('/workspace/komap_model1/data/train')
OOF = Path('/workspace/komap_model1/outputs/model392_widthnorm/seed1')
PALETTE = np.array([(153, 127, 76), (25, 76, 153), (76, 178, 76), (204, 204, 204)], np.uint8)
FEATURES = ['gray', 'mean5', 'std5', 'mean15', 'std15', 'gradient',
            'p_al', 'p_al3ni', 'p_primary', 'p_eutectic', 'p_max', 'margin']


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def target(path):
    rgb = np.asarray(Image.open(path).convert('RGB'), np.uint8)
    y = np.full(rgb.shape[:2], 255, np.uint8)
    for c, color in enumerate(PALETTE):
        y[np.all(rgb == color, axis=2)] = c
    assert not np.any(y == 255), path
    return y


def image_features(name, prob):
    gray = np.asarray(Image.open(DATA / 'images' / f'{name}_image.png').convert('L'), np.float32) / 255.0
    assert prob.shape == (4, *gray.shape) and np.isfinite(prob).all()
    means = [cv2.blur(gray, (k, k)) for k in (5, 15)]
    stds = [np.sqrt(np.maximum(cv2.blur(gray * gray, (k, k)) - m*m, 0))
            for k, m in zip((5, 15), means)]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    top = np.sort(prob, axis=0)
    arrays = [gray, means[0], stds[0], means[1], stds[1], gradient,
              *prob, top[-1], top[-1] - top[-2]]
    x = np.stack(arrays, axis=-1).reshape(-1, len(FEATURES))
    assert np.isfinite(x).all()
    return x


def ious(pred, truth):
    matrix = np.bincount((truth.ravel().astype(np.int64)*4 + pred.ravel()).ravel(),
                         minlength=16).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    return np.divide(intersection, union, out=np.zeros(4), where=union > 0).tolist()


def main():
    cv2.setNumThreads(2)
    archives = {f: OOF / f'fold{f}_oof/valid_probabilities.npz' for f in (1, 2, 3, 4)}
    assert all(p.is_file() for p in archives.values())
    assert not (ROOT / 'screen_result.json').exists()
    rng = np.random.default_rng(39281)
    xs, ys = [], []
    archive_sha = {str(f): sha(p) for f, p in archives.items()}
    ids = {}
    for fold in (1, 2, 3, 4):
        with np.load(archives[fold], allow_pickle=False) as archive:
            ids[fold] = sorted(archive.files)
            assert len(ids[fold]) in (17, 18)
            if fold == 4:
                continue
            for name in ids[fold]:
                prob = np.asarray(archive[name], np.float32)
                truth = target(DATA / 'masks' / f'{name}_mask.png')
                assert prob.shape[1:] == truth.shape
                x = image_features(name, prob)
                flat = truth.ravel()
                n = len(flat)
                uniform = rng.choice(n, min(7000, n), replace=False)
                margin = x[:, -1]
                uncertain = np.argpartition(margin, min(7000, n)-1)[:min(7000, n)]
                chosen = np.unique(np.concatenate([uniform, uncertain]))
                xs.append(x[chosen]); ys.append(flat[chosen])
                print('train', fold, name, len(chosen), flush=True)
    assert len(set(sum(ids.values(), []))) == 70
    xtrain = np.concatenate(xs); ytrain = np.concatenate(ys)
    assert set(np.unique(ytrain)) == set(range(4))
    dataset = lgb.Dataset(xtrain, label=ytrain, feature_name=FEATURES)
    model = lgb.train(dict(objective='multiclass', num_class=4, num_leaves=15,
        learning_rate=0.05, min_data_in_leaf=100, lambda_l2=1.0,
        seed=39281, num_threads=2, verbosity=-1), dataset, num_boost_round=150)
    rows = []
    with np.load(archives[4], allow_pickle=False) as archive:
        for name in ids[4]:
            prob = np.asarray(archive[name], np.float32)
            truth = target(DATA / 'masks' / f'{name}_mask.png')
            x = image_features(name, prob)
            correction = model.predict(x, num_threads=2).T.reshape(prob.shape)
            baseline = ious(prob.argmax(0).astype(np.uint8), truth)
            candidates = {}
            for alpha in (0.25, 0.5, 1.0):
                pred = ((1-alpha)*prob + alpha*correction).argmax(0).astype(np.uint8)
                candidates[str(alpha)] = ious(pred, truth)
            rows.append({'image': name, 'baseline': baseline, 'candidates': candidates})
            print('eval', name, float(np.mean(baseline)), flush=True)
    base = float(np.mean([r['baseline'] for r in rows]))
    scores = {}
    for key in ('0.25', '0.5', '1.0'):
        deltas = {r['image']: 100*(np.mean(r['candidates'][key])-np.mean(r['baseline'])) for r in rows}
        score = float(np.mean([r['candidates'][key] for r in rows]))
        scores[key] = {'miou_percent': 100*score, 'delta_pp': 100*(score-base),
                       'improved_images': sum(v > 0 for v in deltas.values()),
                       'worst_image_delta_pp': min(deltas.values()), 'per_image_delta_pp': deltas}
    passed = any(v['delta_pp'] >= 0.5 and v['improved_images'] >= 12 and
                 v['worst_image_delta_pp'] >= -0.5 for v in scores.values())
    result = {'scope': 'train70 Model392 seed1 OOF; fold1-3 fit, fold4 screen',
              'strict_generalization': False, 'valid_read': False, 'test_read': False,
              'training_images': sum(len(ids[f]) for f in (1, 2, 3)),
              'heldout_images': len(ids[4]), 'sampled_pixels': int(len(ytrain)),
              'archive_sha256': archive_sha, 'baseline_miou_percent': 100*base,
              'scores': scores, 'screen_pass': passed, 'per_image_iou': rows}
    (ROOT / 'screen_result.json').write_text(json.dumps(result, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else TypeError(type(value))) + '\n')
    print(json.dumps({'baseline': 100*base, 'scores': {k: {j:v[j] for j in
          ('delta_pp','improved_images','worst_image_delta_pp')} for k,v in scores.items()},
          'screen_pass': passed}, indent=2))


if __name__ == '__main__':
    main()
