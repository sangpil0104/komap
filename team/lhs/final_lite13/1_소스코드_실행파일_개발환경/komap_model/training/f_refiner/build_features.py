"""Build F (residual refiner) training features from train70 E OOF probabilities. Never reads valid/test.

    python build_features.py --e-output OUT --train DATA/train --palette-config inference/config/r_config.yaml \
        --out FEATURES [--b4-dir b4_w4000]

OUT is the E OOF layout written by run_retrain.sh:
    OUT/<b4-dir>/fold{1..4}_oof/probabilities.npz            (B4, team channel order, one key per image)
    OUT/E4_hrnet/seed{1,2}/fold{1..4}_oof/valid_probabilities.npz  (HRNet, team channel order)
Each train image appears in exactly one fold's OOF archive. Per image (same arithmetic as the original
accuracy81_oof_refiner_20261009/prepare_oof.py):
    E_team = 0.5*B4 + 0.25*HR_seed1 + 0.25*HR_seed2 ; logits = log(max(E_team[[1,2,0,3]], 1e-7))  (route order)
    gray   = L(image)/255 (the trainer applies the z-score transform) ; target = GT mask decoded with the palette
Writes FEATURES/<name>.npz (logits, gray, target) and FEATURES/manifest.json for train_refiner.py.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import yaml
from PIL import Image


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def decode(path, palette):
    rgb = np.asarray(Image.open(path).convert('RGB'))
    out = np.full(rgb.shape[:2], 255, np.uint8)
    for c, col in enumerate(palette): out[np.all(rgb == col, -1)] = c
    assert (out < 4).all(), f'unmapped colours in {path}'
    return out


def miou(pred, gt):
    r = []
    for c in range(4):
        u = np.count_nonzero((pred == c) | (gt == c)); r.append(np.count_nonzero((pred == c) & (gt == c)) / u if u else 0.)
    return float(np.mean(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--e-output', type=Path, required=True)
    ap.add_argument('--train', type=Path, required=True, help='folder with images/ and masks/ of the train images')
    ap.add_argument('--palette-config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--b4-dir', default='b4_w4000', help="B4 OOF family folder (e.g. E1_2_b4 for the warm-started folds)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    palette = np.asarray(yaml.safe_load(a.palette_config.read_text())['palette'], np.uint8)
    O = a.e_output; rows = []; scores = []
    for f in (1, 2, 3, 4):
        paths = [O / a.b4_dir / f'fold{f}_oof/probabilities.npz'] + [O / f'E4_hrnet/seed{s}/fold{f}_oof/valid_probabilities.npz' for s in (1, 2)]
        with np.load(paths[0], allow_pickle=False) as b4, np.load(paths[1], allow_pickle=False) as h1, np.load(paths[2], allow_pickle=False) as h2:
            assert set(b4.files) == set(h1.files) == set(h2.files), f'fold {f}: archives list different images'
            for name in sorted(b4.files):
                im = a.train / 'images' / f'{name}_image.png'; gt = a.train / 'masks' / f'{name}_mask.png'
                gray = np.asarray(Image.open(im).convert('L'), dtype=np.float32) / 255.; target = decode(gt, palette)
                comps = [np.asarray(z[name], dtype=np.float32) for z in (b4, h1, h2)]
                for p in comps:
                    assert p.shape == (4, *gray.shape) and np.isfinite(p).all() and np.max(np.abs(p.sum(0) - 1)) <= .03
                team = .5 * comps[0] + .25 * comps[1] + .25 * comps[2]
                scores.append(miou(np.array([2, 0, 1, 3], np.uint8)[team.argmax(0)], target))
                logits = np.log(np.maximum(team[[1, 2, 0, 3]], 1e-7)).astype(np.float32)
                p = a.out / f'{name}.npz'; np.savez_compressed(p, logits=logits, gray=gray, target=target)
                rows.append({'name': name, 'fold': f, 'image_sha256': sha(im), 'train_mask_sha256': sha(gt), 'features_sha256': sha(p)})
                print('FEATURES', f, name, flush=True)
    n = len(list((a.train / 'images').glob('*_image.png')))
    assert len(rows) == n and len({r['name'] for r in rows}) == n, f'expected {n} distinct train images, got {len(rows)}'
    m = {'split': 'train', 'images': sorted(rows, key=lambda r: r['name']), 'b4_dir': a.b4_dir,
         'raw_e_oof_miou_percent': 100 * float(np.mean(scores)), 'valid_or_test_read': False}
    (a.out / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    print('E OOF70 mIoU (team argmax)', m['raw_e_oof_miou_percent'], flush=True)


if __name__ == '__main__':
    main()
