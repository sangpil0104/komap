"""Step 5: predict new images with every fold x seed member and the configuration chosen on OOF.

    python infer.py --work WORK --images DIR --output NEW_DIR [--config work/select/final_config.json]
                    [--gt MASK_DIR] [--save-probabilities] [--device cuda]

For each member family the probabilities of all runs/fold*/seed* checkpoints are averaged (members
are loaded one at a time). The high-res base uses only the checkpoint family routed for the image's
brightness band; each member is TENT-adapted on the image before averaging. Brightness statistics and
priors come from work/all/calibration.json (all labeled images). --gt only adds scoring.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import fusion as F
import members as M
from common import PALETTE, decode_mask, image_miou, read_json, write_json


def runs(work: Path):
    out = sorted(p for p in (work / 'runs').glob('fold*/seed*') if (p / 'hrnet347' / 'STAGE_DONE.json').exists())
    if not out:
        raise FileNotFoundError('no finished runs under work/runs')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--config', type=Path)
    ap.add_argument('--gt', type=Path)
    ap.add_argument('--save-probabilities', action='store_true')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    if a.output.exists():
        ap.error('--output must be a new directory')
    cfg = read_json(a.config or a.work / 'select' / 'final_config.json')
    cal = read_json(a.work / 'all' / 'calibration.json')
    members = [M.member_paths(r) for r in runs(a.work)]
    for d in ('predicted_labels', 'predicted_masks') + (('probabilities',) if a.save_probabilities else ()):
        (a.output / d).mkdir(parents=True)
    scores = []
    for img in sorted(a.images.glob('*.png')):
        name = img.name.removesuffix('_image.png').removesuffix('.png')
        meta = M.image_meta(img, cal)
        arr = {'z': meta.pop('z'), 'q': meta.pop('q')}
        for fam in list(M.B4_STAGES) + list(M.HR_STAGES) + ['n269']:
            acc = 0
            for m in members:
                acc = acc + (M.b4(m[fam], img, a.device) if fam in M.B4_STAGES else
                             M.hrnet(m[fam], img, a.device) if fam in M.HR_STAGES else M.native269(m[fam], img, a.device))
            arr[fam] = acc / len(members)
        src = meta['routed'] if meta['routed'] in cfg['bands'] else 'restart51'
        acc = 0
        for m in members:
            acc = acc + M.high_preaux(m[f'high_{src}'], img, cal, meta['params'], a.device)
        arr[f'high_{src}'] = acc / len(members)
        pred = F.predict(arr, meta, cfg)
        Image.fromarray(pred).save(a.output / 'predicted_labels' / f'{name}.png')
        Image.fromarray(PALETTE[pred]).save(a.output / 'predicted_masks' / f'{name}_mask.png')
        if a.save_probabilities:
            np.savez(a.output / 'probabilities' / f'{name}.npz', **{k: v.astype(np.float16) for k, v in arr.items()})
        row = {'image': name, 'routed': meta['routed'], 'source': src}
        if a.gt:
            m_iou, cls = image_miou(pred, decode_mask(a.gt / f'{name}_mask.png'))
            row.update(image_miou=m_iou, class_iou=cls)
        scores.append(row)
        print(name, flush=True)
    result = {'members': len(members), 'config': cfg, 'images': scores}
    if a.gt:
        result['mean_image_miou'] = float(np.mean([r['image_miou'] for r in scores]) * 100)
    write_json(a.output / 'completion.json', result)
    print('DONE', result.get('mean_image_miou', ''), flush=True)


if __name__ == '__main__':
    main()
