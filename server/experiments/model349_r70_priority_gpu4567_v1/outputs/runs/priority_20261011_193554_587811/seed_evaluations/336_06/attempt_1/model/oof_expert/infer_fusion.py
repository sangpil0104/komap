"""Stage-1 fusion on new images: R labels (oof_design/infer.py output) overwritten by E where the rule says.

    python infer_fusion.py --work WORK --images DIR --r-labels R_OUT/predicted_labels --output NEW_DIR
                           [--rule work/select_rule/final_rule.json] [--gt MASK_DIR] [--device cuda]

E averages every fold x seed member: B4-329 (D4 TTA, R's b4_329 checkpoints) and HRNet392 (flip4).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from expert_common import apply_rule, b4_329_d4, e_label, hrnet392
from common import PALETTE, decode_mask, image_miou, read_json, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--r-labels', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--rule', type=Path)
    ap.add_argument('--gt', type=Path)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    if a.output.exists():
        ap.error('--output must be a new directory')
    rule = tuple(read_json(a.rule or a.work / 'select_rule' / 'final_rule.json')['rule'])
    members = []
    for hr in sorted((a.work / 'runs_expert').glob('fold*/seed*/hrnet392')):
        fold, seed = hr.parent.parent.name, hr.parent.name
        b4 = a.work / 'runs' / fold / seed / 'b4_329'
        if (hr / 'STAGE_DONE.json').exists() and (b4 / 'STAGE_DONE.json').exists():
            members.append((b4, hr / 'last.pt'))
    if not members:
        raise SystemExit('no finished expert runs')
    for d in ('predicted_labels', 'predicted_masks'):
        (a.output / d).mkdir(parents=True)
    rows = []
    for img in sorted(a.images.glob('*.png')):
        name = img.name.removesuffix('_image.png').removesuffix('.png')
        b4 = sum(b4_329_d4(b, img, a.device) for b, _ in members) / len(members)
        hr = sum(hrnet392(h, img, a.device) for _, h in members) / len(members)
        e = e_label(0.5 * b4 + 0.5 * hr)
        pred = apply_rule(np.asarray(Image.open(a.r_labels / f'{name}.png')), e, rule)
        Image.fromarray(pred).save(a.output / 'predicted_labels' / f'{name}.png')
        Image.fromarray(PALETTE[pred]).save(a.output / 'predicted_masks' / f'{name}_mask.png')
        row = {'image': name, 'changed_pixels': int(np.count_nonzero(pred != np.asarray(Image.open(a.r_labels / f'{name}.png'))))}
        if a.gt:
            m, cls = image_miou(pred, decode_mask(a.gt / f'{name}_mask.png'))
            row.update(image_miou=m, class_iou=cls)
        rows.append(row)
        print(name, flush=True)
    res = {'rule': list(rule), 'members': len(members), 'images': rows}
    if a.gt:
        res['mean_image_miou'] = float(np.mean([r['image_miou'] for r in rows]) * 100)
    write_json(a.output / 'completion.json', res)
    print('DONE', res.get('mean_image_miou', ''))


if __name__ == '__main__':
    main()
