"""CPU-only fold-pair audit using saved Model349 component probability maps.

This is exploratory validation analysis, not an independent generalization test.
It does not modify checkpoints or use GPU time.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import yaml


def iou(pred, gt):
    scores = []
    for cls in range(4):
        union = np.count_nonzero((pred == cls) | (gt == cls))
        intersection = np.count_nonzero((pred == cls) & (gt == cls))
        scores.append(intersection / union if union else 0.0)
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    sys.path.insert(0, str(bundle))
    import infer_lite as inf
    config = yaml.safe_load((bundle / 'configs/model349_lite.yaml').read_text())
    names = sorted(p.stem for p in (args.probe / 'base').glob('*.npz'))
    pairs = list(combinations(range(1, 5), 2))
    variants = [('full', (1, 2, 3, 4), (1, 2, 3, 4))]
    variants += [(f'326={a}{b}', (a, b), (1, 2, 3, 4)) for a, b in pairs]
    variants += [(f'329={a}{b}', (1, 2, 3, 4), (a, b)) for a, b in pairs]
    variants += [(f'both={a}{b}', (a, b), (a, b)) for a, b in pairs]
    rows = []
    for name in names:
        with np.load(args.probe / 'base' / f'{name}.npz') as z:
            b = {key: z[key].astype(np.float32) for key in z.files}
        meta = json.loads((args.probe / 'base' / f'{name}.json').read_text())
        params = dict(meta['params'], pair_weight=0.0)
        high = ((1 - params['style_weight']) * b['high768_raw_tent']
                + params['style_weight'] * b['high768_style_tent'])
        high /= np.maximum(high.sum(0, keepdims=True), 1e-7)
        item = {'preaux': high, 'z': b['z'], 'q': b['q'], 'params': params,
                'prior_z_stats': meta['prior_z_stats'], 'prior_q_stats': meta['prior_q_stats']}
        with np.load(args.probe / '326' / f'{name}.npz') as z:
            p326 = [z[f'f{i}_v1'].astype(np.float32) for i in range(1, 5)]
        with np.load(args.probe / '329' / f'{name}.npz') as z:
            p329 = [z[f'f{i}_v1'].astype(np.float32) for i in range(1, 5)]
        with np.load(args.probe / '336' / f'{name}.npz') as z:
            p336 = z['v1'].astype(np.float32)
        with np.load(args.probe / '269' / f'{name}.npz') as z:
            p269 = z['v1'].astype(np.float32)
        gt = inf.decode_gt(args.gt / f'{name}_mask.png', config['palette'])
        ref = np.asarray(Image.open(args.reference / 'predicted_labels' / f'{name}.png'))
        result = {'image': name, 'reference_miou': iou(ref, gt), 'variants': {}}
        for tag, folds326, folds329 in variants:
            sources = {'326': np.mean([p326[i - 1] for i in folds326], axis=0),
                       '329': np.mean([p329[i - 1] for i in folds329], axis=0),
                       '336': p336, '269': p269}
            pred = inf.predict_fixed(item, sources, config)
            result['variants'][tag] = {'miou': iou(pred, gt),
                                       'agreement_reference': float(np.mean(pred == ref))}
        rows.append(result)
        print(f'{len(rows)}/{len(names)} {name}', flush=True)
    summary = {}
    reference = float(np.mean([x['reference_miou'] for x in rows]))
    for tag, _, _ in variants:
        vals = np.asarray([x['variants'][tag]['miou'] for x in rows])
        summary[tag] = {'miou': float(vals.mean()), 'delta_vs_reference_pp': float(100 * (vals.mean() - reference)),
                        'worst_image_delta_pp': float(100 * np.min(vals - [x['reference_miou'] for x in rows])),
                        'mean_mask_agreement': float(np.mean([x['variants'][tag]['agreement_reference'] for x in rows]))}
    args.output.write_text(json.dumps({'reference_miou': reference, 'summary': summary, 'rows': rows}, indent=2) + '\n')
    for tag, value in sorted(summary.items(), key=lambda x: -x[1]['miou']):
        print(f'{tag}: {100 * value["miou"]:.5f}% (delta {value["delta_vs_reference_pp"]:+.5f}pp)', flush=True)


if __name__ == '__main__':
    main()
