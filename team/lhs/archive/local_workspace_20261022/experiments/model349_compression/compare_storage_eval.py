"""Compare raw-image lite variants with paired image scores and mask parity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_run(path: Path):
    metrics = json.loads((path / 'metrics.json').read_text())
    completion = json.loads((path / 'completion.json').read_text())
    images = {row['image']: row for row in metrics['images']}
    return metrics, completion, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidates', nargs='+', type=Path)
    args = parser.parse_args()
    ref_metrics, ref_completion, ref_rows = load_run(args.reference)
    results = {'metric': ref_metrics['metric'], 'reference': str(args.reference),
               'reference_miou': ref_metrics['image_miou'],
               'reference_seconds': ref_completion['elapsed_seconds'], 'candidates': []}
    for candidate in args.candidates:
        metrics, completion, rows = load_run(candidate)
        assert set(rows) == set(ref_rows) and len(rows) == 20
        deltas = {name: rows[name]['image_miou'] - ref_rows[name]['image_miou'] for name in rows}
        changed = total = 0
        for name in rows:
            ref_mask = np.asarray(Image.open(args.reference / 'predicted_labels' / f'{name}.png'))
            new_mask = np.asarray(Image.open(candidate / 'predicted_labels' / f'{name}.png'))
            assert ref_mask.shape == new_mask.shape
            changed += int(np.count_nonzero(ref_mask != new_mask))
            total += ref_mask.size
        results['candidates'].append({
            'path': str(candidate), 'image_miou': metrics['image_miou'],
            'delta_pp': 100 * (metrics['image_miou'] - ref_metrics['image_miou']),
            'elapsed_seconds': completion['elapsed_seconds'],
            'speedup_vs_reference': ref_completion['elapsed_seconds'] / completion['elapsed_seconds'],
            'mask_changed_pixels': changed, 'mask_total_pixels': total,
            'mask_agreement': 1 - changed / total,
            'worst_image': min(deltas, key=deltas.get),
            'worst_image_delta_pp': 100 * min(deltas.values()),
            'per_image_delta_pp': {name: 100 * delta for name, delta in sorted(deltas.items())},
        })
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
