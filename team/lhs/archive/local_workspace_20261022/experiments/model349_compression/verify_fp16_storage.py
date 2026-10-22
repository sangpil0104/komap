"""Check structural parity and FP16 rounding error for a storage bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def compare(a, b, stats):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape
        if a.dtype == torch.float32:
            assert b.dtype == torch.float16
            assert torch.isfinite(b).all(), 'FP16 overflow detected'
            diff = (a - b.float()).abs()
            stats['max_abs_error'] = max(stats['max_abs_error'], float(diff.max()))
            stats['sum_abs_error'] += float(diff.sum())
            stats['float_elements'] += a.numel()
            stats['changed_elements'] += int(torch.count_nonzero(diff))
        else:
            assert a.dtype == b.dtype and torch.equal(a, b)
        return
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key], stats)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            compare(left, right, stats)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('converted', type=Path)
    args = parser.parse_args()
    source_paths = sorted((args.source / 'weights').rglob('*.pt'))
    converted_paths = sorted((args.converted / 'weights').rglob('*.pt'))
    assert len(source_paths) == len(converted_paths) == 14
    stats = {'max_abs_error': 0.0, 'sum_abs_error': 0.0,
             'float_elements': 0, 'changed_elements': 0}
    for path in source_paths:
        other = args.converted / path.relative_to(args.source)
        assert other.is_file(), other
        compare(torch.load(path, map_location='cpu', weights_only=False),
                torch.load(other, map_location='cpu', weights_only=False), stats)
        print(f'verified {path.relative_to(args.source)}', flush=True)
    stats['mean_abs_error'] = stats.pop('sum_abs_error') / stats['float_elements']
    stats['source_bytes'] = sum(p.stat().st_size for p in source_paths)
    stats['converted_bytes'] = sum(p.stat().st_size for p in converted_paths)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
