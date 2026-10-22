"""Make an exploratory FP16-on-disk copy of a Model349 lite bundle.

Inference still constructs the original FP32 modules. This changes only the
stored checkpoint precision and leaves original checkpoints untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def convert(obj, stats):
    if isinstance(obj, torch.Tensor):
        if obj.dtype == torch.float32:
            stats['float32_tensors'] += 1
            stats['float32_elements'] += obj.numel()
            return obj.to(torch.float16)
        return obj
    if isinstance(obj, dict):
        return type(obj)((key, convert(value, stats)) for key, value in obj.items())
    if isinstance(obj, list):
        return [convert(value, stats) for value in obj]
    if isinstance(obj, tuple):
        return tuple(convert(value, stats) for value in obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir() or not (source / 'weights').is_dir():
        parser.error('source must be a bundle directory containing weights/')
    if output.exists():
        parser.error('output must be a new path; existing bundles are never overwritten')
    if source in output.parents or output in source.parents:
        parser.error('source and output must not contain one another')
    shutil.copytree(source, output, ignore=shutil.ignore_patterns('*.pt', '*.pth'))
    report = []
    for checkpoint in sorted((source / 'weights').rglob('*.pt')):
        target = output / checkpoint.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        stats = {'float32_tensors': 0, 'float32_elements': 0}
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        converted = convert(payload, stats)
        if not stats['float32_elements']:
            raise RuntimeError(f'No FP32 weights found in {checkpoint}')
        torch.save(converted, target)
        report.append({'path': str(checkpoint.relative_to(source)),
                       'source_bytes': checkpoint.stat().st_size,
                       'output_bytes': target.stat().st_size,
                       'source_sha256': digest(checkpoint),
                       'output_sha256': digest(target), **stats})
        print(f'{report[-1]["path"]}: {report[-1]["source_bytes"]} -> {report[-1]["output_bytes"]}', flush=True)
    (output / 'fp16_storage_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Converted {len(report)} checkpoints; original bundle unchanged', flush=True)


if __name__ == '__main__':
    main()
