"""Build exploratory 12-checkpoint FP16 bundle from verified 14-checkpoint bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or source in output.parents or output in source.parents:
        parser.error('output must be a new independent directory')
    for sub in ('weights', 'src', 'configs'):
        if not (source / sub).is_dir():
            parser.error(f'missing source/{sub}')

    def omit(folder, names):
        if Path(folder) == source / 'weights/model326_hybrid_fullstats_group_oof':
            return {'fold2', 'fold4'}
        return set()

    shutil.copytree(source, output, ignore=omit)
    shutil.copy2(args.code, output / 'infer_lite.py')
    shutil.copy2(args.config, output / 'configs/model349_lite.yaml')
    checkpoints = sorted((output / 'weights').rglob('*.pt'))
    if len(checkpoints) != 12:
        raise RuntimeError(f'Expected 12 checkpoints, got {len(checkpoints)}')
    report = {'source': str(source), 'checkpoints':
              [{'path': str(path.relative_to(output)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
               for path in checkpoints]}
    (output / 'fold326_13_bundle_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Built 12-checkpoint independent bundle: {output}')
    print(f'Checkpoint bytes: {sum(row["bytes"] for row in report["checkpoints"])}')


if __name__ == '__main__':
    main()
