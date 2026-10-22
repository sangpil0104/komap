#!/usr/bin/env python3
"""Download and restore archived materials; never downloads model weights."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def download(asset, cache):
    target = cache / asset['name']
    if target.exists():
        if target.stat().st_size == asset['bytes'] and sha256(target) == asset['sha256']:
            return target
        raise RuntimeError(f'Existing download has a different checksum: {target}')
    partial = target.with_suffix(target.suffix + '.partial')
    start = time.monotonic()
    received = 0
    request = urllib.request.Request(asset['url'], headers={'User-Agent': 'komap-archive-restore'})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
        while chunk := response.read(4 * 1024 * 1024):
            output.write(chunk)
            received += len(chunk)
            elapsed = max(time.monotonic() - start, .001)
            eta = max(asset['bytes'] - received, 0) / (received / elapsed)
            print(f'\r{asset["name"]}: {received / asset["bytes"]:6.1%}, elapsed {elapsed:.0f}s, ETA {eta:.0f}s', end='', flush=True)
    print()
    if received != asset['bytes'] or sha256(partial) != asset['sha256']:
        raise RuntimeError(f'Download checksum mismatch: {partial}')
    partial.replace(target)
    return target

def safe_output(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe catalog path: {name}')
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Output escapes repository: {name}')
    return target

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--list', action='store_true', help='List available material groups and sizes')
    group.add_argument('--prefix', help='Restore paths starting with this prefix')
    group.add_argument('--all', action='store_true', help='Restore all non-weight materials')
    parser.add_argument('--cache', type=Path, default=ROOT / '.downloads')
    parser.add_argument('--output-root', type=Path, default=ROOT, help='Separate empty directory may be used for verification')
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'manifests/assets.json').read_text())
    if catalog.get('weights_included') is not False:
        raise RuntimeError('Catalog is not marked as excluding weights')
    if args.list:
        print(f'{len(catalog["files"])} paths; {len(catalog["assets"])} downloads; model weights excluded')
        for asset in catalog['assets']:
            print(f'{asset["bytes"] / 1024**2:8.1f} MiB  {asset["name"]}')
        return
    rows = [row for row in catalog['files'] if args.all or row['path'].startswith(args.prefix)]
    if not rows:
        parser.error('No paths match the requested prefix; use --list or inspect manifests/assets.json')
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    lookup = {asset['name']: asset for asset in catalog['assets']}
    groups = {}
    for row in rows:
        groups.setdefault(row['asset'], []).append(row)
    restored = existing = 0
    for name, members in groups.items():
        archive = download(lookup[name], args.cache)
        with zipfile.ZipFile(archive) as bundle:
            for row in members:
                target = safe_output(args.output_root, row['path'])
                if target.exists():
                    if target.stat().st_size != row['bytes'] or sha256(target) != row['sha256']:
                        raise RuntimeError(f'Refusing to overwrite a different file: {target}')
                    existing += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.restore-partial')
                # Only catalogued regular entries are read; never extract arbitrary archive paths.
                with bundle.open(row['member']) as source, temporary.open('xb') as output:
                    shutil.copyfileobj(source, output, 4 * 1024 * 1024)
                if temporary.stat().st_size != row['bytes'] or sha256(temporary) != row['sha256']:
                    raise RuntimeError(f'Restored file checksum mismatch: {temporary}')
                temporary.replace(target)
                restored += 1
        print(f'{name}: verified {len(members)} paths', flush=True)
    print(f'Done: restored {restored}, already verified {existing}; no training was run.')

if __name__ == '__main__':
    main()
