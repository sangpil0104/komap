"""Read-only verification of the frozen lite12 recipe and separately supplied weights."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / 'checkpoint_manifest.json').read_text())
    assert manifest['count'] == len(manifest['checkpoints']) == 12
    assert manifest['bytes'] == sum(row['bytes'] for row in manifest['checkpoints']) == 683320900
    errors = []
    for row in manifest['checkpoints']:
        path = ROOT / row['path']
        if not path.is_file():
            errors.append(f"missing: {row['path']}")
        elif path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            errors.append(f"weight mismatch: {row['path']}")
    recipe = json.loads((ROOT / 'recipe_manifest.json').read_text())
    for name, digest in recipe['files_sha256'].items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != digest:
            errors.append(f'recipe mismatch: {name}')
    print(json.dumps({'ok': not errors, 'checkpoint_count': manifest['count'],
                      'checkpoint_bytes': manifest['bytes'], 'errors': errors}, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
