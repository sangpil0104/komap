"""Verify this newly trained inference bundle without loading a model."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    manifest = json.loads((ROOT / 'checkpoint_manifest.json').read_text())
    assert manifest['count'] == len(manifest['checkpoints']) == 11
    assert manifest['bytes'] == sum(row['bytes'] for row in manifest['checkpoints'])
    for row in manifest['checkpoints']:
        path = ROOT / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], str(path)
    recipe = json.loads((ROOT / 'recipe_manifest.json').read_text())
    for name, digest in recipe['files_sha256'].items():
        assert sha(ROOT / name) == digest, name
    print(json.dumps(dict(ok=True, checkpoint_count=11, checkpoint_bytes=manifest['bytes'])))


if __name__ == '__main__':
    main()
