"""Verify all files and all 27 weights in the retrained fusion package."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'model_manifest.json').read_text())
assert manifest['checkpoint_count']==27
weights=list((root/'Model349_exh14/weights').rglob('*.pt'))+list((root/'assets').rglob('*.pt'))
assert len(weights)==27
for name,expected in manifest['files_sha256'].items():
    with (root/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    assert actual==expected,name
print(json.dumps(dict(ok=True,checkpoint_count=27,files=len(manifest['files_sha256']))))
