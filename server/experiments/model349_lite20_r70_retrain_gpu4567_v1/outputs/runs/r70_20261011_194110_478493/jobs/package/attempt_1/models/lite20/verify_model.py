"""Strict hash and inventory verification for one portable lite20/r70 model."""
from pathlib import Path
import hashlib,json

def verify(root):
    root=Path(root);manifest=json.loads((root/'model_manifest.json').read_text())
    weights=list((root/'Model349_exh14/weights').rglob('*.pt'))+list((root/'assets').rglob('*.pt'))
    assert len(weights)==manifest['checkpoint_count']
    for name,expected in manifest['files_sha256'].items():
        with (root/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==expected,name
    cfg=json.loads((root/'compact/config.json').read_text())
    selected=json.loads((root/'compact'/cfg['assets_manifest_file']).read_text())
    assert selected['count']==len(selected['checkpoints'])==len(weights)
    assert not selected.get('unchecked_checkpoints')
    for r in selected['checkpoints']+selected['metadata']:
        path=(root/'Model349_exh14' if r['namespace']=='R' else root/'assets')/r['path']
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==r['sha256'] and path.stat().st_size==r['bytes'],r['path']
    return dict(ok=True,variant=manifest['variant'],checkpoint_count=len(weights),files=len(manifest['files_sha256']))

if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent)))
