"""Record build files after review, excluding outputs and large immutable weights."""
import ast,difflib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT, REPO, sha, write_json
old=REPO/'experiments/model349_seed100_gpu4567_v1/source'
diff=[]
for path in sorted((ROOT/'source').rglob('*')):
    if path.is_file():
        parent=old/path.relative_to(ROOT/'source')
        if parent.is_file() and path.read_bytes()!=parent.read_bytes():
            diff.extend(difflib.unified_diff(parent.read_text().splitlines(True),path.read_text().splitlines(True),
                         fromfile='previous/'+str(path.relative_to(ROOT/'source')),
                         tofile='lite12/'+str(path.relative_to(ROOT/'source'))))
(ROOT/'source_changes.diff').write_text(''.join(diff))
files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.sh','.json','.md','.diff']
       and p.name!='code_manifest.json']
for name in ['source','frozen']:
    files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for path in files:
    if path.suffix=='.py': ast.parse(path.read_text(),filename=str(path))
write_json(ROOT/'code_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
print('Sealed',len(files),'source/configuration/documentation files')
