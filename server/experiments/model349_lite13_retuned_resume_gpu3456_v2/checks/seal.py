from pathlib import Path
import ast,difflib,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
parent=Path(json.loads((ROOT/'parent.json').read_text())['parent_run'])
changes=[]
for name in ['timed_predict.py','export_results.py','run.py']:
    old=(parent/'implementation'/name).read_text();new=(ROOT/name).read_text()
    changes.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='parent/'+name,tofile='resume/'+name))
(ROOT/'resume_changes.diff').write_text(''.join(changes))
paths=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.sh','.md','.diff','.json'] and p.name not in ['code_manifest.json','latest_run.json']]
paths += [p for p in (ROOT/'source').rglob('*') if p.is_file()]
for p in paths:
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(ROOT/'code_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Sealed',len(manifest),'resume files; parent training code and weights untouched.')
