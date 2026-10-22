from pathlib import Path
import ast,difflib,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
changes=[];added=[]
for p in sorted((ROOT/'source').rglob('*')):
    if not p.is_file():continue
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
    q=ROOT/'upstream/lite13_retuned_reproduction'/p.relative_to(ROOT/'source')
    if not q.exists():added.append(str(p.relative_to(ROOT)));continue
    if p.read_bytes()!=q.read_bytes():
        changes.extend(difflib.unified_diff(q.read_text().splitlines(True),p.read_text().splitlines(True),fromfile='upstream/'+str(p.relative_to(ROOT/'source')),tofile='source/'+str(p.relative_to(ROOT/'source'))))
(ROOT/'source_changes.diff').write_text(''.join(changes))
with zipfile.ZipFile(ROOT/'downloads/Model349_lite13_retuned_reproduction_code.zip') as z:
    for info in z.infolist():
        if not info.is_dir():assert (ROOT/'upstream'/info.filename).read_bytes()==z.read(info)
paths=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.sh','.md','.diff']]
paths += [ROOT/name for name in ['plan.json','provenance.json']]
paths += [p for p in (ROOT/'source').rglob('*') if p.is_file()]
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(ROOT/'code_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'checks/source_integrity.json').write_text(json.dumps(dict(ok=True,upstream_matches_zip=True,source_files=len(manifest),added_source_files=added,training_started=False),indent=2)+'\n')
print('Sealed',len(manifest),'source/config files; upstream ZIP byte equality verified.')
