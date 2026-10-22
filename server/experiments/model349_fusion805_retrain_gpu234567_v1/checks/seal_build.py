"""Record only this experiment's source/config and exact local fold membership."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import ROOT, read_json, write_json, sha

folds = {str(p.relative_to(ROOT)):str(p.readlink()) for p in sorted((ROOT/'folds').rglob('*.png'))}
write_json(ROOT/'fold_manifest.json',folds)
write_json(ROOT/'expert_fold_manifest.json',{str(p.relative_to(ROOT)):str(p.readlink()) for p in sorted((ROOT/'expert_splits').rglob('*.png'))})
files = []
for folder in ['source','frozen','fusion_source']:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
files.extend(p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.sh','.md','.json']
             and p.name not in ['code_manifest.json','input_manifest.json'])
write_json(ROOT/'code_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
print('Sealed source/config and fold memberships; no model execution.')
