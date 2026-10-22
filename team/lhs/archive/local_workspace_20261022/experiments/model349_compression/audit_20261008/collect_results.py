"""Create a bounded results archive; do not include probability caches or weights."""
from pathlib import Path
import tarfile
ROOT=Path('/workspace/model349_compress/audit_20261008')
paths=[]
for p in ROOT.iterdir():
    if p.is_file() and p.suffix in {'.json','.jsonl','.log','.md','.py'}: paths.append(p)
for bundle in (ROOT/'bundles').iterdir():
    paths.extend(p for p in bundle.iterdir() if p.is_file() and p.suffix in {'.py','.json','.txt','.yaml','.md'})
    paths.extend(p for p in (bundle/'configs').iterdir() if p.is_file())
for folder in [ROOT/'bundles/fp16_15_batched/src',*ROOT.glob('model349_lite15_*_candidate')]:
    if folder.is_dir():
        paths.extend(p for p in folder.rglob('*') if p.is_file() and 'weights' not in p.relative_to(folder).parts and '__pycache__' not in p.parts)
for run in (ROOT/'runs').iterdir():
    for name in ('input_manifest.json','metrics.json','completion.json','run_audit.json'):
        if (run/name).is_file(): paths.append(run/name)
    paths.extend(p for p in (run/'logs').glob('*.log') if p.is_file())
    paths.extend(p for p in (run/'predicted_labels').glob('*.png') if p.is_file())
    if (run/'probabilities').is_dir(): paths.extend(p for p in (run/'probabilities').glob('*/complete.json'))
with tarfile.open(ROOT/'results_archive.tar.gz','w:gz') as tar:
    for p in paths: tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
print('Archived',len(paths),'files', (ROOT/'results_archive.tar.gz').stat().st_size)
