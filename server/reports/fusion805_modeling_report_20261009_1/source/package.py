"""Create complete report and separate figure/table ZIPs, then verify every entry."""
from pathlib import Path
import hashlib,json,zipfile
from tqdm import tqdm
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def deliverable(p):return p.is_file() and '__pycache__' not in p.parts and p.name!='pdf_layout_initial.json'
validation=json.loads((ROOT/'validation/final_validation.json').read_text());assert validation['ok']
for name,digest in validation['document_sha256'].items():assert sha(ROOT/name)==digest,name
files=sorted(p for p in ROOT.rglob('*') if deliverable(p) and p.name!='manifest.sha256.json')
(ROOT/'manifest.sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in files},ensure_ascii=False,indent=2)+'\n')
files.append(ROOT/'manifest.sha256.json')
assets=[p for p in files if p.relative_to(ROOT).parts[0] in ['figures','tables','appendix']]
assets.extend([ROOT/'README.md',ROOT/'source/NanumGothic_LICENSE.txt'])
archives=[]
for name,selected in [(ROOT.name+'.zip',files),('fusion805_figures_tables_20261009_1.zip',assets)]:
    path=ROOT.parent/name
    if path.exists():raise FileExistsError(path)
    temp=path.with_suffix('.zip.partial')
    with zipfile.ZipFile(temp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in tqdm(selected,desc=f'Archive {name}',unit='file'):z.write(p,ROOT.name+'/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        for p in selected:
            item=ROOT.name+'/'+str(p.relative_to(ROOT))
            with z.open(item) as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==sha(p),item
    temp.rename(path);digest=sha(path)
    path.with_suffix('.zip.sha256').write_text(f'{digest}  {path.name}\n')
    archives.append(dict(path=str(path),bytes=path.stat().st_size,files=len(selected),sha256=digest,crc_verified=True,content_hashes_verified=True))
(ROOT.parent/'fusion805_report_archives_20261009_1.json').write_text(json.dumps(archives,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(archives,ensure_ascii=False,indent=2))
