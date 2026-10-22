from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib, json, subprocess

ROOT = Path(__file__).resolve().parent
REPO = Path('/Data3/sppark/komap')
GPUS = [3, 4, 5, 6]
CONDA = Path('/home/sppark/miniconda3/envs/komap')

def now(): return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def read(path): return json.loads(Path(path).read_text())
def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def devices():
    value=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows={}
    for line in value.splitlines():
        i,u,n,m,p=[x.strip() for x in line.split(',')]
        rows[int(i)]=dict(index=int(i),uuid=u,name=n,memory_used_mib=int(m),utilization_percent=int(p))
    return {i:rows[i] for i in GPUS}
