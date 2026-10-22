import os,sys,json,subprocess,runpy
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2)
base=json.loads((ROOT/'gpu0_baseline.json').read_text())['apps']
def apps():
 lines=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
 return {a.split(',')[0].strip():int(a.split(',')[1]) for a in lines if a.strip()}
before=apps();assert set(before)<=set(base) and all(before[p]<=base[p]+128 for p in before)
hold=torch.zeros(1,device='cuda');torch.cuda.synchronize()
after=apps();own=set(after)-set(base);assert len(own)==1 and all(after.get(p,0)<=m+128 for p,m in base.items())
(ROOT/'gpu_identity.json').write_text(json.dumps({'namespace_pid':os.getpid(),'nvml_pid':own.pop()}))
script=sys.argv[1];sys.argv=sys.argv[1:];sys.path.insert(0,str(ROOT/'model392'))
runpy.run_path(str(ROOT/'model392'/script),run_name='__main__')
