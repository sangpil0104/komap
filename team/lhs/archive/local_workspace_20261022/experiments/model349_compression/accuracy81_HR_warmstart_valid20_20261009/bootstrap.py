import os, sys, json, subprocess, runpy
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2)
assert not subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
hold=torch.zeros(1,device='cuda');torch.cuda.synchronize()
apps=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
assert len(apps)==1
(ROOT/'gpu_identity.json').write_text(json.dumps({'namespace_pid':os.getpid(),'nvml_pid':apps[0].strip()}))
script=sys.argv[1];sys.argv=sys.argv[1:]
sys.path.insert(0,str(ROOT.parent/'accuracy81_HR_warmstart_fold4_20261009/model392'))
runpy.run_path(str(ROOT.parent/'accuracy81_HR_warmstart_fold4_20261009/model392'/script),run_name='__main__')
