import importlib.util,json,sys,copy,pickle,os,time,subprocess,hashlib
from pathlib import Path
K=Path('/home/sangpil/komap');D=Path(__file__).resolve().parent
X=K/'lite13_R322share326_20261014/parent_package/Model349_exh14'
sys.path.insert(0,str(X/'src'))
sp=importlib.util.spec_from_file_location('remaining_infer',X/'infer_lite.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
job=json.loads(Path(sys.argv[1]).read_text())
work=Path(job['work']);work.mkdir(parents=True,exist_ok=True)
import yaml
cfg=yaml.safe_load((K/'lite13_R322share326_20261014/selected_R_lite13.yaml').read_text())
cfg['weights_dir']=job['weights']
baseline=json.loads((K/'lite13_R322share326_20261014/e2e_valid/release/input_manifest.json').read_text())
baseline['config']=cfg
(work/'input_manifest.json').write_text(json.dumps(baseline,indent=2))
(work/'probabilities').mkdir(exist_ok=True)
out=work/'probabilities'/job['component']
if (out/'complete.json').exists():sys.exit(0)
if job['component']=='base':
 import torch
 import numpy as np
 from PIL import Image
 out.mkdir(exist_ok=True);torch.set_num_threads(cfg['inference']['threads'])
 device=torch.device('cuda:0');images=baseline['images'];existing=[]
 for row in images:
  p=out/f"{row['name']}.pkl"
  if not p.exists():continue
  with p.open('rb') as f:v=pickle.load(f)
  with Image.open(row['path']) as im:width,height=im.size
  assert v['preaux'].shape==(4,height,width) and np.isfinite(v['preaux']).all(),str(p)
  existing.append({'name':row['name'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 if existing:(work/'resume_existing_predictions.json').write_text(json.dumps(existing,indent=2))
 for row in images:
  if (out/f"{row['name']}.pkl").exists():continue
  # Each image starts from the same checkpoint; completed images can be reused exactly.
  while True:
   pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).split()
   if all(int(pid)==os.getpid() for pid in pids):break
   print('waiting other GPU process before',row['name'],pids,flush=True);time.sleep(10)
  m.base_worker([row],out,device,cfg,Path(job['weights']))
 (out/'complete.json').write_text(json.dumps({'component':'base','images':len(images),'resumed_images':len(existing),'checkpoint':job['weights']}))
else:
 if out.exists():raise RuntimeError('Unreceipted partial inference; inspect before resuming')
 m.worker(work,job['component'])
