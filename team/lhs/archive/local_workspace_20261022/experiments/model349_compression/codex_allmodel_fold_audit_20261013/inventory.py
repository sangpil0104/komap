import os,json,zipfile,sys
from pathlib import Path
import numpy as np
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
NAMES=json.loads(sys.argv[2]);expected=set(NAMES)
roots=[Path(p) for p in sys.argv[3:]];weights=[];groups={};errors=[];directories=[]
def header(stream):
 v=np.lib.format.read_magic(stream)
 return np.lib.format._read_array_header(stream,v)
for root in roots:
 for base,dirs,files in os.walk(root):
  dirs[:]=[d for d in dirs if d not in ['.git','__pycache__','venv','node_modules']]
  p=Path(base)
  if p.parent==root and root.name=='outputs':directories.append(str(p))
  for f in files:
   path=p/f
   if f.endswith(('.pt','.pth')):
    metadata=None
    for pp in [p,p.parent]:
     ap=pp/'args.json'
     if ap.is_file():
      try:metadata=json.loads(ap.read_text())
      except Exception:pass
      break
    weights.append({'path':str(path),'bytes':path.stat().st_size,'args':metadata,'args_path':str(ap) if metadata is not None else None})
   if not f.endswith(('.npy','.npz')):continue
   name=path.stem.removesuffix('_image')
   try:
    if f.endswith('.npy') and name in expected:
     with path.open('rb') as fd:shape,fortran,dtype=header(fd)
     if len(shape)==3 and shape[0]==4:groups.setdefault((str(p),'npy',None),{})[name]={'path':str(path),'shape':shape,'dtype':str(dtype)}
    elif f.endswith('.npz'):
     with zipfile.ZipFile(path) as z:
      for key in z.namelist():
       if not key.endswith('.npy'):continue
       member=key[:-4];imname=name if name in expected else member.removesuffix('_image')
       if imname not in expected:continue
       with z.open(key) as fd:shape,fortran,dtype=header(fd)
       if len(shape)==3 and shape[0]==4:
        folder=str(p) if name in expected else str(path);variant=member if name in expected else 'archive'
        groups.setdefault((folder,'npz',variant),{})[imname]={'path':str(path),'key':member,'shape':shape,'dtype':str(dtype)}
   except Exception as e:errors.append({'path':str(path),'error':str(e)})
caches=[]
for (folder,storage,variant),items in groups.items():
 missing=sorted(expected-set(items));caches.append({'folder':folder,'storage':storage,'variant':variant,'complete20':not missing,'missing':missing,'files':items})
result={'roots':list(map(str,roots)),'output_directories':directories,'checkpoints':weights,'probability_caches':caches,'scan_errors':errors}
(OUT/'inventory.json').write_text(json.dumps(result,indent=2));print(json.dumps({'checkpoint_files':len(weights),'output_directories':len(directories),'cache_groups':len(caches),'complete20':sum(c['complete20'] for c in caches),'errors':len(errors)}),flush=True)
