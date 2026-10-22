import json,sys,re,zipfile,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
O=Path('/workspace/model349_compress/codex_allmodel_fold_audit_20261013');sys.path.insert(0,str(O));import review
coverage=json.loads((O/'coverage.json').read_text());resolved=[];unresolved=[];pal=np.array([[25,76,153],[76,178,76],[153,127,76],[204,204,204]],np.uint8)
def parse_names(v):
 if isinstance(v,list) and len(v)==4 and all(isinstance(x,str) for x in v):
  n=[re.sub(r'[^a-z0-9]','',x.lower()) for x in v]
  if n==['al','al3ni','primarysi','eutecticsi']:return 'team'
  if n==['al3ni','primarysi','al','eutecticsi']:return 'main'
 return None
for c in coverage:
 if not c['complete20'] or c['channel_order'] is not None:continue
 folder=Path(c['folder']);parent=folder.parent if folder.is_file() else folder;proof=None;order=None
 # Use color predictions to establish semantics without consulting GT or maximizing accuracy.
 names=list(c['files'])[:3]
 choices=[]
 for root in [parent,parent.parent]:
  if '/outputs/' not in str(root):continue
  for sub in [root]+[p for p in root.iterdir() if p.is_dir() and any(x in p.name.lower() for x in ['pred','mask','valid']) and 'gt' not in p.name.lower()]:
   ps=[]
   for n in names:
    options=[sub/f'{n}_mask.png',sub/f'{n}.png'];p=next((p for p in options if p.exists()),None)
    if p is None:break
    ps.append(p)
   if len(ps)!=len(names):continue
   ratios={'main':[],'team':[]}
   for n,p in zip(names,ps):
    rgb=np.array(Image.open(p).convert('RGB'));lab=np.full(rgb.shape[:2],255,np.uint8)
    for cl,col in enumerate(pal):lab[np.all(rgb==col,-1)]=cl
    if not (lab<4).all():break
    cc=dict(c,channel_order='main');v=review.read(cc,n)
    if v.shape[1:]!=lab.shape:break
    ratios['main'].append(float((v.argmax(0)==lab).mean()));ratios['team'].append(float((v[[1,2,0,3]].argmax(0)==lab).mean()))
   for key,values in ratios.items():
    if len(values)==len(names) and min(values)>.999 and max(ratios['team' if key=='main' else 'main'])<.95:choices.append((key,str(sub),values))
 if choices and len({x[0] for x in choices})==1:order=choices[0][0];proof='argmax/color-prediction exact-semantic verification: '+str(choices)
 if order is None:
  for root in [parent,parent.parent]:
   for p in root.glob('*.json'):
    if p.stat().st_size>16*1024*1024:continue
    try:d=json.loads(p.read_text())
    except Exception:continue
    if not isinstance(d,dict):continue
    for key in ['class_order','class_names']:
     v=parse_names(d.get(key))
     if v:order=v;proof='recorded '+key+' '+str(p);break
    if order:break
   if order:break
 if order:
  c.update(channel_order=order,channel_proof=proof);resolved.append(c)
 else:unresolved.append(c)
(O/'resolved_channels.json').write_text(json.dumps(resolved,indent=1));(O/'unresolved_channels.json').write_text(json.dumps(unresolved,indent=1));print('Resolved',len(resolved),'unresolved',len(unresolved),flush=True)
