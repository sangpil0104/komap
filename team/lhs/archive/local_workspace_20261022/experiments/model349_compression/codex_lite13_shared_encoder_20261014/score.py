import json,importlib.util,sys
from pathlib import Path
import numpy as np
from PIL import Image
K=Path('/home/sangpil/komap');D=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('base',K/'codex_lite13_E_seed5_reduction_20261014/baseline_snapshot.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
BASE=80.72098350464233;assert abs(b.BASE-BASE)<1e-9
cases={'R_shared':('R_shared',None),'E_shared':(None,'E_shared'),'both_separate':('R_shared','E_shared'),'all_shared':('all_shared','all_shared')};rows={k:{} for k in cases};results={}
for n,r in b.recs.items():
 # Exact old E decomposition from the current HR groups recovers its frozen B4 family.
 exactb=b.rt((r['e14']-.25*r['g1']-.25*r['g2'])/.5)
 err=float(np.max(np.abs(.5*exactb+.25*r['g1']+.25*r['g2']-r['e14'])));assert err==0,(n,err)
 for tag,(rgroup,egroup) in cases.items():
  rr=r['r13'];e=None
  if rgroup:
   p=b.rt(np.load(D/f'probabilities/{rgroup}/{n}_R.npy').astype(np.float32))[[1,2,0,3]];rr=b.predict_r(r,b.CFG,{**r['src'],'329':p})
  if egroup:
   p=b.rt(np.load(D/f'probabilities/{egroup}/{n}_E.npy').astype(np.float32))[[1,2,0,3]];e=r['e14']+.5*(p-exactb)
  pred=b.finalize(r,rr,e);new=b.iou(pred,r['gt']);rows[tag][n]={'miou_percent':100*float(new.mean()),'delta_pp':100*float((new-r['bci']).mean()),'class_delta_pp':(100*(new-r['bci'])).tolist(),'corrected_pixels':int(((r['base']!=r['gt'])&(pred==r['gt'])).sum()),'new_error_pixels':int(((r['base']==r['gt'])&(pred!=r['gt'])).sum())}
  out=D/'predictions'/tag;out.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(out/f'{n}.png')
for tag,rs in rows.items():
 ds=[r['delta_pp'] for r in rs.values()];worst=min(rs,key=lambda n:rs[n]['delta_pp']);pkg=json.loads((D/f'packages/{tag}/manifest.json').read_text())
 results[tag]={'score_percent':BASE+float(np.mean(ds)),'delta_pp':float(np.mean(ds)),'worst_image':worst,'worst_image_delta_pp':min(ds),'worst_class_delta_pp':float(np.min([r['class_delta_pp'] for r in rs.values()])),'better':sum(d>1e-12 for d in ds),'worse':sum(d< -1e-12 for d in ds),'pass_mean_local':bool(np.mean(ds)>=0 and min(ds)>=-.1),'shared_B4_package_bytes':pkg['bytes'],'unique_encoder_count':pkg['unique_encoders'],'output_head_count':pkg['output_heads'],'per_image':rs}
(D/'results.json').write_text(json.dumps({'baseline':BASE,'baseline_exact20':True,'original_E_reconstruction_exact20':True,'cases':results,'test_GT_used':False,'release_replaced':False,'training_used':False,'new_GPU_source_inference':True,'output_members_not_removed':True},indent=2));print(json.dumps({k:{kk:v for kk,v in row.items() if kk!='per_image'} for k,row in results.items()},indent=2))
