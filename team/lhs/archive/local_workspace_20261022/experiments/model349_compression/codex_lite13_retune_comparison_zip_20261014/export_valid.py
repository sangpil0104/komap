import importlib.util,json,sys,shutil,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
K=Path('/home/sangpil/komap');D=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('b',K/'codex_lite13_E_seed5_reduction_20261014/baseline_snapshot.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
state=json.loads((K/'b4tta/search13/retune_full.json').read_text());preds=b.state_eval(state['state'],None);sc=b.score(preds)
assert abs(sc['delta']-state['in_sample']['delta'])<1e-9,(sc['delta'],state['in_sample']['delta'])
rows=[]
for n,p in preds.items():
 r=b.recs[n];old=r['base'];newiou=b.iou(p,r['gt']);rows.append({'image':n,'lite13_miou_percent':100*float(r['bci'].mean()),'retuned_miou_percent':100*float(newiou.mean()),'delta_pp':100*float((newiou-r['bci']).mean()),'original_class_iou_percent':(100*r['bci']).tolist(),'retuned_class_iou_percent':(100*newiou).tolist(),'class_delta_pp':(100*(newiou-r['bci'])).tolist(),'changed_pixels':int((old!=p).sum()),'pixels':p.size,'corrected_pixels':int(((old!=r['gt'])&(p==r['gt'])).sum()),'new_error_pixels':int(((old==r['gt'])&(p!=r['gt'])).sum())})
 for sub,v in [('original_labels',old),('retuned_labels',p),('gt_labels',r['gt'])]:
  o=D/f'assets/valid/{sub}';o.mkdir(parents=True,exist_ok=True);Image.fromarray(v).save(o/f'{n}.png')
 for sub,src in [('inputs',K/f'data/valid/images/{n}_image.png'),('gt_rgb',K/f'data/valid/masks/{n}_mask.png')]:
  o=D/f'assets/valid/{sub}';o.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,o/f'{n}.png')
for img in sorted((K/'data/test/images').glob('*_image.png')):
 n=img.name.removesuffix('_image.png');o=D/'assets/test/inputs';o.mkdir(parents=True,exist_ok=True);shutil.copyfile(img,o/f'{n}.png')
 for sub in ['original_labels','retuned_labels']:
  o=D/f'assets/test/{sub}';o.mkdir(parents=True,exist_ok=True);shutil.copyfile(K/f'codex_lite13_retune_test10_20261014/{sub}/{n}.png',o/f'{n}.png')
report={'baseline_percent':b.BASE,'retuned_percent':b.BASE+sc['delta'],'delta_pp':sc['delta'],'classes':['Al3Ni','Primary Si','Al','Eutectic Si'],'palette':b.CFG['palette'],'baseline_valid_exact':20,'retuned_matches_previous_screen':True,'rows':sorted(rows,key=lambda r:r['delta_pp']),'state':state['state'],'test_gt_used':False}
(D/'valid_comparison.json').write_text(json.dumps(report,indent=2));shutil.copyfile(K/'codex_lite13_retune_test10_20261014/comparison.json',D/'test_comparison.json')
checks={str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (D/'assets').rglob('*.png')};(D/'asset_checksums.json').write_text(json.dumps(checks,indent=2));print('DONE',report['retuned_percent'],len(checks),flush=True)
