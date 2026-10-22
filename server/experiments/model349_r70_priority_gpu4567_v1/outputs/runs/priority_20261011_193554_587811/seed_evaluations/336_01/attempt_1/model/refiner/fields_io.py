"""Extract immutable parent pre-cleanup logits with exact parent mask replay."""
from pathlib import Path
import ast,functools,hashlib,importlib.util,json,pickle
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent;SOURCE=ROOT.parent/'Model349_exh14'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
@functools.lru_cache(None)
def api():
 spec=importlib.util.spec_from_file_location('residual_parent_entry',SOURCE/'infer_lite.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
 fn=next(n for n in ast.parse((SOURCE/'infer_lite.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='predict_fixed')
 fn.name='fusion_fields';call=fn.body[-1].value;assert isinstance(call,ast.Call) and len(call.args)==4
 fn.body[-1]=ast.Return(value=ast.Tuple(elts=call.args,ctx=ast.Load()));ns=dict(entry.__dict__)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE/'infer_lite.py'),'exec'),ns)
 import model198_fast_branch_search as post
 p=SOURCE/'src/model198_fast_branch_search.py';fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='predict');fn.name='precleanup'
 index=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.If));block=fn.body[index];body=[]
 for n in block.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prediction' for t in n.targets):break
  body.append(n)
 assert len(body)==3,'original refinement changed; audit before using'
 fn.body=fn.body[:index]+body+[ast.Return(value=ast.Name(id='refined',ctx=ast.Load()))];pn=dict(post.__dict__)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(p),'exec'),pn)
 return entry,ns['fusion_fields'],pn['precleanup'],post
def finish(logits,config):
 post=api()[3];s=config['route']['base'];pred=post.cleanup_primary(logits.argmax(0).astype(np.uint8),logits,s['cleanup_threshold'])
 if (pred==1).mean()<=.01:
  mask=pred==1;alt=logits.copy();alt[1]=-np.inf;pred[mask]=alt[:,mask].argmax(0).astype(np.uint8)
 return pred
def prepare(run,out,split):
 assert split in ['train','valid'];run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 m=json.loads((run/'input_manifest.json').read_text());entry,fusion,precleanup,_=api();cfg=m['config']
 assert sha(SOURCE/cfg['calibration'])==m['calibration_sha256'];expected=70 if split=='train' else 20;assert len(m['images'])==expected
 receipts=[]
 for row in m['images']:
  name=row['name'];assert sha(row['path'])==row['sha256']
  with (run/'probabilities/base'/f'{name}.pkl').open('rb') as f:item=pickle.load(f)
  sources={c:np.load(run/'probabilities'/c/f'{name}.npy').astype(np.float32) for c in entry.active_components(cfg) if c!='base'}
  item,mixed,setting,weights=fusion(item,sources,cfg);logits=precleanup(item,mixed,setting,weights,True).astype(np.float32)
  old=np.array(Image.open(run/'predicted_labels'/f'{name}.png'));assert np.array_equal(finish(logits,cfg),old),(name,'parent replay differs')
  gray=np.array(Image.open(row['path']).convert('L'),dtype=np.float32)/255.;assert logits.shape==(4,*gray.shape) and np.isfinite(logits).all()
  inputs={'logits':logits,'gray':gray};receipt={'name':name,'image_sha256':row['sha256'],'parent_mask_exact':True}
  if split=='train':
   mask=Path('/workspace/komap_model1/data/train/masks')/f'{name}_mask.png';inputs['target']=entry.decode_gt(mask,np.array(cfg['palette'],dtype=np.uint8));receipt['train_mask_sha256']=sha(mask)
  p=out/f'{name}.npz';np.savez_compressed(p,**inputs);receipt['features_sha256']=sha(p);receipts.append(receipt)
  print('FIELDS',split,name,flush=True)
 write(out/'manifest.json',{'split':split,'images':receipts,'parent_masks_exact':True,'config':cfg,'parent_manifest_sha256':sha(run/'input_manifest.json'),'source_inference_sha256':sha(SOURCE/'infer_lite.py'),'fields_code_sha256':sha(Path(__file__)),'valid_gt_decoded':False})
 return receipts
