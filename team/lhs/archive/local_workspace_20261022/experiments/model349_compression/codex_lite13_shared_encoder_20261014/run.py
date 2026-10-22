import os,sys,json,time,hashlib,traceback,fcntl,inspect,importlib.util,gc
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
D=Path(__file__).resolve().parent;K=Path('/home/sangpil/komap');L=K/'lite13_R322share326_20261014';DEP=K/'codex_lite13_E_seed5_reduction_20261014/completion.json'
PY='/home/sangpil/miniforge3/envs/komap/bin/python'
RROOT=L/'parent_package/Model349_exh14';EROOT=L/'runtime/model397_reproduction_bundle/model329'
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
rmod=module('rb4',RROOT/'src/b4_zcdf.py');emod=module('eb4',EROOT/'common.py')
def dump(name,value):
 p=D/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
folders={**{f'R{f}':RROOT/f'weights/model329_fullstats_width4000_uniform_group_oof/fold{f}' for f in (1,3,4)},**{f'E{f}':L/f'assets/model329_ws/fold{f}' for f in (1,3)}}
def metadata():
 return {tag:json.loads((p/'args.json').read_text()) for tag,p in folders.items()}
def head(model,features):
 params=inspect.signature(model.decoder.forward).parameters
 decoded=model.decoder(features) if len(params)==1 and next(iter(params.values())).kind!=inspect.Parameter.VAR_POSITIONAL else model.decoder(*features)
 return model.segmentation_head(decoded)
def same_schema(a,b):return {k:(tuple(v.shape),str(v.dtype)) for k,v in a.items()}=={k:(tuple(v.shape),str(v.dtype)) for k,v in b.items()}

def export_package(name,encoders,head_tags,members):
 out=D/'packages'/name;out.mkdir(parents=True,exist_ok=True);files=[]
 for tag,enc in encoders.items():
  p=out/f'encoder_{tag}.pt'
  if not p.exists():torch.save({k:v.detach().cpu().clone() for k,v in enc.state_dict().items()},p)
  files.append({'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
 for tag in head_tags:
  model=members[tag]['model'];state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('encoder.')};p=out/f'head_{tag}.pt'
  if not p.exists():torch.save(state,p)
  files.append({'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
 info={'files':files,'bytes':sum(x['bytes'] for x in files),'unique_encoders':len(encoders),'output_heads':len(head_tags),'remaining_lite13_weights_unchanged':True,'single_file_is_not_one_model':True,'inference_only_prototype':True}
 dump(f'packages/{name}/manifest.json',info);return info

@torch.no_grad()
def predict_shared(encoder,heads,tags,gray,meta):
 H,W=gray.shape;scale=meta['target_width']/W;g=rmod.resize_gray(gray,scale);h,w=g.shape;x=torch.from_numpy(rmod.make_input(g,meta['norm'],meta['input'],meta['stats']))[None].to('cuda');x=F.pad(x,(0,(-w)%32,0,(-h)%32),mode='reflect')
 rtags=[t for t in tags if t.startswith('R')];etags=[t for t in tags if t.startswith('E')];acc={'R':None,'E':None};calls=0
 for k in (range(8) if etags else [0]):
  xx=rmod._d4(x,k)
  with torch.autocast('cuda',dtype=torch.float16):feat=encoder(xx)
  calls+=1
  for group,selected in [('R',rtags if k==0 else []),('E',etags)]:
   view=None
   for tag in selected:
    with torch.autocast('cuda',dtype=torch.float16):logits=head(heads[tag],feat)
    p=rmod._d4_inv(logits.float().softmax(1),k);view=p if view is None else view+p
   if view is not None:
    view=view/len(selected);acc[group]=view if acc[group] is None else acc[group]+view
  del feat,xx
 out={}
 for group,value in acc.items():
  if value is None:continue
  value=value[:,:,:h,:w]/(8 if group=='E' else 1)
  if scale!=1:value=F.interpolate(value,size=(H,W),mode='area' if scale>1 else 'bilinear')
  out[group]=value[0].cpu().numpy().astype(np.float32)
 return out,calls


def run():
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 manifest=json.loads((L/'selected_assets.json').read_text());frozen={}
 for row in manifest['checkpoints']+manifest.get('metadata',[]):
  path=(RROOT if row['namespace']=='R' else L/'assets')/row['path'];assert sha(path)==row['sha256'],str(path);frozen[str(path)]=row['sha256']
 meta=metadata();fields=('arch','encoder','norm','scale','target_width','input','feat_stats');first={f:meta['R1'].get(f) for f in fields}
 assert all({f:a.get(f) for f in fields}==first for a in meta.values());assert first['target_width']==4000 and first['input']=='zcdf3'
 sample=np.arange(128*128,dtype=np.uint8).reshape(128,128);assert np.array_equal(rmod.make_input(sample,'image_global','zcdf3'),emod.make_input(sample,'image_global','zcdf3'))
 members={tag:(rmod if tag[0]=='R' else emod).load_member(str(folder),'cpu','last.pt') for tag,folder in folders.items()};models={tag:m['model'] for tag,m in members.items()}
 assert all(same_schema(models['R1'].encoder.state_dict(),m.encoder.state_dict()) for m in models.values())
 controls={}
 with torch.no_grad():
  x=torch.zeros((1,3,64,64))
  for tag in ('R1','E1'):
   m=models[tag];original=m(x);shared=head(m,m.encoder(x));err=float((original-shared).abs().max());assert err==0,(tag,err);controls[tag]=err
 packs={}
 packs['R_shared']=export_package('R_shared',{'R1':models['R1'].encoder},['R1','R3','R4'],members)
 packs['E_shared']=export_package('E_shared',{'E1':models['E1'].encoder},['E1','E3'],members)
 packs['both_separate']=export_package('both_separate',{'R1':models['R1'].encoder,'E1':models['E1'].encoder},list(models),members)
 packs['all_shared']=export_package('all_shared',{'R1':models['R1'].encoder},list(models),members)
 dump('preflight.json',{'source_sha':frozen,'same_preprocessing':first,'encoder_shapes_compatible':True,'decoder_forward_matches_original_cpu':controls,'packages':packs,'anchor_policy':'fixed fold1, no valid selection'})
 dump('status.json',{'phase':'waiting_existing_seed_queue','pid':os.getpid(),'dependency':str(DEP)})
 while not DEP.exists() and not (D/'priority_authorized.json').exists():time.sleep(10)
 while torch.cuda.is_available() and __import__('subprocess').check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(10)
 assert torch.cuda.is_available();dump('status.json',{'phase':'GPU inference','pid':os.getpid()})
 for m in models.values():m.to('cuda').eval()
 # Decoder/head modules are kept separate; unneeded encoders are freed.
 for tag in ('R3','R4','E3'):models[tag].encoder=None
 gpu_controls={}
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
  x=torch.zeros((1,3,64,64),device='cuda')
  for tag in ('R1','E1'):
   original=models[tag](x);shared=head(models[tag],models[tag].encoder(x));err=float((original-shared).abs().max());assert err==0,(tag,err);gpu_controls[tag]=err
 dump('gpu_control.json',gpu_controls)
 names=sorted((K/'data/valid/images').glob('*_image.png'));assert len(names)==20;inputs={str(p):sha(p) for p in names};dump('input_sha.json',inputs)
 groups=[('R_shared',models['R1'].encoder,['R1','R3','R4']),('E_shared',models['E1'].encoder,['E1','E3']),('all_shared',models['R1'].encoder,list(models))];timings={}
 for name,encoder,tags in groups:
  out=D/'probabilities'/name;out.mkdir(parents=True,exist_ok=True);rows=[];start=time.time();torch.cuda.reset_peak_memory_stats()
  for img in names:
   stem=img.name.removesuffix('_image.png');needed=[out/f'{stem}_{role}.npy' for role in ('R','E') if any(t.startswith(role) for t in tags)]
   if all(p.exists() for p in needed):continue
   dump('active_image.json',{'case':name,'image':stem,'pid':os.getpid()})
   torch.cuda.synchronize();tick=time.time();probs,calls=predict_shared(encoder,models,tags,rmod.load_gray(img),members['R1']);torch.cuda.synchronize()
   for role,p in probs.items():assert np.isfinite(p).all();np.save(out/f'{stem}_{role}.npy',p)
   rows.append({'image':stem,'seconds':time.time()-tick,'encoder_calls':calls});print(name,stem,calls,flush=True)
  timings[name]={'seconds_this_attempt':time.time()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'rows':rows,'expected_encoder_calls_per_image':8 if any(t.startswith('E') for t in tags) else 1};dump(f'probabilities/{name}/completion.json',timings[name]);torch.cuda.empty_cache()
 assert all(sha(p)==h for p,h in frozen.items());assert all(sha(p)==h for p,h in inputs.items())
 dump('gpu_completion.json',{'timings':timings,'inputs_sha_unchanged':True,'source_checkpoints_sha_unchanged':True,'test_GT_used':False})
 del models,members;gc.collect();torch.cuda.empty_cache()
 import subprocess
 dump('status.json',{'phase':'CPU final scoring','pid':os.getpid()});subprocess.run([PY,str(D/'score.py')],check=True)
 dump('completion.json',{'finished':time.time(),'phase1_no_training':True,'decoder_realignment_training_started':False,'release_replaced':False,'test_GT_used':False});dump('status.json',{'phase':'complete'})

if __name__=='__main__':
 lock=(D/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (D/'completion.json').exists():sys.exit(0)
 try:run()
 except Exception:dump('failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
