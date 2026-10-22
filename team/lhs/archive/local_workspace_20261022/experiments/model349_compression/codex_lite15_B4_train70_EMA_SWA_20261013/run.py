from pathlib import Path
import os,sys,json,time,subprocess,traceback,copy,gc,importlib.util
import numpy as np
import pipeline_common as pc
K=pc.K;P=pc.P;X=pc.X;O=K/'codex_lite15_B4_train70_EMA_SWA_20261013';PY=sys.executable;COMMON=K/'Model349_lite20_b4ws_bundle/train_route/experiments/model326_hybrid_fullstats/KoMaP'
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=1))
def idle():
 stable=0
 while stable<3:
  ids=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines();stable=0 if any(x.strip() and int(x)!=os.getpid() for x in ids) else stable+1
  if stable<3:time.sleep(10)
def run(cmd,log):
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(COMMON)+':'+str(X/'oof_design/runtime'))
 with log.open('a') as f:
  child=subprocess.Popen(list(map(str,cmd)),stdout=f,stderr=f,cwd=COMMON,env=env);dump('active_child.json',{'pid':child.pid,'log':str(log),'time':time.time()});code=child.wait()
 if code:raise RuntimeError('child failed '+str(log))
def predict_dual(member,gray):
 import torch,torch.nn.functional as F
 import b4_zcdf as b4
 H,W=gray.shape;scale=b4.member_scale(member,W);g=b4.resize_gray(gray,scale);h,w=g.shape;x=torch.from_numpy(b4.make_input(g,member['norm'],member['input'],member['stats']))[None].cuda();x=F.pad(x,(0,(-w)%32,0,(-h)%32),mode='reflect');acc=None;identity=None
 with torch.no_grad():
  for k in range(8):
   with torch.autocast('cuda',dtype=torch.float16):v=member['model'](b4._d4(x,k))
   prob=b4._d4_inv(v.float().softmax(1),k);acc=prob if acc is None else acc+prob
   if k==0:identity=prob[:,:,:h,:w].clone()
  avg=acc[:,:,:h,:w]/8
  if scale!=1:
   mode='area' if scale>1 else 'bilinear';identity=F.interpolate(identity,size=(H,W),mode=mode);avg=F.interpolate(avg,size=(H,W),mode=mode)
 return identity[0].cpu().numpy()[[1,2,0,3]],avg[0].cpu().numpy()[[1,2,0,3]]
def generate_soups():
 import torch
 parents=[X/f'weights/model329_fullstats_width4000_uniform_group_oof/fold{f}' for f in [1,3,4]];weights={f:torch.load(p/'last.pt',map_location='cpu',weights_only=False) for f,p in zip([1,3,4],parents)}
 for f,w in weights.items():
  if 'model' in w:weights[f]=w['model']
 for ids in [(1,3),(1,4),(3,4),(1,3,4)]:
  out=O/('soup_'+''.join(map(str,ids)));out.mkdir(exist_ok=True)
  if (out/'completion.json').exists():continue
  first=weights[ids[0]];assert all(set(weights[f])==set(first) for f in ids);mixed={}
  for key,v in first.items():
   vs=[weights[f][key] for f in ids];assert all(x.shape==v.shape and x.dtype==v.dtype for x in vs)
   mixed[key]=torch.stack(vs).mean(0) if v.is_floating_point() else v.clone()
  meta=json.load(open(parents[0]/'args.json'));keys=['arch','encoder','norm','input','crop','scale','target_width']
  for p in parents:
   other=json.load(open(p/'args.json'));assert all(meta.get(k)==other.get(k) for k in keys)
  dump(str(out.relative_to(O)/'args.json'),meta);torch.save(mixed,out/'last.pt');dump(str(out.relative_to(O)/'completion.json'),{'parent_folds':ids,'state_dict_compatible':True,'checkpoint_sha256':pc.sha(out/'last.pt'),'source_sha256':{str(f):pc.sha(X/f'weights/model329_fullstats_width4000_uniform_group_oof/fold{f}/last.pt') for f in ids}})
 del weights;gc.collect()
def calibrate_soup(folder):
 import torch,random,albumentations as A,cv2
 from types import SimpleNamespace
 import b4_zcdf
 source=(O/'train.py').read_text();args=SimpleNamespace(**json.load(open(folder/'args.json')));args.bs=2;args.crop=1536;args.sample='uniform';args.feat_stats=None;seed=819333;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 sys.path.insert(0,str(COMMON));import common
 scope={'args':args,'A':A,'cv2':cv2,'np':np,'torch':torch,'load_gray':common.load_gray,'load_mask':common.load_mask,'resize_gray':common.resize_gray,'make_input':common.make_input}
 exec(source[source.index('aug = A.Compose'):source.index('def worker_init')],scope)
 dataset=scope['CropDS'](common.list_split(str(K/'data'),'train'),400);loader=torch.utils.data.DataLoader(dataset,batch_size=2,num_workers=0,drop_last=True);member=b4_zcdf.load_member(folder,'cuda','last.pt');net=member['model'];net.eval()
 for mod in net.modules():
  if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):mod.reset_running_stats();mod.momentum=None;mod.train()
 with torch.no_grad():
  for xx,_ in loader:
   with torch.autocast('cuda',dtype=torch.float16):net(xx.cuda())
 net.eval();torch.save(net.state_dict(),folder/'soup_bn.pt');del net,member,loader,dataset;torch.cuda.empty_cache();dump(str(folder.relative_to(O)/'bn_calibration.json'),{'train70_only':True,'batches':200,'seed':seed,'sha256':pc.sha(folder/'soup_bn.pt')})
def evaluate_candidates(candidates):
 import torch
 from PIL import Image
 import b4_zcdf
 G=pc.init_records();names=list(G['records']);rows=json.load(open(O/'per_image.json')) if (O/'per_image.json').exists() else {};info={x['key']:{k:v for k,v in x.items() if k in ['candidate','checkpoint','checkpoint_sha256','role','checkpoint_count','R_TTA','E_TTA']} for x in json.load(open(O/'ranking.json'))} if (O/'ranking.json').exists() else {}
 for tag,folder,file in candidates:
  expected_keys=[tag+'__'+role for role in ['R329_to_one','E_B4pair_to_one','R329_and_E_B4_shared']]
  if all(k in rows and len(rows[k])==20 for k in expected_keys):
   assert all(info[k]['checkpoint_sha256']==pc.sha(folder/file) for k in expected_keys);continue
  dest=O/f'probabilities/{tag}';dest.mkdir(parents=True,exist_ok=True);identity=dest/'single';augmented=dest/'d4';identity.mkdir(exist_ok=True);augmented.mkdir(exist_ok=True)
  if not (dest/'completion.json').exists():
   idle();member=b4_zcdf.load_member(folder,'cuda',file)
   for n in names:
    if (identity/f'{n}.npy').exists() and (augmented/f'{n}.npy').exists():continue
    p0,p8=predict_dual(member,b4_zcdf.load_gray(K/f'data/valid/images/{n}_image.png'));np.save(identity/f'{n}.npy',p0);np.save(augmented/f'{n}.npy',p8)
   del member;torch.cuda.empty_cache();dump(str(dest.relative_to(O)/'completion.json'),{'images':20,'checkpoint_sha256':pc.sha(folder/file),'single_and_D4_same_forward_passes':True})
  for n,r in G['records'].items():
   p0=pc.rt(np.load(identity/f'{n}.npy'));p8=pc.rt(np.load(augmented/f'{n}.npy'));src=dict(r['sources']);src['329']=p0;newr=G['infer'].predict_fixed(r['item'],src,G['cfg']);newe=pc.expert(r['parts'],p8)
   for role,rr,ee,count in [('R329_to_one',newr,r['e'],13),('E_B4pair_to_one',r['r'],newe,14),('R329_and_E_B4_shared',newr,newe,11)]:
    key=tag+'__'+role;pred=pc.final(G,r,rr,ee);rows.setdefault(key,{})[n]=pc.score(r,pred);info[key]={'candidate':tag,'checkpoint':str(folder/file),'checkpoint_sha256':pc.sha(folder/file),'role':role,'checkpoint_count':count,'R_TTA':'single','E_TTA':'D4'}
    target=O/f'predictions/{key}';target.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(target/f'{n}.png')
  dump('status.json',{'stage':'valid20 candidate evaluation','completed_candidate_models':len({v['candidate'] for v in info.values()}),'total':len(candidates),'time':time.time()})
 ranking=[{'key':k,**info[k],**pc.summary(v)} for k,v in rows.items()];ranking.sort(key=lambda x:-x['percent']);dump('ranking.json',ranking);dump('per_image.json',rows);dump('evaluation_completion.json',{'time':time.time(),'candidate_models':len(candidates),'cases':len(ranking),'baseline_percent':pc.BASE,'current15_masks_exact20':True,'release_replaced':False})
def main():
 if (O/'completion.json').exists():return
 dependencies=[K/'codex_lite15_HRtrain70_evaluation_20261013/completion.json',K/'codex_lite15_HRtrain70_evaluation_20261013/test_followup/completion.json',K/'lite15_N3_R326plain3456_adopted_20261013/fresh_completion.json']
 while not all(p.exists() for p in dependencies):
  for p in dependencies:
   f=p.parent/('fresh_failure.json' if p.name=='fresh_completion.json' else 'failure.json')
   if f.exists():raise RuntimeError('prior phase failed '+str(f))
  dump('status.json',{'stage':'waiting current15 HRtrain70 evaluation and adopted fresh validation','time':time.time(),'pid':os.getpid()});time.sleep(20)
 import torch
 torch.set_num_threads(2);sys.path[:0]=[str(X/'src')];inputs={};parents=[]
 # Import the original training implementation explicitly.
 sys.path.insert(0,str(COMMON));from common import list_split
 pairs=list_split(str(K/'data'),'train');assert len(pairs)==70
 for ip,mp in pairs:inputs[str(ip)]=pc.sha(ip);inputs[str(mp)]=pc.sha(mp)
 for fold in [1,3,4]:
  parent=X/f'weights/model329_fullstats_width4000_uniform_group_oof/fold{fold}/last.pt';inputs[str(parent)]=pc.sha(parent);parents.append(parent)
 dump('source_receipt.json',{'sha256':inputs,'train70':True,'official_valid_used_for_training':False})
 candidates=[]
 for fold,parent in zip([1,3,4],parents):
  out=O/f'training/from_fold{fold}';out.mkdir(parents=True,exist_ok=True)
  if not (out/'completion.json').exists():
   idle();dump('status.json',{'stage':'train70 low-LR B4 warmstart','parent_fold':fold,'time':time.time()})
   cmd=[PY,'-u',O/'train.py','--data',K/'data','--out',out,'--train_splits','train','--arch','unet','--encoder','efficientnet-b4','--norm','image_global','--input','zcdf3','--crop','1536','--target_width','4000','--scale','1','--sample','uniform','--bs','2','--iters','1680','--lr','3e-5','--encoder_lr_mult','.2','--wd','1e-4','--dice_w','1','--init_checkpoint',parent,'--freeze_bn_stats','--ema','.995','--ema_start','840','--swa_start','840','--swa_every','120','--workers','4','--deterministic_workers','--seed',str(819329+fold),'--no_valid_eval']
   if (out/'last.pt').exists():raise RuntimeError('unverified partial training; inspect before retry')
   run(cmd,out/'train.log');stats=json.load(open(out/'training_stats.json'));assert stats['optimizer_updates']>=1640 and stats['swa_snapshots']>=6 and stats['ema_updates']>=700;dump(str(out.relative_to(O)/'completion.json'),{'parent_fold':fold,'stats':stats,'sha256':{v:pc.sha(out/v) for v in ['last.pt','ema.pt','swa.pt']}})
  for variant,file in [('plain','last.pt'),('ema','ema.pt'),('swa','swa.pt')]:candidates.append((f'warm_fold{fold}_{variant}',out,file))
 # Warmstart/EMA/SWA are evaluated before the lower-priority cross-fold soups.
 evaluate_candidates(candidates);dump('warm_phase_completion.json',{'time':time.time(),'candidates':9})
 generate_soups();soup_candidates=[]
 for ids in [(1,3),(1,4),(3,4),(1,3,4)]:
  out=O/('soup_'+''.join(map(str,ids)))
  if not (out/'bn_calibration.json').exists():idle();calibrate_soup(out)
  soup_candidates.extend([(out.name+'_originalBN',out,'last.pt'),(out.name+'_recalibratedBN',out,'soup_bn.pt')])
 evaluate_candidates(candidates+soup_candidates)
 for p,h in inputs.items():assert pc.sha(p)==h,'original input/checkpoint changed '+p
 dump('completion.json',{'time':time.time(),'training_trajectories':3,'warm_models':9,'soup_models':8,'all_valid_candidates_evaluated':17,'baseline_percent':pc.BASE,'input_sha_unchanged':True,'release_replaced':False,'test_followup_registered':True});dump('status.json',{'stage':'complete','time':time.time()});dump('warm_phase_ready.json',{'time':time.time(),'source':'B4 warmstart, EMA/SWA, and fold-soup validation complete'})
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',{'trace':traceback.format_exc()});raise
