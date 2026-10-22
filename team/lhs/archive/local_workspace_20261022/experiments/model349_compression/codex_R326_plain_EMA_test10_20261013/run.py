import os,sys,time,json,pickle,hashlib,traceback,signal,importlib.util,subprocess
from pathlib import Path
import numpy as np
K=Path.home()/'komap';O=K/'codex_R326_plain_EMA_test10_20261013';P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805';X=P/'Model349_exh14';C=K/'add/base_test';OLD=K/'codex_lite16_b4_local_20261012/fresh/test';N=K/'codex_N3_L2_seed_retrained_test_20261013';H=K/'codex_lite16_N3_L2_test_20261012';TRAIN=K/'codex_HR392_train70_N3_20261013';BASE=K/'lite16_N3_adopted_20261013'
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def state(pid):
 try:return subprocess.check_output(['ps','-p',str(pid),'-o','stat='],text=True).strip()
 except subprocess.CalledProcessError:return None
def main():
 if (O/'completion.json').exists():return
 import torch,yaml,cv2
 from PIL import Image,ImageDraw,ImageFont
 from scipy import ndimage as ndi
 torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;sys.path[:0]=[str(X),str(X/'src')];import infer_lite
 sp=importlib.util.spec_from_file_location('fu',P/'infer_fusion.py');fu=importlib.util.module_from_spec(sp);sp.loader.exec_module(fu);ref=fu.Residual().eval();ref.load_state_dict(torch.load(K/'weights/all/assets/refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict'])
 cfg=yaml.safe_load((BASE/'selected_R_lite16_N3.yaml').read_text());man=json.load(open(C/'input_manifest.json'));pal=np.array(cfg['palette'],np.uint8);records={};hashes={};rt=lambda p:p.astype(np.float16).astype(np.float32)
 jobs={'EMA3461':(K/'codex_r326_ema_seed4_20261012/training/set3/fold1','ema.pt','074c2bf55886853bcc88d220560ab587770666d5e3986bd8e2067a7f9c49d4d5'),'plain3456':(K/'codex_r326_ema_seed4_20261012/training/set0/fold2','last.pt','06602d7378249de97a0055923861c70d053b7fcbb374597dbd6187e9688605c6')}
 for tag,(folder,file,h) in jobs.items():assert sha(folder/file)==h;hashes[str(folder/file)]=h;hashes[str(folder/'args.json')]=sha(folder/'args.json')
 for row in man['images']:
  n=row['name'];ip=K/f'data/test/images/{n}_image.png';assert sha(ip)==row['sha256'];hashes[str(ip)]=row['sha256'];raw=Image.open(ip).convert('RGB')
  f=C/f'probabilities/base/{n}.pkl';hashes[str(f)]=sha(f)
  with f.open('rb') as file:item=pickle.load(file)
  src={}
  for c in cfg['inference']['active_components']:
   if c=='base':continue
   f=C/f'probabilities/{c}/{n}.npy';hashes[str(f)]=sha(f);src[c]=np.load(f).astype(np.float32)
  rr=infer_lite.predict_fixed(item,src,cfg);assert np.array_equal(rr,np.asarray(Image.open(K/f'l18h/e2e_test/release/predicted_labels/{n}.png'))),'R control mismatch '+n
  parts={}
  for tag in ['hr_s1_f2','hr_s1_f4','hr_s2_f1','hr_s2_f4']:
   f=H/f'probabilities/{tag}/{n}.npy';hashes[str(f)]=sha(f);parts[tag]=np.load(f).astype(np.float32)[[1,2,0,3]]
  f=N/f'probabilities/N3/{n}.npy';hashes[str(f)]=sha(f);np3=np.load(f).astype(np.float32)[[1,2,0,3]];h1=rt((parts['hr_s1_f2']+parts['hr_s1_f4'])/2);newh1=rt((np3+parts['hr_s1_f4'])/2)
  f=OLD/f'expert/b4_13/{n}.npy';hashes[str(f)]=sha(f);e=np.load(f).astype(np.float32)+.25*(newh1-h1);gray=torch.from_numpy(fu.prepare_gray(np.array(raw.convert('L'),np.float32)/255)[None,None])
  with torch.no_grad():v,_=ref(torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),gray)
  ex=fu.labels(e);fl=fu.labels(v.numpy()[0]);base=np.where(fl==2,2,np.where(np.isin(ex,[0,2]),ex,rr)).astype(np.uint8);expected=np.array(Image.open(N/f'results/N3/predicted_labels/{n}.png'));assert np.array_equal(base,expected),'N3 final control mismatch '+n
  records[n]={'raw':raw,'item':item,'sources':src,'ex':ex,'fl':fl,'base':base}
 assert len(records)==10;dump('input_receipt.json',{'sha256':hashes,'R_control_exact10':True,'N3_final_control_exact10':True,'test_GT_used':False,'common_E_F_fixed':True});print('PREFLIGHT_CONTROL10_EXACT',flush=True)
 parent=40249;paused=False
 try:
  if state(parent) and (TRAIN/'status.json').exists() and json.load(open(TRAIN/'status.json'))['stage']=='train70 HR392 training':
   child=json.load(open(TRAIN/'active_child.json'))['pid'];assert str(TRAIN) in os.readlink(f'/proc/{parent}/cwd')
   cmd=Path(f'/proc/{child}/cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'train.py' in cmd and str(TRAIN) in cmd
   assert 'T' not in state(parent);os.kill(parent,signal.SIGSTOP);paused=True;dump('priority_receipt.json',{'paused_controller':parent,'training_child_continues':child,'time':time.time()})
   while (st:=state(child)) and 'Z' not in st:
    dump('status.json',{'stage':'waiting current HR seed training to finish; controller paused','training_child':child,'pid':os.getpid(),'time':time.time()});time.sleep(10)
  else:
   while not (TRAIN/'completion.json').exists():
    if (TRAIN/'failure.json').exists():raise RuntimeError('prior training queue failed')
    dump('status.json',{'stage':'waiting prior queue completion; natural training boundary unavailable','pid':os.getpid(),'time':time.time()});time.sleep(20)
  stable=0
  while stable<3:
   ids=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines();busy=any(x.strip() and int(x)!=os.getpid() for x in ids);stable=0 if busy else stable+1
   if stable<3:time.sleep(10)
  from b4_zcdf import load_member,load_gray,predict_probs
  for tag,(folder,file,h) in jobs.items():
   dest=O/f'probabilities/{tag}';dest.mkdir(parents=True,exist_ok=True)
   if (dest/'completion.json').exists():continue
   dump('status.json',{'stage':'test10 R326 GPU inference','variant':tag,'pid':os.getpid(),'time':time.time()});model=load_member(folder,'cuda',file)
   for n in records:
    f=dest/f'{n}.npy'
    if f.exists():continue
    p=predict_probs([model],load_gray(K/f'data/test/images/{n}_image.png'),tta=True,device='cuda').cpu().numpy().astype(np.float32)[[1,2,0,3]];assert np.isfinite(p).all();np.save(f,p);print(tag,n,flush=True)
   del model;torch.cuda.empty_cache();dump(str(dest.relative_to(O)/'completion.json'),{'images':10,'checkpoint_sha256':h,'class_order':'main','D4':True})
  rows=[];fig=O/'figures';fig.mkdir(exist_ok=True)
  for n,r in records.items():
   preds={}
   for tag in jobs:
    src=dict(r['sources']);src['326']=rt(np.load(O/f'probabilities/{tag}/{n}.npy'));rr=infer_lite.predict_fixed(r['item'],src,cfg);pred=np.where(r['fl']==2,2,np.where(np.isin(r['ex'],[0,2]),r['ex'],rr)).astype(np.uint8);preds[tag]=pred
    for sub,img in [('predicted_labels',Image.fromarray(pred)),('predicted_color',Image.fromarray(pal[pred]))]:dest=O/f'results/{tag}/{sub}';dest.mkdir(parents=True,exist_ok=True);img.save(dest/f'{n}.png')
   a=preds['EMA3461'];b=preds['plain3456'];diff=a!=b;lab,num=ndi.label(diff,structure=np.ones((3,3)));sizes=np.bincount(lab.ravel())[1:];boundary=(b!=ndi.maximum_filter(b,3))|(b!=ndi.minimum_filter(b,3));near=ndi.maximum_filter(boundary,5)
   tr=np.bincount((a*4+b)[diff],minlength=16).reshape(4,4);row={'image':n,'total_pixels':diff.size,'EMA_vs_plain_changed_pixels':int(diff.sum()),'EMA_vs_plain_changed_percent':100*float(diff.mean()),'EMA_to_plain_transitions':tr.tolist(),'plain_minus_EMA_class_area_px':[int((b==c).sum()-(a==c).sum()) for c in range(4)],'changed_region_count_10px':int((sizes>=10).sum()),'largest_changed_region_px':int(sizes.max()) if len(sizes) else 0,'changed_near_plain_boundary_2px':int((diff&near).sum()),'EMA_vs_N3_changed_pixels':int((a!=r['base']).sum()),'plain_vs_N3_changed_pixels':int((b!=r['base']).sum())};rows.append(row)
   # Save full image board and dense change crops. Magenta means disagreement, never an error label.
   raw=np.array(r['raw']);highlight=raw.copy();highlight[diff]=[255,0,255];ims=[r['raw'],Image.fromarray(pal[r['base']]),Image.fromarray(pal[a]),Image.fromarray(pal[b]),Image.fromarray(highlight)];titles=['Input','lite16-N3','fold1 EMA3461','fold2 plain3456','EMA/plain disagreement']
   pw=360;ph=round(raw.shape[0]*pw/raw.shape[1]);can=Image.new('RGB',(pw*5,ph+50),'white');dr=ImageDraw.Draw(can)
   for j,(im,title) in enumerate(zip(ims,titles)):can.paste(im.resize((pw,ph),Image.Resampling.LANCZOS if j in [0,4] else Image.Resampling.NEAREST),(j*pw,35));dr.text((j*pw+4,8),title,fill='black')
   can.save(fig/f'{n}_full.png');remaining=diff.copy();boxes=[];height,width=diff.shape;crop=200
   for _ in range(3):
    if not remaining.any():break
    density=ndi.uniform_filter(remaining.astype(np.float32),160);cy,cx=np.unravel_index(density.argmax(),density.shape);left=max(0,min(width-crop,cx-crop//2));top=max(0,min(height-crop,cy-crop//2));box=(int(left),int(top),int(left+crop),int(top+crop));boxes.append(box);remaining[max(0,top-30):min(height,top+crop+30),max(0,left-30):min(width,left+crop+30)]=False
   board=Image.new('RGB',(crop*5,max(1,len(boxes))*(crop+35)),'white');dr=ImageDraw.Draw(board)
   for i,box in enumerate(boxes):
    for j,(im,title) in enumerate(zip(ims,titles)):board.paste(im.crop(box),(j*crop,i*(crop+35)+30));dr.text((j*crop+3,i*(crop+35)+5),title,fill='black')
   board.save(fig/f'{n}_crops.png')
   dump('status.json',{'stage':'CPU fusion and image changes','images':len(rows),'total':10,'time':time.time()})
  for p,h in hashes.items():assert sha(p)==h,'source changed '+p
  total=sum(r['total_pixels'] for r in rows);changes=sum(r['EMA_vs_plain_changed_pixels'] for r in rows);dump('comparison.json',{'test_GT_available':False,'control_exact10':True,'class_order':['Al3Ni','PrimarySi','Al','EutecticSi'],'summary':{'changed_pixels':changes,'total_pixels':total,'changed_percent':100*changes/total,'EMA_to_plain_transitions':np.sum([r['EMA_to_plain_transitions'] for r in rows],axis=0).tolist()},'per_image':sorted(rows,key=lambda r:-r['EMA_vs_plain_changed_percent']),'new_inference_only_two_R326_candidates':True,'common_R_E_F_reused_after_exact_control':True,'checkpoint_count_each':15,'input_SHA_unchanged':True,'accuracy_claimed':False});dump('completion.json',{'time':time.time(),'images':10,'test_accuracy_claimed':False});dump('status.json',{'stage':'complete','time':time.time()});print('COMPLETE',flush=True)
 finally:
  if paused and state(parent):os.kill(parent,signal.SIGCONT);dump('resume_receipt.json',{'controller_resumed':parent,'time':time.time()})
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',{'time':time.time(),'trace':traceback.format_exc()});raise
