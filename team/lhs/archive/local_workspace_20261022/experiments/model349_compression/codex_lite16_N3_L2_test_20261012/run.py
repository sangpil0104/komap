import os,sys,time,json,hashlib,traceback
from pathlib import Path
K=Path.home()/'komap';O=K/'codex_lite16_N3_L2_test_20261012';P=K/'Model349_lite20_b4ws_bundle/Model349_fusion805';AS=K/'weights/all/assets';BASE=K/'codex_lite16_b4_local_20261012/fresh/test';R=K/'l18h/e2e_test/release'
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def wait_gpu():
 stable=0
 while stable<3:
  import subprocess
  ids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines();busy=any(x.strip() and int(x)!=os.getpid() for x in ids);stable=0 if busy else stable+1
  if stable<3:time.sleep(20)
def main():
 if (O/'completion.json').exists():return
 dependencies=[K/'codex_r326_ema_seed4_20261012/completion.json',K/'codex_r326_ema_seed4_20261012/lite16_evaluation_completion.json',K/'codex_hr392_set5_resume_20261012/completion.json']
 while not (O/"priority_authorized.json").exists() and not all(p.exists() for p in dependencies):
  failures=[p.parent/('lite16_evaluation_failure.json' if p.name.startswith('lite16') else 'failure.json') for p in dependencies]
  if any(p.exists() for p in failures):raise RuntimeError('Preceding experiment failed; inspect before starting GPU')
  dump('queue_status.json',{'state':'waiting 326 EMA and HR392 set5 completion','pid':os.getpid(),'time':time.time()});time.sleep(20)
 import numpy as np,torch,importlib.util
 from PIL import Image,ImageDraw
 from scipy import ndimage as ndi
 sys.path.insert(0,str(P/'model397_reproduction_bundle/model392'))
 from komap_seg.checkpoint import load_model_and_preprocessing,resolve_preprocessing
 from komap_seg.dataset import KomapFullDataset,effective_input_scale,scaled_overlap
 from komap_seg.inference import predict_probs_tta
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device('cuda')
 man=json.loads((R/'input_manifest.json').read_text());names=[r['name'] for r in man['images']];assert len(names)==10;pal=np.array(man['config']['palette'],np.uint8)
 jobs=[(f'hr_s{s}_f{f}',AS/f'model392/seed{s}/fold{f}/last.pt',K/'data/test/images',True) for s,f in [(1,2),(1,4),(2,1),(2,4)]]+ [('N3',O/'assets/outputs/model414_hrnet_zcdf_ft/last.pt',O/'assets/data_zcdf_png/test/images',False),('L2',O/'assets/outputs/model_eut_L2/last.pt',K/'data/test/images',False)]
 inputs={str(p):sha(p) for r in man['images'] for p in [K/f'data/test/images/{r["name"]}_image.png']};assert all(inputs[str(K/f'data/test/images/{r["name"]}_image.png')]==r['sha256'] for r in man['images'])
 for tag,ck,inp,tta in jobs:
  inputs[str(ck)]=sha(ck)
  for n in names:inputs[str(inp/f'{n}_image.png')]=sha(inp/f'{n}_image.png')
 dump('input_receipt.json',{'sha256':inputs,'new_model_tta':'single_view matches legacy v1 screening; original HR392 flip4','recipe':'N3 replaces s1f2; L2 replaces s2f1; original family means FP16-rounded after replacement'})
 for tag,ck,inp,tta in jobs:
  dest=O/f'probabilities/{tag}';dest.mkdir(parents=True,exist_ok=True)
  if (dest/'completion.json').exists():continue
  wait_gpu();dump('queue_status.json',{'state':'GPU probability inference','model':tag,'pid':os.getpid(),'time':time.time()})
  model,saved=load_model_and_preprocessing(ck,device);prep=resolve_preprocessing(saved);ds=KomapFullDataset(inp,masks_dir=None);assert {p['stem'] for p in ds.pairs}==set(names)
  for i,pair in enumerate(ds.pairs):
   n=pair['stem'];f=dest/f'{n}.npy'
   if f.exists():continue
   sample=ds[i];scale=effective_input_scale(prep['input_scale'],prep['target_width'],sample['width'])
   value=predict_probs_tta(model,sample['image'],patch_size=512,overlap=scaled_overlap(512,2.,170,scale),batch_size=2,tta=tta,input_scale=scale,device=device,tta_mode='flip4').numpy().astype(np.float32)
   if tag in ['N3','L2']:value=value.astype(np.float16).astype(np.float32)
   assert np.isfinite(value).all() and np.max(np.abs(value.sum(0)-1))<.03;np.save(f,value);print(tag,n,flush=True)
  del model;torch.cuda.empty_cache();(dest/'completion.json').write_text(json.dumps({'images':10,'checkpoint_sha':inputs[str(ck)],'tta':tta}))
 dump('queue_status.json',{'state':'CPU F fusion and test changes','pid':os.getpid(),'time':time.time()})
 spec=importlib.util.spec_from_file_location('fu',P/'infer_fusion.py');fu=importlib.util.module_from_spec(spec);spec.loader.exec_module(fu);model=fu.Residual().eval();model.load_state_dict(torch.load(AS/'refiner/epoch8.pt',map_location='cpu',weights_only=False)['state_dict'])
 results={tag:[] for tag in ['N3','L2','both']};outfig=O/'figures';outfig.mkdir(exist_ok=True)
 for n in names:
  raw=Image.open(K/f'data/test/images/{n}_image.png').convert('RGB');e0=np.load(BASE/f'expert/b4_13/{n}.npy').astype(np.float32);parent=np.array(Image.open(R/f'predicted_labels/{n}.png'));reference=np.array(Image.open(BASE/f'b4_13/fusion/predicted_labels/{n}.png'));gray=torch.from_numpy(fu.prepare_gray(np.array(raw.convert('L'),dtype=np.float32)/255.)[None,None])
  parts={tag:np.load(O/f'probabilities/{tag}/{n}.npy')[[1,2,0,3]] for tag,_,_,_ in jobs}
  roundtrip=lambda p:p.astype(np.float16).astype(np.float32)
  h1=roundtrip((parts['hr_s1_f2']+parts['hr_s1_f4'])/2);h2=roundtrip((parts['hr_s2_f1']+parts['hr_s2_f4'])/2)
  # Family-level replacement preserves original B4 contribution without rerunning it.
  b4=(e0-.25*h1-.25*h2)/.5;assert np.min(b4)>-.005 and np.max(np.abs(b4.sum(0)-1))<.03
  newh1=roundtrip((parts['N3']+parts['hr_s1_f4'])/2);newh2=roundtrip((parts['L2']+parts['hr_s2_f4'])/2)
  experts={'control':e0,'N3':e0+.25*(newh1-h1),'L2':e0+.25*(newh2-h2),'both':e0+.25*(newh1-h1)+.25*(newh2-h2)};preds={}
  for tag,e in experts.items():
   assert np.min(e)>-.001 and np.max(np.abs(e.sum(0)-1))<.03
   with torch.no_grad():v,_=model(torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),gray)
   ex=fu.labels(e);ref=fu.labels(v.numpy()[0]);pred=np.where(ref==2,2,np.where(np.isin(ex,[0,2]),ex,parent)).astype(np.uint8);preds[tag]=pred
   if tag=='control':assert np.array_equal(pred,reference);continue
   dest=O/f'results/{tag}/predicted_labels';dest.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(dest/f'{n}.png')
   diff=pred!=reference;tr=np.bincount((reference*4+pred)[diff],minlength=16).reshape(4,4);near=ndi.maximum_filter((reference!=ndi.maximum_filter(reference,3))|(reference!=ndi.minimum_filter(reference,3)),5)
   disappeared=[]
   for c in range(4):
    lab,num=ndi.label(reference==c);sizes=np.bincount(lab.ravel());overlap=np.bincount(lab[pred==c],minlength=num+1);ids=np.where((sizes>=10)&(overlap==0))[0];ids=ids[ids!=0]
    disappeared.append({'class':c,'count':int(len(ids)),'largest_px':int(sizes[ids].max()) if len(ids) else 0})
   results[tag].append({'image':n,'changed_pixels':int(diff.sum()),'total_pixels':diff.size,'changed_percent':float(diff.mean()*100),'transitions':tr.tolist(),'class_area_delta_px':[int((pred==c).sum()-(reference==c).sum()) for c in range(4)],'near_boundary_pixels':int((diff&near).sum()),'disappeared_10px_components':disappeared})
  diff=np.logical_or.reduce([p!=reference for tag,p in preds.items() if tag!='control']);remain=diff.copy();boxes=[];h,w=diff.shape
  for _ in range(4):
   if not remain.any():break
   den=ndi.uniform_filter(remain.astype(np.float32),180);cy,cx=np.unravel_index(den.argmax(),den.shape);l=max(0,min(w-240,int(cx)-120));t=max(0,min(h-240,int(cy)-120));box=(l,t,l+240,t+240);boxes.append(box);remain[max(0,t-40):min(h,t+280),max(0,l-40):min(w,l+280)]=False
  canvas=Image.new('RGB',(240*8,270*max(1,len(boxes))),'white');dr=ImageDraw.Draw(canvas)
  for row,box in enumerate(boxes):
   imgs=[('original',raw),('lite16',Image.fromarray(pal[reference]))]+[(tag,Image.fromarray(pal[preds[tag]])) for tag in ['N3','L2','both']]
   for tag in ['N3','L2','both']:
    color=np.array(raw);color[preds[tag]!=reference]=[255,0,255];imgs.append((tag+' changes',Image.fromarray(color)))
   for j,(title,im) in enumerate(imgs):canvas.paste(im.crop(box),(j*240,row*270+28));dr.text((j*240+2,row*270+4),title,fill='black')
  canvas.save(outfig/f'{n}.png')
 summary={tag:{'changed_pixels':sum(r['changed_pixels'] for r in rows),'total_pixels':sum(r['total_pixels'] for r in rows),'changed_percent':100*sum(r['changed_pixels'] for r in rows)/sum(r['total_pixels'] for r in rows),'images':sorted(rows,key=lambda r:r['changed_pixels'],reverse=True),'transitions':np.sum([r['transitions'] for r in rows],axis=0).tolist()} for tag,rows in results.items()}
 for p,h in inputs.items():assert sha(Path(p))==h,'Input modified'
 dump('comparison.json',{'baseline':'lite16_b413','test_GT_used':False,'baseline_masks_exact':True,'replacements':{'N3':'HR392 seed1 fold2','L2':'HR392 seed2 fold1','both':'both replacements'},'results':summary,'input_sha_unchanged':True,'original_R_B4_reused':True,'release_replaced':False})
 report='# N3/L2 교체 test10 변경량\n\nGT없는변화측정. N3=HRseed1fold2교체,L2=HRseed2fold1교체,both=양교체. 모두16가중치. R/B4재사용,F각후보재계산. N3/L2는legacyv1방식single-view,원래HR392는flip4. GT미사용. 자홍색은오류가아닌클래스변경. 소영역소실도정답손실로해석하지않는다.\n\n| 구성 | 변경픽셀 | 비율 |\n|---|---:|---:|\n'
 for tag,v in summary.items():report+=f"| {tag} | {v['changed_pixels']:,} | {v['changed_percent']:.4f}% |\n"
 (O/'report.md').write_text(report);page='<meta charset="utf-8"><title>N3/L2 test10 비교</title><style>body{font-family:system-ui;margin:24px}img{max-width:100%}</style><h1>N3/L2 test10 교체 비교</h1><p>원본/lite16/N3/L2/both/각변경표시. 자홍색은클래스변경이며오류표시가아닙니다. GT없음.</p><a href="report.md">수치보고서</a>'
 for n in names:page+=f'<h2>{n}</h2><img src="figures/{n}.png">'
 (O/'index.html').write_text(page);dump('completion.json',{'time':time.time(),'images':10,'control_exact':True,'test_accuracy_claimed':False});dump('queue_status.json',{'state':'complete','time':time.time()})
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',{'time':time.time(),'trace':traceback.format_exc()});raise
