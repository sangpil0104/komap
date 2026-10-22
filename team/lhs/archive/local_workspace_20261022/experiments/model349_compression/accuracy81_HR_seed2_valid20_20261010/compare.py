"""Fixed full stage1 valid20 candidate; independent saved PNG scoring."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
DATA=Path('/workspace/komap_model1')
OLD=ROOT.parent/'accuracy81_stage1_fresh_20261009/runs'
PAL=np.array([(153,127,76),(25,76,153),(76,178,76),(204,204,204)],np.uint8)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def decode(p):
 rgb=np.asarray(Image.open(p).convert('RGB'));m=(rgb[:,:,None,:]==PAL).all(-1)
 assert (m.sum(-1)==1).all()
 return m.argmax(-1)
def score(p,y):
 return [np.count_nonzero((p==c)&(y==c))/u if (u:=np.count_nonzero((p==c)|(y==c))) else 0. for c in range(4)]
def main():
 assert not (ROOT/'comparison.json').exists()
 c=json.loads((ROOT/'completion.json').read_text());assert c['all_stages_completed'] and c['inputs_unchanged']
 paths=[ROOT/f'fold{f}_eval/valid_probabilities.npz' for f in range(1,5)]
 bpath=DATA/'outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz'
 frozen={str(p):sha(p) for p in paths+[bpath]}
 zs=[np.load(p,allow_pickle=False) for p in paths];b=np.load(bpath,allow_pickle=False)
 try:
  names=sorted(zs[0].files);assert len(names)==20 and all(set(z.files)==set(names) for z in zs) and set(b.files)==set(names)
  rows=[];out=ROOT/'candidate_masks';out.mkdir(exist_ok=False)
  for name in names:
   hr=[]
   for seed in [1,2]:
    p=DATA/f'outputs/model401_model397_blend_valid_once/hrnet_seed{seed}_mean_valid/{name}.npz';frozen[str(p)]=sha(p)
    with np.load(p,allow_pickle=False) as z:hr.append(z[name].astype(np.float32))
   values=[z[name] for z in zs];assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in values)
   newhr=sum(v*.25 for v in values).astype(np.float16).astype(np.float32)
   old=.5*b[name].astype(np.float32)+.25*hr[0]+.25*hr[1]
   new=.5*b[name].astype(np.float32)+.25*hr[0]+.25*newhr
   ep=OLD/f'expert_valid/{name}.npy';rp=OLD/f'release_valid/predicted_labels/{name}.png';gt=DATA/f'data/valid/masks/{name}_mask.png'
   for p in [ep,rp,gt]:frozen[str(p)]=sha(p)
   original=np.load(ep)[[2,0,1,3]];assert np.array_equal(old.argmax(0),original.argmax(0)),name
   r=np.array([1,2,0,3],np.uint8)[np.asarray(Image.open(rp))]
   enew=new.argmax(0);eold=original.argmax(0)
   pred=np.where(np.isin(enew,[0,1]),enew,r);base=np.where(np.isin(eold,[0,1]),eold,r)
   mp=out/f'{name}_mask.png';Image.fromarray(PAL[pred]).save(mp)
   reread=decode(mp);assert np.array_equal(reread,pred)
   truth=decode(gt);assert reread.shape==truth.shape
   rows.append({'name':name,'before':score(base,truth),'after':score(reread,truth),'changed_pixels':int(np.count_nonzero(base!=reread)),'mask_sha256':sha(mp)})
 finally:
  for z in zs+[b]:z.close()
 before=float(100*np.mean([r['before'] for r in rows]));after=float(100*np.mean([r['after'] for r in rows]));assert abs(before-80.47442360179173)<1e-9
 assert all(sha(Path(p))==v for p,v in frozen.items())
 result={'before_percent':before,'after_percent':after,'delta_pp':after-before,'target81_observed':after>=81.,'inputs_unchanged':True,'independent_png_scoring':True,'records':rows,'input_sha256':frozen,'scope':'full fixed R+E valid20; changed HRseed2 raw inference FP32 fold mean; original fresh R/B4/HRseed1 reused; repeated development valid, not independent generalization; test unused'}
 (ROOT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 (ROOT/'report.md').write_text(f'# HR seed2 추가 학습 전체 1단계 valid20\n\n기존 {before:.9f}% → 후보 {after:.9f}%, 변화 {after-before:+.9f}%p. 81% 관측 목표 달성: {after>=81.}.\n\nHRseed2 4fold는 새 가중치로 원본 영상부터 추론하고 FP32로 평균한 뒤 가족 평균에만 FP16 반올림했다. 기존 fresh R/B4/HRseed1은 재사용했다. 고정 E 비율·덮어쓰기 규칙을 유지한 전체 마스크를 저장 후 PNG/GT로 독립 채점했다. valid20은 반복 개발에 사용되어 독립 일반화 점수가 아니다. 릴리스는 교체하지 않았고 test10은 사용하지 않았다.\n')
 (ROOT/'finalizer_completion.json').write_text(json.dumps({'independent_scoring_completed':True,'after_percent':after,'target81_observed':after>=81.},indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['records','input_sha256']}),flush=True)
if __name__=='__main__':main()
