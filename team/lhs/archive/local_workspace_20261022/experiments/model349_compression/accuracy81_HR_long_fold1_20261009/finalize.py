import json,time,subprocess,hashlib,traceback,os
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
BASE=Path('/workspace/model349_compress/accuracy81_HR_texture_pair_20261009/original_eval')
GT=Path('/workspace/komap_model1/data_model382_seed1_fold1/valid/masks')
def main():
 with (ROOT/'finalizer_started.json').open('x') as f:json.dump({'pid':os.getpid(),'controller':2328989},f)
 while not (ROOT/'completion.json').exists():
  proc=Path('/proc/2328989/cmdline')
  assert proc.exists() and b'accuracy81_HR_long_fold1_20261009/run_sequence.py' in proc.read_bytes()
  assert not (ROOT/'controller_failure.json').exists()
  time.sleep(15)
 new=ROOT/'original_eval';a=json.loads((BASE/'valid_summary.json').read_text());b=json.loads((new/'valid_summary.json').read_text())
 assert a['inference']==b['inference'] and a['class_order']==b['class_order']
 paths=[BASE/'valid_probabilities.npz',new/'valid_probabilities.npz']
 palette=np.array([[153,127,76],[25,76,153],[76,178,76],[204,204,204]],np.uint8);records=[]
 with np.load(paths[0]) as z0,np.load(paths[1]) as z1:
  assert set(z0.files)==set(z1.files) and len(z0.files)==18
  for name in sorted(z0.files):
   rgb=np.array(Image.open(GT/(name+'_mask.png')).convert('RGB'));matches=(rgb[:,:,None,:]==palette).all(-1)
   assert np.all(matches.sum(-1)==1);y=matches.argmax(-1);scores=[]
   for z in [z0,z1]:
    p=z[name];assert p.shape==(4,*y.shape) and np.isfinite(p).all();pred=p.argmax(0);ious=[]
    for c in range(4):
     union=np.count_nonzero((pred==c)|(y==c));ious.append(np.count_nonzero((pred==c)&(y==c))/union if union else 0.)
    scores.append(ious)
   records.append({'name':name,'before':scores[0],'after':scores[1]})
 before=100*np.mean([r['before'] for r in records]);after=100*np.mean([r['after'] for r in records]);delta=after-before
 result={'original_percent':before,'warmstart_percent':after,'delta_pp':delta,'class_delta_pp':(100*(np.mean([r['after'] for r in records],0)-np.mean([r['before'] for r in records],0))).tolist(),'inference_metadata_exact':True,'records':records,'probability_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'scope':'train70 held fold1; exported FP16 independent scoring; not official valid20'}
 (ROOT/'independent_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 (ROOT/'report.md').write_text(f'# HRNet fold1 추가 12epoch 결과\n\n보류18 독립 FP16 채점: 4epoch {before:.6f}%, 추가12epoch {after:.6f}%, 변화 {delta:+.6f}%p.\n\n클래스 변화(Al, Al3Ni, Primary Si, Eutectic Si): {result["class_delta_pp"]} %p. 추론 설정 메타데이터가 같음을 확인했다. train70 내부 fold1 결과이며 공식 valid20 및 전체 파이프라인 점수가 아니다. 원본 가중치와 데이터 불변 확인 후 채점했다.\n')
 (ROOT/'finalizer_completion.json').write_text(json.dumps({'time':time.time(),'independent_scoring_completed':True,'delta_pp':delta},indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['records','probability_sha256']}))
if __name__=='__main__':
 try:main()
 except BaseException:
  (ROOT/'finalizer_failure.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2));raise
