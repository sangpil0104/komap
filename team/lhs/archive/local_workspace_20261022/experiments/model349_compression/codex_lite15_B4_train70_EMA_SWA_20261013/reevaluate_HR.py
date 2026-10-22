from pathlib import Path
import json,time,traceback
import numpy as np
from pipeline_common import *
O=K/'codex_lite15_HRtrain70_evaluation_20261013';OLD=K/'codex_HR392_train70_N3_20261013'
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=1))
def main():
 O.mkdir(exist_ok=True)
 if (O/'completion.json').exists():return
 while not (OLD/'completion.json').exists() or not (O/'removal_ready.json').exists():
  if (OLD/'failure.json').exists():raise RuntimeError('original HR train70 failed')
  dump('status.json',{'stage':'waiting removal screen and HR seed1/2 completion','time':time.time()});time.sleep(20)
 G=init_records();data={};receipts={}
 for seed in [1,2]:
  ck=OLD/f'training/seed{seed}/last.pt';receipts[str(ck)]=sha(ck);expected=json.load(open(OLD/f'evaluation/seed{seed}/result.json'));assert expected['seed']==seed
  for n,r in G['records'].items():
   # Match package storage: individual model probabilities are rounded to FP16
   # before family averaging and fusion.
   p=rt(np.load(OLD/f'evaluation/seed{seed}/probabilities/{n}.npy').astype(np.float32)[[1,2,0,3]])
   for targets in [('s1f4',),('s2f1',),('s2f4',),('s2f1','s2f4'),('s1f4','s2f1','s2f4')]:
    parts=dict(r['parts']);parts.update({t:p for t in targets});pred=final(G,r,r['r'],expert(parts,r['b4']));key=f'seed{seed}_'+('_'.join(targets));data.setdefault(key,{})[n]=score(r,pred)
  dump('status.json',{'stage':'CPU re-evaluation on adopted lite15','completed_seeds':seed,'time':time.time()})
 ranking=[]
 for key,rows in data.items():
  ranking.append({'key':key,'checkpoint_count':15-(len(key.split('_'))-1)+1,**summary(rows)})
 ranking.sort(key=lambda x:-x['percent']);dump('ranking.json',ranking);dump('per_image.json',data);dump('completion.json',{'time':time.time(),'baseline_percent':BASE,'candidates':len(ranking),'checkpoint_sha':receipts,'new_GPU_inference':False,'current15_baseline_exact20':True,'release_replaced':False});print('HR15_EVALUATION_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',{'trace':traceback.format_exc()});raise
