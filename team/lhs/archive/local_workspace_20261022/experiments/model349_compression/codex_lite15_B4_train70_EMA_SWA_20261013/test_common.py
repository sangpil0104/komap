from pathlib import Path
import json,textwrap,hashlib
import numpy as np
import pipeline_common as pc
K=pc.K;P=pc.P;X=pc.X

def records(output):
 import torch
 source=(K/'codex_R326_plain_EMA_test10_20261013/run.py').read_text();start=source.index(' import torch,yaml,cv2');end=source.index(' parent=40249;paused=False');code=textwrap.dedent(source[start:end])
 old="rr=infer_lite.predict_fixed(item,src,cfg);assert np.array_equal(rr,np.asarray(Image.open(K/f'l18h/e2e_test/release/predicted_labels/{n}.png'))),'R control mismatch '+n"
 new="src['326']=rt(np.load(K/f'codex_R326_plain_EMA_test10_20261013/probabilities/plain3456/{n}.npy'));rr=infer_lite.predict_fixed(item,src,cfg)";assert old in code;code=code.replace(old,new)
 code=code.replace("expected=np.array(Image.open(N/f'results/N3/predicted_labels/{n}.png'))","expected=np.array(Image.open(K/f'codex_R326_plain_EMA_test10_20261013/results/plain3456/predicted_labels/{n}.png'))")
 code=code.replace("'base':base}","'base':base,'e':e,'gray':gray,'parts':{'N3':np3,'s1f4':parts['hr_s1_f4'],'s2f1':parts['hr_s2_f1'],'s2f4':parts['hr_s2_f4']},'b4':(e-.25*newh1-.25*rt((parts['hr_s2_f1']+parts['hr_s2_f4'])/2))/.5}")
 def dump(n,v):(output/n).write_text(json.dumps(v,indent=1))
 scope={'K':K,'P':P,'X':X,'O':output,'C':K/'add/base_test','OLD':K/'codex_lite16_b4_local_20261012/fresh/test','N':K/'codex_N3_L2_seed_retrained_test_20261013','H':K/'codex_lite16_N3_L2_test_20261012','BASE':pc.A,'np':np,'json':json,'pickle':__import__('pickle'),'importlib':__import__('importlib'),'sys':__import__('sys'),'sha':pc.sha,'dump':dump}
 exec(code,scope)
 for n,r in scope['records'].items():
  f=K/f'codex_R326_plain_EMA_test10_20261013/probabilities/plain3456/{n}.npy';scope['hashes'][str(f)]=pc.sha(f)
 return {'records':scope['records'],'ref':scope['ref'],'fu':scope['fu'],'infer':scope['infer_lite'],'cfg':scope['cfg'],'palette':scope['pal'],'sha':scope['hashes']}
def expert_delta(rec,parts=None,b4=None):
 e=rec['e'].copy()
 if parts is not None:
  for group in [('N3','s1f4'),('s2f1','s2f4')]:
   before=pc.rt((rec['parts'][group[0]]+rec['parts'][group[1]])/2);after=pc.rt((parts[group[0]]+parts[group[1]])/2);e+=.25*(after-before)
 if b4 is not None:e+=.5*(b4-rec['b4'])
 return e
