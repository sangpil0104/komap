import os,json,time,copy,itertools,traceback,importlib.util
from pathlib import Path
import numpy as np
O=Path('/workspace/model349_compress/codex_B4base_REF_design_20261013');S=Path('/workspace/model349_compress/codex_lite16N3_allmodels_20261013')
s=importlib.util.spec_from_file_location('screen',S/'review.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=1))
MEMBERS={'base':{'R329f1','R329f2','R329f3','R329f4'},'B4gray':{'R322f3'},'B4ZCDF':{'R326f1','R326f2'},'HRNet':{'R347'},'ResNet_native':{'R269'},'ResNet_highres':{'Rrestart61'},'HRNet_ZCDF_train70':{'N3'}}
EMEM={'EB4wsf1','EB4wsf3','N3','HRs1f4','HRs2f1','HRs2f4'}
def blend(p,q,alpha,classes):
 out=p.copy();out[list(classes)]=(1-alpha)*p[list(classes)]+alpha*q[list(classes)];out/=out.sum(0,keepdims=True);return out

def main():
 if (O/'completion.json').exists():return
 m.init(False);G=m.audit.GLOBAL;cover=json.load(open('/workspace/model349_compress/codex_lite16_allmodel_complement_20261013/coverage.json'));c=next(x for x in cover['candidates'] if x['id']==146)
 for n,r in G['records'].items():
  r['newbase']=m.audit.read(c,n);r['donors']={'B4gray':r['sources']['322'],'B4ZCDF':r['sources']['326'],'HRNet':r['sources']['347'],'ResNet_native':r['sources']['269'],'ResNet_highres':r['item']['preaux'],'HRNet_ZCDF_train70':r['parts']['N3']}
 baseline_single=100*np.mean([m.audit.iou(G['fu'].labels(r['newbase']),r['gt'])[1] for r in G['records'].values()]);assert abs(baseline_single-78.83342199218438)<1e-8
 def evaluate(steps):
  rows={}
  for n,r in G['records'].items():
   p=r['newbase']
   for st in steps:p=blend(p,r['donors'][st['donor']],st['alpha'],st['classes'])
   lab=G['fu'].labels(p);v=m.score(r,lab);rows[n]=v
  summary=m.summarize(rows);members=set(MEMBERS['base'])
  for st in steps:members|=MEMBERS[st['donor']]
  return {'steps':steps,'summary':summary,'R_checkpoint_count':len(members),'R_members':sorted(members),'per_image':rows}
 current=evaluate([]);rounds=[];used=set();done=0
 for iteration in range(2):
  results=[current]
  for donor in MEMBERS:
   if donor=='base' or donor in used:continue
   for alpha in [.1,.25,.5]:
    for cs in [[0],[1],[2],[3],[0,1,2,3]]:
     v=evaluate(current['steps']+[{'donor':donor,'alpha':alpha,'classes':cs}]);results.append(v);done+=1;dump('progress.json',{'stage':'new R stacking','round':iteration+1,'completed_configs':done,'pid':os.getpid(),'time':time.time()})
  results.sort(key=lambda x:-x['summary']['percent']);rounds.append(results);dump('R_rounds.json',rounds)
  best=results[0]
  if best['summary']['percent']<=current['summary']['percent']+1e-12:break
  current=best;used={x['donor'] for x in current['steps']}
 dump('selected_R.json',{'single_base_percent':baseline_single,'selected':current,'selected_using_development_valid20':True,'method':'two-step forward search; class masks all or one; alpha .1/.25/.5; R mIoU objective'})
 rows_by_rule={}
 import torch
 for n,r in G['records'].items():
  p=r['newbase']
  for st in current['steps']:p=blend(p,r['donors'][st['donor']],st['alpha'],st['classes'])
  rr=G['fu'].labels(p)
  hr=.5*m.roundtrip((r['parts']['N3']+r['parts']['s1f4'])/2)+.5*m.roundtrip((r['parts']['s2f1']+r['parts']['s2f4'])/2)
  for bweight in [.25,.5,.75]:
   e=bweight*r['b4']+(1-bweight)*hr;ex=G['fu'].labels(e)
   with torch.no_grad():fv,_=G['refiner'](torch.from_numpy(np.log(np.maximum(e,1e-7))[None]),r['gray'])
   fl=G['fu'].labels(fv.numpy()[0])
   for bits in range(16):
    cs=[i for i in range(4) if bits&(1<<i)];stage=np.where(np.isin(ex,cs),ex,rr)
    for f_on in [False,True]:
     pred=np.where(fl==2,2,stage) if f_on else stage;tag=f'b4_{bweight}_classes{bits}_F{int(f_on)}';rows_by_rule.setdefault(tag,{})[n]=m.score(r,pred)
  dump('progress.json',{'stage':'new E/F combinations','images':len(next(iter(rows_by_rule.values()))),'total_images':20,'R_configs':done,'pid':os.getpid()})
 ranking=[]
 for tag,rows in rows_by_rule.items():
  bits=int(tag.split('_classes')[1].split('_')[0]);f_on=tag.endswith('F1');members=set(current['R_members'])
  if bits or f_on:members|=EMEM
  if f_on:members.add('F')
  ranking.append({'rule':tag,**m.summarize(rows),'checkpoint_count':len(members),'members':sorted(members)})
 ranking.sort(key=lambda x:-x['percent']);dump('ranking.json',ranking);dump('EF_per_image.json',rows_by_rule)
 dump('completion.json',{'time':time.time(),'R_configs':done,'EF_configs':len(ranking),'base_single_percent':baseline_single,'current_lite16N3_percent':m.BASE,'selected_R_percent':current['summary']['percent'],'best_screen_final':ranking[0],'valid20_development_selection':True,'F_original_OOF_weights_not_retrained':True,'old_R_route_and_postprocessing_not_applied':True,'no_new_GPU_inference':True,'automatic_adoption':False,'greedy_R_search_not_global_grid':True});print('COMPLETE',json.dumps(ranking[:3]),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',{'trace':traceback.format_exc()});raise
