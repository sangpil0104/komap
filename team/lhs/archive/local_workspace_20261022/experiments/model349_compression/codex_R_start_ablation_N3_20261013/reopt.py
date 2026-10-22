import os,json,time,copy,traceback,importlib.util
from pathlib import Path
import numpy as np
O=Path('/workspace/model349_compress/codex_R_start_ablation_N3_20261013')
s=importlib.util.spec_from_file_location('ablation',O/'run.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2))
def main():
 if (O/'reopt_completion.json').exists():return
 while not (O/'completion.json').exists():
  if (O/'failure.json').exists():raise RuntimeError('Fixed comparison failed')
  time.sleep(10)
 a.m.init(False);G=a.m.audit.GLOBAL;cover=json.load(open('/workspace/model349_compress/codex_lite16_allmodel_complement_20261013/coverage.json'));c=next(x for x in cover['candidates'] if x['id']==146)
 probabilities={n:a.m.audit.read(c,n) for n in G['records']};allresults={};best={};done=0
 def evaluate(tag,cfg):
  nonlocal done
  rows={}
  for n,rec in G['records'].items():
   it=dict(rec['item'])
   if tag=='B4_4fold':it['preaux']=probabilities[n]
   rr=G['infer'].predict_fixed(it,rec['sources'],cfg);pred=a.m.finish(rec,rec['ex'],rec['fl'],rr);v=a.m.score(rec,pred);v['R_percent']=100*a.m.audit.iou(rr,rec['gt'])[1];v['R_plus_E_percent']=100*a.m.audit.iou(np.where(np.isin(rec['ex'],[0,2]),rec['ex'],rr),rec['gt'])[1];rows[n]=v
  summary=a.m.summarize(rows);summary['R_percent']=float(np.mean([x['R_percent'] for x in rows.values()]));summary['R_plus_E_percent']=float(np.mean([x['R_plus_E_percent'] for x in rows.values()]));done+=1;dump('reopt_progress.json',{'completed_configs':done,'base':tag,'time':time.time()});return {'config':cfg,'summary':summary,'per_image':rows}
 for tag in ['ResNet_control','B4_4fold']:
  cfg=copy.deepcopy(G['cfg']);current=evaluate(tag,cfg);allresults[tag]=[current]
  coordinates=[('base',k) for k in ['primary_gamma','al3ni_gamma','eutectic_gamma','al_gamma']]+[('gamma329',0),('gamma329',1)]
  for section,key in coordinates:
   incumbent=cfg['route'][section][key]
   trials=[current]
   for v in sorted(set([0.,float(incumbent),1.])):
    if v==incumbent:continue
    trial=copy.deepcopy(cfg);trial['route'][section][key]=v;result=evaluate(tag,trial);trials.append(result);allresults[tag].append(result);dump('reopt_partial.json',allresults)
   chosen=max(trials,key=lambda x:x['summary']['R_percent']);cfg=copy.deepcopy(chosen['config']);current=chosen
  best[tag]=current
 dump('reopt_result.json',{'best':best,'all_trials':allresults,'objective':'R valid20 image mIoU','method':'single coordinate sweep; levels 0/current/1; identical ordered budget for both bases','not_global_optimum':True,'HR_coefficients_postprocessing_E_F_fixed':True,'valid20_reused_for_selection_not_independent_generalization':True,'automatic_adoption':False})
 dump('reopt_completion.json',{'time':time.time(),'configs':done});print('REOPT_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump('reopt_failure.json',{'trace':traceback.format_exc()});raise
