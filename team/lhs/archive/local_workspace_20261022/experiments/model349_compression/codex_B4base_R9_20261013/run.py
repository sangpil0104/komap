import os,json,time,hashlib,traceback,importlib.util
from pathlib import Path
import numpy as np
O=Path('/workspace/model349_compress/codex_B4base_R9_20261013');PREV=Path('/workspace/model349_compress/codex_B4base_REF_design_20261013')
s=importlib.util.spec_from_file_location('design',PREV/'run.py');d=importlib.util.module_from_spec(s);s.loader.exec_module(d);m=d.m

def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=1))
def main():
 if (O/'completion.json').exists():return
 m.init(False);G=m.audit.GLOBAL;cover=json.load(open('/workspace/model349_compress/codex_lite16_allmodel_complement_20261013/coverage.json'));c=next(x for x in cover['candidates'] if x['id']==146)
 for n,r in G['records'].items():
  r['newbase']=m.audit.read(c,n);r['donors']={'B4gray':r['sources']['322'],'B4ZCDF':r['sources']['326'],'HRNet':r['sources']['347'],'ResNet_native':r['sources']['269'],'ResNet_highres':r['item']['preaux'],'HRNet_ZCDF_train70':r['parts']['N3']}
 seed=json.load(open(PREV/'selected_R.json'))['selected'];seen={};done=0
 def predict(r,steps):
  p=r['newbase']
  for st in steps:p=d.blend(p,r['donors'][st['donor']],st['alpha'],st['classes'])
  return p
 def evaluate(steps):
  nonlocal done
  key=json.dumps(steps,sort_keys=True)
  if key in seen:return seen[key]
  rows={};members=set(d.MEMBERS['base'])
  for st in steps:members|=d.MEMBERS[st['donor']]
  for n,r in G['records'].items():rows[n]=m.score(r,G['fu'].labels(predict(r,steps)))
  v={'steps':steps,'summary':m.summarize(rows),'checkpoint_count':len(members),'members':sorted(members),'per_image':rows};seen[key]=v;done+=1
  dump('progress.json',{'completed_configs':done,'last_checkpoint_count':len(members),'time':time.time(),'pid':os.getpid()});return v
 control=evaluate(seed['steps']);assert abs(control['summary']['percent']-79.44550467674868)<1e-9
 pools={6:[control],7:[],8:[],9:[]};beams={}
 for count in [6,7,8]:
  choices=sorted(pools[count],key=lambda x:-x['summary']['percent'])[:2];beams[count]=choices
  for node in choices:
   used={x['donor'] for x in node['steps']}
   for donor,ms in d.MEMBERS.items():
    if donor=='base' or donor in used or count+len(ms)>9:continue
    for alpha in [.1,.25,.5]:
     for cs in [[0],[1],[2],[3],[0,1,2,3]]:
      result=evaluate(node['steps']+[{'donor':donor,'alpha':alpha,'classes':cs}]);pools[result['checkpoint_count']].append(result)
  dump('all_results.json',list(seen.values()));dump('best_by_count.json',{str(k):max(v,key=lambda x:x['summary']['percent']) for k,v in pools.items() if v})
 best=max(pools[9],key=lambda x:x['summary']['percent']);paired={}
 import model335_exhaustive_b4_route_search as route
 for n,r in G['records'].items():
  p=predict(r,best['steps']);raw=G['fu'].labels(p);post=route.predict(r['item'],p,G['cfg']['route']['base'],route.bilateral_weights(r['item']['z']),exact=True)
  for tag,rr in [('R9_raw',raw),('R9_existing_postprocessing',post)]:
   v=m.score(r,rr);v['R_percent']=v['miou']*100;fp=m.finish(r,r['ex'],r['fl'],rr);v['fixed_existing_E_F_percent']=m.audit.iou(fp,r['gt'])[1]*100;paired.setdefault(tag,{})[n]=v
 summary={}
 for tag,rows in paired.items():
  v=m.summarize(rows);v['delta_vs_existing_R_pp']=v['percent']-79.79930889538244;v['fixed_existing_E_F_percent']=float(np.mean([x['fixed_existing_E_F_percent'] for x in rows.values()]));summary[tag]=v
 dump('paired_postprocessing.json',{'summary':summary,'per_image':paired,'postprocessing_keeps_historical_valid_selected_rules':True})
 dump('completion.json',{'time':time.time(),'configs':done,'control_R6_exact':True,'best_R9':{k:v for k,v in best.items() if k!='per_image'},'paired_summary':summary,'beam_width_per_count':2,'selection_uses_valid20':True,'global_optimum_claimed':False,'release_replaced':False});print('COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',{'trace':traceback.format_exc()});raise
