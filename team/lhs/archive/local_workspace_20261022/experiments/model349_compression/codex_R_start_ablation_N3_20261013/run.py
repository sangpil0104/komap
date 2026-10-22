import os,json,time,hashlib,importlib.util,traceback,re
from pathlib import Path
import numpy as np
O=Path('/workspace/model349_compress/codex_R_start_ablation_N3_20261013');S=Path('/workspace/model349_compress/codex_lite16N3_allmodels_20261013')
spec=importlib.util.spec_from_file_location('screen',S/'review.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2))
def main():
 if (O/'completion.json').exists():return
 m.init(False);G=m.audit.GLOBAL
 cover=json.load(open('/workspace/model349_compress/codex_lite16_allmodel_complement_20261013/coverage.json'));c=next(x for x in cover['candidates'] if x['id']==146)
 data={};hashes={};baseR={};baseS={};single={}
 for n,rec in G['records'].items():
  baseR[n]=m.audit.iou(rec['r'],rec['gt']);baseS[n]=m.audit.iou(np.where(np.isin(rec['ex'],[0,2]),rec['ex'],rec['r']),rec['gt'])
  vals={'ResNet_highres_control':rec['item']['preaux'],'B4_gray_1fold':rec['sources']['322'],'B4_ZCDF_2fold':rec['sources']['326'],'B4_width4000_3fold':rec['sources']['329'],'B4_width4000_4fold_report':m.audit.read(c,n),'HRNet_W48':rec['sources']['347'],'ResNet_native':rec['sources']['269']}
  for tag,p in vals.items():
   hashes.setdefault(tag,hashlib.sha256()).update(p.tobytes());single.setdefault(tag,{})[n]=100*m.audit.iou(G['fu'].labels(p),rec['gt'])[1]
   item=dict(rec['item']);item['preaux']=p;rr=G['infer'].predict_fixed(item,rec['sources'],G['cfg']);pred=m.finish(rec,rec['ex'],rec['fl'],rr)
   if tag=='ResNet_highres_control':assert np.array_equal(rr,rec['r']) and np.array_equal(pred,rec['base'])
   v=m.score(rec,pred);v['R_percent']=100*m.audit.iou(rr,rec['gt'])[1];v['R_plus_E_percent']=100*m.audit.iou(np.where(np.isin(rec['ex'],[0,2]),rec['ex'],rr),rec['gt'])[1]
   data.setdefault(tag,{})[n]=v
  dump('progress.json',{'images':len(baseR),'total':20,'pid':os.getpid()});print('IMAGE',len(baseR),n,flush=True)
 summary={}
 names=list(baseR);groups={re.match(r'A\d+',n).group() for n in names};groups=sorted(groups);rng=np.random.default_rng(819349);sample=rng.integers(0,len(groups),(10000,len(groups)))
 for tag,rows in data.items():
  v=m.summarize(rows);v['R_percent']=float(np.mean([r['R_percent'] for r in rows.values()]));v['R_plus_E_percent']=float(np.mean([r['R_plus_E_percent'] for r in rows.values()]));v['starting_distribution_single_percent']=float(np.mean(list(single[tag].values())))
  sums=np.array([sum(rows[n]['delta_pp'] for n in names if re.match(r'A\d+',n).group()==g) for g in groups]);counts=np.array([sum(re.match(r'A\d+',n).group()==g for n in names) for g in groups]);boot=sums[sample].sum(1)/counts[sample].sum(1);v['specimen_bootstrap95_delta_pp']=np.quantile(boot,[.025,.975]).tolist();summary[tag]=v
 dump('result.json',{'baseline_percent':m.BASE,'summary':summary,'per_image':data,'probability_sha256':{k:v.hexdigest() for k,v in hashes.items()},'report_fourfold_cache':c,'fixed_coefficients':True,'E_F_fixed':True,'TENT_not_applied_to_alternative_base':True,'cached_probability_ablation_not_fresh_GPU':True,'development_valid20':True,'specimen_groups':groups,'bootstrap_replicates':10000,'release_replaced':False})
 dump('completion.json',{'time':time.time(),'variants':len(summary),'images':20,'control_masks_exact':True});print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump('failure.json',{'trace':traceback.format_exc()});raise
