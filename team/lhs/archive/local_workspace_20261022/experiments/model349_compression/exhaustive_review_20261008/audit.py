"""Read-only audit of the supplied cached search; not a raw-image inference."""
import hashlib,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; SRC=ROOT.parent/'exhaustive_20261008'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def weight_count(c):
 w=set()
 for tag,m in zip(['322','326','329'],c[:3]):w.update(f'{tag}f{i+1}' for i in range(4) if m>>i&1)
 for tag,on in zip(['336','347','269'],c[3:6]):
  if on:w.add(tag)
 for i,tag in enumerate(['97','98','109']):
  if c[6]>>i&1:w.add(tag)
 if c[7]:w.add('pairwise')
 w.add('restart51')
 for i,tag in enumerate(['109','seed456','pairwise','ema']):
  if c[8]>>i&1:w.add(tag)
 return len(w)
datasets={k:np.load(SRC/f'stage{k}.npz') for k in 'ABC'}
names=list(map(str,datasets['A']['names'])); rows={};duplicates=[]
for stage,d in datasets.items():
 assert list(map(str,d['names']))==names and d['iou'].shape==(len(d['combos']),20,4)
 for c,v in zip(d['combos'],d['iou']):
  c=tuple(map(int,c));v=v.astype(float)
  if c in rows:duplicates.append({'combo':c,'first':rows[c][1],'second':stage,'max_class_delta_pp':float(abs(rows[c][0]-v).max()*100),'mean_delta_pp':float((v-rows[c][0]).mean()*100)})
  else:rows[c]=(v,stage)
combos=list(rows); arr=np.stack([rows[c][0] for c in combos]);img=arr.mean(2)*100;mean=img.mean(1);idx={c:i for i,c in enumerate(combos)}
base=(0,5,15,1,0,1,0,0,7);lite=(0,15,15,1,0,1,0,0,7);best=int(np.argmax(mean));nw=np.array([weight_count(c) for c in combos]);groups=np.array([re.match(r'A\d+',n)[0] for n in names]); unique=sorted(set(groups))
def compare(i,r):
 d=img[i]-img[r];cl=(arr[i]-arr[r])*100
 return {'mean_delta_pp':float(d.mean()),'class_delta_pp':cl.mean(0).tolist(),'negative_images':int((d<0).sum()),'worst_image':names[int(d.argmin())],'worst_delta_pp':float(d.min()),'per_image':{n:{'delta_pp':float(x),'class_delta_pp':y.tolist()} for n,x,y in zip(names,d,cl)}}
def loso(cands):
 held=np.zeros(20);picks={}
 for g in unique:
  tr=groups!=g;te=~tr;i=int(cands[np.argmax(img[cands][:,tr].mean(1))]);held[te]=img[i,te];picks[g]=combos[i]
 delta=held-img[idx[lite]];rng=np.random.default_rng(0);b=[]
 for _ in range(4000):
  chosen=rng.choice(unique,len(unique));inds=np.concatenate([np.where(groups==g)[0] for g in chosen]);b.append(delta[inds].mean())
 return {'delta_pp':float(delta.mean()),'cluster_bootstrap95_pp':np.percentile(b,[2.5,97.5]).tolist(),'picks':picks,'strict_nested':False}
# Determine how many training-only top50 x top40 candidates the globally screened C missed.
leak=[];cset={tuple(map(int,c)) for c in datasets['C']['combos']}
for g in unique:
 tr=groups!=g;a=datasets['A'];b=datasets['B'];at=a['combos'][np.argsort(-a['iou'][:,tr].astype(float).mean((1,2)))[:50]];bt=b['combos'][np.argsort(-b['iou'][:,tr].astype(float).mean((1,2)))[:40]]
 needed={tuple(map(int,x[:3]))+tuple(map(int,y[3:])) for x in at for y in bt}
 leak.append({'held_specimen':g,'training_only_crosses':len(needed),'missing_from_C':len(needed-cset),'missing_from_union':len(needed-set(rows))})
# Within-stage exact matched toggle differences, not pooled selected-candidate correlations.
matched={}
for label,stage,field,off,on in [('HR336','B',3,0,1),('HR347','B',4,0,1),('269','B',5,0,1),('pair_flag','B',7,0,1),('326_all4','A',1,0,15),('322_all4','A',0,0,15),('329_all4','A',2,0,15)]:
 d=datasets[stage];lookup={tuple(map(int,c)):v.astype(float).mean(1)*100 for c,v in zip(d['combos'],d['iou'])};diff=[]
 for c,v in lookup.items():
  if c[field]!=off:continue
  cc=list(c);cc[field]=on
  if tuple(cc) in lookup:diff.append(lookup[tuple(cc)]-v)
 vals=np.array(diff);matched[label]={'stage':stage,'matched_pairs':len(vals),'mean_effect_pp':float(vals.mean()),'context_effect_min_pp':float(vals.mean(1).min()),'context_effect_max_pp':float(vals.mean(1).max())}
result={'input_hashes':{p.name:sha(p) for p in SRC.iterdir() if p.is_file()},'rows_by_stage':{k:len(v['combos']) for k,v in datasets.items()},'unique_combos':len(combos),'duplicate_count':len(duplicates),'duplicate_max_class_delta_pp':max(x['max_class_delta_pp'] for x in duplicates),'duplicates':duplicates,'class_order':['Al3Ni','Primary Si','Al','Eutectic Si'],'best':{'combo':combos[best],'checkpoint_count':int(nw[best]),'miou_percent':float(mean[best]),'class_iou_percent':(arr[best].mean(0)*100).tolist(),'vs_base12':compare(best,idx[base]),'vs_lite14':compare(best,idx[lite])},'baseline_scores':{'base12_cached_percent':float(mean[idx[base]]),'lite14_cached_percent':float(mean[idx[lite]])},'loso_leaky_all':loso(np.arange(len(combos))),'loso_AB_fixed_grid':loso(np.array([i for i,c in enumerate(combos) if rows[c][1] in ['A','B']])),'loso_weight_budgets':{str(k):loso(np.where(nw<=k)[0]) for k in [6,8,10,12,14]},'training_only_shortlist_coverage':leak,'matched_component_effects':matched}
(ROOT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
assert result['input_hashes']=={p.name:sha(p) for p in SRC.iterdir() if p.is_file()}
print(json.dumps({k:result[k] for k in ['unique_combos','duplicate_count','duplicate_max_class_delta_pp','baseline_scores','matched_component_effects']},indent=2))
print('BEST',result['best']['miou_percent'],result['best']['vs_base12']['mean_delta_pp'],result['best']['vs_base12']['worst_delta_pp'])
print('SHORTLIST missing range',min(r['missing_from_C'] for r in leak),max(r['missing_from_C'] for r in leak))
