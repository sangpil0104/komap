import json,csv,statistics,hashlib
from pathlib import Path
from collections import defaultdict
R=Path('/tmp/komap_team_seed_review_20261012');O=Path(__file__).parent
candidates=json.loads((R/'metadata/all_candidates.json').read_text());groups=defaultdict(list)
for c in candidates:
 assert len(c['images'])==c['image_count']==20
 assert max(abs(sum(x['class_iou'])/4-x['image_miou']) for x in c['images'])<1e-12
 assert abs(statistics.mean(x['image_miou'] for x in c['images'])-c['image_miou'])<1e-12
 key=c['family']+((' / '+c['hr336_parent']['task_id']) if c['family']=='HRNet347' else '')
 groups[key].append(c)
rows=[];detail={}
for group,cs in groups.items():
 xs=[100*c['image_miou'] for c in cs];mean=statistics.mean(xs)
 row={'family_context':group,'n_candidates':len(cs),'mean_percent':mean,'sample_sd_pp':statistics.stdev(xs),'min_percent':min(xs),'max_percent':max(xs),'range_pp':max(xs)-min(xs),'mean_delta_baseline_pp':statistics.mean(c['delta_baseline_percentage_points'] for c in cs),'best_id':max(cs,key=lambda c:c['image_miou'])['id'],'worst_id':min(cs,key=lambda c:c['image_miou'])['id']};rows.append(row)
 images=defaultdict(list)
 for c in cs:
  for x in c['images']:images[x['image']].append(x['image_miou']*100)
 detail[group]={'rows':[{'id':c['id'],'score_percent':c['image_miou']*100,'training_seeds':[x['training_seed'] for x in c['checkpoints']]} for c in cs],'largest_image_seed_ranges':sorted([{'image':n,'range_pp':max(v)-min(v)} for n,v in images.items()],key=lambda d:d['range_pp'],reverse=True)[:5]}
base=candidates[0]['baseline_image_miou']*100
plan=json.loads((R/'metadata/original_plan.json').read_text())
result={'repository':'https://github.com/CNUKoMaP/giveItToHyeonSeok','commit':'b59fc743d6aca93acf1234a9ce3819076afa7551','baseline_percent':base,'scope':'recorded full R+E+F scores; weights not downloaded or independently inferred; 28 candidate metadata means verified','provided_candidates':len(candidates),'provided_checkpoints':sum(len(c['checkpoints']) for c in candidates),'planned_training_counts':plan['group_counts'],'sd_policy':'sample standard deviation ddof1; units percentage points; HR347 grouped by HR336 parent','statistics':rows,'details':detail}
(O/'seed_statistics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
with (O/'seed_statistics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for row in rows:print(row['family_context'],f"{row['mean_percent']:.6f} ± {row['sample_sd_pp']:.6f}",f"range {row['range_pp']:.6f}")
for group,d in detail.items():print(group,d['largest_image_seed_ranges'][:2])
