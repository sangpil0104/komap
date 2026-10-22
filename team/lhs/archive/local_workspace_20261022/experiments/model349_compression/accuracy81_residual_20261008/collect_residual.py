"""Collect residual pipeline artifacts and rescore completed valid20 runs, verify archive hash, rescore PNGs independently."""
from pathlib import Path
import hashlib,importlib.util,json,shlex,subprocess,tarfile,time
import numpy as np
ROOT=Path(__file__).resolve().parent;REMOTE='/workspace/model349_compress/accuracy81_residual_20261008'
SSH=['ssh','-p','20670','root@168.188.128.36'];SCP=['scp','-P','20670'];stamp=str(time.time_ns())
code=r'''import pathlib,json,tarfile,hashlib
r=pathlib.Path(%r);tag=%r;files=[]
for p in r.glob('*.json'):files.append(p)
for p in r.glob('*.log'):files.append(p)
for p in r.glob('*.py'):files.append(p)
for p in r.glob('*.md'):files.append(p)
for directory in ['migration_snapshot_gpu7','deterministic_ce_snapshot','attempt1_failed_deterministic_ce']:
 for p in (r/directory).rglob('*'):
  if p.is_file():files.append(p)
for run in (r/'runs').glob('*'):
 if not (run/'completion.json').exists():continue
 if run.name=='teacher_train70':
  for key in ['input_manifest.json','completion.json','metrics.json']:
   p=run/key
   if p.exists():files.append(p)
 else:
  for p in run.rglob('*'):
   if p.is_file() and 'probabilities' not in p.relative_to(run).parts:files.append(p)
for key in ['recipe.json','completion.json','progress.json','epoch8.pt']:
 p=r/'training'/key
 if p.exists():files.append(p)
for split in ['train','valid']:
 p=r/'features'/split/'manifest.json'
 if p.exists():files.append(p)
for bundle in (r/'bundles').glob('*'):
 for key in ['infer_lite.py','checkpoint_manifest.json','accuracy_recipe.json','configs/model349_lite.yaml']:
  p=bundle/key
  if p.exists():files.append(p)
parent_sizes=r/'parent_checkpoint_sizes.json'
if (r/'parent_frozen.json').exists():
 frozen=json.loads((r/'parent_frozen.json').read_text())
 sizes={key:(r.parent/'Model349_exh14'/key).stat().st_size for key in frozen if key.startswith('weights/')}
 assert len(sizes)==14
 parent_sizes.write_text(json.dumps(sizes,indent=2)+'\n');files.append(parent_sizes)
archive=r/('collection_'+tag+'.tar.gz')
with tarfile.open(archive,'x:gz') as t:
 for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
print(json.dumps({'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(set(files))}))
'''%(REMOTE,stamp)
metadata=json.loads(subprocess.check_output(SSH+['python3 -c '+shlex.quote(code)],text=True))
archive=ROOT/Path(metadata['archive']).name
subprocess.run(SCP+['root@168.188.128.36:'+metadata['archive'],str(archive)],check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==metadata['sha256']
# Keep historical logs/state/code inside snapshot, never overwrite local running source.
snapshot=ROOT/'collected'/stamp;snapshot.mkdir(parents=True)
with tarfile.open(archive) as t:
 for m in t.getmembers():
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts,m.name
 t.extractall(snapshot,filter='data')
(ROOT/('collection_'+stamp+'.json')).write_text(json.dumps(metadata,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('audit',ROOT.parent/'audit_20261008/analyze_runs.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
gt=Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad/gt/valid/masks')
refpath=ROOT.parent/'exh14_base_search_20261008/exh14_valid20';ref=audit.score(refpath,gt)
report={'metric':'mean of per-image 4 class IoUs; empty-union=0','scope':'fresh parent raw valid20 plus supervised train70 residual correction; repeated-development evaluation, generalization unverified','base_percent':100*ref['miou'],'runs':{},'archive':metadata}
for run in sorted((snapshot/'runs').glob('*')):
 if not (run/'completion.json').exists() or run.name=='teacher_train70':continue
 b=audit.score(run,gt);pair=audit.pair(refpath,run,ref,b)
 report['runs'][run.name]={'miou_percent':100*b['miou'],'class_iou_percent':(np.array(b['classes'])*100).tolist(),'vs_base':pair,'completion':audit.load(run/'completion.json'),'independent_png_scoring_agrees':True,'independent_generalization_verified':False}
if 'refined_valid20' in report['runs']:
 cp=snapshot/'runs/teacher_valid20';lp=snapshot/'runs/refined_valid20';control=audit.score(cp,gt);refined=audit.score(lp,gt)
 report['vs_same_fresh_parent']=audit.pair(cp,lp,control,refined)
 train=audit.load(snapshot/'training/completion.json');assert train['epochs']==8 and train['optimizer_updates']==1120 and train['valid_read'] is False and train['test_read'] is False
 checkpoint=snapshot/'training/epoch8.pt';assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==train['final_sha256']
 assert report['runs']['refined_valid20']['completion']['zero_refiner_parent_masks_exact']
 fields=audit.load(snapshot/'features/train/manifest.json');assert fields['split']=='train' and len(fields['images'])==70 and fields['parent_masks_exact']
 lock=audit.load(snapshot/'preflight_source_lock.json');assert all(hashlib.sha256((snapshot/key).read_bytes()).hexdigest()==h for key,h in lock.items())
 complete=audit.load(snapshot/'completion.json');assert complete['parent_source_weights_unchanged'] and complete['sequence_finished']
 sizes=audit.load(snapshot/'parent_checkpoint_sizes.json');assert len(sizes)==14
 report['training_receipt']=train;report['residual_checkpoint_bytes']=checkpoint.stat().st_size;report['checkpoint_count']=15;report['total_checkpoint_bytes']=sum(sizes.values())+checkpoint.stat().st_size
 report['source_snapshot_matches_preflight']=True
audit.dump(ROOT/'residual_comparison.json',report)
lines=['# 작은 잔차 보정 모델 원본 valid20 검증','','빈 union=0인 이미지별4클래스평균 후20장평균. PNG/GT 독립재채점과실행metrics일치 확인. 동일valid반복개발평가이며 독립일반화 증거가 아니다.','','| 구성 | image mIoU | base 대비 | 최악이미지손실 | 단회시간 |','|---|---:|---:|---:|---:|',f'| base14 | {100*ref["miou"]:.7f}% | — | — | 별도환경단회 |']
for key,v in report['runs'].items():
 p=v['vs_base'];timing=f'{v["completion"]["elapsed_seconds"]:.2f}s'
 if v['completion'].get('resumed'):timing+=' (재개 구간만; 전체시간 아님)'
 lines.append(f'| {key} | {v["miou_percent"]:.7f}% | {p["delta_pp"]:+.7f}%p | {p["worst_image_delta_pp"]:+.7f}%p ({p["worst_image"]}) | {timing} |')
lines+=['','원래14가중치는보존하고11,876parameter 잔차모델1개를추가한다. checkpoint bytes는training receipt를검증해별도기록한다. refined elapsed_seconds는전체fresh부모valid+feature추출+잔차평가이며학습시간을제외한다. 시간은단회이고속도개선근거가아니다. train70 teacher 자료는부모가학습한이미지여서held-out/OOF가아니다. 같은valid반복개발평가/독립일반화미확인. 기존릴리스/원본보존/test mIoU 없음.','']
(ROOT/'residual_report.md').write_text('\n'.join(lines));print(json.dumps({k:{'miou':v['miou_percent'],'delta':v['vs_base']['delta_pp'],'worst':v['vs_base']['worst_image_delta_pp']} for k,v in report['runs'].items()},indent=2))
