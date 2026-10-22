"""Collect completed component-replay runs only, verify archive hash, rescore PNGs independently."""
from pathlib import Path
import hashlib,importlib.util,json,shlex,subprocess,tarfile,time
import numpy as np
ROOT=Path(__file__).resolve().parent;REMOTE='/workspace/model349_compress/accuracy81_hrscale_20261008'
SSH=['ssh','-p','20670','root@168.188.128.36'];SCP=['scp','-P','20670'];stamp=str(time.time_ns())
code='''import pathlib,json,tarfile,hashlib
r=pathlib.Path(%r);tag=%r;files=[]
for p in r.glob('*.json'):files.append(p)
for p in r.glob('*.log'):files.append(p)
for p in r.glob('*.py'):files.append(p)
for p in r.glob('*.md'):files.append(p)
for run in (r/'runs').glob('*'):
 if not (run/'completion.json').exists():continue
 for p in run.rglob('*'):
  if p.is_file() and 'probabilities' not in p.relative_to(run).parts:files.append(p)
for bundle in [r/'bundle']:
 for key in ['infer_lite.py','checkpoint_manifest.json','accuracy_recipe.json','configs/model349_lite.yaml']:
  p=bundle/key
  if p.exists():files.append(p)
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
report={'metric':'mean of per-image 4 class IoUs; empty-union=0','scope':'raw HR336 component inference with immutable original branch replay; not whole fresh inference or independent generalization','base_percent':100*ref['miou'],'runs':{},'archive':metadata}
for run in sorted((snapshot/'runs').glob('*')):
 if not (run/'completion.json').exists():continue
 b=audit.score(run,gt);pair=audit.pair(refpath,run,ref,b)
 report['runs'][run.name]={'miou_percent':100*b['miou'],'class_iou_percent':(np.array(b['classes'])*100).tolist(),'vs_base':pair,'completion':audit.load(run/'completion.json'),'independent_png_scoring_agrees':True,'independent_generalization_verified':False}
audit.dump(ROOT/'scale_comparison.json',report)
lines=['# HR336 배율 component replay 검증','','빈 union=0인 이미지별4클래스평균 후20장평균. PNG/GT 독립재채점과실행metrics일치 확인. 동일valid반복개발평가이며 독립일반화 증거가 아니다.','','| 구성 | image mIoU | base 대비 | 최악이미지손실 | 측정 구간 시간 |','|---|---:|---:|---:|---:|',f'| base14 | {100*ref["miou"]:.7f}% | — | — | 별도환경단회 |']
for key,v in report['runs'].items():
 p=v['vs_base'];timing=f'{v["completion"]["elapsed_seconds"]:.2f}s ('+v['completion']['elapsed_kind']+')'
 if v['completion'].get('resumed'):timing+=' (재개 구간만; 전체시간 아님)'
 lines.append(f'| {key} | {v["miou_percent"]:.7f}% | {p["delta_pp"]:+.7f}%p | {p["worst_image_delta_pp"]:+.7f}%p ({p["worst_image"]}) | {timing} |')
lines+=['','가중치14개/1,755,428,956바이트 유지. HR336만 새 입력 배율로 원본 재추론하고 변경 없는6분기의SHA검증된확률을재사용했다. 전체fresh재추론/독립일반화성적이 아니다. 시간은 신규component 또는CPU평균구간만이며 전체모델시간/속도개선근거가 아니다. 기존릴리스/원본가중치보존. test mIoU 없음.','']
(ROOT/'scale_report.md').write_text('\n'.join(lines));print(json.dumps({k:{'miou':v['miou_percent'],'delta':v['vs_base']['delta_pp'],'worst':v['vs_base']['worst_image_delta_pp']} for k,v in report['runs'].items()},indent=2))
