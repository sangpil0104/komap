"""Collect only a completed stage1 fresh run, verify source/parents, score PNGs."""
from pathlib import Path
import subprocess,json,hashlib,tarfile,time,importlib.util
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent
SSH=['ssh','-p','20670','root@168.188.128.36']
SERVER=Path('/workspace/model349_compress')/HERE.name
GT=Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad/gt/valid/masks')
stamp=str(time.time_ns())
script="""
from pathlib import Path
import json,hashlib,tarfile
root=Path(ROOT_VALUE)
assert (root/'completion.json').is_file(), 'Run is not complete; do not collect partial results'
assert not (root/'failure.json').exists()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
completion=json.loads((root/'completion.json').read_text())
assert completion['finished'] and completion['whole_fresh_pipeline'] and not completion['test_gt_read']
lock=json.loads((root/'source_lock.json').read_text())
assert all(sha(root/p)==h for p,h in lock.items())
frozen=json.loads((root/'parent_frozen.json').read_text())
assert all(sha(Path(p))==h for p,h in frozen.items())
for split,n in [('valid',20),('test',10)]:
 for kind in ['release','stage1']:
  run=root/'runs'/f'{kind}_{split}'
  assert len(list((run/'predicted_labels').glob('*.png')))==n
  assert (run/'completion.json').is_file() and (run/'input_manifest.json').is_file()
archive=root/ARCHIVE_VALUE
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(root.rglob('*')):
  rel=p.relative_to(root)
  if not p.is_file() or p.is_symlink():continue
  if any(x in rel.parts for x in ['assets','Model349_exh14','model397_reproduction_bundle','__pycache__']):continue
  if p.suffix in ['.npy','.npz','.pt','.tar','.gz'] or p==archive:continue
  t.add(p,arcname=str(Path(root.name)/rel))
print(json.dumps({'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'source_lock':lock,'parents_unchanged':True}))
""".replace('ROOT_VALUE',repr(str(SERVER))).replace('ARCHIVE_VALUE',repr(f'collection_{stamp}.tar.gz'))
r=subprocess.run([*SSH,'/workspace/venv-komap/bin/python -'],input=script,text=True,capture_output=True,check=True)
receipt=json.loads(r.stdout)
assert receipt['source_lock']==json.loads((HERE/'source_lock.json').read_text())
archive=HERE/f'collection_{stamp}.tar.gz'
subprocess.run(['scp','-P','20670',f"root@168.188.128.36:{receipt['archive']}",str(archive)],check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
dest=HERE/'collected'/stamp;dest.mkdir(parents=True)
with tarfile.open(archive) as t:
 for m in t.getmembers():
  assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not m.issym() and not m.islnk()
  assert m.isfile() or m.isdir()
 t.extractall(dest,filter='data')
run=dest/HERE.name
# Verify submission RGB masks as well as label PNGs, including native size.
for split,count in [('valid',20),('test',10)]:
 for kind in ['release','stage1']:
  folder=run/'runs'/f'{kind}_{split}'
  manifest=json.loads((folder/'input_manifest.json').read_text())
  palette=np.asarray(manifest['config']['palette'],dtype=np.uint8)
  assert len(manifest['images'])==count
  assert len(list((folder/'predicted_masks').glob('*.png')))==count
  for row in manifest['images']:
   labels=np.asarray(Image.open(folder/f"predicted_labels/{row['name']}.png"))
   rgb=np.asarray(Image.open(folder/f"predicted_masks/{row['name']}_mask.png").convert('RGB'))
   assert labels.shape==tuple(reversed(row['size'])) and np.isin(labels,range(4)).all()
   assert np.array_equal(rgb,palette[labels]),row['name']
spec=importlib.util.spec_from_file_location('audit',HERE.parent/'audit_20261008/analyze_runs.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
freshR=run/'runs/release_valid';candidate=run/'runs/stage1_valid'
canonical=HERE.parent/'exh14_base_search_20261008/exh14_valid20'
old=HERE.parent/'accuracy81_candidate_oracle_20261009/fusion_candidate_valid20'
scores={name:audit.score(p,GT) for name,p in [('fresh_R',freshR),('candidate',candidate),('canonical_R',canonical),('cached_candidate',old)]}
def agreement(ref,other):
 a=json.loads((ref/'input_manifest.json').read_text());b=json.loads((other/'input_manifest.json').read_text())
 assert {r['name']:r['sha256'] for r in a['images']}=={r['name']:r['sha256'] for r in b['images']}
 rows=[];changed=total=0
 for r in a['images']:
  n=r['name'];x=np.asarray(Image.open(ref/f'predicted_labels/{n}.png'));y=np.asarray(Image.open(other/f'predicted_labels/{n}.png'))
  assert x.shape==y.shape and np.isin(x,range(4)).all() and np.isin(y,range(4)).all()
  d=int(np.count_nonzero(x!=y));changed+=d;total+=x.size
  rows.append({'image':n,'changed_pixels':d,'pixels':int(x.size),'agreement':1-d/x.size,'reference_class_fraction':[(x==c).mean().item() for c in range(4)],'candidate_class_fraction':[(y==c).mean().item() for c in range(4)]})
 return {'changed_pixels':changed,'pixels':total,'agreement':1-changed/total,'images':rows,'input_sha_exact':True}
report={'whole_fresh_pipeline':True,'candidate_percent':scores['candidate']['miou']*100,'fresh_R_percent':scores['fresh_R']['miou']*100,
 'vs_fresh_R':audit.pair(freshR,candidate,scores['fresh_R'],scores['candidate']),
 'vs_canonical_R':audit.pair(canonical,candidate,scores['canonical_R'],scores['candidate']),
 'vs_cached_candidate':audit.pair(old,candidate,scores['cached_candidate'],scores['candidate']),
 'test_vs_fresh_R':agreement(run/'runs/release_test',run/'runs/stage1_test'),
 'test_vs_canonical_R':agreement(HERE.parent/'exh14_base_search_20261008/exh14_test10',run/'runs/stage1_test'),
 'independent_png_gt_scoring':True,'submission_RGB_palette_and_native_size_verified':True,'parent_source_weights_unchanged':receipt['parents_unchanged'],
 'independent_generalization_verified':False,'test_miou_available':False,'base_changed':False,'collected':str(run)}
(HERE/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(HERE/'collection_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['candidate_percent','fresh_R_percent','test_miou_available']},indent=2))

text = f"""# 1단계 R + E 원본 영상 재추론 결과

원본 영상부터 R의 릴리스 14가중치 및 E의 B4 4-fold/HRNet 2-seed × 4-fold를 다시 계산했다. F와 확신도 제한은 사용하지 않았다. 원본 소스·가중치 SHA는 불변이며, archive SHA/안전 추출 및 PNG/GT 20장 80개 IoU 독립 채점을 통과했다.

- 새 1단계 valid20: **{report['candidate_percent']:.10f}%**
- 같은 실행의 R: {report['fresh_R_percent']:.10f}%
- 같은 실행 R 대비: {report['vs_fresh_R']['delta_pp']:+.10f}%p
- 기존 canonical R 대비: {report['vs_canonical_R']['delta_pp']:+.10f}%p
- 저장 부모 출력 기반 후보 대비: {report['vs_cached_candidate']['delta_pp']:+.10f}%p
- 최악 영상: {report['vs_canonical_R']['worst_image']}, {report['vs_canonical_R']['worst_image_delta_pp']:+.10f}%p
- 기존 영상별 손실 기준 통과: {report['vs_canonical_R']['quality_pass']}

test10은 새 R 대비 {report['test_vs_fresh_R']['changed_pixels']:,} / {report['test_vs_fresh_R']['pixels']:,} 픽셀이 달라, 일치율 {report['test_vs_fresh_R']['agreement']*100:.8f}%다. 기존 canonical exh14 대비 일치율은 {report['test_vs_canonical_R']['agreement']*100:.8f}%다. 영상별 변화 및 클래스 면적 비율은 comparison.json에 기록했다. test에는 정답이 없어 mIoU 또는 성능 증감을 주장하지 않는다.

1단계 시편 제외 80.4088%는 69개 실행 × 30개 규칙 전체를 매번 다시 선택한 값이다. 후보와 R 자체의 과거 valid 기반 개발 선택 편향까지 제거하지는 못한다. F는 조건부 CV에서 1단계 대비 손해로 미채택이다. 기존 base는 유지하며, 81% 목표 달성 여부는 별도 전체 요구사항 검증이 필요하다.
"""
(HERE/'report.md').write_text(text)
