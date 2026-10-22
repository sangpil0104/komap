"""Post-completion independent PNG scoring, repeated timing and test mask audit."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import hashlib,importlib.util,json,statistics,tarfile
from pathlib import Path
ROOT=Path('/workspace/model349_compress/exh14_base_search_20261008');GT=Path('/workspace/komap_model1/data/valid/masks')
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 done=read(ROOT/'completion.json');assert done['base_and_weights_unchanged'];score=module('scores',ROOT.parent/'audit_20261008/analyze_runs.py');agreement=module('agreement',ROOT.parent/'base12_followup_20261008/finalize.py').agreement
 paths={'base14':ROOT/'runs/control14_valid20','base12':ROOT.parent/'audit_20261008/runs/cpu2_fp16_12','lite15':ROOT.parent/'audit_20261008/runs/cpu2_lite15','original349':ROOT.parent/'audit_20261008/runs/historical349'};refs={k:score.score(p,GT) for k,p in paths.items()};runs={}
 for p in sorted((ROOT/'runs').iterdir()):
  if not (p/'metrics.json').exists():continue
  r=score.score(p,GT);runs[p.name]={'miou_percent':100*r['miou'],'class_iou_percent':[100*v for v in r['classes']],'completion':read(p/'completion.json'),**{'vs_'+k:score.pair(paths[k],p,a,r) for k,a in refs.items()}}
 frames=[json.loads(x) for x in (ROOT/'monitor.jsonl').read_text().splitlines()];uuid=next(line.split(',')[1].strip() for line in frames[0]['gpu'].splitlines() if line.startswith('7,'));interference=[f['time'] for f in frames if sum(uuid in line for line in f['processes'].splitlines())>1];timing=None;selected=done['time_candidate'];tests={}
 if selected:
  rows=read(ROOT/'timing_runs.json');times={k:[r['completion']['elapsed_seconds'] for r in rows if r['candidate']==k] for k in ['control14',selected]};assert all(len(v)==3 for v in times.values());med={k:statistics.median(v) for k,v in times.items()};reduction=100*(1-med[selected]/med['control14']);quality=all(runs[r['run']]['vs_base14']['quality_pass'] for r in rows);timing={'seconds':times,'median_seconds':med,'median_reduction_percent':reduction,'all_repeats_quality_pass':quality,'ten_percent_gate_pass':quality and reduction>=10 and not interference}
  for k in done['test_candidates']:
   p=ROOT/'runs'/(k+'_test10');assert read(p/'completion.json')['mean_image_miou'] is None;tests[k]={'vs_base14':agreement(ROOT.parent/'test_compare_20261008/exh14_test10',p),'vs_base12':agreement(ROOT.parent/'generalization_20261008/runs/cpu2_fp16_12_test10',p),'completion':read(p/'completion.json')}
 result={'runs':runs,'timings':timing,'tests':tests,'checkpoint_manifests':{p.parent.name:read(p) for p in (ROOT/'bundles').glob('*/checkpoint_manifest.json')},'gpu7_interference_frames':interference,'png_gt_rescoring_verified':True,'original_base14_files_unchanged':True,'independent_generalization_verified':False};write(ROOT/'comparison.json',result)
 lines=['# exh14 base 후속 실행 결과','','사용자 지정 base14 대비 고정 조건으로 검증했다. valid 재사용이며 일반화 향상이 아니다. base12/lite15/original349 비교는 보조 진단이다.','','| 실행 | valid image mIoU | base14 평균 변화 | 최악 이미지 변화 | 품질 |','|---|---:|---:|---:|---|']
 for key,r in runs.items():
  b=r['vs_base14'];lines.append(f"| {key} | {r['miou_percent']:.7f}% | {b['delta_pp']:+.7f}%p | {b['worst_image_delta_pp']:+.7f}%p ({b['worst_image']}) | {b['quality_pass']} |")
 lines+=['','공식 metric은 이미지별4IoU 평균 후20장 평균, empty union=0. 모든 클래스·이미지 손실은comparison.json. FP32→FP16 저장은 반올림을 포함하고 ZIP은 그FP16 저장판에 대해 무손실이다. 로드FP32이므로 FP16 저장 자체는 연산/VRAM 절감이 아니다. fast14에는 별도 HRNet FP16autocast/배치8이 포함된다.','',f'반복 시간: {timing}',f'GPU7 외부 compute 감지 frames: {len(interference)}. 공유 서버 측정이며 성능/속도의 다른 환경 보장은 없다.','','## test10','']
 for key,r in tests.items():
  for ref in ['base14','base12']:
   x=r['vs_'+ref];lines.append(f"- {key} / {ref}: {100*x['agreement']:.7f}% 일치, {x['changed_pixels']}/{x['pixels']}픽셀 변화. test정답/IoU 없음.")
 lines+=['','기존 base/릴리스/가중치를 교체하지 않는다. 이 결과로 자동 모델 채택이나 일반화 보장을 하지 않는다.',''];(ROOT/'report.md').write_text('\n'.join(lines))
 files=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix not in ['.pt','.pyc'] and 'probabilities' not in p.parts and '__pycache__' not in p.parts and p.name not in ['gpu7.lock','results_archive.tar.gz','archive_receipt.json']]
 with tarfile.open(ROOT/'results_archive.tar.gz','w:gz') as f:
  for p in sorted(files):f.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
 p=ROOT/'results_archive.tar.gz';write(ROOT/'archive_receipt.json',{'files':len(files),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});print(json.dumps({'timings':timing,'selected':selected},indent=2))
if __name__=='__main__':main()
