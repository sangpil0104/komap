"""Collect only a completed P100-weight exh14 transfer and score its PNGs."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shlex
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parent
REMOTE = '/workspace/model349_compress/' + ROOT.name
SSH = ['ssh', '-o', 'BatchMode=yes', '-p', '20670', 'root@168.188.128.36']
STAMP = str(time.time_ns())

remote_code = r'''
from pathlib import Path
import hashlib,json,tarfile
r=Path(%r)
d=json.loads((r/'completion.json').read_text())
assert d['finished'] and d['fresh_valid20'] and d['weights']==14
manifest=json.loads((r/'bundle/checkpoint_manifest.json').read_text())
assert manifest['count']==14 and len(manifest['checkpoints'])==14
for row in manifest['checkpoints']:
 p=Path(row['source_path'])
 assert p.is_file() and p.stat().st_size==row['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
pre=json.loads((r/'prepared.json').read_text())
assert pre['source_inference_sha256']==pre['transfer_inference_sha256']
assert pre['source_config_route']==pre['candidate_config_route']
files=[p for p in r.iterdir() if p.is_file() and p.suffix in ('.json','.md','.py','.log')]
files += [p for p in (r/'runs/p100_exh14_valid20').rglob('*') if p.is_file() and 'probabilities' not in p.parts]
files += [p for p in (r/'bundle').rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.yaml')]
a=r/('collection_'+%r+'.tar.gz')
with tarfile.open(a,'x:gz') as t:
 for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
print(json.dumps({'archive':str(a),'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),
                  'p100_weights_verified':14,'source_route_exact':True}))
''' % (REMOTE, STAMP)

assert not (ROOT / 'comparison.json').exists()
meta = json.loads(subprocess.check_output(SSH + [
    '/workspace/venv-komap/bin/python -c ' + shlex.quote(remote_code)], text=True))
archive = ROOT / Path(meta['archive']).name
subprocess.run(['scp', '-P', '20670', 'root@168.188.128.36:' + meta['archive'], str(archive)], check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == meta['sha256']
snapshot = ROOT / 'collected' / STAMP
snapshot.mkdir(parents=True)
with tarfile.open(archive) as t:
    for member in t.getmembers():
        assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
    t.extractall(snapshot, filter='data')
spec = importlib.util.spec_from_file_location('audit', ROOT.parent / 'audit_20261008/analyze_runs.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
gt = Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad/gt/valid/masks')
release = ROOT.parent / 'exh14_base_search_20261008/exh14_valid20'
candidate = snapshot / 'runs/p100_exh14_valid20'
a, b = audit.score(release, gt), audit.score(candidate, gt)
paired = audit.pair(release, candidate, a, b)
assert len(a['images']) == len(b['images']) == 20
assert abs(100*b['miou']-audit.load(snapshot/'completion.json')['miou_percent_reported']) < 1e-10
source_lock = audit.load(snapshot/'source_lock.json')
assert all(hashlib.sha256((snapshot/name).read_bytes()).hexdigest()==digest
           for name,digest in source_lock.items())
gt_sha = {}
for name in a['images']:
    path = next(p for p in [gt/f'{name}_mask.png',gt/f'{name}.png'] if p.is_file())
    gt_sha[name] = hashlib.sha256(path.read_bytes()).hexdigest()
old_p100_score = 79.90951887388398
report = {'metric':'20 image means of four class IoUs; empty union=0',
          'release_exh14_miou_percent':100*a['miou'],
          'p100_exh14_transfer_miou_percent':100*b['miou'],
          'vs_release_exh14':paired,
          'historical_p100_model349_miou_percent':old_p100_score,
          'vs_historical_p100_model349_pp':100*b['miou']-old_p100_score,
          'historical_comparison_scope':'different route and earlier cached implementation; contextual only',
          'p100_weight_archive_sha256':meta['sha256'],
          'p100_weights_verified':meta['p100_weights_verified'],
          'source_route_exact':meta['source_route_exact'],
          'source_lock_exact':True,
          'gt_sha256':gt_sha,
          'fresh_completion':audit.load(candidate/'completion.json'),
          'independent_png_gt_scoring':True,
          'independent_generalization_verified':False,
          'test_used':False}
audit.dump(ROOT/'comparison.json', report)
lines=['# 80.2% 방법의 P100 가중치 이식 결과','',
       '원본영상 valid20을 fresh 추론하고 PNG/GT 20장 80개 IoU를 독립 채점했다. 공식 image별4class 평균, empty union=0.','',
       f"릴리스 가중치 exh14 {100*a['miou']:.7f}%, P100 가중치 이식 {100*b['miou']:.7f}%, 변화 {paired['delta_pp']:+.7f}%p.",
       f"과거 P100 원형 Model349 {old_p100_score:.7f}%와의 수치 차이 {report['vs_historical_p100_model349_pp']:+.7f}%p는 참고값이다. 이 비교에는 route와 과거 캐시 구현 차이가 함께 있다.",'',
       f"14개 P100 checkpoint SHA/bytes 및 이식 추론 코드·route 동일성 확인. 전체 fresh 단회 {report['fresh_completion']['elapsed_seconds']:.2f}초.",
       '반복 사용한 valid20 결과이므로 독립 일반화 또는 GPU만의 효과가 아니다. test 정답은 사용하지 않았다.','',
       '| 클래스 | 릴리스 대비 P100 이식 변화 |','|---|---:|']
for name,delta in zip(('Al3Ni','Primary Si','Al','Eutectic Si'),paired['class_delta_pp']):
    lines.append(f'| {name} | {delta:+.7f}%p |')
lines += ['', '| 이미지 | P100 이식 mIoU | 릴리스 대비 변화 |', '|---|---:|---:|']
for name in sorted(b['images']):
    lines.append(f"| {name} | {100*b['images'][name]['image_miou']:.7f}% | {paired['per_image_delta_pp'][name]:+.7f}%p |")
(ROOT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'p100_exh14_miou_percent':100*b['miou'],
                  'vs_release_pp':paired['delta_pp'],
                  'vs_historical_p100_pp':report['vs_historical_p100_model349_pp']},indent=2))
