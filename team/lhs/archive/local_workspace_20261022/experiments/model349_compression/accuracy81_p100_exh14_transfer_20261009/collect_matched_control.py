"""Independently score the P100 original-route control and paired transfer."""
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
done=json.loads((r/'control_completion.json').read_text())
assert done['finished'] and done['fresh_valid20'] and done['weights']==20
m=json.loads((r/'matched_control_bundle/checkpoint_manifest.json').read_text())
assert m['count']==20 and len(m['checkpoints'])==20
for row in m['checkpoints']:
 p=Path(row['source_path'])
 assert p.is_file() and p.stat().st_size==row['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
pre=json.loads((r/'matched_control_prepared.json').read_text())
assert hashlib.sha256((r/'matched_control_bundle/infer_lite.py').read_bytes()).hexdigest()==pre['control_inference_sha256']
files=[p for p in r.iterdir() if p.is_file() and p.suffix in ('.json','.md','.py','.log')]
files += [p for p in (r/'runs/p100_original_matched_valid20').rglob('*') if p.is_file() and 'probabilities' not in p.parts]
files += [p for p in (r/'matched_control_bundle').rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.yaml')]
a=r/('matched_collection_'+%r+'.tar.gz')
with tarfile.open(a,'x:gz') as t:
 for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
print(json.dumps({'archive':str(a),'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),
                  'p100_control_weights_verified':20}))
''' % (REMOTE, STAMP)

assert (ROOT/'comparison.json').is_file() and not (ROOT/'matched_comparison.json').exists()
meta = json.loads(subprocess.check_output(SSH + [
    '/workspace/venv-komap/bin/python -c ' + shlex.quote(remote_code)], text=True))
archive = ROOT / Path(meta['archive']).name
subprocess.run(['scp', '-P', '20670', 'root@168.188.128.36:' + meta['archive'], str(archive)], check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == meta['sha256']
snapshot = ROOT / 'matched_collected' / STAMP
snapshot.mkdir(parents=True)
with tarfile.open(archive) as t:
    for member in t.getmembers():
        assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
    t.extractall(snapshot, filter='data')
spec = importlib.util.spec_from_file_location('audit', ROOT.parent/'audit_20261008/analyze_runs.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
gt = Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad/gt/valid/masks')
release = ROOT.parent/'exh14_base_search_20261008/exh14_valid20'
candidate_snapshot = sorted((ROOT/'collected').iterdir())[-1]
candidate = candidate_snapshot/'runs/p100_exh14_valid20'
control = snapshot/'runs/p100_original_matched_valid20'
a, b, c = (audit.score(p,gt) for p in (release, candidate, control))
pair = audit.pair(control,candidate,c,b)
release_pair = audit.pair(release,candidate,a,b)
historical=json.loads((ROOT/'historical_p100_model349_results.json').read_text())
old=100*historical['raw_best_valid_image_mean_miou_empty0']
assert abs(100*b['miou']-audit.load(candidate_snapshot/'completion.json')['miou_percent_reported'])<1e-10
assert abs(100*c['miou']-audit.load(snapshot/'control_completion.json')['miou_percent_reported'])<1e-10
control_lock=audit.load(snapshot/'control_source_lock.json')
assert all(hashlib.sha256((snapshot/name).read_bytes()).hexdigest()==digest for name,digest in control_lock.items())
report={'metric':'20 image means of four class IoUs; empty union=0',
        'release_exh14_miou_percent':100*a['miou'],
        'p100_original_matched_miou_percent':100*c['miou'],
        'p100_exh14_transfer_miou_percent':100*b['miou'],
        'historical_p100_model349_cached_miou_percent':old,
        'matched_control_vs_historical_cached_pp':100*c['miou']-old,
        'p100_transfer_vs_matched_control':pair,
        'p100_transfer_vs_release_exh14':release_pair,
        'control_archive':meta,'control_source_lock_exact':True,
        'independent_png_gt_scoring':True,'independent_generalization_verified':False,
        'test_used':False}
audit.dump(ROOT/'matched_comparison.json',report)
lines=['# 같은 P100 가중치의 원래 route와 80.2% 방법 이식 비교','',
       '원본영상 fresh 추론 20장 PNG/GT의 80개 IoU를 두 P100 경로에서 독립 채점했다. 공식 image별4class 평균, empty union=0.','',
       '| 경로 | valid20 mIoU |','|---|---:|',
       f"| 원래 P100 route (같은 실행기 fresh) | {100*c['miou']:.7f}% |",
       f"| P100 가중치 + exh14 방법 (fresh) | {100*b['miou']:.7f}% |",
       f"| 릴리스 가중치 + exh14 방법 | {100*a['miou']:.7f}% |",
       f"| 과거 P100 Model349 캐시 (참고) | {old:.7f}% |",'',
       f"같은 P100 가중치에서 route 이식 변화 {pair['delta_pp']:+.7f}%p. 릴리스 가중치 exh14 대비 P100 이식 변화 {release_pair['delta_pp']:+.7f}%p.",
       f"같은 실행기의 원래 route가 과거 P100 캐시 점수와 다른 정도 {100*c['miou']-old:+.7f}%p. 이 차이가 크면 route 순수 효과로 해석하기 전에 재현 원인을 감사해야 한다.",'',
       '14/20개의 각 대응 P100 checkpoint SHA/bytes, 소스·route lock, archive SHA, 동일 입력 SHA/GT, 실행 metrics와 독립 PNG 채점을 확인했다. valid20은 반복 개발에 사용되어 독립 일반화 증거가 아니며 test 점수는 없다.','',
       '| 클래스 | P100 원래 route 대비 이식 변화 |','|---|---:|']
for name,delta in zip(('Al3Ni','Primary Si','Al','Eutectic Si'),pair['class_delta_pp']):
    lines.append(f'| {name} | {delta:+.7f}%p |')
lines += ['', '| 이미지 | P100 이식 mIoU | P100 원래 route 대비 |', '|---|---:|---:|']
for name in sorted(b['images']):
    lines.append(f"| {name} | {100*b['images'][name]['image_miou']:.7f}% | {pair['per_image_delta_pp'][name]:+.7f}%p |")
(ROOT/'matched_report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'p100_original_matched_percent':100*c['miou'],
                  'p100_exh14_percent':100*b['miou'],
                  'matched_route_delta_pp':pair['delta_pp'],
                  'release_exh14_percent':100*a['miou']},indent=2))
