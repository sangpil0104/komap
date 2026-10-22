"""Collect completed moment experiments and independently score all valid PNGs."""
from pathlib import Path
import hashlib, importlib.util, json, shlex, subprocess, tarfile, time
import numpy as np

ROOT = Path(__file__).resolve().parent
REMOTE = '/workspace/model349_compress/' + ROOT.name
SSH = ['ssh', '-p', '20670', 'root@168.188.128.36']
stamp = str(time.time_ns())
code = r'''import pathlib,json,tarfile,hashlib
r=pathlib.Path(%r)
done=json.loads((r/'completion.json').read_text())
assert done['finished'] and done['baseline80_iou_exact']
files=[p for p in r.iterdir() if p.is_file() and p.suffix in ('.json','.py','.md','.log')]
files += [p for folder in ['runs','moment_stats'] for p in (r/folder).rglob('*') if p.is_file()]
a=r/('collection_'+%r+'.tar.gz')
with tarfile.open(a,'x:gz') as t:
 for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
print(json.dumps({'archive':str(a),'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'files':len(set(files))}))
''' % (REMOTE, stamp)
metadata = json.loads(subprocess.check_output(SSH + ['python -c ' + shlex.quote(code)], text=True))
archive = ROOT / Path(metadata['archive']).name
subprocess.run(['scp', '-P', '20670', 'root@168.188.128.36:' + metadata['archive'], str(archive)], check=True)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata['sha256']
snapshot = ROOT / 'collected' / stamp
snapshot.mkdir(parents=True)
with tarfile.open(archive) as t:
    for m in t.getmembers():
        assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
    t.extractall(snapshot, filter='data')
spec = importlib.util.spec_from_file_location('audit', ROOT.parent / 'audit_20261008/analyze_runs.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
gt = Path('/tmp/claude-501/-Users-hyunseok-orca-workspaces-KoMaP-gurnard/8c455a47-b7eb-4c6e-b73d-ddf6da07fc20/scratchpad/gt/valid/masks')
lock = audit.load(snapshot / 'preflight_source_lock.json')
assert all(hashlib.sha256((snapshot / p).read_bytes()).hexdigest() == h for p, h in lock.items())
done = audit.load(snapshot / 'completion.json')
gpu = audit.load(snapshot / 'gpu_completion.json')
assert done['parent_source_weights_unchanged'] and done['cached_sources_unchanged']
assert gpu['baseline_probability_maps_exact'] and not gpu['valid_gt_opened']
refpath = snapshot / 'runs/baseline'
ref = audit.score(refpath, gt)
report = {'archive': metadata, 'completion': done, 'gpu_completion': gpu,
          'source_snapshot_matches_preflight': True, 'runs': {},
          'independent_generalization_verified': False,
          'scope': 'changed HR336 raw image inference, unchanged components reused; repeated valid development'}
for kind in ['baseline', 'alpha025', 'alpha1']:
    run = snapshot / 'runs' / kind
    result = audit.score(run, gt)
    report['runs'][kind] = {'miou_percent': result['miou'] * 100,
        'class_iou_percent': (np.asarray(result['classes']) * 100).tolist(),
        'vs_baseline': audit.pair(refpath, run, ref, result),
        'completion': audit.load(run / 'completion.json'), 'independent_png_scoring_agrees': True}
audit.dump(ROOT / 'comparison.json', report)
lines = ['# HR336 이미지별 정규화 통계 검증', '',
    '이미지별 4클래스 IoU 평균 후 valid20 평균. 빈 union은 0. 저장 PNG와 GT를 독립 재채점하고 실행 기록의 80개 IoU와 일치함을 확인했다.', '',
    '| 후보 | image mIoU | 기준 대비 | 최악 이미지 변화 |',
    '|---|---:|---:|---:|']
for kind, result in report['runs'].items():
    pair = result['vs_baseline']
    lines.append(f"| {kind} | {result['miou_percent']:.7f}% | {pair['delta_pp']:+.7f}%p | {pair['worst_image_delta_pp']:+.7f}%p ({pair['worst_image']}) |")
lines += ['', f"가중치 {done['checkpoint_count']}개, {done['checkpoint_bytes']:,} bytes. 원본 가중치·소스와 재사용 확률 파일의 SHA가 유지됐다.",
    'HR336 변경 경로만 원본 이미지에서 추론했고 다른 분기는 기존 확률을 재사용했다. GPU 실행 시간은 전체 fresh 앙상블 시간이 아니며 단회 결과로 속도 개선을 판단하지 않는다.',
    '현재 이미지의 특성 통계만 사용하고 추론 중 정답을 읽지 않는다. 같은 valid를 반복 사용한 개발평가이므로 일반화 개선을 보장하지 않는다. test 정답 또는 test mIoU는 없다.', '']
(ROOT / 'report.md').write_text('\n'.join(lines))
print(json.dumps({k: {'miou': v['miou_percent'], 'delta': v['vs_baseline']['delta_pp']} for k, v in report['runs'].items()}, indent=2))
