"""Collect a completed fresh context run and independently audit valid PNGs."""
from pathlib import Path
import hashlib, importlib.util, json, shlex, subprocess, tarfile, time
import numpy as np
ROOT = Path(__file__).resolve().parent
REMOTE = '/workspace/model349_compress/' + ROOT.name
SSH = ['ssh', '-p', '20670', 'root@168.188.128.36']
stamp = str(time.time_ns())
code = r'''from pathlib import Path
import json,tarfile,hashlib
r=Path(%r);done=json.loads((r/'completion.json').read_text())
assert done['finished'] and done['whole_fresh_pipeline']
receipt=json.loads((r/'training/context/completion.json').read_text())
assert hashlib.sha256(Path(receipt['final_checkpoint']).read_bytes()).hexdigest()==done['final_checkpoint_sha256']==receipt['final_sha256']
files=[p for p in r.iterdir() if p.is_file() and p.suffix in ('.json','.py','.md','.log')]
files += [p for p in (r/'training').rglob('*.json')]
files += [p for p in (r/'runs').rglob('*') if p.is_file() and 'probabilities' not in p.relative_to(r/'runs').parts]
files += [p for p in (r/'bundle').rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.yaml')]
a=r/('collection_'+%r+'.tar.gz')
with tarfile.open(a,'x:gz') as t:
 for p in sorted(set(files)):t.add(p,arcname=str(p.relative_to(r)),recursive=False)
print(json.dumps({'archive':str(a),'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'remote_final_checkpoint_sha_verified':True}))
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
refpath = ROOT.parent / 'exh14_base_search_20261008/exh14_valid20'
controlpath = sorted((ROOT.parent / 'accuracy81_hrlovasz_gpu3_20261008/collected').glob('*/runs/control_valid20'))[-1]
run = snapshot / 'runs/context_valid20'
ref, control, result = (audit.score(p, gt) for p in [refpath, controlpath, run])
receipt = audit.load(snapshot / 'training/context/completion.json')
assert receipt['epochs'] == 4 and receipt['optimizer_updates'] == 420
assert not receipt['valid_read'] and not receipt['test_read']
lock = audit.load(snapshot / 'preflight_source_lock.json')
assert all(hashlib.sha256((snapshot / p).read_bytes()).hexdigest() == h for p, h in lock.items())
complete = audit.load(snapshot / 'completion.json')
assert complete['parent_source_weights_unchanged'] and complete['whole_fresh_pipeline']
invariants = audit.load(snapshot / 'final_invariant_audit.json')
assert invariants['epoch'] == 4 and invariants['optimizer_updates'] == 420
assert invariants['candidate_sha256'] == receipt['final_sha256'] == complete['final_checkpoint_sha256']
assert invariants['bn_buffers_unchanged'] and invariants['boundary_head_unchanged']
assert invariants['all_tensors_finite'] and invariants['parent_checkpoint_unchanged']
report = {'archive': metadata, 'miou_percent': result['miou'] * 100,
    'class_iou_percent': (100 * np.asarray(result['classes'])).tolist(),
    'vs_original_base': audit.pair(refpath, run, ref, result),
    'vs_previous_matched_training_control': audit.pair(controlpath, run, control, result),
    'training': receipt, 'fresh_inference': audit.load(run / 'completion.json'),
    'completion': complete, 'final_checkpoint_invariants': invariants, 'source_snapshot_matches_preflight': True,
    'independent_png_scoring_agrees': True, 'independent_generalization_verified': False}
data_audit = audit.load(ROOT / 'context_data_audit.json')
overlap = set(data_audit['train_valid_specimen_overlap'])
report['context_data_audit'] = data_audit
report['descriptive_specimen_subgroups'] = {}
for label, overlapping in [('overlaps_training_specimen', True), ('disjoint_from_training_specimen', False)]:
    names = [n for n in result['images'] if (n.split('_')[0] in overlap) == overlapping]
    assert names
    means = {key: 100 * np.mean([values['images'][n]['image_miou'] for n in names])
             for key, values in [('base', ref), ('control', control), ('context', result)]}
    report['descriptive_specimen_subgroups'][label] = {'images': names, 'count': len(names),
        'means_percent': means, 'delta_vs_base_pp': means['context'] - means['base'],
        'delta_vs_control_pp': means['context'] - means['control'],
        'independent_generalization_verified': False}
audit.dump(ROOT / 'comparison.json', report)
lines = ['# HR336 학습 문맥 확대 검증', '',
    '공식 per-image4IoU 평균 후20장평균, emptyunion0. PNG/GT 독립 재채점과 저장 metrics의 모든 IoU 일치를 확인했다.', '',
    f"원본 base {100*ref['miou']:.7f}%, 기존 512 학습 control {100*control['miou']:.7f}%, 이번 1024 학습 {100*result['miou']:.7f}%.", '',
    'valid 반복 개발평가이며 독립 일반화를 보장하지 않는다. 기존 대조군과 동시 실행은 아니고 학습 픽셀 수·연산량은 4배다. 초기화/손실/학습률/epoch/샘플 수/배치 구조/추론 레시피는 유지했다. 원본과 기존 릴리스를 보존했고 test 정답·test mIoU는 없다.', '',
    f"최종 checkpoint SHA: `{receipt['final_sha256']}`. 4epochs/420updates, 학습 {receipt['elapsed_seconds']:.2f}초. 전체 fresh valid 추론 {report['fresh_inference']['elapsed_seconds']:.2f}초(단회, 속도 개선 근거 아님).", '',
    f"가중치14개, 총 {complete['checkpoint_bytes']:,} bytes. 구조·파라미터 수를 줄이지 않았으며 원본 대비 미세한 파일 크기 차이는 저장 형식 차이로 취급한다. 최종 BN1029 buffers와 동결 boundary2 tensors가 원본과 exact이고 모든 state가 finite다.", '',
    '| 클래스 | 원본 base 대비 | 기존 control 대비 |', '|---|---:|---:|']
for i, name in enumerate(['Al3Ni', 'Primary Si', 'Al', 'Eutectic Si']):
    lines.append(f"| {name} | {report['vs_original_base']['class_delta_pp'][i]:+.7f}%p | {report['vs_previous_matched_training_control']['class_delta_pp'][i]:+.7f}%p |")
lines += ['', '| 이미지 | 최종 mIoU | 원본 base 대비 | 기존 control 대비 |', '|---|---:|---:|---:|']
for name in sorted(result['images']):
    lines.append(f"| {name} | {100*result['images'][name]['image_miou']:.7f}% | {report['vs_original_base']['per_image_delta_pp'][name]:+.7f}%p | {report['vs_previous_matched_training_control']['per_image_delta_pp'][name]:+.7f}%p |")
lines += ['', '| valid 부분 | 장수 | 원본 base 대비 | 기존 control 대비 |', '|---|---:|---:|---:|']
for key, label in [('overlaps_training_specimen', '학습 시편 겹침'), ('disjoint_from_training_specimen', '학습 시편 비겹침')]:
    row = report['descriptive_specimen_subgroups'][key]
    lines.append(f"| {label} | {row['count']} | {row['delta_vs_base_pp']:+.7f}%p | {row['delta_vs_control_pp']:+.7f}%p |")
lines += ['', '부분집합 결과는 기술 통계다. 비겹침13장도 모델 개발·선택에 반복 사용된 valid의 일부이므로 독립 평가로 주장하지 않는다. 17장 학습 이미지의 reflection padding32픽셀과 시편 겹침을 입력 감사에 기록했다.', '']
(ROOT / 'report.md').write_text('\n'.join(lines))
print(json.dumps({'miou_percent': report['miou_percent'], 'vs_base_pp': report['vs_original_base']['delta_pp'], 'vs_control_pp': report['vs_previous_matched_training_control']['delta_pp']}, indent=2))
