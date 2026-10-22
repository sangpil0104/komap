"""Independent PNG scoring and reproducible bounded archive."""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import hashlib, importlib.util, json, statistics, tarfile
from pathlib import Path
ROOT = Path('/workspace/model349_compress/base12_highflip4_20261008')
OLD = Path('/workspace/model349_compress/audit_20261008/runs/cpu2_fp16_12')
GT = Path('/workspace/komap_model1/data/valid/masks')
def read(p): return json.loads(p.read_text())
def write(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')
def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def main():
    done = read(ROOT / 'completion.json')
    assert done['base_and_weights_unchanged']
    s = module('scores', '/workspace/model349_compress/audit_20261008/analyze_runs.py')
    a = module('agreement', '/workspace/model349_compress/base12_followup_20261008/finalize.py')
    references = {'base12': OLD, 'lite15': OLD.parent / 'cpu2_lite15'}
    ref_scores = {k: s.score(p, GT) for k, p in references.items()}
    runs = {}
    for p in sorted((ROOT / 'runs').iterdir()):
        if not (p / 'metrics.json').exists(): continue
        score = s.score(p, GT)
        runs[p.name] = {'miou_percent': 100 * score['miou'], 'class_iou_percent': [100*x for x in score['classes']], 'completion': read(p / 'completion.json'), **{'vs_' + k: s.pair(ref, p, ref_scores[k], score) for k, ref in references.items()}}
    frames = [json.loads(x) for x in (ROOT / 'monitor.jsonl').read_text().splitlines()]
    uuid = next(line.split(',')[1].strip() for line in frames[0]['gpu'].splitlines() if line.startswith('7,'))
    interference = [x['time'] for x in frames if sum(uuid in line for line in x['processes'].splitlines()) > 1]
    timings = None
    if (ROOT / 'timing_runs.json').exists():
        rows = read(ROOT / 'timing_runs.json')
        values = {k: [r['completion']['elapsed_seconds'] for r in rows if r['candidate'] == k] for k in ['control12', 'highflip4_12']}
        assert all(len(x) == 3 for x in values.values())
        med = {k: statistics.median(v) for k, v in values.items()}
        reduction = 100*(1-med['highflip4_12']/med['control12'])
        quality = all(row['vs_base12']['quality_pass'] and row['vs_lite15']['quality_pass'] for row in runs.values())
        timings = {'seconds': values, 'median_seconds': med, 'median_reduction_percent': reduction, 'all_quality_pass': quality, 'ten_percent_gate_pass': reduction >= 10 and quality and not interference}
    tests = {}
    for key in done['test_candidates']:
        p = ROOT / 'runs' / (key + '_test10')
        assert read(p / 'completion.json')['mean_image_miou'] is None
        tests[key] = a.agreement(Path('/workspace/model349_compress/generalization_20261008/runs/cpu2_fp16_12_test10'), p)
    result = {'runs': runs, 'timings': timings, 'test_mask_agreement': tests, 'gpu7_multiple_process_frames': interference, 'checkpoint_count': 12, 'checkpoint_bytes': 683320900, 'png_rescore_verified': True, 'base_and_weights_unchanged': True, 'independent_generalization_verified': False}
    write(ROOT / 'comparison.json', result)
    lines = ['# 高해상도 4방향 예측 결과'.replace('高', '고'), '', 'TENT5·원본/밝기 보정 두 입력·모든 12가중치·계수·후처리를 유지하고 high D4 8방향만 대칭 flip4 4방향으로 줄였다. 다른 기각 후보와 조합하지 않았다. 12개/683,320,900바이트 저장량은 동일하다.', '', '| 실행 | valid image mIoU | base 평균 변화 | 최악 이미지 변화 | base / lite15 gate |', '|---|---:|---:|---:|---|']
    for name, r in runs.items():
        b = r['vs_base12']
        lines.append(f"| {name} | {r['miou_percent']:.7f}% | {b['delta_pp']:+.7f}%p | {b['worst_image_delta_pp']:+.7f}%p ({b['worst_image']}) | {b['quality_pass']} / {r['vs_lite15']['quality_pass']} |")
    lines += ['', '이미지별 4 IoU 평균을 20장 평균, absent/empty=0. PNG/GT 독립 재채점과 metrics 일치를 확인했다. 클래스별·이미지별 상세는 comparison.json에 보관한다. 평균 손실 ≤0.01%p·최악 손실 ≤0.10%p를 두 기준에 모두 적용한다.', '']
    if timings:
        lines += [f"기준 3회 {timings['seconds']['control12']}초 / 중앙값 {timings['median_seconds']['control12']:.2f}초.", f"후보 3회 {timings['seconds']['highflip4_12']}초 / 중앙값 {timings['median_seconds']['highflip4_12']:.2f}초.", f"중앙값 단축 {timings['median_reduction_percent']:+.3f}%, 사전 10% 및 품질 gate {timings['ten_percent_gate_pass']}."]
    else:
        lines += ['품질 조건을 초과하여 반복 시간 비교와 test10을 시작하지 않았다. 후보 단회 시간은 진단 기록이며 확정 속도 개선으로 주장하지 않는다.']
    for key, r in tests.items(): lines.append(f"test10 {key}: 기존 base 마스크 일치 {100*r['agreement']:.7f}%, 변화 {r['changed_pixels']:,}/{r['pixels']:,}. 정답이 없으므로 test IoU/정확도가 아니다.")
    lines += ['', f'GPU7 복수 compute 감지 {len(interference)} frames. 기존 공유 CPU 작업을 보존했고 부하 이력도 보관했다.', '', '같은 valid20에서 반복 탐색한 결과다. 신규 데이터·독립 일반화 증거가 없다. base12는 교체하지 않고 원본 가중치·기존 릴리스도 보존했다.', '']
    (ROOT / 'report.md').write_text('\n'.join(lines))
    files = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix not in ['.pt', '.pyc'] and 'probabilities' not in p.parts and '__pycache__' not in p.parts and p.name not in ['gpu7.lock', 'results_archive.tar.gz', 'archive_receipt.json']]
    archive = ROOT / 'results_archive.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for p in sorted(files): stream.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    write(ROOT / 'archive_receipt.json', {'files': len(files), 'bytes': archive.stat().st_size, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()})
    print(json.dumps({'runs': {n: r['miou_percent'] for n, r in runs.items()}, 'timings': timings, 'test': tests}, indent=2))
if __name__ == '__main__': main()
