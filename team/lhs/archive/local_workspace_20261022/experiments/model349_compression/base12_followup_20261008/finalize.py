"""Independently rescore PNG/GT and collect a bounded follow-up report."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import hashlib, importlib.util, json, statistics, tarfile, re, collections
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path('/workspace/model349_compress/base12_followup_20261008')
OLD = Path('/workspace/model349_compress/audit_20261008/runs/cpu2_fp16_12')
BASE_TEST = Path('/workspace/model349_compress/generalization_20261008/runs/cpu2_fp16_12_test10')
GT = Path('/workspace/komap_model1/data/valid/masks')


def read(path): return json.loads(path.read_text())
def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def agreement(ref, candidate):
    a, b = read(ref / 'input_manifest.json'), read(candidate / 'input_manifest.json')
    assert {x['name']: x['sha256'] for x in a['images']} == {x['name']: x['sha256'] for x in b['images']}
    assert a['calibration_sha256'] == b['calibration_sha256']
    rows = []
    for row in a['images']:
        name = row['name']
        x = np.asarray(Image.open(ref / 'predicted_labels' / (name + '.png')))
        y = np.asarray(Image.open(candidate / 'predicted_labels' / (name + '.png')))
        assert x.shape == y.shape and np.isin(y, range(4)).all()
        changed = int(np.count_nonzero(x != y))
        rows.append({'image': name, 'pixels': int(x.size), 'changed_pixels': changed, 'agreement': 1 - changed / x.size})
    total = sum(x['pixels'] for x in rows)
    changed = sum(x['changed_pixels'] for x in rows)
    return {'images': rows, 'pixels': total, 'changed_pixels': changed, 'agreement': 1 - changed / total,
            'accuracy_or_test_iou': False}


def main():
    completed = read(ROOT / 'completion.json')
    assert completed['diagnostics_completed'] and completed['base_and_weights_unchanged']
    spec = importlib.util.spec_from_file_location('independent_scoring',
        '/workspace/model349_compress/audit_20261008/analyze_runs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ref = module.score(OLD, GT)
    lite15_path = OLD.parent / 'cpu2_lite15'
    lite15 = module.score(lite15_path, GT)
    runs = {}
    for path in sorted((ROOT / 'runs').iterdir()):
        if not (path / 'metrics.json').is_file(): continue
        val = module.score(path, GT)
        runs[path.name] = {'miou_percent': 100 * val['miou'],
                           'class_iou_percent': (100 * np.asarray(val['classes'])).tolist(),
                           'completion': read(path / 'completion.json'),
                           'vs_base12': module.pair(OLD, path, ref, val),
                           'vs_lite15': module.pair(lite15_path, path, lite15, val)}
    tests = {}
    coverage = {}
    for key, path in [('base_valid20', OLD), ('base_test10', BASE_TEST)]:
        chosen = dict(re.findall(r'DONE base (\S+) source=(\S+)', (path / 'logs/base.log').read_text()))
        assert len(chosen) == (20 if key == 'base_valid20' else 10)
        coverage[key] = {'selected_sources': chosen, 'counts': dict(collections.Counter(chosen.values()))}
    for key in completed['test_candidates']:
        path = ROOT / 'runs' / (key + '_test10')
        assert read(path / 'completion.json')['mean_image_miou'] is None
        tests[key] = {'vs_base12': agreement(BASE_TEST, path), 'completion': read(path / 'completion.json')}
    timings = None
    if (ROOT / 'timing_runs.json').is_file():
        rows = read(ROOT / 'timing_runs.json')
        values = {key: [x['completion']['elapsed_seconds'] for x in rows if x['candidate'] == key]
                  for key in ['control12', completed['time_candidate']]}
        assert all(len(x) == 3 for x in values.values())
        med = {k: statistics.median(v) for k, v in values.items()}
        speedup = 100 * (1 - med[completed['time_candidate']] / med['control12'])
        quality = all(runs[x['run']]['vs_base12']['quality_pass'] and runs[x['run']]['vs_lite15']['quality_pass'] for x in rows)
        timings = {'candidate': completed['time_candidate'], 'seconds': values, 'median_seconds': med,
                   'median_reduction_percent': speedup, 'all_repeats_quality_pass': quality,
                   'ten_percent_gate_pass': speedup >= 10 and quality,
                   'interpretation': 'shared GPU exploratory paired timing; inspect monitored load before claiming improvement'}
    frames = [json.loads(x) for x in (ROOT / 'monitor.jsonl').read_text().splitlines()]
    gpu7_uuid = next(line.split(',')[1].strip() for line in frames[0]['gpu'].splitlines() if line.startswith('7,'))
    interference = [{'time': row['time'], 'processes': row['processes']} for row in frames
                    if sum(gpu7_uuid in line for line in row['processes'].splitlines()) > 1]
    write(ROOT / 'comparison.json', {'metric': '20-image mean of per-image four-class IoU; empty/absent=0',
                                     'base': 'cpu2_fp16_12', 'runs': runs, 'test_comparisons': tests,
                                     'timings': timings, 'gpu7_multiple_process_frames': interference,
                                     'high_source_coverage': coverage,
                                     'independent_generalization_verified': False,
                                     'png_scores_recomputed_and_metrics_verified': True})
    raw = read(ROOT / 'raw_comparison.json')
    lines = ['# 12개 base 후속 경량화·성능 탐색 결과', '',
             '세 후보 변경을 분리하고 결과 전 범위를 고정했다. 모든 결과는 반복 사용한 valid20의 탐색·구현 재현이며 독립 일반화 검증이 아니다. 기존 base 및 가중치는 그대로 보존했다.', '',
             '| 후보 | 가중치/바이트 | valid image mIoU | base 대비 평균 | 최악 이미지 변화 | base gate / 기존 lite15 gate |',
             '|---|---:|---:|---:|---:|---|']
    for key, item in raw.items():
        row = runs[key + '_valid20']; p = row['vs_base12']; storage = item['checkpoints']
        lines.append(f"| {key} | {storage['count']} / {storage['bytes']:,} | {row['miou_percent']:.7f}% | {p['delta_pp']:+.7f}%p | {p['worst_image_delta_pp']:+.7f}%p ({p['worst_image']}) | {'통과' if p['quality_pass'] else '기각'} / {'통과' if row['vs_lite15']['quality_pass'] else '기각'} |")
    lines += ['', 'gate: base12 대비 평균 손실 ≤0.01%p 및 최악 이미지 손실 ≤0.10%p. lite15 대비 원래 gate도 JSON에 별도로 기록했다. 클래스별·이미지별 손실, 원본349 비교는 `raw_comparison.json`과 `comparison.json`을 확인한다.', '']
    lines += ['## 결합 계수 CPU 탐색', '',
              '기존 base 확률을 고정한 탐색 채점이다. gamma0.5의 base 점수는 정확히 재현했다. 아래 둘은 국소 손실 gate 초과로 기각했으므로 원본 영상 재추론 후보로 올리지 않았다.', '',
              '| Al3Ni gamma329 | 탐색 mIoU | 평균 변화 | 최악 이미지 변화 |',
              '|---|---:|---:|---:|']
    for row in read(ROOT / 'offline_gamma_probe.json')['candidates']:
        if row['gamma'] == 0.5: continue
        p = row['vs_base12']
        lines.append(f"| {row['gamma']} | {100*row['metrics']['image_miou']:.7f}% | {p['delta_pp']:+.7f}%p | {p['worst_image_delta_pp']:+.7f}%p ({p['worst_image']}) |")
    lines += ['', 'Al3Ni 클래스 계수를 올려도 `replace_classes`가 네 클래스 확률을 다시 정규화하므로 다른 클래스가 독립적으로 보존되는 것은 아니다. gamma0.6에서 A153 Eutectic Si IoU는 약 −0.770925%p로 악화했다. 평균만 높아진 조합을 성능 개선판으로 채택하지 않았다.', '']
    if timings:
        lines += ['## 실제 시간 반복 비교', '', f"선택 후보: {timings['candidate']}. 기준·후보를 A-B, B-A, A-B로 각각 3회 원본 valid20부터 순차 실행했다.", '',
                  f"- 기준 시간: {timings['seconds']['control12']}초; 중앙값 {timings['median_seconds']['control12']:.2f}초.",
                  f"- 후보 시간: {timings['seconds'][timings['candidate']]}초; 중앙값 {timings['median_seconds'][timings['candidate']]:.2f}초.",
                  f"- 중앙값 변화: {timings['median_reduction_percent']:+.3f}% 단축. 10% gate: {timings['ten_percent_gate_pass']}. 반복 품질 gate 전부 통과: {timings['all_repeats_quality_pass']}.",
                  f"- GPU7 복수 compute process 감지 frame: {len(interference)}. 공유 서버 CPU/GPU 부하 이력은 telemetry와 monitor JSONL에 보존했다.", '']
    else:
        lines += ['실행 전 동결한 선택 코드에서는 base12와 기존 lite15의 gate를 모두 통과해야 반복 시간/후속 test 후보로 올리도록 구현했다. 프로토콜 본문의 “품질 통과”는 이 이중 조건을 명확하게 풀어 쓰지 않아 문구가 불완전했다. TENT4는 base12 기준만 통과했고 lite15 대비 A74에서 −0.1177126%p로 기존 국소 손실 한도를 넘었다. 따라서 이번 단계에서는 반복 시간과 test를 시작하지 않았다. TENT4가 base gate 자체에 실패했다고 표시하지 않으며, 탐색 단회 시간을 속도 개선 근거로 쓰지 않는다.', '']
    lines += ['## test10 출력 안정성', '', '정답이 없어 아래는 출력 일치율이며 정확도나 test IoU가 아니다. test 결과로 후보를 다시 고르지 않았다.', '']
    lines += ['기존 base의 test10 고해상도 선택은 restart51 8장·pairwise 2장이고 seed456 구간은 0장이다. 따라서 11개 seed 대체 후보의 test 일치가 높아도 제거한 밝기 구간의 성능을 검증한 것은 아니다.', '']
    for key, item in tests.items():
        a = item['vs_base12']; lines.append(f"- {key}: base12 마스크 일치 {100*a['agreement']:.7f}%, {a['changed_pixels']:,}/{a['pixels']:,}픽셀 차이; {item['completion']['elapsed_seconds']:.2f}초 단회.")
    lines += ['', '## 한계와 사용 상태', '',
              '계수 후보는 동일 valid의 저장 확률로 선택한 뒤 원본 재추론했다. 재추론은 구현 재현이지 독립 성능 확인이 아니다. 고해상도 seed 대체도 밝기별 모델 선택을 추가 탐색한 것이므로 신규 자료에서 손실을 보증할 수 없다. 100장만 있으며 train-valid specimen 중복과 동결 가중치 학습 이력의 불완전성은 기존 일반화 감사와 동일하다.', '',
              '12개 base는 자동 교체하지 않는다. 통과한 새 후보를 별도 실험 버전으로 보관하고 사용자에게 결과를 보고한다. TENT 변경은 계산량에 영향을 주며 고해상도 밝기 모델 한 개 제거는 저장량만큼 전체 순전파 시간을 줄이지 않을 수 있다.', '']
    (ROOT / 'report.md').write_text('\n'.join(lines))
    files = [p for p in ROOT.rglob('*') if p.is_file() and p.name not in ['results_archive.tar.gz', 'archive_receipt.json']
             and 'probabilities' not in p.parts and p.suffix not in ['.pt', '.pyc']
             and '__pycache__' not in p.parts and p.name not in ['gpu7.lock']]
    archive = ROOT / 'results_archive.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for p in sorted(files): stream.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    write(ROOT / 'archive_receipt.json', {'files': len(files), 'bytes': archive.stat().st_size, 'sha256': digest})
    print(json.dumps({'raw_results': {k: {'score': v['miou_percent'], 'delta': v['vs_base12']['delta_pp'],
                                       'worst': v['vs_base12']['worst_image_delta_pp']} for k, v in runs.items()},
                      'timings': timings, 'archive_sha256': digest}, indent=2))


if __name__ == '__main__': main()
