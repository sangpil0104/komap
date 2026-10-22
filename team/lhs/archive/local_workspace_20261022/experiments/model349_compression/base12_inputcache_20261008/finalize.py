"""PNG rescore, actual timing and packaging of unchanged-model input reuse."""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import hashlib, importlib.util, json, statistics, tarfile
from pathlib import Path
ROOT = Path('/workspace/model349_compress/base12_inputcache_20261008')
PREVIOUS = Path('/workspace/model349_compress/base12_followup_20261008')
OLD = Path('/workspace/model349_compress/audit_20261008/runs/cpu2_fp16_12')
TEST = Path('/workspace/model349_compress/generalization_20261008/runs/cpu2_fp16_12_test10')
GT = Path('/workspace/komap_model1/data/valid/masks')


def read(path): return json.loads(path.read_text())
def write(path, obj): path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    done = read(ROOT / 'completion.json')
    assert done['base_and_weights_unchanged']
    s = module('score', '/workspace/model349_compress/audit_20261008/analyze_runs.py')
    helper = module('agreement', PREVIOUS / 'finalize.py')
    base = s.score(OLD, GT)
    lite15_path = OLD.parent / 'cpu2_lite15'
    lite15 = s.score(lite15_path, GT)
    runs = {}
    for path in sorted((ROOT / 'runs').iterdir()):
        if (path / 'metrics.json').exists():
            val = s.score(path, GT)
            runs[path.name] = {'miou_percent': 100 * val['miou'], 'completion': read(path / 'completion.json'),
                               'vs_base12': s.pair(OLD, path, base, val),
                               'vs_lite15': s.pair(lite15_path, path, lite15, val)}
    timings = None
    frames = [json.loads(x) for x in (ROOT / 'monitor.jsonl').read_text().splitlines()]
    uuid = next(x.split(',')[1].strip() for x in frames[0]['gpu'].splitlines() if x.startswith('7,'))
    interference = [x['time'] for x in frames if sum(uuid in line for line in x['processes'].splitlines()) > 1]
    if (ROOT / 'timing_runs.json').exists():
        rows = read(ROOT / 'timing_runs.json')
        seconds = {key: [x['completion']['elapsed_seconds'] for x in rows if x['candidate'] == key]
                   for key in ['control12', 'cache12']}
        assert all(len(v) == 3 for v in seconds.values())
        med = {key: statistics.median(values) for key, values in seconds.items()}
        reduction = 100 * (1 - med['cache12'] / med['control12'])
        paired = [100 * (1 - seconds['cache12'][i] / seconds['control12'][i]) for i in range(3)]
        quality = all(x['vs_base12']['quality_pass'] and x['vs_lite15']['quality_pass'] for x in runs.values())
        timings = {'seconds': seconds, 'median_seconds': med, 'median_reduction_percent': reduction,
                   'round_paired_reduction_percent': paired, 'all_quality_pass': quality,
                   'ten_percent_gate_pass': reduction >= 10 and quality and not interference,
                   'scope': 'three cross-ordered runs per side on shared P100 with existing CPU workloads'}
    tests = {}
    for key in done['test_candidates']:
        path = ROOT / 'runs' / (key + '_test10')
        assert read(path / 'completion.json')['mean_image_miou'] is None
        tests[key] = helper.agreement(TEST, path)
    result = {'runs': runs, 'timings': timings, 'test_mask_agreement': tests,
              'probe': read(ROOT / 'probe_result.json'), 'gpu7_multiple_process_frames': interference,
              'component_probability_parity': read(ROOT / 'component_probability_parity.json'),
              'checkpoint_count': 12, 'checkpoint_bytes': 683320900,
              'base_and_weights_unchanged': True, 'png_rescore_verified': True,
              'independent_generalization_verified': False}
    write(ROOT / 'comparison.json', result)
    lines = ['# 12개 base B4 입력 재사용 결과', '',
             '가중치·모델 순서·출력 결합은 그대로 두고 동일 영상·동일 전처리 설정의 B4 입력만 fold 사이 재사용했다. 원래 base는 변경하지 않았다. 체크포인트 12개/683,320,900바이트는 같다. 정확도 상승 또는 모델 수 감소 실험이 아니다.', '',
             '## 동일성 및 공식 품질', '']
    for row in result['probe']:
        lines.append(f"- {row['component']}: 출력 tensor 정확히 동일={row['exact_tensor_equal']}, 최대 오차={row['max_absolute_error']}; 전처리 호출 {row['preprocessing_calls']['old']}→{row['preprocessing_calls']['new']}.")
    for key, row in result['component_probability_parity'].items():
        lines.append(f"- valid20 전체 저장 확률 {key}: 20장 모두 기존 base와 정확히 동일={row['all20_exact_equal']}.")
    for name, row in runs.items():
        p = row['vs_base12']; lines.append(f"- {name}: valid20 image mIoU {row['miou_percent']:.7f}%, base 대비 평균 {p['delta_pp']:+.7f}%p / 최악 {p['worst_image_delta_pp']:+.7f}%p, gate {p['quality_pass']}.")
    lines += ['', '평가는 PNG/GT를 별도 재채점하고 metrics와 1e-12 이내 일치를 확인했다. 이미지별 4 IoU 평균 후 20이미지 평균, absent/empty=0. 클래스별·이미지별 변화는 comparison.json에 포함한다. 변경하지 않은 TENT 고해상도 경로의 반복 수치 흔들림이 있어 미세한 평균 상승을 정확도 개선으로 주장하지 않는다.', '', '## 실제 시간', '']
    if timings:
        lines += [f"- 기준3회: {timings['seconds']['control12']}초, 중앙값 {timings['median_seconds']['control12']:.2f}초.",
                  f"- 후보3회: {timings['seconds']['cache12']}초, 중앙값 {timings['median_seconds']['cache12']:.2f}초.",
                  f"- 중앙값 {timings['median_reduction_percent']:+.3f}% 단축, 각 교차 쌍 {paired}%. 사전 10% gate 통과={timings['ten_percent_gate_pass']}."]
    lines += [f'- GPU7 복수 compute process 감지: {len(interference)} frames.',
              '- 기존 exhaustive CPU 탐색 부하가 있는 공유 서버다. 단회 과거 364.28초와 직접 비교하지 않는다. 이 부하 조건의 반복 결과이며 깨끗한 서버/다른 GPU에서 같은 속도를 보증하지 않는다.', '', '## test10', '']
    for key, item in tests.items():
        lines.append(f"- {key}: base 대비 마스크 일치 {100*item['agreement']:.7f}% ({item['changed_pixels']:,}/{item['pixels']:,}픽셀 변화). 정답이 없어 정확도·test IoU가 아니다.")
    lines += ['', '## 상태', '',
              '새 입력 캐시는 이미지별 로컬 객체라 다른 영상/전처리 설정 사이 공유하지 않는다. 모델 학습과 valid 계수/fold 재선택은 없다. base12가 가진 valid 선택 편향·일반화 미확인·학습 이력/OOF 한계는 그대로다. 실험 구현을 별도 보관하며 원래 base/릴리스를 덮어쓰지 않는다.', '']
    (ROOT / 'report.md').write_text('\n'.join(lines))
    files = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix not in ['.pt', '.pyc']
             and 'probabilities' not in p.parts and '__pycache__' not in p.parts
             and p.name not in ['gpu7.lock', 'results_archive.tar.gz', 'archive_receipt.json']]
    archive = ROOT / 'results_archive.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for p in sorted(files): stream.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    write(ROOT / 'archive_receipt.json', {'files': len(files), 'bytes': archive.stat().st_size,
                                        'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()})
    print(json.dumps({'timings': timings, 'raw_miou': runs['cache12_valid20']['miou_percent'], 'test': tests}, indent=2))


if __name__ == '__main__': main()
