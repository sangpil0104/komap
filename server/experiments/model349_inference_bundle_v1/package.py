"""Finalize the audited Model349 inference bundle and make a verified ZIP64."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from tqdm import tqdm

ROOT = Path('/Data3/sppark/komap')
HERE = Path(__file__).resolve().parent
BUNDLE = ROOT / 'results/model349_inference_bundle_20261007_1'
RUN = ROOT / 'experiments/model349_train_route_v2/outputs/runs/model349_train_20260930T143930_234475Z'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    assert 'PASS: bundle hashes/imports and all 23 checkpoint loads' in (HERE / 'verify.log').read_text()
    cached = json.loads((HERE / 'checks/cached_parity.json').read_text())
    fresh = json.loads((HERE / 'checks/fresh_parity.json').read_text())
    sources = json.loads((BUNDLE / 'provenance/source_manifest.json').read_text())
    weights = json.loads((BUNDLE / 'provenance/weights_manifest.json').read_text())
    for row in tqdm(sources, desc='Original source preservation', unit='file'):
        assert digest(RUN / 'source_snapshot' / row['source']) == row['source_sha256']
        assert digest(BUNDLE / 'src' / (row['module'].replace('.', '/') + '.py')) == row['derived_sha256']
    for row in tqdm(weights, desc='Original checkpoint preservation', unit='file'):
        assert digest(RUN / row['original_relative_path']) == row['sha256']
        assert digest(BUNDLE / row['path']) == row['sha256']
    verification = {'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                    'environment': 'Conda komap, Linux x86_64, RTX 4090',
                    'source_imports': {'module_count': len(sources), 'passed': True},
                    'strict_checkpoint_loads': {'checkpoint_count': len(weights), 'passed': True},
                    'pip_check': 'No broken requirements found.',
                    **cached, 'fresh_inference': fresh, 'original_source_and_weights_preserved': True,
                    'fresh_environment_install_tested': False,
                    'verification_scope': '20 valid cached probability routes; 10 test and 1 valid exported probability routes; 1 image freshly inferred from all component weights; no training launched'}
    (BUNDLE / 'provenance/verification.json').write_text(json.dumps(verification, indent=2, ensure_ascii=False) + '\n')
    md = '''# 패키지 검증 결과

| 확인 항목 | 결과 |
|---|---|
| 추론 의존 모듈 27개 import | 모두 통과 |
| 원본 가중치 23개 strict loading | 모두 통과 |
| 기존 검증 20장 저장 확률 → 새 코드의 최종 결합 | 원래 마스크와 모두 픽셀 단위 일치 |
| 기존 test 10장 + valid 1장 저장 확률 → 새 코드의 최종 결합 | 기존 내보내기 마스크와 모두 일치 |
| 새 패키지로 valid 1장 실제 재추론 | 구성 요소 7개의 확률 배열과 최종 마스크 모두 완전 일치 |
| 재추론 입력 | A117_step4_sample2_image.png |
| 프로젝트 외부 작업 디렉터리에서 실행 | /tmp에서 run.sh 실행 통과 |
| 원본 소스·모델 가중치 보존 | SHA256 일치 |
| 현재 환경 pip check | No broken requirements found. |
| 새 빈 Conda 환경 설치 | 수행하지 않음; 환경 파일은 실제 설치된 버전에서 추출 |

20장 전체 재학습·재추론을 수행한 검증과 혼동하지 마세요. 전체 20장은 기존 저장 확률의 최종 결합을 확인했고, 체크포인트에서 시작하는 전체 추론은 대표 이미지 1장에 대해 확인했습니다. 자세한 수치와 항목별 결과는 `verification.json`에 있습니다. 검증용 입력·정답·기존 확률 캐시는 패키지에 포함하지 않았습니다.
'''
    (BUNDLE / 'provenance/VERIFICATION.md').write_text(md)
    files = sorted(p for p in BUNDLE.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt')
    for p in files:
        assert not p.is_symlink(), p
        assert '__pycache__' not in p.parts and p.suffix not in ['.pyc', '.npz', '.npy', '.pkl', '.png'], p
    hashes = {str(p.relative_to(BUNDLE)): digest(p) for p in tqdm(files, desc='Final file hashes', unit='file')}
    (BUNDLE / 'SHA256SUMS.txt').write_text('\n'.join(f'{h}  {name}' for name, h in hashes.items()) + '\n')
    files.append(BUNDLE / 'SHA256SUMS.txt')
    hashes['SHA256SUMS.txt'] = digest(BUNDLE / 'SHA256SUMS.txt')
    target = BUNDLE.with_suffix('.zip')
    total_bytes = sum(p.stat().st_size for p in files)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        with tqdm(total=total_bytes, desc='Create Model349 ZIP', unit='B', unit_scale=True) as bar:
            for p in files:
                info = zipfile.ZipInfo.from_file(p, arcname=BUNDLE.name + '/' + str(p.relative_to(BUNDLE)))
                info.compress_type = zipfile.ZIP_DEFLATED
                info._compresslevel = 1
                with p.open('rb') as source, z.open(info, 'w', force_zip64=True) as dest:
                    for b in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        dest.write(b)
                        bar.update(len(b))
    # Reading every member verifies ZIP CRC and independently verifies SHA256.
    with zipfile.ZipFile(target) as z:
        assert len(z.infolist()) == len(files)
        with tqdm(total=total_bytes, desc='Verify ZIP contents', unit='B', unit_scale=True) as bar:
            for info in z.infolist():
                relative = info.filename.removeprefix(BUNDLE.name + '/')
                h = hashlib.sha256()
                with z.open(info) as f:
                    for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
                        h.update(b)
                        bar.update(len(b))
                assert h.hexdigest() == hashes[relative], info.filename
    result = {'zip': str(target), 'bytes': target.stat().st_size, 'sha256': digest(target),
              'files': len(files), 'checkpoints': len(weights), 'zip_crc_and_sha256_verified': True}
    (HERE / 'package_result.json').write_text(json.dumps(result, indent=2) + '\n')
    target.with_suffix('.zip.sha256').write_text(result['sha256'] + '  ' + target.name + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
