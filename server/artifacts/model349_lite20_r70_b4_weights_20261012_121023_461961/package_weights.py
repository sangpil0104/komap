"""Build three separately verified weight ZIPs; never edit training artifacts."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
REPO = Path('/Data3/sppark/komap')
SUMMARY = REPO / 'results/model_results_20261011_3/summary.json'
SUMMARY_DATA = json.loads(SUMMARY.read_text())
RUN = Path(SUMMARY_DATA['r70_run'])
DIRECTORIES = {
    '322': 'model322_team_b4_group_oof_full70',
    '326': 'model326_hybrid_fullstats_group_oof_full70',
    '329': 'model329_fullstats_width4000_uniform_group_oof_full70',
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def build(model):
    receipt = json.loads((RUN / f'completed/train_{model}.json').read_text())
    checkpoint = Path(receipt['checkpoint'])
    args = checkpoint.with_name('args.json')
    sources = {'last.pt': checkpoint, 'args.json': args}
    hashes = {name: sha(path) for name, path in sources.items()}
    for name, path in sources.items():
        assert hashes[name] == receipt['artifacts'][str(path)], path
    for candidate in SUMMARY_DATA['candidates']:
        deployed = Path(candidate['package']) / 'Model349_exh14/weights' / DIRECTORIES[model] / 'last.pt'
        assert sha(deployed) == hashes['last.pt'], deployed
    config = json.loads(args.read_text())
    assert config['seed'] == 0 and config['iters'] == 21000
    stem = f'Model349_lite20_r70_B4_{model}'
    filename = stem + '_weights.zip'
    manifest = dict(
        model=stem, seed=0, training_images=70, iterations=21000,
        checkpoint='last.pt', checkpoint_policy='final iteration',
        architecture=config['arch'], encoder=config['encoder'],
        input=config['input'], normalization=config['norm'],
        scale=config['scale'], target_width=config['target_width'], sampling=config['sample'],
        source_run=str(RUN), source_checkpoint=str(checkpoint),
        source_result='results/model_results_20261011_3/model_results_20261011_3.md',
        shared_across_hr336_seeds=list(range(3407, 3415)),
        deployed_weight_path=f'Model349_exh14/weights/{DIRECTORIES[model]}/last.pt',
        files={name: dict(bytes=path.stat().st_size, sha256=hashes[name]) for name, path in sources.items()},
        note='Single R-component B4 weight; requires the matching model code and other ensemble weights.',
    )
    readme = f'''# Model349 lite20 r70 — B4-{model}

train70 전체로 재학습한 B4-{model}의 최종 가중치입니다.
HR336 seed 3407~3414의 B4 교체 평가에 공통으로 사용한 동일한 가중치입니다.
B4 자체의 학습 seed는 0이며, 21,000 iterations의 마지막 checkpoint입니다.

- `last.pt`: 원본 학습 가중치 (SHA256 검증 완료)
- `args.json`: 원본 전처리/학습 설정 (로컬 경로는 학습 당시 기록)
- `manifest.json`: 출처, 적용 경로, SHA256
- `SHA256SUMS.txt`: `last.pt`, `args.json` 무결성 확인

구성: UNet / efficientnet-b4 / input={config['input']} / norm={config['norm']} / scale={config['scale']} / target_width={config['target_width']} / sampling={config['sample']}.
앙상블 적용 위치: `Model349_exh14/weights/{DIRECTORIES[model]}/last.pt`.
이 ZIP은 단일 구성 모델의 가중치와 설정입니다. 전체 R/E/F 앙상블의 나머지 가중치와 실행 코드는 별도로 필요합니다.
8개 HR336 seed의 점수는 앙상블 결과이며, 이 B4 한 개의 단독 점수가 아닙니다.

일반 ZIP(Deflate) 형식입니다. 압축 해제 후 이 폴더에서 `sha256sum -c SHA256SUMS.txt`로 확인할 수 있습니다.
'''
    with ZipFile(ROOT / filename, 'x', compression=ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for name, path in sources.items():
            z.write(path, f'{stem}/{name}')
        z.writestr(f'{stem}/README.md', readme)
        z.writestr(f'{stem}/manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
        z.writestr(f'{stem}/SHA256SUMS.txt', ''.join(f'{value}  {name}\n' for name, value in hashes.items()))
    with ZipFile(ROOT / filename) as z:
        assert z.testzip() is None
        for name, expected in hashes.items():
            with z.open(f'{stem}/{name}') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
    result = dict(model=model, filename=filename, bytes=(ROOT/filename).stat().st_size,
                  sha256=sha(ROOT/filename), original_weight_sha256=hashes['last.pt'],
                  matches_all_eight_seed_packages=True, archive_verified=True)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(build, DIRECTORIES))
    (ROOT/'archives.json').write_text(json.dumps(results, indent=2)+'\n')
    (ROOT/'Model349_lite20_r70_B4_SHA256SUMS.txt').write_text(''.join(f'{r["sha256"]}  {r["filename"]}\n' for r in results))
    rows = '\n'.join(f'| B4-{r["model"]} | [{r["filename"]}]({r["filename"]}) | {r["bytes"]/1024**2:.2f} MiB |' for r in results)
    (ROOT/'Model349_lite20_r70_B4_weights_README.md').write_text('''# Model349 lite20 r70 — full-train70 B4 가중치 3개

기존 HR336 seed 3407~3414의 B4 교체 평가에서 공통으로 사용한 B4-322/326/329 가중치입니다.
각 B4의 학습 seed는 **0**, 학습 데이터는 **train70 전체**, 학습 길이는 **21,000 iterations**입니다.
각 ZIP은 마지막 checkpoint `last.pt`, 원본 `args.json`, 설명 및 SHA256을 포함합니다.
세 파일 모두 원본 완료 기록 및 8개 시드 평가 패키지의 가중치와 SHA256이 일치합니다.

| 모델 | 다운로드 파일 | 크기 |
|---|---|---:|
'''+rows+'''

Git LFS로 저장됩니다. GitHub 파일 페이지에서 Download raw file을 누르거나 `git lfs pull`로 받으세요.
ZIP 형식은 일반 Deflate입니다. 체크섬은 `Model349_lite20_r70_B4_SHA256SUMS.txt`에 있습니다.

| 모델 | 입력 | 정규화 | 크기 처리 | 샘플링 |
|---|---|---|---|---|
| B4-322 | gray3 | imagenet | scale 3 | area |
| B4-326 | zcdf3 | image_global | scale 3 | area |
| B4-329 | zcdf3 | image_global | target width 4000 | uniform |

이는 R 구성의 full70 B4 가중치 3개이며, 전체 R/E/F 앙상블과 HR336 가중치는 별도입니다.
8개 시드 평가의 최고 R+E+F image mIoU 79.899812%는 HR336 seed 3412와 나머지 구성 모델을 결합한 점수입니다.
개별 B4 단독 점수를 의미하지 않습니다. seed 탐색 결과는 valid20에서 선택한 개발용 결과입니다.
''')
