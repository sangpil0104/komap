# 추가 탐색 실행·재현 안내

이 폴더는 기존 `Model349_lite12_base`와 분리된 실험이다. 채택 여부는 `report.md`/`comparison.json`의 공식 품질과 반복 시간 조건으로 판단한다. 완료 전 로그나 단회 값으로 채택하지 않는다.

서버 위치: `/workspace/model349_compress/base12_resident_packed_20261008`.

- `protocol.md`: 실행 전 후보/선택/판정 조건.
- `run_experiment.py`: GPU7 점유/잠금 확인 → 후보 준비 → 상태 복원 probe → 원본 valid 후보 → 조건부 무손실 결합 → 선택 후보와 기준 각각 3회 교차 추론 → test10.
- `probe.py`: 실제 GPU에서 source 전환 및 고의 변조 후 전체 원본 parameter/buffer/grad/eval 복원 확인.
- `finalize.py`: 완료된 실행의 PNG/GT 독립 채점, 클래스·이미지별 손실, component/전체 시간, 부하/복원/저장 감사, 결과 아카이브.
- `bundles/*/source_hashes.json` 및 `checkpoint_manifest.json`: 실행한 소스·설정 및 실제 가중치 SHA/바이트. `soup_provenance.json`은 합성 출처, `lossless_storage_verification.json`은 ZIP 및 tensor 동일성 기록.

컨트롤러는 기존 GPU7 잠금을 모두 확보한다. 이미 실행 중이면 또 실행하지 않는다. 원본 PDF 매뉴얼을 읽고 타 사용자 작업/가중치를 유지한다. 원래 폴더에서 전체 실행 스크립트를 다시 실행하면 기존 출력과 충돌하도록 설계했다. 재현하려면 새로운 실험 폴더 이름으로 복사하고 ROOT를 새 폴더로 지정한다.

단일 bundle의 추론은 아래처럼 별도 **새 출력 폴더**에 실행한다. GPU7 유휴와 작업 중복을 먼저 확인하며 다른 사용자의 프로세스를 종료하지 않는다.

```bash
cd /workspace/model349_compress/base12_resident_packed_20261008/bundles/<candidate>
/workspace/venv-komap/bin/python -B infer_lite.py \
  --input /workspace/komap_model1/data/valid/images \
  --gt /workspace/komap_model1/data/valid/masks \
  --output /workspace/model349_compress/<new-output-folder> \
  --gpus 7 --keep-probabilities
```

test는 input을 `/workspace/komap_model1/data/test/images`로 바꾸고 `--gt`를 생략한다. 점수가 아니라 기존 base 대비 마스크 일치율만 평가한다. 결과 archive에는 체크포인트 및 대용량 원본 확률을 포함하지 않는다. 실제 가중치/확률은 서버에 보존하며 원본을 덮어쓰지 않는다.

평가 데이터 100장 외 독립 정답 자료가 없다. 여기서 valid 평균이 높아져도 일반화 개선이 확정되는 것은 아니다. 합성 모델의 provenance와 무손실 저장 포맷 변경을 FP16 연산·적응 단계 변경과 구분한다.
