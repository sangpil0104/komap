# 준비 검증 — 2026-09-26

**본 학습/시험 학습은 실행하지 않았다.**

- 실제 직전 실행의 학습/추론/진행 코드 사본과 SHA256 일치.
- 10개 기존 job 설정과 새 base_config 일치. 변경 대상은 seed와 실행 결과 경로뿐.
- 동일 Python 패키지 버전, train70/valid20 파일/팔레트/크기/SHA256 확인.
- 동일 ImageNet 인코더 SHA256 확인. CPU strict load 및 유한한 forward 출력 확인.
- GPU 0–5 각 RTX 4090, 여유 메모리 조건 통과.
- 기존 소스/모델/개별 결과 472개 경로의 SHA256을 preservation.json에 기록/확인.
- 실제 결과 준비 계획: 기존 10개 보존, 추가 90개(10–99), 초기 ETA 15:11:31.
- Python/shell 구문 확인.
- **4 tests passed in 17.47s.**

테스트는 학습 없이 수행했다.

1. 실제 직전 학습 소스/설정 일치, 10–99 범위와 과거 실측 기반 ETA.
2. 저장된 예측을 임시 폴더에서 렌더링해 기존 수정 이미지와 전 픽셀 일치 확인.
   Error rate 수식과 비교 패널 검증.
3. 기존 시드 결과 누락 시 거부, 기존 metrics 덮어쓰기 거부.
4. 임시 폴더의 가짜 subprocess **90개**로 GPU 0–5 배정/동시 점유 방지/통합 로그,
   기존 10개와 합산, 하나 실패 후 그 시드만 재시도, 최종 100개 보고서 생성 확인.
   기존 개별 결과 파일의 내용 유지 및 부분 결과 백업 확인.

검증 명령:

```bash
NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  experiments/seonghyeon_seed10_v1/.venv/bin/python -B -m pytest -q \
  -p no:cacheprovider experiments/seonghyeon_seed10_99_v1/tests
```

실제 학습 시간/90개 시드의 성능은 사용자가 시작한 뒤 확인한다.
