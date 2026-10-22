# 준비 및 검증 기록

2026-09-26 UTC. **본 학습과 시험 학습은 실행하지 않았다.**

- 원본 `seonghyeon_v1` 파일 20개의 SHA256 유지, vendor 사본 4개 바이트 일치.
- 별도 `.venv` 설치 및 `pip check` 통과. 기존 komap 환경 패키지는 변경하지 않음.
- EfficientNet-B4 ImageNet 인코더를 CPU에서 준비하고 SHA256 기록.
- train 70 / valid 20의 파일 수, 마스크 색상, 이미지/마스크 크기 및 SHA256 확인.
- 물리 GPU 0–5: RTX 4090, 각각 24 GB, 준비 검사 시 여유 메모리 15 GB 이상.
- 모델의 CPU strict weight load, 64×64 입력 forward의 출력 (1,4,64,64), 유한값 확인.
- Python 구문 및 두 shell 스크립트 `bash -n` 통과.
- 실행 계획 확인: seed 0–9, GPU 0–5, 병렬 6, 시드당 16,000회.
- 테스트 **6 passed in 3.14s**. GPU 학습 없이 수행.

검증 범위:

1. 이미지별 지표 평균, 4클래스 고정 분모, 빈 클래스와 오검출 처리.
2. 저장된 PNG의 원본/예측/정답 패널 전 픽셀 및 오류 패널의 흰색/정답 색.
3. 병렬 6개와 대기 4개를 고려하는 ETA, 최종 D4 진행률 반영.
4. 기존 결과를 덮어쓰지 않는 날짜/번호 폴더 할당.
5. 실제 DataLoader 2개 worker의 같은 시드 재현 및 다른 시드의 증강 차이.
6. 가짜 subprocess 10개로 GPU 0–5 배정, GPU 중복 점유 방지, 한 작업 실패 후
   나머지 9개 완료, 통합/개별 로그, 실패 종료 코드, 비교 그림 누락 시 완료 거부.

```bash
NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  experiments/seonghyeon_seed10_v1/.venv/bin/python -B -m pytest -q \
  -p no:cacheprovider experiments/seonghyeon_seed10_v1/tests
```

GPU backward·전체 16,000회·최종 200장 생성은 사용자가 실행할 본 학습에서 수행한다.
실제 학습 시간과 최종 성능은 아직 측정하지 않았다. 재개 체크포인트 기능은 제공하지 않는다.
실험 소스와 환경, 변경된 worker RNG 정책 때문에 이전 서버 결과와 비트 단위 재현을 주장하지 않는다.
