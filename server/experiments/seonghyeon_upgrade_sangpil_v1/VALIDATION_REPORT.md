# 검증 보고서

실제 데이터 학습은 시작하지 않았습니다. 새 모델의 성능 향상을 주장하지 않습니다.

| 검사 | 실측 결과 |
|---|---|
| 데이터/팔레트/짝 검사 | train70 / valid20 / test10 통과 |
| 기존100 checkpoint | SHA-256 기록, 최종 보존 검사 통과 |
| 기존 seed32 strict load | 통과 |
| valid20 재추론 image mIoU | 80.2265456654% |
| 기존 점수와 차이 | -2.220e-16 (0–1 단위) |
| A71 기존/새 어댑터 확률 최대 차이 | 0.0 |
| A71 기존/새 어댑터 argmax 불일치 | 0 pixels |
| A71 image mIoU | 71.1385026339% |
| CPU 수학·통합 검사 | 44 passed in 5.38s |
| 실제 B4 합성 입력 크기 | batch2, detail1536, context768 |
| 새 head0 초기 출력 차이 | 0.0 |
| GPU forward/backward 최대 메모리 | 18.0115 GiB |
| 합성 GPU smoke optimizer updates | 0 |
| 실제 seed32 확률 캐시/진단 | valid20 완료 |
| baseline test PNG | test10, 원본 H/W·4색 검사 통과 |
| 원본 보존 최종 검사 | 306 files, 변경0 |

CPU 검사에는 A71 fixture, 네 색상/unknown/ignore, empty union, image/global 평균 차이, 중복 이미지 ID 처리, D4, native 좌표 ramp/gradient, tile coverage, cache 손상/재개, K1/동일100-member 평균, train-only sampler·online hard refresh·resume, 실제 B4 분기 on/off·zero-init·BN freeze·native gradient, refiner zero-init, 합성 training loop의 중단/재개 일치, 합성 OOF teacher registry→cache→deployment→refiner 입력 연결과 오염 차단을 포함합니다.

GPU smoke는 새 texture/context/native loss를 모두 켠 실제 B4 구조에서 실행했습니다. 첫 backward에서 마지막 head에 gradient가 생기고, 테스트용 head를 메모리에서 미세하게 바꾼 다음 backward에서 texture/context에도 finite gradient가 전달됐습니다. 이 과정은 합성 입력이며 optimizer 업데이트나 가중치 저장을 하지 않았습니다. 학습 optimizer 상태와 실제 데이터 증강을 포함한 장시간 최고 메모리 측정은 아닙니다.

합성 resume 테스트는 작은 모델/가짜 PNG만 사용했습니다. 실제 프로젝트 데이터 학습을 실행한 것으로 보고하지 않습니다. 실제 train 패치 overfit 명령은 README에 준비했고 사용자가 시작하도록 남겨뒀습니다.

아직 실행하지 않은 항목: 새 모델4000/16000 update 학습, 실제 train-patch overfit, 전체100-model 앙상블 sweep, 실제 OOF teacher/refiner 학습, outer-fold 평가. 실제 OOF/refiner 가중치는 아직 없으며 학습 후 생성됩니다. 기본 test PNG는 기존 seed32 기준선입니다.

기록: `reports/pytest.log`, `reports/seed32_reproduction.json`, `reports/smoke_candidate_unvalidated_full.json`, `reports/seed32_baseline/`, `reports/preservation.json`, `outputs/submission/validation.json`.
