# 준비·검증 결과 — 2026-09-29

실제 train70으로 새 B4 모델을 학습하지 않았습니다. 사용자가 `run.sh`로 시작할 수 있도록 준비했습니다.
T 모델의 초기화는 ImageNet encoder + seed 32로 새로 만든 decoder입니다. 원본 seed32 segmentation checkpoint는
R 모델의 추론에만 사용합니다. 학습 모델 10개와 추론 비교 3개를 구성했습니다.

## 확인한 항목

| 요구사항 | 실행한 검사 / 결과 |
|---|---|
| 하나의 변경 요소 | 각 arm과 명시한 대조군의 config 차이가 정확히 한 key인지 검사; 잘못된 복합 변경을 거부하는 테스트 통과 |
| 새 모델 학습 | ImageNet 경로만 T 초기화에 허용; T00/T05의 기본 가중치가 동일하고, 새 decoder가 학습된 seed32 decoder와 다른 것을 검사 |
| 8 GPU DDP | RTX4090 8개에서 10개 구조 모두 실제 crop1536, GPU당 1장, 두 번 연속 forward/backward 통과 |
| 학습 메모리 점검 | 합성 forward/backward 검사 최대 7,773 MiB/GPU; optimizer state를 포함한 전체 장기 학습의 메모리 실측은 아님 |
| 공통 batch Dice | DDP의 미분 가능한 전체 배치 CE/Dice gradient를 단일 직접 계산과 비교해 일치 확인 |
| 중단·재개 | 실제 trainer 함수를 작은 합성 CNN에 적용: 2 updates 후 중단, 4까지 재개. 연속 실행과 모든 parameter/BN buffer가 비트 단위로 동일 |
| 샘플 일관성 | index 기반 샘플링·rank 분할·spawn DataLoader·원본/확대 라벨 좌표 정렬 테스트 통과 |
| 순차 실행 | 실제 runner로 R00 → R01 → R02를 각각 8 GPU에서 완료; launcher의 비중첩·실패 전파 테스트 통과 |
| 완료 실험 재사용 | R suite 재개 시 세 결과의 파일/가중치 hash 확인 후 모두 건너뜀 |
| 평가 지표 | 사용자 예시의 0.8/0.8/0.8/0 → 0.6 검증; 없는 클래스 포함 4개 고정 평균; Global mIoU 미계산 |
| 결과 그림 | 각 R 결과마다 valid20의 원본 크기 2×2 비교 그림 생성. 오류 색·패널 위치를 픽셀 단위로 검사하고 실제 그림도 확인 |
| 기존 결과 보존 | baseline source/checkpoint/ImageNet 초기 가중치 등 22개 파일의 SHA-256을 기록하고 재확인 |
| 환경 | conda komap; torch2.7.1+cu128, SMP0.5.0, timm1.0.29, OpenCV4.13.0.92, Albumentations2.0.8 |

CPU 계약·데이터·모델·결과·실행기 테스트: **19 passed**. [실행 로그](checks/pytest_final.log)

```bash
conda run -n komap --no-capture-output python -B -m pytest \
  -c experiments/seed32_boundary_ablation_v1/pytest.ini \
  experiments/seed32_boundary_ablation_v1/tests -q
```

GPU 계산 검증 자료:

- [모든 구조의 8 GPU forward/backward 기록](checks/gpu_backward_01/backward.json)
- [실제 trainer의 합성 CNN 중단·재개 검사](checks/trainer_resume_01/verification.json)
- [저장된 예측 마스크 60개 독립 재계산](checks/mask_recalculation.json)
- [완료된 R 실험 재사용 로그](checks/reference_resume.log)

첫 GPU 검사에서 NCCL의 암묵적 device 선택 경고가 나와 이후 코드에는 `device_id`를 명시했습니다.
그 코드로 실제 R suite와 합성 trainer의 8 GPU 실행을 완료했습니다.

## 학습 없이 얻은 추론 비교

기존 seed32 checkpoint를 그대로 사용한 valid20 평가입니다. 저장한 RGB 예측 마스크를 다시 읽어
클래스별 intersection/union으로 독립 재계산했으며, 세 결과의 60개 image mIoU가 모두 일치했습니다.

| 추론 설정 | image mIoU, empty=0 | 기존 D4+area 대비 |
|---|---:|---:|
| R00: D4 + area 복원 | 78.9765456654% | 기준값, 기존 점수 정확히 재현 |
| R01: TTA 제거 + area | 78.7298050244% | −0.2467406410%p |
| R02: D4 + bilinear 복원 | 79.1478089690% | +0.1712633036%p |

원본 기준 1픽셀 경계 띠의 이미지별 평균 오류율도 R00의 12.751399%에서 R02의 12.552253%로 낮아졌습니다.
T 모델의 기본 area 복원은 변경하지 않았습니다. 추론 개선을 새 학습 구조의 이득으로 섞지 않기 위함입니다.

- [전체 R 비교표](outputs/20260929T110331_662976Z/comparison.md)
- [R00 결과](/Data3/sppark/komap/results/model_results_0929_3/model_results_0929_3.md)
- [R01 결과](/Data3/sppark/komap/results/model_results_0929_4/model_results_0929_4.md)
- [R02 결과](/Data3/sppark/komap/results/model_results_0929_5/model_results_0929_5.md)

## 실측 범위

아직 실행하지 않은 것은 train70을 사용한 T00–T09의 16,000-update 학습과 그 모델들의 최종 valid20 평가입니다.
성능 개선을 검증한 것은 위 R 추론 비교뿐입니다. 합성 B4 검사에서는 optimizer step을 실행하지 않았습니다.
optimizer·scheduler·checkpoint 재개 검사는 실제 사진이나 B4 대신 별도의 작은 합성 CNN으로 수행했습니다.

새 공통 유효 배치는 8 / SyncBatchNorm이며 원본 seed32의 배치2와 다릅니다.
모든 T 실험이 같은 조건을 사용하고, ablation 효과는 지정된 T 대조군끼리 비교합니다.
한 seed·같은 valid20을 사용한 개발 실험이므로 서로 다른 seed에서의 안정성이나 독립 테스트 성능까지 확인한 결과는 아닙니다.
