# HRNet_taeho 6-GPU 순차 실행 검증

- 검증일: 2026-09-28 UTC.
- 실제 데이터 학습을 실행하지 않았고 optimizer update는 0회입니다. 합성 입력만 사용한 GPU 점검을 수행했습니다.
- 기존 두 실험의 81개 파일: 변경 전후 SHA-256 일치. 기존 모델·결과 경로를 변경하지 않았습니다.
- 원본 18개 파일의 사본 SHA-256 검사 통과.
- preflight: train 70 / valid 20 / test 10 데이터·마스크·팔레트 검사 통과.
- 단위·통합 테스트: **10 passed** (`reports/tests.log`). 전체 batch 손실/gradient, 증강 차이, batch 분배, GPU 번호 검사, 순차 실행/실패 중단, 결과 모델명 검사 포함.

## 실제 GPU 0~5의 분산 검증

- DDP + SyncBatchNorm, global batch 8, local batch [2, 2, 1, 1, 1, 1].
- 작은 검증 모델에서 단일 GPU 전체 batch와 6-GPU 분산 결과 비교: 손실·gradient·BatchNorm running statistics 일치 검사 통과.
- 검증 대상: 전부 같은 클래스인 마스크, 띠 모양 경계 마스크, 랜덤 4클래스 마스크.
- 최대 손실 절대 차이: 2.38418579e-07. 최대 parameter gradient 절대 차이: 4.47034836e-08.
- 수학적 일치 검사는 batch 크기에 따른 TF32 커널 차이를 분리하기 위해 TF32를 비활성화했습니다. 실제 학습의 기본 CUDA 설정은 기존 코드와 같습니다. 단일 GPU와 비트 단위로 동일한 학습 결과를 보장하지 않습니다.
- 기록: `reports/distributed_equivalence.json`.

## 실제 HRNet 구조 및 데이터 전달

- 파라미터 76,964,949개, FP32, global batch 8 × 3 × 512 × 512.
- 6개 GPU에서 순전파·역전파 두 차례 완료. DDP reducer의 다음 iteration도 정상 동작.
- 학습과 동일한 4-worker spawn DataLoader 및 전체 batch 전달 함수를 합성 TensorDataset으로 확인.
- 모든 rank의 손실·gradient 유한값. optimizer를 만들거나 가중치를 갱신하지 않았습니다.
- 아래 메모리는 이 합성 순전파·역전파의 최대 할당량이며 AdamW 상태는 포함하지 않습니다.

| GPU | local batch | 최대 할당 메모리 (GiB) |
|---|---:|---:|
| 0 | 2 | 4.5226 |
| 1 | 2 | 4.5226 |
| 2 | 1 | 2.5639 |
| 3 | 1 | 2.5639 |
| 4 | 1 | 2.5639 |
| 5 | 1 | 2.5639 |

- 기록: `reports/smoke.json`.

## 평가 및 결과 파일

- 임시 폴더의 합성 이미지 20장을 6개 GPU에 분담하여 원본 타일/flip TTA 및 결과 출력 경로 전체를 확인.
- 이미지 누락/중복 없음, 원래 H×W 보존, RGB 예측 마스크 20장과 2×2 비교 그림 20장 및 보고서 생성 확인.
- 테스트 파일은 임시 경로에서 생성 후 제거했습니다. 로그의 합성 점수 100%는 실제 HRNet 성능이 아닙니다.
- 기록: `reports/distributed_evaluation.json`.
- `run_sequence.sh`의 모의 실행으로 baseline 완료 후 no_gaussian 실행 및 첫 작업 실패 시 다음 작업 미실행을 확인.
- 실행 스크립트 Bash 구문 검사와 Python AST 검사 통과. 학습 시작 명령은 `README.md` 참조.

실제 학습 완료 후 각 모델의 image mIoU와 결과 그림은 `results/model_results_0928_<빈 번호>/`에 기록됩니다.
