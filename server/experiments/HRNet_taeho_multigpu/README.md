# HRNet_taeho — GPU 0~5 사용 및 두 실험 순차 실행

**한 실험이 GPU 0~5를 모두 사용하며, 학습과 평가를 마치면 다음 실험을 시작합니다.** 기존 `HRNet_taeho/`와 `HRNet_taeho_no_gaussian/`의 소스·설정·모델·결과는 수정하지 않았습니다. 이 폴더에 분산 실행용 소스와 결과를 독립적으로 보존합니다. 생성일 태그는 기존과 같은 `0928`입니다.

## 사용자가 nohup으로 시작

```bash
cd /Data3/sppark/komap/experiments/HRNet_taeho_multigpu
./start_nohup.sh
```

실행 순서:

1. `HRNet_taeho`: GPU **0,1,2,3,4,5**를 사용해 70 epoch 학습 → valid 20장 평가 → 보고서 저장.
2. 첫 작업이 성공하면 `HRNet_taeho_no_gaussian`: 같은 GPU **0,1,2,3,4,5**로 70 epoch 학습 → 평가 → 보고서 저장.

두 번째 실험에서 Gaussian noise와 Gaussian blur 두 증강만 제거합니다. 첫 작업이 실패하면 다음 실험을 시작하지 않습니다. GPU 6·7은 사용하지 않습니다. 학습 시작은 사용자가 하며, 준비 과정에서는 실제 학습을 실행하지 않았습니다.

`start_nohup.sh`는 내부에서 `nohup ./run_sequence.sh`를 실행하고 PID·로그 경로·`tail -f` 명령을 출력합니다. 로그에서 epoch·전체 update 진행률·경과 시간·예상 남은 시간을 확인합니다. 시작 없이 순서만 확인하려면:

```bash
./run_sequence.sh --dry-run
```

GPU 목록을 바꾸려면 `HRNET_GPUS`를 지정합니다. 예: `HRNET_GPUS=0,1,2 ./start_nohup.sh`. 0~5 중 중복 없는 번호만 허용합니다.

## 학습 설정 유지 방법

| 항목 | 값 |
|---|---|
| 모델 | HRNet-W48 + U-Net++ + 경계 보조 head |
| 총 batch size | **8**, GPU 수와 무관하게 유지 |
| GPU 0~5의 local batch | **2, 2, 1, 1, 1, 1** |
| 데이터 생성 | GPU 0 프로세스의 기존 Dataset/DataLoader가 총 8장을 생성하고 각 GPU에 분배 |
| BatchNorm | SyncBatchNorm으로 6개 GPU 전체 8장의 통계를 합산 |
| 손실 | 전체 batch의 CE·Dice·경계 BCE·경계 Dice 통계를 합산 후 계산 |
| gradient | DDP로 동기화. 불균등한 local batch도 전체 batch 기준 가중치 유지 |
| 학습률·최적화 | lr 3e-4, AdamW, weight decay 1e-4, cosine, FP32 |
| 학습량 | seed 0, 70 epoch, epoch당 105 update, 총 7,350 update |
| 최종 모델 | 원본과 동일하게 마지막 epoch의 `last.pt` |
| 추론 | 원본의 2배 확대 타일·Gaussian 병합·4종 flip TTA 유지 |
| 검증 분배 | 이미지 20장을 GPU별로 분담하고 각 이미지의 mIoU를 같은 가중치로 평균 |

분산 손실은 각 GPU의 Dice를 따로 평균하는 방식이 아니라 전체 batch의 통계를 먼저 합쳐 계산합니다. BatchNorm 통계도 전체 batch 기준으로 동기화하여 GPU별 작은 batch 때문에 학습 방식이 달라지는 것을 방지합니다.

부동소수점 합산 순서와 CUDA 커널 선택 등이 달라질 수 있으므로 단일 GPU와 비트 단위로 같은 학습 결과를 보장하지는 않습니다. 6장을 사용해도 통신 비용 때문에 속도가 6배가 되지는 않습니다. 실제 완료 예상 시간은 학습 로그의 tqdm 표시로 확인하세요.

[PyTorch DDP 문서](https://docs.pytorch.org/docs/2.7/generated/torch.nn.parallel.DistributedDataParallel.html)의 프로세스당 GPU 1장 방식으로 구성했습니다. 설치된 PyTorch의 SyncBatchNorm 구현이 서로 다른 local batch 크기를 처리하는 것도 확인했습니다. NCCL과 DataLoader의 조합에 맞춰 DataLoader worker는 `spawn` 방식으로 시작합니다.

## 저장 위치

```text
experiments/HRNet_taeho_multigpu/
  configs/baseline.json
  configs/no_gaussian.json
  source/                     # 기존 원본 18개 파일 복사본
  distributed_core.py         # batch 분배, 전체 batch 손실, 증강 변형
  worker.py                   # 분산 학습·평가
  outputs/
    logs/                     # 두 실험을 연속 실행하는 nohup 로그
    latest_baseline.json      # baseline 최근 실행 위치·상태
    latest_no_gaussian.json    # noise/blur 제거 실험 최근 실행 위치·상태
    runs/<모델명>_0928_ddp6_<UTC timestamp>/
      last.pt
      config.json
      execution.json          # GPU 목록, global/local batch, 실험 종류
      preflight.json
      source/ / adapter/      # 실제 사용한 소스 스냅샷
      history.jsonl
      status.json
      completed.json          # 학습·평가 완료 및 결과 위치
```

각 실행은 새 경로를 사용합니다. 학습된 checkpoint는 기본 모델의 state_dict 형식으로 저장하므로 `module.` 접두사가 붙지 않습니다. 기존 방식과 같이 optimizer/scheduler 상태를 저장하지 않으므로 정확한 중단 재개 기능은 없습니다.

최종 결과는 각각 `results/model_results_0928_<빈 번호>/`에 저장합니다. MD 제목과 `summary.json`의 모델명으로 두 실험을 구분하며, `execution` 항목에서 GPU 실행 정보를 확인합니다. 기존 결과 폴더를 덮어쓰지 않습니다.

- image mIoU: 각 이미지의 4개 클래스 IoU 평균 → 20장 평균. 정답/예측 양쪽에 없는 클래스 IoU=1.
- 비교 그림: 원본 / 오류 / 예측 / 정답의 2×2 배치. 맞힌 픽셀은 흰색, 틀린 픽셀은 정답 색. image mIoU 및 `100% − image mIoU` 표시.
- 학습 중 `quick_native1x_image_miou`는 원본 크기에서 한 번에 추론한 진행 참고값으로, 최종 타일+TTA 평가와 조건이 다릅니다.

## 개별 실행과 점검

한 실험만 6개 GPU에서 실행하려면:

```bash
./run_distributed.sh train --variant baseline --execute
./run_distributed.sh train --variant no_gaussian --execute
```

위 명령은 터미널에서 직접 실행합니다. 둘을 nohup으로 연속 실행하려면 상단의 `./start_nohup.sh`를 사용합니다.

```bash
./run_distributed.sh train --variant baseline  # 설정 확인, 학습 시작 안 함
./run_distributed.sh preflight --variant baseline
./run_distributed.sh evaluate --variant baseline --checkpoint outputs/runs/<실행 폴더>/last.pt
```

설정을 바꾸어 학습했다면 `--config outputs/runs/<실행 폴더>/config.json`도 지정합니다. noise/blur 제거 모델은 `--variant no_gaussian`을 지정합니다.

기존 `../HRNet_taeho/.venv/bin/python` 환경을 공유합니다. 다른 Python 경로가 필요하면 `HRNET_PYTHON`으로 지정합니다. 환경 패키지와 기존 실험 코드를 변경하지 않습니다.

검증 기록: [VALIDATION_REPORT.md](VALIDATION_REPORT.md). `check-loss`, `smoke`, `check-evaluation` 명령은 합성 데이터로 검사하며 optimizer update를 수행하지 않습니다.
