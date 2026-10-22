# ResNet18 자기지도학습 A·B·C 실험

원본 조직 이미지의 일부를 가리고 복원하는 자기지도 사전학습이 **4상 분할과 테두리 오류에 도움이 되는지** 비교합니다. 최종 추론 모델은 각각 하나의 ResNet18 U-Net입니다. 기존 모델·학습 소스·결과를 수정하지 않고 이 디렉터리에서 독립적으로 실행합니다.

## 비교 구성

| 실험 | 초기값과 학습 | 비교 목적 |
|---|---|---|
| A | ImageNet ResNet18 → 지도학습 4,000 updates | 기준 모델 |
| B | A와 같은 초기값 → 원본 가림 복원 2,000 updates → encoder만 이전 → 지도학습 4,000 updates | 자기지도 사전학습 효과 |
| C | A의 마지막 모델·optimizer에서 추가 지도학습 | B의 추가 계산량에 대한 대조군 |

A와 B의 분할 decoder·head 초기값, 분할 학습 패치 순서, D4 증강, optimizer 설정, update 수, 유효 배치가 같습니다. B에서 사전학습으로 바뀌는 부분은 encoder뿐입니다. 사전학습용 복원 decoder·head는 분할 단계에서 버립니다.

C는 A의 학습 구간을 공유하고 그 이후를 이어 학습합니다. C의 전체 학습 예산은 **A 계산 시간 + C 추가 계산 시간**이며, 이를 **B 복원 계산 시간 + B 분할 계산 시간**에 맞춥니다. 계산 시간은 각 update의 실제 입력 전송·forward·backward·optimizer 처리 구간을 측정한 뒤 가장 느린 GPU rank의 값을 합산합니다. 데이터 미리 읽기, 로그, 체크포인트 저장, 검증 시간은 제외합니다. 동일한 GPU 수를 유지하므로 `시간 × GPU 수`도 비교 가능합니다.

C는 마지막 update 한 번만큼 목표 시간을 초과할 수 있습니다. A의 측정 시간이 이미 B 전체 이상이면 추가 학습 없이 A를 C로 사용하고 그 차이를 기록합니다. update 수·FLOPs까지 정확히 일치시키는 실험은 아닙니다. C 추가 구간은 A의 optimizer 상태를 유지하고 LR warmup+cosine을 다시 시작합니다. 계획 update 수는 A의 측정 속도로 추정하며 실제 종료는 시간 예산으로 결정합니다. 보호 상한 12,000 extra updates에 도달하면 미완료 오류로 종료해 예산 미달을 성공으로 보고하지 않습니다.

## 모델과 학습

- Encoder: torchvision **ResNet18 IMAGENET1K_V1**, BatchNorm 통계 고정, 가중치·BN affine parameter 학습.
- U-Net decoder: skip 채널 256/128/64/64/16, 출력 채널 128/64/48/32/16. GroupNorm 사용.
- 원해상도 16채널 skip 분기를 사용해 미세 조직 정보를 전달합니다. 총 분할 모델 파라미터 12,523,764개.
- 흑백 원본을 3채널로 복제하고 ImageNet 정규화. 크기 확대 없음.
- 원본 기준 중심 512×512, 사방 halo64를 포함해 입력 640×640. 손실은 실제 영상의 중심 영역에만 적용합니다.
- 모든 학습 단계에 D4 회전·반전 적용. 강한 명암 변환·블러·형태학적 후처리 없음.
- 유효 배치 12, GPU당 micro-batch 1, 기본 6 GPU에서 2회 누적. GPU 수 변경 시 dummy slot의 손실을 0으로 두어 같은 유효 배치를 유지합니다. 실행 중 재개는 같은 GPU 수만 허용합니다.
- AdamW, encoder LR 1e-4 / decoder LR 3e-4, weight decay 0.01, warmup 200 + cosine, gradient clipping 1.0, BF16.
- 분할 손실: CE + Dice. 경계 손실을 추가하지 않아 사전학습의 효과를 비교합니다.
- 설정 파일: [configs/base.yaml](configs/base.yaml). 모든 실행은 실제 적용 설정을 보존합니다.

자기지도 단계에서는 train 원본만 유지하는 별도 Dataset을 사용합니다. **정답 마스크 경로를 저장하거나 정답을 읽지 않습니다.** 입력을 16×16 픽셀 블록으로 나누고 약 50%를 명암 0.5로 가립니다. 출력은 1채널 복원 영상이며, **가려진 실제 중심 픽셀에서만 L1 손실**을 계산합니다. 보이는 픽셀·halo·원본 바깥 padding은 복원 손실에서 제외합니다. 이는 마스킹 복원 아이디어를 소형 CNN에 적용한 실험입니다. 관련 개념: [MAE](https://arxiv.org/abs/2111.06377), [전자현미경 자기지도 사전학습 연구](https://arxiv.org/abs/2402.18286).

## 데이터와 평가

기존 official split train70/valid20/test10을 보존합니다. **valid와 test 원본은 자기지도 사전학습에도 사용하지 않습니다.** 이 버전은 추가 외부 원본 없이 train70만 이용합니다. 원본 파일 해시와 ID로 split 간 중복을 검사합니다. 같은 시편에서 얻은 서로 다른 이미지가 split에 존재할 가능성은 기존 official split의 특성이므로, 별도 시편 일반화 성능으로 해석하지 않습니다.

검증 점수로 best checkpoint를 고르지 않고 각 학습 단계의 종료 checkpoint를 평가합니다. 원본 해상도에서 core512/halo64, 50% 겹침 타일 추론 및 Gaussian 확률 결합을 사용합니다. 세 모델의 추론 조건은 같고 TTA는 끕니다.

주 지표는 이미지별 4개 상 IoU 평균을 검증 20장에서 다시 평균하는 **image mIoU (%)**입니다. 상별 값도 이미지별 IoU의 평균입니다. 빈 클래스는 정답·예측 모두 없을 때 IoU=1, 정답은 없지만 예측이 있으면 IoU=0입니다. 전체 mIoU는 빈 클래스 0 처리·제외 처리도 함께 저장합니다. Global mIoU는 사용하지 않습니다.

보조 지표는 조직 경계 1/2/4px 주변 오류율과 boundary IoU, 이미지 외곽 16/32/64px 띠의 오류율·mIoU입니다. boundary IoU의 양쪽에 없는 상은 제외 처리하며, 영상 프레임을 클래스 경계에 포함합니다. 주 지표의 빈 클래스 규칙과 구분합니다.

## 실행

프로젝트 루트 `/Data3/sppark/komap`에서 실행합니다. 기존 Python 환경과 기본 GPU **0,1,2,5,6,7**을 사용합니다. A·B·C를 차례로 실행하며, 학습 중인 하나의 모델을 6 GPU DDP로 학습합니다. GPU 3·4는 기본 실행에 포함하지 않습니다.

데이터 검사와 ImageNet 가중치 준비(학습 없음):

```bash
/home/sppark/miniconda3/envs/komap/bin/python experiments/selfsup_unet_v1/run.py prepare
```

사용자가 전체 실험을 시작하는 명령:

```bash
bash experiments/selfsup_unet_v1/start_nohup_6gpu.sh
```

통합 로그 확인:

```bash
tail -f experiments/selfsup_unet_v1/logs/latest.log
```

표준 출력·오류·학습·검증 기록을 실행마다 새 로그 하나에 모읍니다. 20 updates마다 loss, 진행률, 단계 경과 시간, 남은 시간, 종료 시각, 전체 실험 예상 시간·종료 시각을 출력합니다. 첫 단계에서 다른 단계의 속도는 아직 측정되지 않아 잠정 추정하며 검증 시간은 제외한다고 표시합니다. A 검증 후부터 측정한 검증 시간을 포함합니다. C의 추가 update 수와 예상 시간도 B의 실측으로 갱신됩니다. 체크포인트 저장 및 시스템 부하로 추정에 오차가 생길 수 있습니다. 서버나 웹 대시보드는 실행하지 않습니다.

포그라운드 또는 다른 GPU 수:

```bash
bash experiments/selfsup_unet_v1/run_6gpu.sh
KOMAP_GPUS=0 bash experiments/selfsup_unet_v1/run_6gpu.sh
```

다른 설정은 `run_6gpu.sh suite --config /path/to/config.yaml`로 지정합니다. Python 경로는 `KOMAP_PYTHON`으로 바꿀 수 있습니다. 같은 nohup 실행기를 중복 실행하면 잠금으로 막습니다.

## 체크포인트와 재개

기본 1,000 updates마다, 각 단계 종료 시, 저장 후 중단 요청 시 새 checkpoint를 저장합니다. 모델·optimizer·각 rank의 난수 상태·데이터 진행 위치·계산 시간·설정·학습 출처를 보존합니다. checkpoint 파일은 덮어쓰지 않고 `latest.json` 포인터만 갱신합니다. 완료한 단계는 재개할 때 다시 학습하지 않습니다.

로그의 `SUITE=...`에 나온 실행 경로를 사용합니다.

```bash
touch /path/to/suite/SAVE_AND_STOP
```

`PAUSED after checkpoint`가 출력되어 이전 실행이 종료된 것을 확인하고 재개합니다.

```bash
bash experiments/selfsup_unet_v1/start_nohup_6gpu.sh suite \
  --resume-suite /path/to/suite
```

재개 시 저장된 소스·설정·데이터 해시와 GPU 수를 검사하고 `SAVE_AND_STOP` 요청을 해제합니다. 검증 중 중단 요청은 다음 학습 단계에서 처리합니다. 마지막 검증 중이면 전체 실행이 완료될 수 있습니다. 강제 종료 후에는 마지막으로 저장된 update부터 다시 시작합니다.

완료 checkpoint만 별도 재평가하려면 다음을 사용합니다. 기존 결과를 변경하지 않고 새 번호에 저장합니다.

```bash
KOMAP_GPUS=0 bash experiments/selfsup_unet_v1/run_6gpu.sh evaluate \
  --checkpoint /path/to/checkpoints/update_004000.pt --arm B
```

## 산출물

- `outputs/suite_<시간>/`: 공통 초기값 `initial.pt`, 설정·데이터·환경·소스 사본, A/B/C 단계별 checkpoint와 `training.jsonl`.
- `B_ssl/`: 1채널 복원 모델. 최종 4상 예측에 사용하는 모델은 `B_finetune/`입니다.
- `results/model_results_MMDD_N/`: A·B·C 개별 평가 결과를 각각 다음 빈 번호로 생성합니다. 이어 종합 비교 보고서를 별도 번호로 저장합니다.
- 개별 결과: 동명 Markdown, `summary.json`, `per_image.csv/json`, `masks/`, `valid_mask_comparison/`, `verification.json`, `provenance.json`, 평가 소스 사본.
- 종합 결과: A/B/C 전체·상별 점수, 경계·외곽 오류, 학습 계산 시간 및 `budget.json`, 개별 보고서 링크.

검증 사진은 **좌상단 원본 / 우상단 오류 / 좌하단 예측 / 우하단 정답**입니다. 우상단은 맞는 픽셀을 흰색, 틀린 픽셀을 **정답 상의 색**으로 표시합니다. Eutectic Si 회색 204와 흰색 255를 구분합니다. 원본 픽셀 크기를 유지하고 저장 파일에서 네 패널을 다시 검증합니다.

## 구현 확인

```bash
/home/sppark/miniconda3/envs/komap/bin/python -m unittest discover \
  -s experiments/selfsup_unet_v1/tests -p 'test_*.py' -v
```

합성 데이터만으로 마스크를 읽지 않는 SSL 경로, 가려진 픽셀의 손실, encoder만 이전하는 동작, 이미지별 평균, 타일 외곽 커버, 분산 배치, 전체 결과 생성, checkpoint 재개를 검사합니다. 테스트의 학습·보고서는 임시 폴더에서 생성·삭제합니다. GPU 분산 검사는 `tests/verify_ddp.py`로 가짜 입력만 사용합니다. 실제 데이터 학습은 위 실험 실행 명령으로 사용자가 시작합니다.

복원 성능이 좋아져도 상 분할·테두리가 개선된다고 단정할 수 없습니다. B−A와 B−C, Eutectic Si, 경계·외곽 오류를 함께 판단해야 하며, 한 seed 결과의 안정성은 추후 반복 실험으로 확인해야 합니다.
