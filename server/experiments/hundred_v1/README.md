# 100개 모델 실험 — hundred_v1

기존 모델·소스·결과를 보존하며 `komap/spec.md`, `AGENTS.md`를 따릅니다. 진행률과 경과 시간·ETA는 터미널 tqdm에 표시합니다.

## 실행 명령 하나

```bash
bash /Data3/sppark/komap/experiments/hundred_v1/run_all.sh 8
```

어느 작업 디렉터리에서 실행해도 됩니다. 기존 `komap` conda 환경을 자동 선택하고 **100개를 하나씩 순서대로 학습한 뒤 각각 plain/D4 검증**합니다. 한 모델 안에서 GPU 8장을 사용합니다. 마지막 인자는 `1`, `2`, `4`, `8` 중 선택할 수 있습니다. 추가 패키지 설치나 외부 사전학습 가중치 다운로드는 필요 없습니다.

100개가 모두 성공하면 **`/Data3/sppark/komap/model_results_0920.md`**를 자동 생성합니다. 이미 같은 이름의 파일이 있으면 이 작업 공간의 `outputs/report_history/`에 복사해 보존한 뒤 갱신합니다.

중단됐을 때 **같은 명령을 다시 실행**하세요. 동일 소스·설정·데이터인지 확인하고, 완료 모델은 재사용하며 미완료 모델은 `last.pt`에서 재개합니다. 미완료 학습은 처음과 같은 GPU 수를 사용해야 합니다. `이미 학습 완료 … 재사용`은 오류가 아닙니다.

한 실험이 실패하면 실패를 기록하고 다음 실험을 시도합니다. 실패/미완료가 있으면 100개 성공으로 표시하지 않으며, 중간 보고서는 `outputs/reports/model_results_0920.md`에 둡니다. 모든 모델과 두 평가가 완료된 경우에만 프로젝트 루트의 최종 보고서를 생성합니다.

## 구성

| 번호 | 개수 | 내용 |
|---|---:|---|
| 001 | 1 | 추천: 기존 CNN에서 패치 단계를 생략한 전체 영상 미세조정 |
| 002–020 | 19 | 기존/동료 결과를 고려한 학습률·순서·부모 모델·ASPP·약한 손실/샘플링 조정 |
| 021–044 | 24 | ASPP, pyramid pooling, 큰 커널, 축 방향 depthwise, SE/ECA/CBAM, skip gate, decoder/고해상도 보완 등 |
| 045–064 | 20 | Focal, Tversky, generalized Dice, label smoothing, hard-pixel CE, 경계·보조 손실 |
| 065–076 | 12 | AdamW/Adam/SGD, weight decay, beta, AMSGrad, 학습률 일정 |
| 077–088 | 12 | 밝기·대비·감마·노이즈·흐림·채널 dropout·선명도·국소 대비·가림·이미지 샘플링 |
| 089–100 | 12 | residual/separable/ConvNeXt형/SE attention U-Net, 16/24/32 폭, 무작위 초기화부터 학습 |

**100개의 서로 다른 학습 설정입니다.** 전부 서로 다른 공식 아키텍처 100종을 재현한 것은 아닙니다. 구조가 같은 후보도 손실·최적화·데이터·일정이 실제로 다르고, 각각 별도 가중치를 학습합니다. 이름과 설명을 제외한 유효 설정의 중복 검사를 수행합니다. 전체 목록과 출처는 [EXPERIMENTS.md](EXPERIMENTS.md), 실행 가능한 설정은 `configs/resolved/*.yaml`입니다.

추가 80개는 이전 점수 순위를 따르지 않고 원 논문의 아이디어와 자체 가설로 넓게 탐색하도록 정했습니다. 논문 기반 모듈도 현재 경량 CNN에 맞춘 변형이며 원 논문 전체 모델/학습 결과와 동일하지 않습니다. 마지막 attention U-Net 계열은 **SE 채널 attention을 붙인 자체 U-Net**이며 Oktay 등의 Attention U-Net을 그대로 구현한 모델이 아닙니다. skip gate 후보는 별도로 제공됩니다.

## 학습·선택 정책

- 공식 train 70장, validation 20장과 기존 클래스/팔레트를 유지합니다. 추가 seed를 이름만 바꾸어 100개로 세지 않습니다.
- 모델 001–088은 기존 CNN 가중치에서 시작합니다. 011/012/013은 각각 seed2028/photo_mild/fullframe 가중치를 읽습니다. 모두 동일 공식 분할로 학습됐는지 확인합니다.
- 추천 001: 전체 영상 600 updates, LR `2e-6`, EMA `0.99`, D4 학습 증강, 광도 증강 없음.
- 021–088: 기본 전체 영상 1,200 updates. 새 구조 모듈은 보통 LR `2e-5`, 기존 가중치는 `2e-6`. 015/016은 새 모듈만 선행 학습 후 함께 미세조정합니다.
- 089–100: 512 patch 1,600 → 768 patch 1,400 → 전체 영상 1,500 updates. 외부 사전학습 가중치 없이 시작합니다.
- 총 **148,900 optimizer updates**입니다. effective batch 8, micro batch 1, BF16, 기본 200 updates마다 raw/EMA 검증, 100 updates마다 재개 체크포인트를 저장합니다.
- 학습 전 update 0도 평가합니다. `best.pt`는 초기 모델까지 포함한 최고 **plain** 검증 가중치라 시작점보다 낮은 모델로 자동 교체되는 일을 막습니다.
- **`best_trained.pt`**는 추가 학습 이후 후보 중 최고 plain 검증 가중치입니다. 100개 비교표와 plain/D4 평가는 이 파일을 사용해 미세조정 실패도 숨기지 않습니다. 보고서의 `초기 가중치 유지` 열은 `best.pt`가 초기 가중치를 유지했는지 뜻합니다.
- `last.pt`는 학습 재개용입니다. 최고 체크포인트가 중간 update에서 선택됐으면 completed=false일 수 있습니다. 최종 완료 상태는 `last.pt`와 `status.json`으로 확인합니다.
- 단계 전환은 직전 단계에서 가장 좋았던 가중치로 시작하고 새 optimizer를 만듭니다. 새 모듈은 가능한 경우 초기 출력이 원 모델과 같도록 초기화합니다.
- m019만 기존 `refinement_v1/outputs/mining/`의 train 70장 오류 지도를 읽습니다. 해시와 train ID를 검증하며 validation/test를 mining에 사용하지 않습니다. OOF mining은 아닙니다.

## 예상 시간과 저장 공간

이전 RTX 4090 8장 실행에서 전체 영상 학습은 update당 약 0.22–0.24초였습니다. 이번 100개의 예산과 새 구조 비용을 고려하면 **약 8–14시간을 계획 예산**으로 잡을 수 있습니다. 100개 전체 실측값은 아니며 후보 구조·데이터 로딩·다른 GPU 작업에 따라 달라집니다. 터미널에는 현재 학습 단계 ETA와 전체 모델 완료 수 기반 ETA가 표시됩니다. 처음에는 전체 ETA가 미정이고, scratch 후보가 뒤에 있어 초기 추정이 낙관적일 수 있습니다.

원본 모델을 복사해 덮어쓰지 않습니다. 새 실험마다 optimizer 상태를 포함한 여러 체크포인트를 보관하므로 **수십~100 GB 규모**의 추가 공간을 예상하세요. preflight가 실제 남은 공간을 표시합니다. 자동으로 과거 체크포인트를 삭제하지 않습니다.

## 결과 경로

```text
komap/model_results_0920.md              # 100개 성공 완료 후 최종 보고서
experiments/hundred_v1/
  configs/resolved/                     # 100개 실제 설정
  outputs/runs/<ID>/                    # best.pt, best_trained.pt, last.pt, 로그
  outputs/evaluations/<ID>/{plain,d4}/   # 이미지별 CSV, 클래스 점수, provenance
  outputs/reports/results.csv           # 중간에도 확인 가능한 비교표
  outputs/reports/model_results_0920.md  # 진행 중/실패를 포함한 중간 보고서
  outputs/suite_status.json             # 전체 실행 상태
  outputs/smoke*/                       # 짧은 기능 점검; 본 결과와 분리
```

보고서는 순위, 이미지 평균/Global mIoU, 기준 대비 변화, 동일 19장 변화, 클래스별 IoU, 초기 모델 대비 변화, 학습 시간, 실패 목록, 100개 전체 조건을 정리합니다. 기존 모델을 자동으로 교체하거나 테스트 마스크를 자동 제출하지 않습니다.

100개를 같은 validation에서 선택하므로 최고점은 개발 검증 점수입니다. 새 검증셋/그룹 CV와 반복 seed에서 확인해야 합니다. warm-start와 scratch의 예산·초기화가 달라 아키텍처만의 공정한 비교로 해석할 수 없습니다. A71의 빈 클래스 처리에 따른 지표 민감도도 보고서에 표시합니다.

## 개별 명령

```bash
conda run --no-capture-output -n komap python -B experiments/hundred_v1/run.py list
conda run --no-capture-output -n komap python -B experiments/hundred_v1/run.py preflight --gpus 8
conda run --no-capture-output -n komap python -B experiments/hundred_v1/run.py train m001_recommended_fullframe --gpus 8
conda run --no-capture-output -n komap python -B experiments/hundred_v1/run.py train m001_recommended_fullframe --gpus 8 --resume
conda run --no-capture-output -n komap python -B experiments/hundred_v1/run.py report
```

학습 코드·설정은 실행 이후 수정하지 마세요. 소스 해시가 바뀌면 잘못된 재개를 방지하기 위해 거부합니다. 변경 실험은 새 작업 공간에서 실행해야 합니다.

검증 기록은 [BUILD_REPORT.md](BUILD_REPORT.md)에 있습니다. 전체 학습은 사용자가 위 명령으로 시작합니다.
