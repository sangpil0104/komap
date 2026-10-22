# HRNet_taeho — 0928

`CNUKoMaP/HRNet_best`를 현재 프로젝트에서 실행할 수 있도록 가져왔습니다. 모델명은 **HRNet_taeho**, 생성일은 **2026-09-28 (0928)**입니다. 원본 모델·증강·손실·추론 코드는 수정하지 않고 `source/`에 보존했습니다.

- 원본: https://github.com/CNUKoMaP/HRNet_best.git
- 가져온 브랜치/커밋: `main` / `97adcdf337ab78fb7ec1469fe48943ff9a6363ea`
- 원본 18개 파일의 해시: [source_manifest.json](source_manifest.json)
- Git 이력을 포함한 clone: `outputs/upstream/` (상위 저장소 추적 제외)
- **학습은 시작하지 않았습니다.** 원본 저장소에는 학습 완료된 segmentation 체크포인트가 포함되어 있지 않습니다.

## 모델과 기본 설정

| 항목 | 설정 |
|---|---|
| 모델 | HRNet-W48 인코더 + U-Net++ 디코더 + 학습용 경계 보조 head |
| 초기화 | HRNet-W48 ImageNet 사전학습 가중치 |
| 데이터 | 현재 저장소의 `dataset/`: train 70장, valid 20장, test 10장 |
| 입력 | 원본 256×256 crop → 512×512 확대, ImageNet RGB 정규화 |
| 샘플링 | 50% 확률로 Al3Ni/Eutectic Si 주변 crop |
| 손실 | 0.5 CE + 0.5 Dice + 경계 head loss (BCE + Dice), 클래스 가중치 없음 |
| 학습 | batch 8, 70 epoch, seed 0, FP32, AdamW lr 3e-4, weight decay 1e-4, cosine |
| 학습량 | epoch당 105 update, 총 7,350 update |
| 최종 체크포인트 | 원본과 같이 마지막 epoch의 `last.pt` |
| 추론 | 원본 256px 타일, 겹침 170px, 2배 확대, Gaussian 확률 병합, flip TTA 4종 |
| GPU | 한 번에 GPU 1개 사용. 기본 물리 GPU 0번, `HRNET_GPU`로 0~5번 중 지정 |

증강과 추론의 자세한 내용은 [원본 README](source/README.md)를 참고하세요. 통합 실행 설정은 [configs/default.json](configs/default.json)에 있습니다. 사용자 설정 파일의 `data_root`는 해당 JSON 파일의 위치를 기준으로 해석합니다.

## 현재 프로젝트에 맞춘 부분

평가 지표는 이미지마다 4개 클래스 IoU를 평균한 뒤 valid 20장에 같은 가중치를 주어 평균하는 **image mIoU**입니다. 정답·예측 양쪽에 없는 클래스는 `union=0 → IoU=1`입니다. 나머지 세 클래스가 각각 0.8이면 `(0.8+0.8+0.8+1)/4 = 0.85`가 됩니다.

원본 README의 81.77% 및 재학습 81.62%는 20장의 confusion matrix를 합친 **global mIoU**입니다. 이 저장소에서 실제 학습 후 산출할 image mIoU 점수로 사용하지 않습니다.

기존 파일을 덮어쓰지 않도록 실행별 디렉터리를 생성하고, 학습 종료 후 valid 평가와 결과 보고서를 자동 생성합니다. 학습 중 `quick_native1x_image_miou`는 원본 방식의 1배 전체 이미지 추론을 사용한 진행 참고값입니다. 최종 보고서의 2배 타일+flip TTA 평가와 조건이 다릅니다. 중간 점수로 체크포인트를 선택하지 않습니다.

## 실행 환경

현재 서버의 `.venv`에 설치했습니다. 다른 환경에서 준비하려면:

```bash
cd /Data3/sppark/komap/experiments/HRNet_taeho
bash setup.sh
```

기존 `komap` 환경의 PyTorch를 공유하고 추가 패키지는 이 실험의 `.venv`에 설치합니다. 기본 Python 경로는 `/home/sppark/miniconda3/envs/komap/bin/python`이며, 필요하면 `HRNET_BASE_PYTHON`으로 지정합니다.

현재 환경은 Python 3.11 / PyTorch 2.7.1+cu128 / SMP 0.5.0 / timm 1.0.29 / Albumentations 2.0.8입니다. 원본의 Python 3.12 / PyTorch 2.14.0+cu126 환경과 다릅니다. 원본 환경 명세는 `source/pyproject.toml`, `source/uv.lock`, `source/requirements.txt`에 그대로 보존했으며, 현재 설치 목록은 [environment.freeze.txt](environment.freeze.txt)에 기록했습니다. 원본 점수의 재현 여부는 학습 후에 확인할 수 있습니다.

학습 없이 확인하는 명령:

```bash
./run.sh preflight   # 원본 해시, 데이터 개수·크기·팔레트 검사
./run.sh prepare     # ImageNet 초기 가중치 로딩/캐시 준비; 학습 없음
./run.sh smoke       # 합성 batch 8×3×512×512 순전파·역전파; optimizer update 없음
./run.sh train       # 실행 설정만 출력하는 dry-run
```

## 사용자가 학습 시작

GPU 0번에서 nohup 실행:

```bash
cd /Data3/sppark/komap/experiments/HRNet_taeho
./start_nohup.sh
```

GPU 3번을 사용하려면:

```bash
HRNET_GPU=3 ./start_nohup.sh
```

`start_nohup.sh`는 내부에서 `nohup ./run.sh train --execute`를 실행하고 PID·로그 경로·진행 확인 명령을 출력합니다. 터미널에서 직접 실행하려면 `./run.sh train --execute`를 사용하세요. 진행률·경과 시간·예상 남은 시간은 `tqdm` 로그로 제공합니다.

## 저장 위치와 재평가

```text
experiments/HRNet_taeho/
  source/                 # 변경하지 않은 원본 소스
  configs/default.json    # 통합 실행 설정
  run.py / runtime.py     # 실행·image mIoU·결과 출력 연결 코드
  reports/                # 학습 전 점검 기록
  outputs/
    upstream/             # 원본 Git clone
    logs/                 # nohup 로그
    latest_run.json        # 최근 실행 위치·상태
    runs/HRNet_taeho_0928_seed0_<UTC timestamp>/
      last.pt             # 해당 실행의 마지막 epoch 가중치
      config.json         # 사용한 설정
      source/ / adapter/  # 사용한 원본·연결 코드 스냅샷
      history.jsonl       # 학습 손실과 중간 검증 기록
      status.json         # 실행 상태
      completed.json      # 학습 및 최종 평가 완료 후 결과 위치
```

같은 실행 안에서 `last.pt`를 갱신합니다. 새 실행은 새 폴더를 사용하므로 이전 실행의 모델과 소스를 보존합니다. 원본과 같이 optimizer/scheduler 상태를 저장하지 않으므로 학습 중단 후 정확히 이어 학습하는 기능은 제공하지 않습니다.

학습이 끝나면 다음 경로에서 아직 사용하지 않은 번호로 결과를 생성합니다.

```text
results/model_results_0928_<번호>/
  model_results_0928_<번호>.md
  summary.json
  per_image.csv
  per_class.csv
  config.json
  predictions/             # 원본 해상도·지정 RGB 팔레트 예측 마스크
  valid_mask_comparison/   # valid 20장 각각의 2×2 비교 그림
```

비교 그림은 왼쪽 위 원본, 오른쪽 위 오류, 왼쪽 아래 예측, 오른쪽 아래 정답입니다. 오류 패널은 맞힌 픽셀이 흰색이고 틀린 픽셀은 해당 위치의 정답 색입니다. 오류 패널 위에 image mIoU와 `100% − image mIoU`를 표시합니다. 후자는 픽셀 오류율과는 다른 값입니다.

평가만 다시 실행하거나 test 마스크를 만들려면 실제 체크포인트 경로를 넣으세요.

```bash
./run.sh evaluate --checkpoint outputs/runs/<실행 폴더>/last.pt
./run.sh predict --checkpoint outputs/runs/<실행 폴더>/last.pt --output outputs/test_predictions_01
```

설정을 변경해 학습했다면 `--config outputs/runs/<실행 폴더>/config.json`을 함께 지정합니다. 평가 결과는 새 번호로 생성하며, `predict`의 출력 폴더는 기존에 없는 경로여야 합니다.

검증 기록은 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)에 정리했습니다.
