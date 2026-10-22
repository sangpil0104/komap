# lhs: Model326 학습 + Model338 실행 환경

원본: <https://github.com/CNUKoMaP/lhs> · 기준 commit `aa970bef8d16b267c8d7bfbffe3ca1b4c4a4813f` · 생성일 2026-09-29.

원본 Git 추적 파일 862개는 `source/`에 그대로 보존했습니다. 전체 clone은 `outputs/upstream/`, 원본 파일 SHA-256은 `source_manifest.json`에 있습니다.
기존 HRNet·다른 실험·이전 결과를 수정하지 않았습니다. **실제 학습은 시작하지 않았습니다.**

| 구성 | 준비 상태 | 역할 |
|---|---|---|
| Model326 | 환경·ImageNet 초기 가중치·데이터·GPU 메모리 검사 완료 | EfficientNet-B4 U-Net 4-fold 학습 및 자동 결과 생성 |
| Model338 | 실행기·원본 경로·합성 입력 통합 검사 완료, 실제 입력 4개 누락 | 여러 모델의 캐시 확률을 클래스별로 혼합하고 공간 보정 |

## Model326 실행

```bash
cd /Data3/sppark/komap/experiments/lhs
./start_nohup.sh
```

이 명령을 **사용자가 실행하면** 학습이 시작됩니다. PID와 로그 경로가 출력됩니다.
출력된 `tail -f ...` 명령으로 진행률, 경과 시간, ETA를 확인할 수 있습니다. 각 fold의 상세 tqdm은 해당 run의 `fold1.log`~`fold4.log`에 기록합니다.

기본 GPU pool은 `0,1,2,3,4,5`입니다. 원본은 독립적인 4개 fold이므로 기본 학습은 **GPU 0–3에서 fold 하나씩 동시에** 진행합니다.
GPU 4–5는 기본 학습에서 사용하지 않고, GPU 6–7은 사용하지 않습니다. batch size 2를 6개 GPU에 분산시키는 DDP 구성이 아닙니다.
GPU를 지정하거나 동시 작업 수를 줄일 수 있습니다.

```bash
LHS_GPUS=2,3,4,5 ./start_nohup.sh   # 네 fold 동시 학습
LHS_GPUS=0,1 ./start_nohup.sh       # 두 fold씩 순차 실행
LHS_GPUS=0 ./start_nohup.sh         # 한 GPU에서 네 fold 연속 실행
```

동일 실행기의 중복 학습을 파일 잠금으로 차단합니다. 다른 실험과의 GPU 공유 여부는 사용자가 GPU 목록으로 조정합니다.
하나의 fold가 실패하면 이 실행기가 시작한 다른 fold도 중단하고 로그와 생성된 체크포인트를 보존합니다.
새 실행마다 새 run 디렉터리를 만들며, 실패한 fold의 중간 학습을 자동 재개하지 않습니다. 원본처럼 각 fold의 최종 체크포인트를 저장합니다.

학습 없이 점검하려면:

```bash
./run.sh check
./run.sh train326                 # --execute 없으면 점검만 수행
```

학습 → 각 fold의 OOF D4 예측 → 학습 이미지 70장 OOF 통합 → 검증 이미지 20장의 4-fold 평균 + D4 추론 → 결과 저장 순서입니다.
검증 ensemble 추론은 지정한 GPU 목록의 첫 번째 GPU에서 실행합니다. 학습 중 valid 데이터로 체크포인트를 선택하지 않습니다.
결과는 `results/model_results_실행완료날짜_다음번호/`에 생성됩니다. 기존 폴더가 있으면 다음 번호를 사용합니다.

체크포인트는 `outputs/runs/Model326_YYYYMMDD_HHMMSS_.../fold1/last.pt`~`fold4/last.pt`입니다.
완료된 체크포인트가 있으면 재학습 없이 검증 결과만 다시 만들 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh evaluate326 --run /절대/경로/outputs/runs/Model326_...
```

평가를 다시 하면 새로운 평가 확률 폴더와 새로운 results 번호가 만들어집니다.

## Model326 원본 설정

- SMP U-Net, EfficientNet-B4, ImageNet 초기화, 출력 4개 클래스.
- 입력 `zcdf3`: 밝기, 영상 전체 통계로 정규화한 z-score, 영상 전체 CDF. 원본 영상을 3배 확대합니다.
- crop 1536×1536, batch 2, fold마다 16,000 iterations, seed 0, workers 4.
- AdamW(lr 0.0003, weight decay 0.01), 300-step warmup + cosine, FP16 AMP, CE + Dice, gradient clipping 1.
- 원본의 affine, D4, brightness/contrast, gamma, Gaussian blur/noise 증강을 유지합니다.
- Model303에서 고정한 시편 단위 4-fold를 유지합니다. train/OOF 이미지 수는 각각 52/18, 52/18, 53/17, 53/17입니다.
- 원본처럼 worker seed에 시간을 포함합니다. seed 0만으로 비트 단위 재현을 보장하지 않습니다.

`components326.py`의 CropDS, worker 초기화, Dice 및 augmentation은 원본에서 그대로 추출했고 AST 일치 검사를 통과했습니다.
원본의 이전 부모 모델과의 OOF 비교 단계는 부모 모델 확률이 없어 실행하지 않습니다. Model326 자체의 학습·OOF·valid 평가는 수행합니다.
실행 시 코드, 설정, fold 분할, 원본 및 환경 목록을 run에 복사해 기록합니다.
원본처럼 확률 NPZ는 float16으로 저장합니다. valid 마스크와 점수는 저장 전 float32 확률로 계산하며, 통합 OOF 점수는 저장된 OOF 확률을 사용합니다.

## Model338 실행

Model338은 단일 체크포인트가 아닌 **기록된 최종 ensemble 경로**입니다.
Primary Si는 Model326, Al은 Model322, Al3Ni는 Model322/326 평균, Eutectic Si는 Model269를 사용하며 클래스별 혼합 계수가 다릅니다.
밝기 prior, Gaussian/bilateral 보정 및 작은 Primary Si 영역 후처리는 원본 함수를 사용합니다.
설정은 `configs/model338.json`에 고정했으며, 로컬 valid에서 후보 탐색을 다시 하지 않습니다.

**실행에 필요한 4개 파일과 준비 방법은 [MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md)를 확인하세요.**
파일을 준비한 후:

```bash
./run.sh check338 --deep
./run.sh evaluate338
```

Model338 고정 경로는 CPU로 실행하며 학습하지 않습니다. 검증 20장에 대한 캐시 실행기이며, 임의의 새 이미지에 대한 모든 상위 모델의 end-to-end 추론을 제공하는 구성은 아닙니다.

## 지표와 결과 형식

현재 프로젝트의 `AGENTS.md`에 맞춰 이미지별 4개 클래스 IoU 평균을 구하고 20장의 평균을 냅니다.
**정답·예측에 모두 없는 클래스는 IoU=0**입니다. 예시 `(0.8+0.8+0.8+0)/4=60%`에 해당합니다. global mIoU는 계산하지 않습니다.
원본 보존 파일에는 이전 지표 구현과 기록이 남아 있지만, 새 실행기는 로컬 `runtime.metric()`을 사용합니다.

새 결과에는 같은 이름의 Markdown, `metrics.json`, 색상 예측 마스크, 0–3 label PNG와 `valid_mask_comparison/`이 포함됩니다.
비교 이미지는 왼쪽 위 Original, 오른쪽 위 Error, 왼쪽 아래 Prediction, 오른쪽 아래 Ground truth입니다.
오류 지도는 맞은 픽셀을 흰색, 틀린 픽셀을 정답 클래스 색으로 표시합니다. 오른쪽 위에 image mIoU와 100%−image mIoU를 표기합니다.

## 환경과 검증

별도 `.venv`가 기존 komap 환경의 PyTorch 2.7.1+cu128을 읽어 공유하며, 새 의존성은 `.venv`에 설치합니다.
SMP 0.5.0, timm 1.0.29, Albumentations 2.0.8, OpenCV 4.13.0.92를 사용합니다. 원본 기록의 PyTorch/OpenCV 버전과 달라 같은 점수를 보장하지 않습니다.
현재 환경은 설치되어 있습니다. 환경을 다시 준비하려면 `./setup_env.sh`를 실행합니다. 이 스크립트도 학습하지 않습니다.

검증 내용은 [VALIDATION_REPORT.md](VALIDATION_REPORT.md), 로그는 `reports/`에 있습니다.
