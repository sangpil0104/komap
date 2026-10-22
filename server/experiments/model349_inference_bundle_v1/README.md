# Model349 실행 패키지

Validation 20장 image mIoU **80.068799%**를 기록한 `model349_train_20260930T143930_234475Z`의 학습된 가중치와 고정 추론 경로를 담았습니다. 입력 이미지로부터 4클래스 label PNG와 RGB mask PNG를 생성합니다.

## 포함 범위

- `infer.py`, `run.sh`: 단일 GPU 또는 여러 GPU에서 실행하는 추론 진입점.
- `src/`: Model349 실행에 필요한 네트워크, 입력 처리, 확률 결합, 후처리 정의만 추출한 코드.
- `weights/`: 실제 실행 가중치 23개. B4 12개, HRNet 2개, ResNet 계열 9개입니다.
- `configs/model349.yaml`: 최종 클래스별 결합 계수와 추론 설정.
- `configs/calibration.json`: 학습 70장에서 구한 고정 prior 및 밝기 통계. 실행 시 학습 데이터가 필요하지 않습니다.
- `environment.yaml`, `requirements.lock.txt`: 실행에 필요한 Conda 환경과 하위 의존성의 정확한 버전.
- `environment/`: 전체 개발환경 YAML, Conda 패키지 목록, 원래 학습 당시의 패키지 기록, 환경 버전 정보.
- `verify.py`, `SHA256SUMS.txt`, `provenance/`: 무결성 확인 및 원본 대응 기록.

다른 실험의 학습 실행 파일, 탐색 진입점, 로그, 결과, 데이터셋은 포함하지 않습니다. `src/`와 `weights/`에 남아 있는 97·181·269·322·326·329·336·347 등의 번호는 **Model349 앙상블에 실제 필요한 구성 요소와 코드의 원래 식별자**입니다. 해당 요소를 빼면 Model349가 달라집니다. 기존 다중 아키텍처 factory는 실제 사용하는 ASPP 경로만 남겼으며 네트워크 연산은 유지했습니다.

## 개발환경 설치

기준 환경: **Linux x86_64, Python 3.11.16, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, SMP 0.5.0, timm 1.0.29**, NVIDIA RTX 4090 24GB. CUDA 지원 NVIDIA GPU 및 호스트 드라이버가 필요합니다. 전체 영상 B4 추론 때문에 입력 크기에 따라 GPU 메모리 사용량이 증가합니다. 검증에 사용한 GPU는 RTX 4090 24GB이며 다른 장치의 최소 필요 메모리는 별도 측정하지 않았습니다.

압축을 풀고 이 README가 있는 디렉터리에서 실행합니다.

```bash
conda env create -f environment.yaml
conda activate komap
python -B verify.py --load-models
bash run.sh --check
```

이미 `komap` 환경이 있다면 우선 활성화하고 검증 명령을 실행하세요. 새 환경 생성 명령은 기존 환경을 덮어쓰지 않습니다. 더 넓은 개발 도구까지 복원하려면 `environment/environment.full.yaml`을 사용할 수 있습니다. 설치 파일에는 다른 실험 코드에 의존하는 editable 저장소 패키지를 제외했습니다. Model349는 그 패키지를 import하지 않습니다.

환경 YAML/버전 기록은 현재 설치된 환경에서 추출했습니다. 새 빈 환경에서의 다운로드·설치는 별도 검증하지 않았습니다. 패키지 설치에는 인터넷과 해당 버전의 배포 파일이 필요합니다. 설치가 끝나면 제공 가중치로 추론하며 사전학습 가중치를 추가로 내려받지 않습니다. 운영체제·드라이버·Conda 설치 프로그램·패키지 wheel 자체는 ZIP에 포함하지 않습니다.

## 실행

GPU 한 개에서 폴더 내 이미지를 추론합니다. 폴더에는 원본 이미지만 넣고 mask 파일은 넣지 마세요.

```bash
bash run.sh --input /path/to/test/images --output ./runs/test_001 --gpus 0
```

Validation 평가도 원하면 정답 마스크 폴더를 함께 지정합니다.

```bash
bash run.sh --input /path/to/valid/images --gt /path/to/valid/masks \
  --output ./runs/valid_001 --gpus 0
```

사용 가능한 GPU가 여러 개라면 각 구성 요소를 서로 다른 GPU에서 실행합니다.

```bash
bash run.sh --input /path/to/test/images --output ./runs/test_multi_001 --gpus 0,1,2
```

`--gpus`는 물리 CUDA 장치 번호 또는 GPU UUID 목록입니다. 기본값은 기존 `CUDA_VISIBLE_DEVICES`의 첫 장치이며, 설정이 없으면 `0`입니다. 사용 가능한 장치를 지정하세요. 실행기는 GPU를 자동 예약하지 않습니다. 단일 이미지 경로와 `--limit 1`도 지원합니다. 진행률·경과 시간·예상 잔여 시간은 터미널 `tqdm`과 20초 간격 로그 요약으로 제공됩니다. 이미지 크기와 분기별 추론 시간이 달라 ETA는 추정값입니다.

`--output`은 **아직 없는 새 폴더**여야 합니다. 기존 결과를 덮어쓰지 않습니다. 중간 확률은 성공 시 삭제하며, 보관하려면 `--keep-probabilities`를 추가합니다. 오류가 발생하면 해당 출력 폴더와 로그를 보존합니다.

## 출력과 클래스

| 경로 | 내용 |
|---|---|
| `predicted_labels/이름.png` | 원본 해상도, 클래스 ID 0~3 |
| `predicted_masks/이름_mask.png` | 원본 해상도, RGB 예측 마스크 |
| `metrics.json` | `--gt` 지정 시 이미지별 점수와 평균 image mIoU |
| `input_manifest.json` | 입력 SHA256, 실제 사용한 설정 |
| `completion.json` | 완료 여부, 이미지 수, 소요 시간 |
| `logs/` | 구성 요소별 상세 로그 |

입력 파일명이 `A117_step4_sample2_image.png`라면 출력 이름은 `A117_step4_sample2`입니다. 하위 폴더는 재귀 검색하지 않습니다. 동일한 기본 이름의 입력은 오류로 처리합니다.

| ID | 클래스 | RGB |
|---:|---|---|
| 0 | Al3Ni | (25, 76, 153) |
| 1 | Primary Si | (76, 178, 76) |
| 2 | Al | (153, 127, 76) |
| 3 | Eutectic Si | (204, 204, 204) |

정답 마스크는 위 RGB 색상 또는 클래스 ID 0~3의 단일 채널 PNG를 지원합니다. 각 이미지에서 네 클래스 IoU를 평균하고, 그 값을 이미지 수로 평균합니다. 정답·예측 모두 없는 클래스의 IoU는 0이며 global mIoU는 계산하지 않습니다. Test에는 정답을 지정하지 않으면 점수를 계산하지 않습니다.

## 재현성 및 모델 실행 범위

최종 라우팅·확률 FP16 저장 정밀도·후처리는 기존 Model349 설정을 유지합니다. `configs/model349.yaml`을 수정하면 다른 실행 조건이 됩니다. 일부 저수준 타일 설정은 원본 함수 기본값과 `infer.py`에 고정되어 있으며 YAML과 코드가 함께 실행 레시피를 구성합니다.

기존 모델의 TENT 절차도 유지합니다. 각 입력 이미지에서 BatchNorm 파라미터를 entropy 최소화로 5회 임시 적응시키며, 정답 마스크를 사용하거나 가중치 파일을 수정하지 않습니다. 새로운 지도학습·모델 선택·조합 탐색은 실행하지 않습니다. 학습 시작 명령은 이 패키지에 포함하지 않았습니다.

검증 결과는 `provenance/verification.json`에 기록합니다. 원래 검증 20장에 대한 저장 확률 재조합 검증과 새 패키지의 실제 재추론 검증은 서로 구분하여 기록합니다. GPU·라이브러리 버전 차이에 따른 수치 차이는 있을 수 있습니다.
