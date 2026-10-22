# 12개 base 재현 레시피

## 1. 환경과 파일

Linux/CUDA/P100에서 실행한 코드다. Python 3.11.16, PyTorch 2.7.1+cu128 등 당시 환경 스냅샷은 `environment.yaml`과 `requirements.lock.txt`에 있다. 기록 환경에서 추론을 확인했으며 새 환경에서의 설치 성공이나 성능 동일성을 보장하는 문서는 아니다.

```bash
conda env create -f environment.yaml
conda activate komap
python -B infer_lite.py --check
```

저장소 루트에 `weights/`를 두고 manifest 상대 경로대로 **실제로 검증된 FP16 저장 체크포인트**를 별도 복사한다. B4의 6개 fold `args.json`은 이 저장소에 포함한다. 현재 서버에서 새로 학습한 같은 이름의 파일은 배포판과 텐서가 다르므로 대신 쓰지 않는다. 원본 릴리스에 덮어쓰지 않는다.

```bash
python verify_bundle.py
```

검증기는 체크포인트 수·바이트·SHA-256, 추론 코드·설정·fold args 해시를 검사한다. 가중치 미제공 상태에서는 missing으로 실패하는 것이 정상이다. FP16 저장을 FP32로 다시 저장하면 값이 유사하더라도 바이트 해시가 달라 별도 아티팩트가 된다.

## 2. GPU 사용 전

사용할 서버의 GPU 사용자 매뉴얼을 먼저 읽는다. 실행 중인 자신의 추론·예약 작업과 GPU 점유를 확인하고 같은 실험을 중복 실행하지 않는다. 다른 사용자의 프로세스를 종료하거나 설정을 바꾸지 않는다.

```bash
nvidia-smi
ps -ef | rg 'infer_lite|run_test10|model349'
```

아래의 `GPU_ID`를 실제 비어 있는 물리 GPU 번호로 설정한다. 기록 실험은 단일 P100에서 수행했다. 추론 실행기도 각 component 시작 전 GPU memory >64 MiB 또는 utilization >0이면 중단하지만 사전 사용자 점유 확인을 대체하지 않는다. 이 코드가 시작한 worker만 종료 처리한다. 기존 출력 폴더는 재사용하지 않는다.

## 3. valid20 원본 재추론

```bash
GPU_ID=0
python -B infer_lite.py \
  --config configs/model349_lite.yaml \
  --input /path/to/valid/images \
  --gt /path/to/valid/masks \
  --output /path/to/new_lite12_valid20_run \
  --gpus "$GPU_ID"
```

입력은 이미지 파일 하나 또는 비재귀 폴더다. 20장 폴더를 사용하고 `--limit`를 주지 않는다. GT는 `<image_stem>_mask.png` 또는 `<image_stem>.png`다. 출력 `metrics.json`의 count=20 및 `completion.json`을 확인한다. `image_miou`는 0~1이므로 백분율은 100을 곱한다. 각 클래스는 이미지별 IoU를 평균한 `class_image_mean_iou`로 비교한다. 빈 union 및 해당 영상에 없는 클래스는 0으로 채점한다.

출력에는 `predicted_labels/`(class ID PNG), `predicted_masks/`(palette PNG), `logs/`, 입력 SHA와 calibration SHA를 포함한 `input_manifest.json`, 완료 파일이 있다. 원본 확률을 보존하려면 `--keep-probabilities`를 추가한다. 기본 실행은 완료 후 중간 확률을 삭제한다.

## 4. test10

```bash
GPU_ID=0
python -B infer_lite.py \
  --config configs/model349_lite.yaml \
  --input /path/to/test/images \
  --output /path/to/new_lite12_test10_run \
  --gpus "$GPU_ID"
```

test에는 `--gt`를 주지 않는다. completion의 images=10, mean_image_miou=null이 정상이다. 기존 모델과 class ID 마스크의 동일 픽셀 수/전체 픽셀 수를 비교할 수 있으나 그 값을 정확도 또는 IoU로 표시하지 않는다.

## 5. 설정을 그대로 유지할 항목

- 326 fold `[1,3]`, 329 fold `[1,2,3,4]`; TENT steps=5, lr=1e-5.
- 고해상도 4개 후보, EMA 제거 구간의 restart51 대체, 원본·밝기 보정 예측 및 D4.
- B4/269/HRNet 실제 TTA off; torch/OpenCV threads=2. HRNet batch=2. D4 batching/HRNet autocast 탐색판은 이 base에 포함하지 않는다.
- `configs/calibration.json`, route 계수, prior·beta·경계 혼합·cleanup 후처리 전부 동결.
- 저장 tensor FP16, 로드 후 연산 FP32. 중간 확률도 기존 경로와 같이 FP16으로 저장한다.

## 6. 다음 경량화 실험 기록

새 후보마다 별도 코드·config·출력 폴더를 만들고 입력·가중치·calibration·코드 해시를 기록한다. 후속 주 비교판은 이 12개 base이며 원본349/lite15/14 FP16 비교를 함께 남기면 기존 결과와 연결할 수 있다. 기존 lite15 대상 사전 판정 기준을 새 base 기준으로 소급 적용했다고 주장하지 않는다. 새로운 기준은 실행 전에 명시한다.

정확도는 공식 이미지 평균 지표, 클래스별·이미지별 손실을 함께 확인한다. 시간은 같은 GPU/영상/환경에서 기준판·후보를 교차 반복 측정한 중앙값과 실행별 값을 기록한다. checkpoint 크기 개선과 실제 연산 시간 개선은 별도 판정한다. 지금 valid/test 기록은 단회다.

100장 안에서 valid를 다시 골라 fold/비율을 맞추면 탐색이다. specimen 비중복 valid 부분집합도 이미 선택에 노출됐고, 전체 앙상블 train70 추론은 strict OOF가 아니다. 새 독립 정답 없이 일반화 개선 확정 문구를 추가하지 않는다.
