# Model349 경량판(D3·표현모델 제거) 재현 패키지

이 ZIP은 **추론 코드·고정 설정·15개 체크포인트의 무결성 목록·검증 기록**을 담은 코드 재현 패키지입니다. 학습된 체크포인트와 대회 데이터는 포함하지 않습니다. 따라서 ZIP만 풀어서 즉시 추론하는 패키지는 아닙니다. 비공개 연구 저장소의 데이터/가중치 비포함 정책을 따릅니다.

## 확인된 결과

P100 한 장, valid 20장의 공식 지표(이미지마다 네 클래스 IoU 평균, 양쪽 모두 없는 클래스는 0):

| 구성 | 가중치 | image mIoU | 벽시계 시간 |
|---|---:|---:|---:|
| 원형 Model349 | 23개 | 80.0683465% | 1895.94초 |
| 기존 D3 | 16개 | 80.0642960% | 407.52초 |
| 이 경량판 | **15개** | **80.0643596%** | 366.56초 |

경량판은 원형보다 `-0.0039868%p` 낮습니다. 시간은 단회 측정이므로 표현 모델 하나를 제거한 효과를 별도로 추정하지 않습니다. test 10장의 기존 D3 마스크와는 9,074,964픽셀 중 14픽셀만 달랐지만, test 정답이 없어 test mIoU는 알 수 없습니다. valid 20장은 모델 선택에 반복 사용됐으므로 독립 일반화 검증이 아닙니다.

## 가중치 준비

`provenance/weights_manifest.json`에 나열된 **정확히 15개** `.pt`를 같은 상대 경로의 `weights/` 아래에 넣습니다. B4 8개 fold의 `args.json`은 ZIP에 포함됩니다. 체크포인트는 기존 Model349 실행 릴리스에서 가져오거나, 별도 데이터와 학습 코드로 생성할 수 있습니다. 같은 저장소의 `Model349_train_route_code.zip`에는 구성요소 학습 소스가 있으나 전체 15개 lineage를 자동으로 학습하는 단일 명령은 제공하지 않습니다. 새로 학습한 가중치는 수치나 SHA-256이 이 패키지의 기록과 같을 필요는 없지만, **기록된 80.06436% 재현을 주장하려면 manifest와 동일한 체크포인트**가 필요합니다.

원본 23가중치 실행 릴리스가 있다면, 그 디렉터리를 `--release`로 지정해 이 패키지의 `build_rep_free_bundle.py`로 15가중치 독립 묶음을 만들 수 있습니다. 이 명령은 원본 파일을 수정하지 않고, 같은 파일시스템이면 하드링크를 사용합니다.

```bash
python build_rep_free_bundle.py \
  --release /path/to/original_model349_release \
  --script ./infer_lite.py \
  --output /path/to/new_model349_lite_bundle
python verify_rep_free_bundle.py /path/to/new_model349_lite_bundle
```

기존 릴리스가 없다면 이 ZIP의 디렉터리를 그대로 사용하고 15개 가중치를 위 경로에 배치하세요. `verify_rep_free_bundle.py`는 원본 체크포인트의 SHA-256을 확인하므로, 새로 학습한 다른 가중치에는 통과하지 않는 것이 정상입니다.

## 추론

`environment.yaml` 또는 `requirements.lock.txt`를 참고해 CUDA 지원 환경을 준비합니다. 코드가 원본 영상에서 바로 마스크를 만들며 확률 캐시는 필요하지 않습니다. 가중치 준비 후 이 디렉터리에서:

```bash
python -B infer_lite.py --check
python -B infer_lite.py --input /path/to/valid/images \
  --gt /path/to/valid/masks --output /path/to/new_valid_run --gpus 0
python -B infer_lite.py --input /path/to/test/images \
  --output /path/to/new_test_run --gpus 0
```

`--output`은 **존재하지 않는 새 디렉터리**여야 합니다. 추론 라벨은 `predicted_labels/`, RGB 마스크는 `predicted_masks/`, 정답을 제공한 경우 점수는 `metrics.json`에 저장됩니다. `--gt`를 쓰지 않으면 점수를 계산하지 않습니다. 원래 대회 test 정답은 제공되지 않았습니다.

## 구성과 추가 축소

활성 네트워크는 B4 Model326 4-fold, B4 Model329 4-fold, HRNet336, native269, 밝기에 따라 선택되는 고해상도 기초 모델 5개 후보(이미지당 1개)입니다. Model322·347, 기초 분기3개·별도 pair 예측, 표현 모델181은 생략합니다. 고해상도 후보 중 pairwise 체크포인트는 별도 pair 예측용이 아니라 **선택 가능한 기초 모델**이므로 아직 필요합니다. 15개 전체가 valid/test 30장에서 각각 적어도 한 번 선택됐습니다.

`results/bundle15_audit.json`과 `results/bundle15_valid20_metrics.json`은 기록된 실행 결과입니다. 재학습한 다른 가중치에 대한 점수를 보증하지 않습니다.
