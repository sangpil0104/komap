# Model349 lite23 + B4 BN 고정 추가 학습 — 통합 재현 패키지

이 ZIP 하나에 **추론, 채점, 시드별 전체 재학습** 코드가 모두 들어 있습니다. 가중치와 대회 데이터는 들어 있지 않습니다.

| 구성 | valid20 (image-wise 4-class mIoU) |
|---|---:|
| lite23 (기준) | 80.5172705% |
| **lite23 + E의 B4 교체 (이 패키지)** | **80.5464316%** (+0.0292%p, 시편 bootstrap 95% [+0.003, +0.067]) |

valid20을 반복해서 써 온 개발 결과이므로 독립 일반화 점수가 아닙니다.

## 폴더

| 폴더 | 내용 |
|---|---|
| `Model349_fusion805/` | 공통 코드: R(exh14) 추론, E 학습·추론(`model397_reproduction_bundle/`), B4 추가 학습(`b4ws/`), F 추론, 채점 |
| `Model349_fusion805_lite23_b4ws/` | 이 구성의 설정과 실행기(`infer_compact.py`), 재학습 키트(`retrain/`, `refiner_retrain/`), 결과(`results_b4ws/`) |
| `train_route/` | R 14개의 학습 코드(`Model349_train_route_code.zip` 내용) |
| `weights_manifest.json` | 가중치 23개(+ args 메타데이터)의 위치와 SHA256 |

## 환경
Python 3.10, PyTorch 2.6.0 + CUDA 11.8 (P100에서 검증): `pip install -r Model349_fusion805/requirements.txt`

## 1. 추론 (공개 가중치)

```sh
bash run_infer.sh /path/WEIGHTS /path/test/images /path/out_test 0
bash run_infer.sh /path/WEIGHTS /path/valid/images /path/out_valid 0
bash score.sh /path/out_valid/fusion/predicted_labels /path/valid/masks /path/out_valid/metrics.json   # 80.5464316% 기대
```

- `WEIGHTS`는 `weights_manifest.json`대로 놓습니다.
  - `WEIGHTS/R/...`: exh14 가중치 14개
  - `WEIGHTS/assets/model329_ws/foldK/`, `WEIGHTS/assets/model392/seedS/foldK/`, `WEIGHTS/assets/refiner/epoch8.pt`
- 팀 P100 서버에는 이 구조로 맞춰 둔 폴더가 있습니다: `/workspace/model349_compress/lite23_b4ws_weights` (심볼릭 링크, SHA 34개 확인).
- 실행기가 SHA를 검사합니다. `args.json`이 없으면 동봉 메타데이터에서 복원합니다.
- 결과 마스크는 `OUT/fusion/predicted_masks/`에 생깁니다.

## 2. 시드별 전체 재학습

```sh
bash run_retrain.sh SEED /path/DATA /path/runs/seedSEED 0,1,2,3      # GPU 목록. DATA/train, DATA/valid, (DATA/test)
```

- 23개를 모두 다시 학습한 뒤 lite23 고정 설정으로 valid20 채점(`pred_valid/metrics.json`)과 test 예측(`pred_test/fusion/predicted_masks`)까지 합니다.
- 설정·규칙은 다시 고르지 않습니다. 시드는 기록된 시드 + 1000×SEED입니다.
- 시드 하나에 P100 약 76 GPU시간입니다(R·E가 같은 B4-329를 공유). GPU 4장에 나눠 돌리면 실제로는 약 21시간 걸리고, 끊기면 같은 명령으로 이어집니다.
- 자세한 내용은 `Model349_fusion805_lite23_b4ws/retrain/README.md`에 있습니다.
- 재학습한 가중치를 `run_infer.sh`로 추론할 때는 다섯 번째 인자로 `--allow-retrained`를 줍니다(SHA 검사 생략).

## 검증한 범위와 한계
- **80.5464316%**: 같은 가중치·코드로 E를 원본 영상부터 새로 추론하고 F를 적용해 얻은 값입니다(저장 확률 결과와 0픽셀 차이). R은 이전 원본 추론 결과를 재사용했습니다.
- **F 재학습 스크립트**: 원래 F와 SHA까지 같은 가중치를 재현했습니다(P100).
- **아직 하지 않은 것**
  - 이 ZIP을 새로 풀어 끝까지 실행하는 패키지 검증
  - `run_retrain.sh`로 시드 하나를 처음부터 끝까지 학습하기
- **재학습 예상:** 재학습 R은 80.02~80.05%였습니다. 그래서 전체는 약 80.2~80.3%로 예상하고, 80.5464%가 그대로 나오지는 않습니다.
