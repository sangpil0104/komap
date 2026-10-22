# Model349 lite20 + B4 BN 고정 추가 학습 — 통합 재현 패키지 (2026-10-11)

**lite20**은 lite23_b4ws(23개)에서 R 가중치 3개를 뺀 구성입니다. 가중치는 20개입니다(R 11 + E 8 + F 1).

| 뺀 R 가중치 | 이유 |
|---|---|
| 고해상도 model109, restart59(seed456) | 검증 데이터로 고른 좁은 밝기 구간(검증 4장, 시험 0장)에서만 쓰임 |
| B4 흑백(322) fold 4 | 빼도 최종 점수가 그대로(+0.002%p) |

| 구성 | 가중치 | valid20 최종 (image-wise 4-class mIoU) | test10 마스크 (lite23 대비) |
|---|---:|---:|---|
| lite23_b4ws | 23 | 80.5464316% | — |
| **lite20 (이 패키지)** | **20** | **80.5301323%** | 9,074,964픽셀 중 836픽셀 다름 (0.009%) |

- 위 lite20 점수는 이 패키지의 코드로 P100에서 **원본 영상부터 R을 새로 추론**하고, 기존 E·F와 결합해 채점한 값입니다. 저장 확률로 낸 예상(80.5304%)과 같았습니다. 결과는 `Model349_fusion805_lite23_b4ws/results_lite20/`에 있습니다.
- valid20을 반복해 쓴 개발 결과이므로 독립 일반화 점수가 아닙니다.

## 실행

```sh
pip install -r Model349_fusion805/requirements.txt              # Python 3.10, PyTorch 2.6.0 + CUDA 11.8 (P100에서 검증)
bash run_infer_lite20.sh /path/WEIGHTS /path/test/images /path/out_test 0
bash run_infer_lite20.sh /path/WEIGHTS /path/valid/images /path/out_valid 0
bash score.sh /path/out_valid/fusion/predicted_labels /path/valid/masks /path/out_valid/metrics.json   # 약 80.530% 기대
```

- `WEIGHTS`: lite23과 같은 배치(`WEIGHTS/R/...`, `WEIGHTS/assets/...`)를 그대로 씁니다. 실제로 쓰는 20개는 `weights_manifest_lite20.json`에 있습니다. 팀 P100 서버: `/workspace/model349_compress/lite23_b4ws_weights`
- 마지막 숫자는 **실제 GPU 번호**입니다. 시작할 때 그 GPU가 비어 있어야 합니다. 시작한 뒤에는 다른 사용자가 같은 GPU를 쓰기 시작해도 멈추지 않고 경고만 남깁니다.
- 다른 축소판도 같은 방식으로 돌릴 수 있습니다: `bash run_infer_variant.sh lite21|lite20|lite19 WEIGHTS IMAGES OUT GPU`. 세 구성을 GPU 3장에서 동시에 검증하려면 `verify_variants.sh`를 씁니다.

## 이전 패키지와 바뀐 점
- `Model349_fusion805/Model349_exh14/infer_lite.py`
  - 선택 설정 `inference.pruned_high_bands` 추가(설정이 없으면 lite23과 같게 동작).
  - GPU 점유 검사를 시작 시점에만 하도록 변경(이전에는 모델마다 다시 검사해, 다른 사용자가 GPU를 쓰기 시작하면 중간에 멈췄음).
- **`run_infer.sh`(lite23) GPU 지정 수정:** 이전 버전은 GPU를 0이 아닌 번호로 지정해도 R·E가 실제 0번 GPU에서 돌았습니다. 이제 지정한 GPU를 씁니다. lite23·lite21 ZIP에는 이 버그가 남아 있습니다.
- 설정: `Model349_fusion805_lite23_b4ws/selected_{R,config,assets}_{lite21,lite20,lite19}.*`
- 축소 후보 탐색(하나씩 빼기·다시 넣기, E·F 포함 최종 점수): `Model349_fusion805_lite23_b4ws/results_lite21/`

## 한계
- 재학습 키트(`run_retrain.sh`)는 여전히 23개를 모두 학습하고 lite23 설정으로 예측합니다. lite20으로 재학습 결과를 보려면 학습 후 `run_infer_lite20.sh ... --allow-retrained`로 추론하면 됩니다.
- 축소 선택도 같은 valid20으로 했으므로, 0.01%p 수준의 차이는 우연 범위입니다.

---


# Model349 lite21 + B4 BN 고정 추가 학습 — 통합 재현 패키지

**lite21**은 lite23_b4ws에서 R의 고해상도 학습본 2개(model109, restart59 seed456)를 뺀 구성입니다(가중치 23 → 21개). 두 학습본은 검증 데이터로 고른 좁은 밝기 구간(152.3~152.7, 174.6~175.1)에서만 쓰였고, 그 구간의 이미지는 이제 기본 학습본(restart51)으로 처리합니다.

| 구성 | 가중치 | valid20 (image-wise 4-class mIoU) |
|---|---:|---:|
| lite23_b4ws | 23 | 80.5464316% |
| **lite21 (이 패키지 기본)** | **21** | **80.5287696%** (−0.0177%p) |

- 바뀌는 valid 이미지는 4장뿐입니다(A74 −0.19, A11_step2 −0.13, A40 −0.03, A67 −0.00%p). test10에는 두 밝기 구간에 드는 이미지가 없어 **test 예측은 lite23과 같습니다**(추론과 같은 채널0 평균 밝기로 확인).
- 실행: `bash run_infer_lite21.sh WEIGHTS IMAGES OUT GPU`. 가중치 배치는 lite23과 같고, 실제로 쓰는 21개는 `weights_manifest_lite21.json`에 있습니다.
- 코드 변경: `Model349_fusion805/Model349_exh14/infer_lite.py`에 `inference.pruned_high_bands` 설정을 추가했습니다(없으면 lite23과 동일하게 동작). 설정은 `Model349_fusion805_lite23_b4ws/selected_R_lite21.yaml`, `selected_config_lite21.json`, `selected_assets_lite21.json`입니다.
- 검증 범위
  - 80.5287696%: 영향받는 4장의 R 라벨을 exh14 저장 확률로 다시 만든 뒤(기존 경로로 만들면 공개 R 라벨과 1~6픽셀 차이), 같은 E·F로 결합해 채점했습니다. 같은 절차로 lite23 80.5464316%가 정확히 재현됩니다.
  - 수정한 `infer_lite.py`로 영향받는 4장을 P100에서 원본부터 추론해, 4장 모두 restart51을 쓰고 위 R 라벨과 1~5픽셀만 다름을 확인했습니다. valid20·test10 전체를 원본부터 끝까지 다시 돌린 검증은 하지 않았습니다.
  - 재학습 키트(`run_retrain.sh`)는 여전히 23개를 모두 학습하고 lite23 설정으로 예측합니다.
- 추가 제거 후보 탐색 결과는 `Model349_fusion805_lite23_b4ws/results_lite21/`에 있습니다(R 구성원 하나씩·조합 제거, E·F까지 결합한 valid20 점수).

---

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
