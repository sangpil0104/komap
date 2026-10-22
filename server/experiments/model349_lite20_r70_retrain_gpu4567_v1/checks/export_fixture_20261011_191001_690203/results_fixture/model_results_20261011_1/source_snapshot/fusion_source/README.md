# Model349 R + E 대표 후보와 fusion805 탐색 보존 코드

[재현 ZIP](../Model349_fusion805_reproduction_code.zip) · [SHA256](../Model349_fusion805_reproduction_code.zip.sha256)

## 개선 후보: E의 B4를 BN 고정 추가 학습판으로 교체 (2026-10-10, Claude)

**1단계 구조·규칙은 그대로 두고, E 안의 B4-329 4 fold 가중치만 바꾼 후보입니다. 원본 영상부터 다시 추론한 valid20은 80.50167745%입니다.**

| 구성 | valid20 | 1단계 대비 |
|---|---:|---:|
| 1단계 R + E (위 대표 후보) | 80.4744236% | 기준 |
| **1단계 R + E′ (B4 추가 학습)** | **80.5016774%** | **+0.0273%p** (시편 bootstrap 95% [+0.006, +0.056], 14/20장 개선, 최악 −0.048%p) |

- **바뀐 것**
  - E의 B4-329 fold k를 같은 fold의 train70 학습 이미지로 2,000회 더 학습했습니다(lr 3e-5, seed 819349, 결정적 worker).
  - 이때 **BatchNorm은 eval 모드로 고정**해 running 통계는 그대로 두고, 가중치와 BN의 scale·shift만 학습합니다.
  - 나머지(R 14개, HRNet392 seed1·2, 비율 0.5/0.25/0.25, 덮어쓰기 클래스 {Al₃Ni, Al})는 모두 같습니다.
- **근거**
  - 학습에 쓰지 않은 시편 기준(OOF70)으로 B4 단독이 79.123 → 79.499%(**+0.376%p**, 51/70장 개선)입니다. fold별로는 +0.72 / +0.21 / −0.06 / +0.62%p입니다.
  - 같은 추가 학습을 BN 고정 없이 하면 OOF70이 −0.104%p로 떨어졌습니다. BN 고정이 핵심입니다.
  - valid20·test10 정답은 학습에 쓰지 않았습니다.
- **test10 마스크**는 1단계 대비 0.169%(15,335 / 9,074,964 픽셀)가 바뀝니다. 정답이 없어 정확도는 주장하지 않습니다.
- **한계**
  - valid20은 개발 과정에서 반복해서 사용했습니다. +0.027%p는 고정 후보 하나의 차이이며, 선택 편향이 없는 시험 추정치가 아닙니다.
  - 추가 학습 레시피는 Codex의 HRNet 추가 학습 결과(train70 보류 fold)를 본 뒤 정했습니다.

### 실행

```sh
# 1) 추가 학습 (fold당 P100 약 34분). 부모는 ASSETS/model329/foldK/last.pt, 결과는 ASSETS/model329_ws/foldK
bash b4ws/train_b4ws.sh /path/model397-output/splits/model329_cv /path/assets 0
#    공개된 가중치를 쓰는 경우에는 학습 대신 ASSETS/model329_ws/foldK/{last.pt,args.json}에 두면 됩니다.
#    args.json은 b4ws/args/foldK/args.json입니다. SHA는 config/b4ws_assets.json으로 확인합니다.
# 2) E′ 확률 → 1단계 → 채점
CUDA_VISIBLE_DEVICES=0 python infer_model401.py --b4-warmstart \
  --input /path/valid/images --assets /path/assets --output /path/new_expert_ws --device cuda
python infer_stage1.py --release /path/new_release --expert /path/new_expert_ws --output /path/new_stage1_ws
python score_masks.py --labels /path/new_stage1_ws/predicted_labels --gt /path/valid/masks \
  --output /path/new_stage1_ws/metrics.json
```

- 직접 다시 학습한 가중치는 SHA가 다르므로 `--b4ws-retrained`를 함께 줍니다. 이 경우 SHA 대신 추론 관련 args(구조·입력·배율·폭)만 확인합니다.
- `--b4-warmstart`를 주지 않으면 기존 1단계(80.4744%)와 완전히 같은 경로로 동작합니다.
- 결과·이미지별 점수·라벨 SHA: `results/b4ws_stage1_fresh.json`

## 검토 반영: 대표 후보는 1단계 R + E

**대표 후보는 F 없는 1단계 80.4746%입니다.** 이 ZIP 이름의 fusion805는 2단계 탐색 결과를 보존한 이름이며 제출 권장 점수가 아닙니다.

| 구성 | valid20 | 시편 제외 선택 |
|---|---:|---:|
| exh14 | 80.2312% | 기준 |
| 1단계 R + E | **80.4746% (+0.2434%p)** | **80.4088% (+0.1776%p)** |
| 2단계 + F | 80.5007% (+0.2695%p) | 80.3160% (1단계 대비 **−0.1586%p**) |

F는 조건부 교차검증에서 추가 단계의 손해가 관측돼 **미채택**입니다. 릴리스 대비 +0.0848%p만으로 F의 추가 이득을 주장하지 않습니다. 1단계는 보정기 가중치나 학습이 필요 없습니다.

1단계의 시편 제외 선택은 매번 해당 시편의 모든 영상을 제외하고 **69개 실행 × 30개 규칙 전체를 다시 선택**합니다. E를 바깥에서 고정한 평가가 아닙니다. 별도 재계산의 20개 선택 내역과 80.4087871%가 모두 일치했습니다. 후보 모델과 R 자체의 과거 valid 기반 개발 선택 편향까지 제거한 것은 아닙니다. 2단계 CV는 전체 valid에서 고른 1단계를 고정한 조건부 평가로 범위가 다릅니다.

1단계 최악 A71_step2_sample1은 **−0.2300%p**, 기존 영상별 손실 기준 −0.10%p를 통과하지 못해 **quality_pass: false**입니다. valid에서 184,749 / 15,757,260 픽셀(1.1725%)이 바뀌었습니다. Primary Si가 없는 영상이라는 특징으로 기준 미통과를 면제하지 않습니다. 기존 base는 유지합니다.

1단계의 고정 가중치 원본 영상 valid20·test10 전체 재추론과 독립 채점을 완료했습니다. 새 valid20은 **80.4744236018%**, 같은 실행의 R은 80.2309863330%, 차이는 **+0.2434372688%p**입니다. 저장 1단계 후보와 7픽셀 차이가 있습니다. test10은 새 R 대비 **161,731 / 9,074,964 픽셀(1.7822%)**이 바뀌며 일치율은 **98.21783315%**입니다. 원본 source·부모 가중치 SHA 및 RGB 팔레트·크기를 확인했습니다. 이는 고정 가중치 추론 검증이며 **처음부터 재학습한 성능의 재현성은 미확인**입니다. A71 기준 미통과와 반복 valid 선택 편향은 남습니다. test GT가 없어 시험 정확도 개선이나 mIoU를 주장하지 않습니다. [전체 추론 검증 보고서](model349_stage1_fresh_report.md)를 참고하세요.

### 1단계 실행 (F 불필요)

외부 릴리스 R의 새 추론 폴더와 `infer_model401.py`가 생성한 E 확률 폴더를 사용합니다. 부모 R/E 명령은 아래 B 경로를 참고하되 최종 결합을 다음 명령으로 실행합니다.

```sh
python infer_stage1.py \
  --release /path/new_release \
  --expert /path/new_model401_probabilities \
  --output /path/new_stage1
python score_masks.py \
  --labels /path/new_stage1/predicted_labels \
  --gt /path/valid/masks --output /path/new_stage1/metrics.json
```

1단계에서 `--check-reference`는 사용하지 않습니다. 이 옵션의 보존된 기준은 2단계 80.5007%입니다. 아래의 F 관련 명령·기록은 2단계 탐색 보존본입니다.

## 재학습 재현 설계 추가 (2026-10-09, Claude)

- **`oof_expert/`와 `Model349_exh14/oof_design/`**
  - R과 E를 시편 단위 fold × 시드로 다시 학습합니다(결정적 시드).
  - 1단계 덮어쓰기 클래스 집합(16가지 후보, 원래 규칙 {Al₃Ni, Al} 포함)은 OOF에서 고릅니다.
  - 기본은 A 방식입니다. train 70장으로 학습·선택하고, valid 20장은 마지막에 한 번만 채점합니다.
  - 2단계 F는 이 설계에 포함하지 않습니다.
  - 보고하는 점수는 OOF 개발 점수, 바깥 fold 추정치(선택 과정 포함), 시드별 점수의 평균·표준편차입니다. 사용법은 [oof_expert/README.md](oof_expert/README.md)를 참고하세요.
- **검증 범위:** smoke로 코드 경로를 일부만 확인한 뒤 팀원에게 인계했습니다(`Model349_exh14/oof_design/SMOKE_RESULT.md`). 전체 규모의 학습 결과는 없습니다.
- **팀원 환경 재학습 결과 (사용자 보고, 독립 채점 안 함)**
  - OOF 설계 적용 전 구성으로 다시 학습한 결과입니다.

    | 구성 | valid20 | 직전 단계 대비 |
    |---|---|---|
    | R | 80.051924% | - |
    | 1단계 | 80.236507% | +0.185%p |
    | 2단계 | 80.267268% | +0.031%p |

  - 고정 1단계 규칙의 이득이 새 가중치에서도 대부분 유지됐습니다.
  - 고정 2단계 규칙도 다시 소폭 양수였습니다. F의 미채택 판단은 유지하되, 재검토 근거로 기록합니다.

## 2단계 탐색 결과의 보존 기록

valid20 공식 image-wise 4-class mIoU **80.5006890946%**를 관측한 탐색 후보입니다. 기존 Model349_exh14 80.2312313198% 대비 **+0.2694577748%p**입니다. 원래 릴리스와 base를 교체하지 않았습니다.

## 포함 범위와 확인한 것

- 추론 코드, 부모 코드, 고정 설정, 외부 가중치 SHA256, train70 잔차 보정 학습 기록, 독립 PNG/GT 채점 결과를 포함합니다.
- 저장소의 기존 재현 ZIP 정책과 동일하게 **가중치·대회 데이터·확률맵·예측 마스크는 포함하지 않습니다**. 원래 checkpoint를 별도로 준비해야 관측 점수를 재현할 수 있습니다. 재학습은 동일 가중치/점수를 보장하지 않습니다.
- 80.5006891%는 저장된 부모 출력으로 두 단계 마스크를 생성한 뒤 PNG 20장/GT 80 IoU를 독립 채점한 값입니다. 이 후보의 모든 부모를 한 번에 원본 영상부터 다시 실행해 같은 점수가 나오는지 확인하는 단계는 아직 완료하지 않았습니다. 전체 fresh 시간 및 test 점수는 보고하지 않습니다.
- ZIP을 새 디렉터리에 풀어 동봉 실행기로 20장을 생성했습니다. 원래 후보의 픽셀 SHA 20개 및 독립 GT 80 IoU가 전부 일치해80.5006891%를 재현했습니다. `results/package_verification.json`과 `results/portable_reproduction_metrics.json` 참고.
- 같은 잔차 보정 checkpoint를 CPU에서 실행한 valid20 마스크는 기존 GPU 실행과 전체 20장 픽셀 단위로 같습니다.
- valid20을 여러 실험에 반복 사용했습니다. 독립 일반화 검증값이 아닙니다. 첫 단계 시편 제외 선택 80.4087871%, 첫 단계를 전체 valid에서 고정한 두 번째 단계의 조건부 시편 제외 선택 80.3160176%입니다. 후자는 기존 릴리스보다 +0.0847863%p 높지만 첫 단계 선택 편향이 남습니다.

## 고정 규칙

route 클래스 순서는 `0=Al3Ni, 1=Primary Si, 2=Al, 3=Eutectic Si`입니다.

1. 부모 R: 동봉한 Model349_exh14의 14개 릴리스 가중치·융합·TENT5·후처리.
2. 부모 E(Model401): P100 Model329 B4 4fold 평균0.50 + Model392 HRNet seed1 4fold 평균0.25 + seed2 4fold 평균0.25. B4는 D4, HRNet은 flip4. 각 가족의 저장 FP16 반올림을 재현한 뒤 FP32로 결합합니다. 팀 채널 `[Al,Al3Ni,Primary,Eutectic]`을 route로 변환합니다. argmax 동률은 팀 클래스 우선순위를 유지합니다.
3. 첫 결합: E가 Al3Ni(0) 또는 Al(2)로 예측한 픽셀은 E, 나머지는 R.
4. 보정 F: train70 OOF로 8 epoch 학습한 11,876 parameter 잔차 보정기. 전체 영상 gray z-score·±3clip 후 [0,1], log probability 입력, 잔차±1. checkpoint SHA는 `config/auxiliary_assets.json` 참고.
5. 최종 결합: F가 Al(2)로 예측한 픽셀은 F, 나머지는 첫 결합.

추론은 GT를 읽지 않습니다. `score_masks.py`만 평가용 GT를 읽습니다.

## 환경과 외부 자산

검증 환경은 Linux/P100, Python3.10.12, torch2.6.0+cu118입니다. `config/verified_environment.json`과 `requirements.txt` 참고. 부모 ZIP의 과거 환경 lock은 출처 기록으로 남겼으며 이 후보의 검증 환경은 위 파일이 기준입니다.

```sh
python -m pip install -r requirements.txt
```

- 릴리스14가중치: `Model349_exh14/checkpoint_manifest.json`의 경로와 SHA를 그대로 준비합니다. 가중치를 `Model349_exh14/weights/`에 둡니다.
- 보조12가중치와 보정1가중치: 별도 자산 디렉터리 `/path/assets/` 아래 `config/auxiliary_assets.json`의 상대 경로대로 준비합니다.
- B4 각fold의 `args.json`은 `model401_args/model329/foldN/args.json`에서 해당 자산 fold 폴더로 복사합니다. 파일을 임의 수정하지 않습니다. 예: `cp -R model401_args/model329 /path/assets/` (기존 last.pt를 보존).
- 보정 checkpoint 경로는 `/path/assets/refiner/epoch8.pt`입니다. 관측 결과에 사용한 SHA를 실행기가 강제 확인합니다.

## A. 2단계 보존값 재현: 기존 부모 출력이 있을 때

원래 릴리스 valid20의 `predicted_labels`, Model401의 B4 `probabilities.npz`, HRNet seed별 평균 `.npz` 20개와 원본 영상을 외부에서 준비합니다.

```sh
python infer_fusion.py \
  --input /path/valid/images \
  --release-labels /path/canonical_exh14/predicted_labels \
  --b4-archive /path/model329/valid_ensemble/probabilities.npz \
  --hr1 /path/model401/hrnet_seed1_mean_valid \
  --hr2 /path/model401/hrnet_seed2_mean_valid \
  --refiner /path/assets/refiner/epoch8.pt \
  --output /path/new_fusion_valid --device cpu --strict-valid
python score_masks.py \
  --labels /path/new_fusion_valid/predicted_labels \
  --gt /path/valid/masks --output /path/new_fusion_valid/metrics.json \
  --check-reference
```

`--strict-valid`는 원래20입력 SHA를 확인하고 `--check-reference`는 예측 픽셀 SHA 및 이미지별80IoU를 대조합니다. CPU재추론 control 출력은 canonical과 극미세한 차이가 있으므로 정확 재현에는 원래 canonical R을 사용합니다.

## B. 부모의 원본 영상 추론 경로 (1단계 검증 완료, F 포함 2단계 미완료)

이 경로의 코드는 포함됐지만 전체 후보 fresh 실행의 80.5007% 재현 검증은 미완료입니다. 각 모델 source/kernel·장치·수치환경에 따라 미세한 차이가 생길 수 있으므로 결과를 A의 관측값과 조용히 동일 취급하지 않습니다. 다른 사용자가 쓰지 않는 GPU를 확인해 실행합니다.

```sh
python Model349_exh14/infer_lite.py \
  --input /path/valid/images --output /path/new_release --gpus 0
CUDA_VISIBLE_DEVICES=0 python infer_model401.py \
  --input /path/valid/images --assets /path/assets \
  --output /path/new_model401_probabilities --device cuda
python infer_fusion.py \
  --input /path/valid/images \
  --release-labels /path/new_release/predicted_labels \
  --expert-probabilities /path/new_model401_probabilities \
  --refiner /path/assets/refiner/epoch8.pt \
  --output /path/new_fusion --device cpu
python score_masks.py \
  --labels /path/new_fusion/predicted_labels \
  --gt /path/valid/masks --output /path/new_fusion/metrics.json
```

test 등 새 영상에도 B의 추론 경로를 적용할 수 있습니다. test에는 정답이 없어 mIoU를 계산하지 않습니다.

## 학습 출처와 한계

`model397_reproduction_bundle`은 B4/HRNet의 기존 학습·OOF 재현 코드입니다. `refiner/train_residual.py` 등은 원래 서버 실험 코드의 보존본입니다. 입력 특징 생성에는 외부 OOF 확률과 원래 부모 디렉터리 구조가 필요합니다. 동봉 추론 코드를 실행할 때 재학습할 필요는 없습니다.Train70만 감독학습에 사용했고 valid/test 정답을 학습에 넣지 않았습니다. 팀원 학습 환경과 P100 가중치 차이를 GPU만의 인과 효과로 해석하지 않습니다.

81%에는 아직 도달하지 않았습니다. 결과와 조건부 검증을 구분해 보고합니다.
