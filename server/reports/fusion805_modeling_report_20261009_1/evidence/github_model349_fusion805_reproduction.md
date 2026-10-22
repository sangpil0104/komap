# Model349 R + E 대표 후보와 fusion805 탐색 보존 코드

> **2026-10-09 갱신:** 권장 후보는 1단계(R + E의 Al₃Ni·Al 덮어쓰기)입니다. 원본부터 재추론한 valid 점수는 80.4744%, 시편 제외 선택은 +0.178%p입니다. 2단계는 1단계를 고정한 뒤 시편 제외로 고르면 1단계보다 −0.159%p라 권장하지 않습니다. `infer_fusion.py --stage1-only`로 1단계만 실행할 수 있습니다. fold × 시드 OOF 재학습 설계(`oof_expert/`, `Model349_exh14/oof_design/`)도 추가했고, 현재 smoke로 검증하는 중입니다. 자세한 내용은 ZIP 안 README의 "2026-10-09 갱신" 절에 있습니다.

[재현 ZIP](../Model349_fusion805_reproduction_code.zip) · [SHA256](../Model349_fusion805_reproduction_code.zip.sha256)

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
