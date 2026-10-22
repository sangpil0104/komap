# Model404: tiny Eutectic-Si component recall fine-tuning

## 상태

**CPU preflight·합성 smoke 완료, 학습 전.** Model403의 고정 OOF run이 P100에서 진행 중이므로, 해당 run과 GPU를 겹쳐 쓰지 않는다. Model402/397 오류 audit에서 `component_any_fn_rate`는 성분 안에 FN이 하나라도 있는 비율이었지, 절반 이상 놓친 성분 비율은 아니었다. 별도 Model397 고정 OOF 재감사에서 실제 `<50%` recall 비율은 1–64 native-px 성분의 31.90%, 64–256px의 12.13%였고, 완전 미검출은 각각 21.51%, 3.32%였다. 따라서 처치는 가장 약한 tiny `[1,64)` native-px 범위로 한정하고 64–256px 영역은 loss 대상에서 제외하되 OOF 안전 지표로 감시한다. 이 진단은 기존 train OOF를 재사용했으므로 Model404도 selection-conditional development experiment이며 독립 confirmatory test가 아니다. Pre-run lock v1은 runner/학습기 arm 이름 불일치로 보존, v2는 preflight에 따른 target 범위 수정으로 보존하고, GPU optimizer step 전에 revision-3 lock을 만든다.

## 동기와 범위

Model402의 고정 Model397 grouped train-OOF 분석(70장/59 specimen groups)은 1–64 native-px 성분 43,545개 중 89.83%, 64–256px 성분 16,336개 중 99.19%에서 어떤 FN 픽셀이라도 관측했다고 보고했다. 단, 이 any-FN 비율은 완전 또는 과반 누락률이 아니다. 동일한 고정 Model397 확률을 새로 분석한 결과, `<50%` recall은 tiny 31.90%(그룹 bootstrap hit-rate 95% CI 64.80–71.35%), small/medium 12.13%(hit-rate CI 84.72–90.69%)였고 완전 미검출은 각각 21.51%와 3.32%였다. fold별 tiny hit-rate도 64.17–71.51%로 모두 낮은 편이라 이 실험은 loss 감독을 1–64 native-px로 제한한다. 작은 성분 oracle correction gain은 달성 가능한 상한이며 성능 예측으로 해석하지 않는다. 원자료: `results/model404_component_recall_preflight.json`.

이 처치는 기존 HRNet-W48/U-Net++ CE+Dice 및 boundary loss를 유지하면서 정답 Eutectic-Si component의 평균 정답 확률을 직접 높이는 보조 손실 하나만 추가한다. 기존 Al3Ni/Primary-Si boundary 문제는 이번 대상이 아니다. 과거 작은 예측 영역을 제거한 component cleanup 및 Eutectic hard-crop mining은 채택 가능한 이득이 없었으므로 재사용하지 않는다.

## 고정 처치

- 부모: fold별 Model392 seed1 `last.pt`; 아키텍처/preprocessing key를 검증하고 outer fold의 해당 Model392 checkpoint로 초기화한다.
- 네 specimen-group fold, seed 1, fold별 고정 학습 이미지만 사용. 각 arm은 per-item RNG(seed 1)를 사용해 worker timing 차이와 무관하게 같은 crop/augmentation 및 batch order를 받는다. 이는 Model392의 legacy RNG 스트림을 그대로 재현하기 위한 설정이 아니라, 두 continuation arm 사이에서 처치 외 차이를 통제하기 위한 paired 설정이다.
- 두 arm 모두 8 epochs fine-tuning, patch 512, width-normalized input scale 2.0, 12 crops/image, batch 4, accumulation 2, AdamW weight decay `1e-4`, cosine schedule, AMP, gradient clipping 1.0, CE+Dice + boundary loss.
- encoder LR `3e-5`, decoder/head LR `1e-4`; treatment와 matched control에 동일 적용.
- **control:** 추가 component loss 없음. **treatment:** 아래 component recall loss를 추가.
- 정답 mask의 class-3(Eutectic Si)를 8-connectivity로 분리한다. 학습 crop/augmentation이 끝난 512×512 target에서 area `[4,16)`, `[16,64)`, `[64,256)` px에 속하고 crop 경계에 닿지 않는 성분만 포함한다. 경계 접촉 성분은 crop truncation 가능성 때문에 제외한다. Nominal 2× linear scale에선 대략 native 1–64 px에 해당하지만 width normalization/augmentation 때문에 정확한 native-area 대응으로 주장하지 않는다. Native 64–256px 성분은 component loss에서 제외한다.
- 성분별 손실은 `-log(mean pixel p(Eutectic | component))`. 각 crop×area-bin에서 성분 평균을 먼저 구한 뒤 관측된 crop×bin cell들을 동등 평균한다. 따라서 한 crop의 특정 크기 성분 수가 다른 크기 bin을 압도하지 않는다. 기존 pixel CE+Dice는 precision/false-positive 감독으로 유지한다.
- coefficient는 mIoU가 아닌 outer-train-only calibration으로 fold별 고정한다. 학습 전 8개 batch에서 full CE+Dice+boundary와 component loss의 shared decoder activation gradient norm 비율 중앙값 `r_f`를 측정하고, `lambda_f = 0.10 / r_f`로 고정한다. calibration은 optimizer step 0회이며 weight 범위 `[1e-5, 1]`, component 포함 gradient batch 최소 6/8을 요구한다. 범위를 벗어나거나 calibration이 비유한 경우 학습하지 않고 처치 재설계로 돌린다.
- 두 arm의 training steps/augmentation order/epochs는 같다. OOF inference는 patch512, overlap170, scale2, flip4를 고정한다. threshold, TTA, routing, calibration search는 하지 않는다.

## 데이터 누수·평가

- P100의 기존 locked `model392_cv/seed1_foldN` 분할에서 outer-train 이미지만 학습한다. 해당 fold의 held-out training images/masks는 OOF 평가에만 쓴다.
- 70개 train images를 59 specimen groups 안에서 정확히 한 번씩 OOF 채점한다. 공식 점수인 이미지별 네 class IoU 평균 후 이미지 평균, empty-union=0을 사용한다.
- primary comparison은 **component treatment vs 같은 compute의 matched continuation control**이다. Model392 고정 OOF는 기준 맥락으로 별도 보고한다.
- paired specimen-group bootstrap 20,000회, fold별/class별/Step4 paired delta를 보고한다. 1–64 및 64–256 native-pixel Eutectic 성분 각각에 대해 GT component 내부 Eutectic pixel recall `>=50%`를 hit로 정의하고, group-bootstrap hit-rate delta도 보고한다.
- 단일 seed adoption gate는 모두 만족해야 한다: overall mIoU `>= +0.20 pp`; paired group-bootstrap 95% CI 하한 `>0`; 4 folds 중 3개 이상 양수이며 최악 fold `>=−0.30 pp`; 모든 class IoU delta `>=−0.50 pp`, Eutectic-Si delta `>=0`; Step4 delta `>=−0.30 pp`; tiny `[1,64)` component hit-rate delta 점추정 `>=0` 및 95% CI 하한 `>=−5 pp`; loss 대상에서 제외한 `[64,256)` medium-bin hit-rate도 같은 비열화 기준을 통과한다.
- seed1 통과는 채택이 아니다. 같은 lock을 seed2로 재현해 동일 gate를 통과해야 최종 full-train fit을 검토한다. 그 전에는 공식 valid/test를 읽지 않는다. 두 seed를 통과한 경우에만 full train fit과 공식 valid 20장 단회 평가를 별도 사전등록한다. test.zip은 봉인한다.

## 중단 조건 및 재현 파일

잘못된 fold inventory, baseline/checkpoint metadata mismatch, 비유한 loss/gradient, optimizer step이 calibration에서 발생, calibration 범위 이탈, OOF 누락/중복이면 해당 run은 무효 처리한다. 중간 validation/OOF로 학습 checkpoint·weight·threshold를 고르지 않는다.

- component loss: `model404_component_recall_loss.py`
- synthetic CPU test: `model404_component_recall_smoke.py`
- paired fine-tuner: `model404_component_recall_train.py`
- train-only gradient calibration: `model404_component_recall_calibrate.py`
- paired OOF scorer: `score_model404_component_recall_oof.py`
- train-only component preflight: `model404_component_recall_preflight.py`; result: `results/model404_component_recall_preflight.json`.
- P100 runner/hash receipt generator: `run_model404_component_recall_oof.sh`, `model404_preregistration_lock.py`; revision-3 source lock은 calibration/학습 전에 만들며 revision-1/2 receipts는 보존한다. GPU calibration/학습은 Model403이 P100을 해제한 다음에만 실행한다.

현재 구현만으로 점수 향상을 주장하지 않는다.
