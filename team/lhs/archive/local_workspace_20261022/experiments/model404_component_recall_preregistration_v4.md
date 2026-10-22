# Model404 v4: tiny Al3Ni + Eutectic-Si component recall

## 변경 사유와 사전 상태

v3는 Eutectic-Si만 대상으로 했고 아직 calibration/optimizer step에 들어가지 않았다. 따라서 v3 대기 작업을 취소한 뒤, immutable v3 파일·lock은 보존하면서 두 취약상에 균형 적용하는 v4를 별도 파일과 새 lock으로 준비한다. Model403 grouped OOF가 P100을 해제하기 전에는 GPU 작업을 시작하지 않는다.

고정된 Model397 train-OOF 사후 진단(70장/59 specimen groups, empty-union=0)에서 native 1–<64 px GT 성분의 oracle image-mIoU 상한은 Al3Ni `+1.7374%p`(66,342개 성분), Eutectic Si `+2.6216%p`(43,545개)였다. 작은 Al은 `+0.0894%p`, Primary Si는 `+0.0093%p`였다. 이 값은 해당 GT cell을 완벽히 고친다는 비현실적 상한이지 예상 이득이 아니다. Al3Ni/Eutectic을 1차 처치 대상으로 하고 다른 두 클래스는 보조 component loss에서 제외한다. 사후 OOF 진단에 조건부인 개발 가설이며 공식 valid/test는 읽지 않았다.

## 고정 처치

- 부모: 네 fold 각각의 Model392 seed1 `last.pt`; HRNet-W48 encoder + U-Net++ decoder + boundary head와 preprocessing metadata를 그대로 유지한다.
- 네 specimen-group fold, seed 1. Matched control과 candidate는 같은 고정 split, 같은 item-seeded crops/augmentations, batch order, 8-epoch continuation, optimizer/LR/schedule, inference recipe를 사용한다.
- 학습 조건: patch 512, input scale 2.0, width normalization, 12 crops/image, batch 4, accumulation 2, AdamW weight decay `1e-4`, cosine schedule, AMP, gradient clip 1.0; encoder LR `3e-5`, decoder/head LR `1e-4`.
- control은 기존 CE+Dice+boundary loss만 사용한다. Candidate는 이를 유지하면서 추가 component loss 하나를 적용한다.
- 증강 완료 뒤 CPU에서 8-connectivity로 Al3Ni(class 1)와 Eutectic Si(class 3)를 분리한다. augmented-mask area `[4,16)`, `[16,64)`, `[64,256)` px의 crop 경계 비접촉 component만 포함한다. 이는 nominal 2× 선형 scale에서 native `[1,64)` px를 근사하며 정확한 변환은 아니다. 경계 접촉 성분은 truncation 가능성 때문에 제외한다. Al과 Primary-Si small component, 두 target class의 native 64–256 px component는 auxiliary loss에서 제외한다.
- 성분별 손실은 `-log(mean pixel p(true class | component))`. component 수로 crop/bin이 지배되지 않도록 crop×target-class×size-bin cell별 성분 평균을 낸 뒤 nonempty cell들을 동일 가중 평균한다. 두 target class도 cell 존재 시 동등한 cell 가중을 받는다. Pixel CE+Dice와 boundary loss는 정밀도 및 다른 클래스를 계속 감독한다.
- component coefficient는 fold별 outer-train-only 8-batch decoder-feature gradient norm 비율로 `0.10 / median(aux/main)`을 계산한다. optimizer step 0회, nonempty batch 최소 6/8, coefficient `[1e-5,1]`를 요구한다. mIoU, valid, test로 calibration하지 않는다.
- inference: patch 512, overlap 170, scale 2.0, flip4 TTA. Threshold/routing/TTA/weight search는 하지 않는다.

## 평가와 사전고정 채택 gate

70 train images를 59 specimen groups 단위로 정확히 한 번 OOF 채점한다. 공식 image-wise 지표(이미지마다 네 class IoU 평균 후 이미지 평균, empty-union=0)를 쓴다. Primary comparison은 component candidate 대 compute-matched continuation control이며, Model392 seed1 OOF는 context로만 추가 보고한다. Paired specimen-group bootstrap 20,000회와 fold/class/Step4 결과를 기록한다.

component hit는 각 원본 해상도 GT component에서 true-class prediction recall이 `>=50%`인 것으로 정의한다. Target tiny `[1,64)` 성분과 loss에서 제외한 medium `[64,256)` guard를 Al3Ni/Eutectic 각각 계산한다.

다음을 모두 만족해야 seed1 내부 gate 통과로 본다: overall mIoU delta `>=+0.20%p`, paired group-bootstrap CI 하한 `>0`, fold 4개 중 3개 이상 양수 및 최악 fold `>=−0.30%p`; Al3Ni/Eutectic class-IoU delta 각각 `>=0`, 나머지 class delta `>=−0.50%p`; Step4 delta `>=−0.30%p`; 네 개 class×size component hit-rate delta 점추정 `>=0`, 각 paired CI 하한 `>=−5%p`. 통과해도 최종 채택 전 같은 lock으로 seed2에서 동일 gate를 재현해야 한다. 그 전 공식 valid/test는 읽지 않으며 test는 봉인한다.

## 재현 산출물

- loss: `model404_component_recall_loss_v4.py`; CPU smoke: `model404_component_recall_smoke_v4.py`.
- v4 trainer/calibrator wrapper는 이전에 잠근 공통 optimizer/data recipe를 재사용하되 v4 loss 모듈을 명시적으로 주입한다. 공유 trainer/calibrator와 모든 관련 source hash도 v4 lock에 포함한다.
- scorer: `score_model404_component_recall_oof_v4.py`; queue/runner: `queue_model404_component_recall_v4.sh`, `run_model404_component_recall_oof_v4.sh`; immutable source receipt: `results/model404_preregistration_lock_v4.json`.
- Model403의 P100 file lock을 획득하지 못하면 즉시 종료한다. 현재 valid/test는 열지 않았다.

현재까지는 구현 및 합성 smoke 단계이며 성능 향상을 주장하지 않는다.
