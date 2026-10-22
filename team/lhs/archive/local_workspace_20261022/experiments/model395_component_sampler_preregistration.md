# Model395: component-balanced minority-anchor sampler

## 목적과 해석 범위

Model388의 train-only expected-dose 감사와 held-out OOF component-headroom 감사에서 제안 후보가 겨냥할 수 있는 신호가 관찰됐다. 이 결과를 바탕으로 **한 가지 변경만** 비교한다: positive crop 확률, minority class 선택 규칙, 나머지 학습 recipe는 유지하고, Al3Ni/Eutectic Si anchor를 class 안의 연결 성분 중 면적 `>=64 native pixels`인 성분에서 균등 선택한 뒤 그 성분 안의 pixel에서 균등 선택한다. 그런 eligible component가 없으면 기존 pixel-uniform sampler로 fallback한다.

기존 Model382 OOF의 세부 오류 분석이 이미 열렸으므로 이 비교는 독립 confirmatory test가 아니라 선택 편향이 남는 **development-only paired experiment**다. OOF 성과는 유효성 확인에 쓰되 독립 일반화 성능처럼 표현하지 않는다. 공식 valid/test 성능은 OOF 결과를 보기 전에는 측정하지 않으며 test 데이터와 라벨은 끝까지 열지 않는다.

## 데이터와 분할

- 고정 grouped CV: `/workspace/komap_model1/data_model322_group_cv`의 `train_fold1..4`와 대응하는 `oof_fold1..4`만 사용한다. 한 specimen group은 한 fold에만 속한다.
- 각 fold에서 candidate는 해당 `train_foldN`만 학습하고, OOF 추론/paired 평가는 해당 `oof_foldN`만 사용한다. 네 fold 결과는 각 학습 이미지에 대해 정확히 한 번의 out-of-fold 예측을 구성한다.
- CPU 노출량 사전검증은 training-fold 이미지/마스크만 읽는다. OOF 확률, valid.zip, test.zip은 읽지 않는다.
- 공식 valid와 test는 모든 seed OOF adoption gate가 끝날 때까지 사용하지 않는다. test.zip은 계속 봉인한다.

## 고정 학습 recipe

- Model336 HRNet-W48 encoder + SMP U-Net++, ImageNet 초기화, boundary auxiliary head 사용.
- patch 512, native crop 256 (`input_scale=2`), image당 12 crops/epoch.
- batch 4, accumulation 2, workers 4, drop-last, AMP.
- 70 epochs, AdamW lr `3e-4`, encoder lr multiplier `1.0`, weight decay `1e-4`, cosine schedule, boundary loss weight `1.0`; CE/Dice는 기존 무가중치 recipe.
- seed1이 첫 비교다. `val_every=0`; 학습 중 quick validation을 실행하지 않는다.
- treatment 외에 source, optimizer, schedule, augmentation, split, inference/TTA를 변경하지 않는다.
- Model382의 `random.Random` dataset RNG가 fork된 worker 사이에서 같은 상태로 복제되고 `seed_worker`가 이 객체를 reseed하지 않는 기존 동작을 양 arm에 그대로 둔다. 이 실험에서는 worker RNG 수정(Model390)을 섞지 않는다.
- 한 번에 P100 한 장에서 fold1→2→3→4 순서로 실행한다. GPU 점유율/메모리 조회는 하지 않는다. 모든 실험 산출물은 `/workspace/komap_model1` 아래 둔다.

## CPU-only realized-dose gate (학습 전)

실제 DataLoader 동작(4 workers, shuffle, batch 4, drop-last, 매 epoch worker 재생성)을 그대로 재현해 70 epochs의 anchor 선택 수와 native crop 내 GT pixel 노출량을 측정한다. 이건 점수 예측이 아니며 모델을 학습하지 않는다. folds3–4가 주 gate, folds1–2는 설명용이다.

모두 만족할 때만 seed1 OOF 학습을 시작한다:

1. folds3와 4 각각에서 `[64,256)` 성분 anchor의 candidate/legacy 실현 비율이 Al3Ni와 Eutectic Si 모두 `>=2.0x`.
2. folds3와 4 각각에서 두 목표 class 중 적어도 하나의 같은 bin crop GT-pixel 노출 비율이 `>=1.25x`.
3. folds3와 4 각각에서 두 class 모두 목표 성분 anchor가 5개 이상 specimen groups에 분포하고, 단일 group의 anchor 비중은 50% 미만.

실패하거나 data/hash/smoke 확인이 맞지 않으면 학습은 시작하지 않는다. 기준은 결과를 본 뒤 완화하지 않는다.

## Paired OOF adoption gate

candidate seed1의 70-epoch 네 fold OOF 결과를 기존 Model382 seed1과 동일한 image-wise 4-class mIoU(`empty union=0`, 이미지별 class IoU 네 개 평균 후 이미지 평균)로 비교한다. 이 비교용 evaluator와 source/data/artifact lock은 candidate OOF 결과를 열기 전에 고정한다. 불확실성은 outer fold 안에서 specimen groups를 복원추출하는 paired bootstrap 20,000회(seed 395)로 계산한다.

seed1을 다음 조건을 **모두** 만족할 때에만 추가 seed2 실험으로 진행한다:

1. pooled OOF image-mean mIoU 향상 `>= +0.005` absolute.
2. paired group-bootstrap 95% CI 하한 `> 0`.
3. 네 fold 중 3개 이상에서 delta 양수이고, 최악 fold delta `>= -0.003`.
4. 네 class 각각의 IoU delta `>= -0.005`; Al3Ni와 Eutectic Si의 paired CI 하한은 각각 `>= -0.010`.
5. Step4 이미지 subset의 image-mean mIoU delta `>= -0.005`.
6. `<64 px` GT component의 50%-recall hit-rate paired CI 하한이 `>= -0.05`.

모든 area bin의 component hit-rate(50% recall 기준), GT-component FN pixel rate, 해당 bin이 하나 이상 존재하는 이미지에서의 class FP pixel 수를 보고한다. 사전 지정 mechanism check는 `[64,256)` Al3Ni/Eutectic Si FN pixel rate 변화다. 이 기전 통계만으로 adoption하지 않는다.

seed1 gate를 통과하면 같은 recipe와 고정 seed2로 네 fold를 실행한다. seed2는 paired CI 하한 `>0`, 3/4 folds positive 및 같은 guardrail을 독립 통과해야 하며, 두 seed 평균 향상은 `>=+0.005`여야 한다. 실패하면 sampler 채택을 중단한다.

두 seed gate를 모두 통과한 경우에만 고정 recipe로 전체 train set을 학습하고 공식 valid를 **한 번** 평가한다. valid 결과를 보고 threshold, TTA, routing, calibration을 재조정하지 않는다. 이 단계에서도 test는 읽지 않는다.

## 실행 및 보고 제한

- 학습 시작 전 candidate/legacy synthetic bitwise check, component sampling smoke, worker-faithful simulation smoke, source diff, checkpoint/OOF provenance 검사를 통과해야 한다.
- 출력에는 fold 단위 진행 정보만 표시한다. per-image OOF 점수는 stdout/commentary에 출력하지 않는다.
- 중간 체크는 10분 간격으로 process/state metadata만 확인한다. GPU occupancy/memory를 질의하지 않는다. Claude Code Opus는 읽기 전용으로 현재 병목/실험 해석을 검토한다.
- 각 실행은 별도 output directory와 append-only manifest를 사용한다. 기존 파일을 덮어쓰지 않으며 lock/hash 불일치 시 중단한다.
- E2 oracle headroom은 달성 가능한 성능의 상한이지 기대 성능 증가가 아니다. 시뮬레이션된 anchor 비율도 성능 향상 예측치가 아니다.
