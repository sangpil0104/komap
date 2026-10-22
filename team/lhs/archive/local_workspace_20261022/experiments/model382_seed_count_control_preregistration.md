# Model382 사전등록: HRNet seed 변동과 모델 수 효과의 분리

## 목적

Model379의 고정 3-model 평균이 Model374 대비 `+0.4034%p`였지만 사전등록된 `+0.50%p` 채택 문턱에는 미달했다. Model374 seed 복제를 추가해 같은 recipe의 모델 수 증가만으로 비슷한 이득이 나는지, Model376/377 recipe 다양성이 같은 모델 수의 seed 다양성보다 낫는지 분리한다. Claude Code Opus는 Model379 결과 검토 뒤 이 matched-count 비교를 우선 권고했다.

## 고정 학습 설정

- Recipe는 Model374와 완전히 동일한 HRNet-W48/U-Net++: patch `512`, input scale `2.0`, 12 samples/image, batch 4 × accumulation 2, 70 epochs, AdamW LR `3e-4`, weight decay `1e-4`, boundary-head weight `1.0`, 동일 augmentation/loss/fold.
- P100 현재 Model336/Model374 training code tree SHA-256은 `89c6d41b7ad5bb88e5cf876731954ac2aeedf3aed495b9a6ae8d3e446a183d00`이며, Model374 fold1·2 OOF inference manifest의 기록과 정확히 일치한다. Model382는 이 코드 tree를 고정 사용한다.
- 유일한 변경은 seed이며 **seed 1과 2를 사전에 고정**한다. 기존 seed0 Model374, Model376, Model377 OOF maps는 그대로 쓴다.
- 각 seed의 4개 고정 specimen-group fold를 모두 학습해 총 8회 학습한다. 각 fold는 마지막 epoch 70 checkpoint만 쓰고, validation checkpoint 선택은 없다. 70개 train image를 각 seed에서 exactly once OOF로 예측한다.
- 출력/체크포인트/OOF를 seed·fold 별도 디렉터리에 보존하고 SHA-256 manifest를 확인한다. 현재 Google Drive 권한 복구를 기다리고 있으므로 학습은 P100의 영구 작업공간에 우선 저장하고, Drive가 연결되면 파일을 복사한 뒤 SHA-256을 대조한다. Drive 복구 여부는 학습 설정·fold·seed를 바꾸지 않는다. 중단 복구는 같은 seed/fold만 재개한다. 모델 품질에 따른 seed 교체·재시도는 하지 않는다.
- valid/test 및 라벨은 학습·선택·중간 gate에 쓰지 않는다. Gamma, calibration, TTA, morphology, route 탐색도 하지 않는다.

## 고정 비교 arm

각 이미지에는 그 이미지의 held-out fold에서 나온 예측만 사용하며, 아래 모든 평균은 softmax 확률의 산술평균 후 argmax다.

| Arm | 구성 | 수 | 의미 |
|---|---|---:|---|
| A | Model374 seed0 | 1 | 기준 |
| S1 / S2 | Model374 seed1 / seed2 각각 단독 | 1 | seed 간 변동 |
| B | Model374 seed0 + seed1 + seed2 | 3 | 같은 recipe, 3-seed 평균 |
| C | Model374 seed0 + Model376 seed0 + Model377 seed0 (기존 Model379) | 3 | 서로 다른 recipe, 3-model 평균 |

B와 C는 모델 수·동일 이미지·외부 fold를 맞춘 primary 비교다. C는 이미 고정된 Model379이며 다시 튜닝하지 않는다. Arm C−B, B−A, S1/S2−A만 보고하고, B/C 중 결과가 더 좋은 쪽을 사후 선택하지 않는다.

## 지표 및 불확실성

- 공식 metric: image-wise 4-class mIoU, empty-union=0, 70 images/59 specimen groups.
- 매 OOF image에서 class IoU를 다시 계산하고, 모델/앙상블간 paired delta를 쓴다.
- paired group bootstrap은 **outer fold 층화**한다. 각 fold 안에서 그 fold의 specimen groups 수만큼 복원추출하고, 해당 group의 모든 image를 포함한다. 20,000회, bootstrap seed 382, percentile 95% CI로 고정한다. 각 replicate에서 A/B/C의 같은 image/group 표본을 짝지어 비교한다.
- 기존 Model379의 비층화 CI와 직접 같은 숫자로 취급하지 않는다. 관측값은 두 절차 모두 별도로 보고해 비교 가능성과 재현성을 보존한다.
- S0/S1/S2 단독 OOF 점수의 범위·표준편차는 seed noise의 거친 기술통계로만 보고한다 (`n=3`임을 명시). 각 seed의 paired per-image 차이·rank correlation도 함께 보고한다. Group bootstrap은 dataset 표본 불확실성이며 seed 변동 CI가 아니다.

## 계층형 판정 규칙 (결과 열람 전 고정)

1. **Diversity 대 같은 recipe의 seed 평균:** `C−B`의 point estimate가 `≥+0.20%p`, group-bootstrap 95% CI 하한 `>0`, 4개 fold 중 3개 이상 양수이면 recipe diversity 신호로 본다. 세 조건 중 하나라도 미달하면 diversity를 입증하지 못한 것으로 보고, 사후 기준 완화는 금지한다.
2. **모델 수 효과:** diversity 조건 미달, `C−B` CI가 0을 포함, 그리고 `B−A` CI 하한 `>0`일 때만 seed-count 효과 신호로 본다. 그 외에는 원인 불확실로 판정한다.
3. **후보 adoption:** 1단계가 diversity 신호면 고정 C, 2단계가 count 효과 신호면 고정 B만 후보로 삼는다. 해당 arm은 A 대비 다음 Model379 adoption gate를 모두 만족해야 후속 후보가 된다: Δ `≥+0.50%p`, paired 95% CI 하한 `>0`, 4개 fold 중 최소 3개 개선, 최악 fold `≥−0.30%p`, 어떤 class도 `−0.50%p`보다 크게 하락하지 않음, Step4 delta `≥−0.50%p`.
4. `C−B` CI 상한 `<0`이면 B가 상대 우위인 evidence로 보고하되, B가 별도의 adoption gate를 통과하지 않으면 채택하지 않는다. `B`와 `C`를 결과로 비교해 더 높은 점수를 고르는 best-of 선택은 하지 않는다.
5. 채택 기준 미달 시 seed/ensemble 계열을 이 결과만으로 반복 확장하지 않는다. `+0.5%p` 미만의 양수 효과는 통계적으로 양수여도 실용 gate 실패로 미채택한다.

## 순서와 유의사항

- **2026-10-02 사전등록 보완(새 seed 결과 열람 전):** Claude Code Opus 검토에서 Model381은 기하 진단이고 Model382는 고정 학습 recipe의 seed-count 대조이므로 서로 의존하지 않는다고 판단했다. 따라서 Model381의 기존 OOF 복구·CPU 진단과 Model382의 seed 1/2 학습을 병렬로 진행할 수 있다. P100에서는 비어 있는 GPU 7 한 장만 사용하고, 8개 seed×fold 학습은 순차 실행한다. 이 변경은 학습 recipe, fold, 결과 지표, 기존 adoption gate를 바꾸지 않는다. Model382의 B/C 비교와 adoption 판정은 Model374/376/377의 모든 동일-fold OOF 확률 아카이브를 복구하고 fold·이미지·metric 재현 검증을 통과할 때까지 실행하지 않는다.
- 학습 전에 Model381을 완료해야 한다는 선행조건은 제거한다. 대신 Model381의 CPU geometry diagnosis는 아카이브 복구 뒤 병렬로 진행한다. Model382의 여덟 학습 결과를 점수 기준으로 중간 중단·seed 교체하지 않는다.
- Model378 SVLS의 기존 recipe와 adoption/mechanism gates는 바꾸지 않는다. 실행은 Model381 gate 및 Model382 seed/count 분석 뒤에 순차 판단하며, 이 문서의 결과 arm에는 포함하지 않는다.
- OOF gate 통과는 independent test나 공식 validation 성능을 증명하지 않는다. Gate를 통과한 고정 arm에 한해서만 full-train 고정 학습 및 official valid 20장 단회 최종 보고를 별도로 계획한다. valid 결과로 arm/seed/가중치를 다시 고르지 않는다.
- 이 비교는 과거 loss/model 선택에 조건부이며, held-out fold가 가리는 것은 학습 image/group이지 실험 아이디어 선택 과정이 아니다.
