# Model383 사전등록: 3×3 Boundary Label Relaxation

작성: 2026-10-02, Model383 실데이터 결과 열람 전. Model382는 별도 GPU 작업으로 진행 중이며, 이 문서는 Model382 프로토콜을 변경하지 않는다.

## 근거와 가설

Model374 grouped train-OOF에서 오류의 약 86%가 기존 GT 경계 띠 안에 있고, Al3Ni↔Eutectic Si 경계 혼동은 비대칭이다. Model377처럼 특정 오류 방향을 직접 줄이는 항은 반대 방향 오류를 함께 늘려 채택되지 않았다. 따라서 한쪽 클래스를 밀기보다 GT 경계의 이웃 클래스 집합 중 어느 쪽이든 확률 질량을 허용하면 annotation boundary ambiguity에 더 견고할 수 있다는 가설을 한 번 시험한다.

Zhu et al.은 경계 픽셀에서 주변 3×3 GT label 집합의 확률 합을 최대화하는 Boundary Label Relaxation을 제시했다. 여기서는 그 아이디어만 고정 HRNet recipe에 적용하며, 원 논문의 도메인·모델·성능 결과를 KoMaP에 그대로 일반화한다고 가정하지 않는다. [논문(arXiv, CVPR 2019)](https://arxiv.org/abs/1812.01593)

## CPU 사전 게이트

학습 전에 Model374의 **전체 70장/59-group train-OOF 확률맵**으로만 아래를 계산한다. 부분 fold는 탐색 요약만 허용하며 GO/NO-GO 결정에 사용할 수 없다.

1. 각 픽셀의 allowed class set을 그 픽셀 중심의 3×3 GT 이웃에 실제로 존재하는 class 집합으로 정의한다. 이는 학습 시 relax되는 픽셀 집합과 동일하다.
2. **경계 oracle 상한**은 이 band의 예측 label을 GT로 바꾸고, empty-union=0 image-wise four-class mIoU를 다시 잰다. 상승폭이 `+1.00%p` 미만이면 BLR을 학습하지 않는다.
3. band에서 Euclidean distance가 3px보다 큰 잘못된 픽셀 비율이 `50% 이상`이면, 오류가 경계보다 큰 내부 영역에 집중된 것으로 보고 BLR을 학습하지 않는다.

oracle은 GT를 직접 사용하므로 배포 가능한 모델 점수가 아니며, 오직 학습 비용 전 feasibility 상한이다. 계산 코드: [gate](model383_boundary_relaxation_gate.py), [합성 smoke](model383_boundary_relaxation_gate_smoke.py).

## 고정 학습 설정

- Model374와 같은 HRNet-W48/U-Net++, 동일 4-fold specimen-group split, seed `0`, 70 epochs, patch `512`, input scale `2.0`, 12 crops/image, batch 4 × accumulation 2, AdamW 및 동일 augmentation/scheduler.
- 바꾸는 것은 segmentation loss의 CE 부분 한 가지다. 각 위치에서
  `L_BLR = logsumexp(all logits) − logsumexp(logits for classes present in the 3×3 GT neighborhood)`
  를 평균한다. 주변 class가 하나뿐인 interior 위치에서는 표준 CE와 같고, 경계에서는 이웃 class 집합의 총 확률만 높인다.
- Model374의 Dice 항(`0.5`)과 보조 boundary-head loss/weight(`1.0`)는 그대로 둔다. CE와 Dice의 계수도 기존 `0.5/0.5` 그대로다. label-radius, loss weight, seed, checkpoint epoch 탐색은 없다.
- 네 fold 모두 마지막 epoch만 평가한다. train OOF 이외 valid/test는 학습·선택·중간 점수 측정에 쓰지 않는다. OOF prediction은 Model374와 동일 fold assignment에서 각 이미지가 held-out인 모델에서만 생성한다.
- 예상 계산량은 현재 전용 P100 GPU7 측정 속도에 따라 약 8 GPU-hours(4 fold × 약 2시간)지만, Model382가 끝나고 난 뒤에만 실행한다. 중간 seed 성능에 따른 중단·재시도는 없다.

## 사전 고정 판정

### 단일 모델의 기전 확인

Model383을 Model374와 짝지은 official OOF mIoU 차이와 59 specimen-group paired bootstrap 95% CI(10,000회, seed 383)를 보고한다. 다음을 전부 만족해야 단일 모델 후보로 인정한다.

- 전체 delta `≥ +0.40%p`, CI 하한 `> +0.10%p`.
- 4개 fold 중 적어도 3개에서 양수.
- 각 class mean IoU 하락이 `0.50%p`를 넘지 않음.
- Al3Ni→Eutectic 및 Eutectic→Al3Ni 경계 오류율이 각각 1.00%p 초과 악화되지 않음.
- 개선분이 relaxed boundary band 주변에 나타나고, interior만의 개선에 의존하지 않음.

하나라도 미달하면 Model383 단독은 미채택이다. 이 평가는 과거 OOF를 이용해 연구 아이디어를 고른 selection-conditional 실험이며 완전한 nested-CV 추정치가 아니다.

### model-count matched 보조 비교

Model382 seed1 OOF 확률이 무결성 검사를 통과해 있을 경우에만, 아래 고정 4-member 평균을 한 번 비교한다.

- BLR 후보: `{374 seed0, 376 seed0, 377 seed0, 383 seed0}`
- 동일-count control: `{374 seed0, 376 seed0, 377 seed0, 374 seed1}`

두 arm은 같은 이미지/fold의 softmax 확률 uniform mean 후 argmax다. 후보의 OOF가 `80.00% 이상`, control 대비 `≥ +0.25%p`, paired group-bootstrap CI 하한 `>0`, 그리고 fold별·class별 no-harm 조건(각 class 하락 `≤0.50%p`)을 모두 만족해야만 80% 목표 후보로 본다. Model376/377 자체의 과거 OOF 선택 편향은 양 arm에 공유되므로 결과는 selection-conditional로만 해석한다. seed1 자료 또는 원본 OOF가 없으면 이 비교는 생략하며 다른 조합 탐색으로 대체하지 않는다.

## 실행 순서

1. Model382 8개 seed×fold 학습 완료 및 SHA-256 manifest 검증.
2. Model376의 실제 loss 구현을 확보해 BLR와 동등한 neighborhood-union likelihood인지 확인한다. 동등하면 중복 실험이므로 Model383 학습을 취소한다. 현재 P100에는 Model376 코드가 없으므로 Drive 복구 또는 provenance가 확인되는 소스 번들이 필요하다.
3. Model374/376/377 동일-fold archive 복구, image ID/fold/group 일치와 각 raw mIoU 재현성 확인.
4. [추가 no-training gate](model383_no_training_gate_addendum.md)의 untouched folds3–4에서 기하, confidence, Model376 attribution 조건을 모두 검증한다.
5. Model383 CPU 사전 게이트의 전체 70장 판정. 어느 gate든 no-go면 학습하지 않는다.
6. 앞선 gate를 모두 통과한 경우에만 Model383 4-fold 학습을 시작하고, 사전 고정된 단일·matched-count 지표를 한 번 계산한다.
7. 결과와 방법 선택 편향을 기록한 후, 최종 고정 후보만 official valid 20장으로 한 차례 보고한다. valid 점수로 재튜닝하지 않는다.
