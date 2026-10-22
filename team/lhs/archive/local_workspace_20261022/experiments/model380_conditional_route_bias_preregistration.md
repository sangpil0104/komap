# Model 380 사전등록: Model341 anchor 이후 HRNet 라우트 선택 편향 audit

## 목적과 범위

- Model349 grouped-LOSO는 gamma347 선택을 held-out group 바깥에서 했지만, gamma336 및 Model341/339 anchor 값은 전체 valid에서 정한 채 고정했다. 따라서 기존 `−0.0864%p`는 부분 cross-fit이지 전체 route search의 무편향 추정치가 아니다.
- Model380은 **기존에 고정된 Model341 anchor에 조건부로만** gamma336 및 gamma347 선택을 outer specimen-group 밖에서 하여 Model346 절차와 Model349 절차를 paired 비교한다. 과거 Model341/339 route 선택 편향이나 checkpoint/학습 선택 편향을 제거하지 않으며, 독립 test 성능 추정으로 주장하지 않는다.
- 분석은 CPU-only, 저장된 valid 확률맵과 valid mask만 사용한다. GPU 추론·재학습·test 입력/라벨 사용은 하지 않는다.

## 고정 outer-group 절차

각 valid specimen group을 한 번씩 held-out으로 두고, 나머지 groups만 route coefficient 선택에 사용한다. Metric은 official image-wise four-class mIoU, empty-union=0이다.

1. 기존 Model346 결과의 625-gamma336 grouped-LOSO에서, 각 held-out group에 대해 선택된 gamma336을 그대로 취한다. 기존 결과상 18/19 groups는 `[0,0,1,0.25]`, A71은 `[0.25,0,1,0.25]`를 선택했다.
2. **Arm 346:** 선택 gamma336을 held-out group의 이미지에 적용한다.
3. **Arm 349:** gamma336은 Arm346과 동일하게 두고, 해당 gamma336을 고정한 상태에서 나머지 groups로 gamma347의 동일 625개 grid를 고른 뒤 held-out group에 적용한다. 기존 gamma336과 같은 18개 group은 저장된 Model349 grouped-LOSO 선택을 쓴다. 다른 gamma336을 고른 A71 fold는 해당 subset에서 gamma347 grid를 한 번 다시 계산한다.
4. 두 arm 모두 per-image 예측을 공식 지표로 다시 채점한다. 결과 비교는 같은 held-out images/groups의 paired delta이며, 19-group bootstrap과 group-equal sensitivity를 함께 보고한다. gamma 선택 빈도와 A71의 점수 변화를 별도로 공개한다.

기존 full-grid 선택값은 gamma347 `[0,0,0.25,0]` (17 groups), `[0.25,0,0.25,0]` (2 groups)뿐이다. 이 선택을 재채점하는 global-anchor 행렬은 해당 두 설정이 모두 포함되는 순서상 앞 131개를 저장한다. 이 제한된 행렬로 새 gamma를 고르지 않으며, 선택값은 기존 full 625-grid grouped-LOSO 결과에서만 가져온다. A71의 alternate gamma336에 대해서만 full 625-grid를 다시 계산한다.

## 해석 규칙

- 주 결과는 **Model341 anchor에 조건부인** cross-fitted gamma336/gamma347 추가 절차 간 paired 차이다. 전체 Model349 파이프라인의 독립 성능이나 validation 80% 달성을 증명하지 않는다.
- 사후에 gamma grid, anchor, group assignment, metric 또는 후보를 바꾸지 않는다. raw valid 최고점으로 다시 선택하지 않는다.
- 19 groups라 CI가 넓을 수 있다. CI가 0을 포함하면 개선/열위를 주장하지 않고 불확실로 보고한다.
- P100에서는 CPU worker를 제한하고 `CUDA_VISIBLE_DEVICES`를 비워 GPU를 사용하지 않는다. 임시 입력/코드 및 결과는 `/workspace/komap_model1` 하위에만 둔다.
