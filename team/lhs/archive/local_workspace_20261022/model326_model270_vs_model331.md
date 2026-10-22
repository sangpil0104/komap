# Model326+Model270 융합과 Model331 상세 비교

작성 기준: 2026-09-28. 점수는 모두 `valid.zip` 20장 기준이며, `test.zip`에는 정답 mask가 없어 test 점수가 아니다.

## 요약

- **Model326 단독**은 EfficientNet-B4 U-Net에 full-image 통계 입력을 넣어 새로 학습한 branch다. 입력 표현 변경만으로는 기존 Model322 대비 70장 OOF 개선이 확인되지 않았다.
- **Model326+Model270**은 두 모델을 함께 재학습한 것이 아니라, Model270 inference route의 일부 클래스 확률을 Model326 B4 확률로 섞은 후처리/앙상블이다. empty-union=0 기준 valid **79.1937%**, absent-class 제외 기준 **80.0186%**다.
- **Model331**은 새 신경망 학습이 아니라, **Model322 B4**와 Model270 route의 클래스별 혼합비를 CPU에서 다시 탐색한 route다. valid는 **79.1914% / 80.0359%**(empty=0 / absent 제외)다.
- 보수적인 empty=0 점수 최고이자 grouped held-out 개선 근거가 있는 쪽은 **Model326+Model270**이다. Model331의 absent 제외 raw 점수는 가장 높지만, grouped 신뢰구간이 0을 포함해 채택하지 않았다.

## 평가 지표와 비교 시 주의점

대회 image-wise mIoU는 각 이미지에서 네 상(Al3Ni, Primary Si, Al, Eutectic Si)의 IoU를 평균하고, 그 image score를 평가 이미지들에 대해 평균한다. 다만 어떤 이미지에서 특정 클래스의 union(TP+FP+FN)이 0일 때 처리법이 결과에 영향을 준다.

- `empty-union=0`: 해당 클래스 IoU를 0으로 놓고 네 클래스 모두 평균
- `absent 제외` 또는 `NaN-skip`: union이 0인 클래스는 그 이미지의 클래스 평균에서 제외

이 문서에서는 두 방식을 함께 쓴다. 따라서 absent 제외로 80%를 넘는 결과를 empty=0 기준의 80% 달성이라고 바꿔 말하지 않는다. 점수는 픽셀 전체를 한 confusion matrix로 합산하는 global mIoU와도 다르다.

## Model326: full-image-stat B4 branch

### 구조와 학습

- 네트워크: ImageNet 사전학습 EfficientNet-B4 encoder를 사용하는 U-Net, 4-class segmentation head
- 입력: 한 장의 grayscale 이미지를 scale 3으로 전체 resize한 뒤 만든 세 채널 `[raw grayscale, full-image z-score, full-image empirical CDF]`
- 정규화 통계와 CDF: crop별이 아니라 resize된 전체 이미지에서 계산하고 해당 이미지 crop과 추론 입력에 동일하게 적용
- 학습: 4-fold specimen-group split, fold마다 train 52–53장 / OOF holdout 17–18장; 70장 전체에서 이미지별 OOF 예측 1회
- 고정 recipe: crop 1536, batch 2, 16,000 optimizer updates/fold, AdamW `lr=3e-4`, `weight_decay=1e-2`, unweighted CE + Dice (`dice_w=1`), clDice 비활성, seed 0
- crop sampling은 area-weighted다. 16,000은 epoch 수가 아니라 Model322와 예산을 맞춘 update 수이며, valid 점수로 중간 checkpoint를 고르지 않고 마지막 checkpoint를 쓴다.

실행 코드: [`model326_hybrid_fullstats/KoMaP/train.py`](model326_hybrid_fullstats/KoMaP/train.py), [`run_model326_hybrid_fullstats_group_oof.sh`](run_model326_hybrid_fullstats_group_oof.sh).

### Model326 단독 결과

| 범위 | empty-union=0 | absent 제외 | 해석 |
|---|---:|---:|---|
| Valid 20장 | 78.0400% | 78.7090% | 단일 B4 fold ensemble |
| Train 70장 grouped OOF | 약 78.53% | 78.78% | Model322와 paired 비교용 |
| Model322 대비 paired OOF 차이 | -0.0050%p | -0.0034%p | group-bootstrap 95% CI 각각 `[-0.2850,+0.2915]`, `[-0.2837,+0.2935]`%p |

따라서 full-image 통계 입력은 단독으로는 Model322보다 낫다는 증거가 없었다. 이후의 이점은 이 branch가 Model270 route와 보완적으로 섞일 때 나타났다.

## Model326 + Model270: 클래스별 확률 융합

### 무엇을 섞었나

이는 공동 학습이나 loss 결합이 아니다. 이미 생성된 확률 지도를 inference 시점에 클래스별로 혼합한다. Model270 route를 기준으로 Al3Ni, Primary Si, Al에는 Model326 B4를 각각 `gamma=0.50`으로 섞고, Eutectic Si는 Model269 branch를 `gamma=0.50`으로 유지한다. 확률 혼합은 클래스 채널마다 `(1-gamma) × 기존 확률 + gamma × 새 branch 확률`을 계산한 뒤 채널 합을 다시 정규화한다.

팀 B4 확률 archive는 `[Al, Al3Ni, Primary Si, Eutectic Si]` 순서이므로, 혼합 전에 route의 `[Al3Ni, Primary Si, Al, Eutectic Si]` 순서로 재배열한다. 이 명시적 mapping으로 클래스 ID permutation 오류를 방지한다.

Model270 기준 route의 나머지 inference 설정도 유지한다: `prior_scale=1.25`, `beta_scale=0.80`, `al_delta=0`, `eut_delta=0.15`, `gaussian_blend=0.50`, `eut_edge_blend=0.25`, `cleanup_threshold=128`. 따라서 점수는 B4 단독 성능이 아니라, 해당 base route와 확률 보정·공간 후처리를 모두 포함한 결과다.

### 점수와 grouped 평가

| 항목 | 결과 |
|---|---:|
| Valid image-wise mIoU, empty=0 | **79.1937%** |
| Valid image-wise mIoU, absent 제외 | **80.0186%** |
| Model270 단독 대비 grouped held-out route 개선 | +0.4728%p; 15/20장 개선, 19 specimen groups |
| Group-bootstrap 95% CI | **[+0.0933,+0.8838]%p** |

이 비교에서는 19개 specimen group을 하나씩 holdout하고 나머지 group으로 route를 선택한 뒤 held-out group을 평가했다. 이 방식은 같은 시편의 이미지가 train/test 양쪽에 섞이는 누수를 줄인다. CI 하한이 0보다 커 Model270 대비 보완 개선 근거가 Model331보다 강하다. 다만 20장 validation에 기반한 선택이므로 완전히 독립된 test 성능을 보증하는 것은 아니다.

결과 원본: `/workspace/komap_model1/outputs/model326_model270_fusion/route_probe.json`.

## Model331: Model322 B4의 클래스별 혼합비 미세 탐색

### 구조와 탐색 방식

- 별도 네트워크를 학습하지 않았다. 입력 branch는 기존 **Model322 팀 B4** valid 확률이고, 비교 기준은 Model270 route다. Model326 B4 branch를 쓴 실험이 아니다.
- 앞선 Model327 탐색에서 Al3Ni/Primary Si/Al의 B4 혼합률 후보를 더 세밀하게 보려고 각 클래스 gamma를 `{0.50, 0.70, 0.80, 0.90, 1.00}`에서 독립 조합했다. Eutectic Si는 약한 B4 branch로 바꾸지 않고 Model269 `gamma=0.50`으로 고정했다.
- 기존 후보와 전체 교체 후보를 포함해 157개 route를 exact spatial-refinement 평가했다. GPU 학습이 아니라 저장된 확률을 사용하는 CPU 후처리 탐색이다.
- 최고 route: Al3Ni Model322 B4 `0.50`, Primary Si Model322 B4 `0.70`, Al Model322 B4 `0.80`, Eutectic Si Model269 `0.50`. Model270의 prior/edge/Gaussian/cleanup 설정은 유지했다.

탐색 코드는 [`model323_model270_branch_route_probe.py`](model323_model270_branch_route_probe.py)이고, Model331의 157개 후보 설정은 `/workspace/komap_model1/outputs/model331_b4_fine_gamma_route/route_probe.json`에 보존돼 있다.

### 점수와 채택 여부

| 항목 | 결과 |
|---|---:|
| Valid image-wise mIoU, empty=0 | 79.1914% |
| Valid image-wise mIoU, absent 제외 | **80.0359%** |
| Model270 대비 grouped held-out route 개선 | +0.4408%p; 16/20장 개선, 19 specimen groups |
| Group-bootstrap 95% CI, empty=0 | `[-0.070,+0.960]%p` |
| Group-bootstrap 95% CI, absent 제외 | `[-0.112,+1.156]%p` |

Model331은 Model327 대비 valid 최고점이 empty=0 `+0.0016%p`, absent 제외 `+0.0029%p`만 올랐다. 두 grouped CI 모두 0을 포함하므로 개선이 재현된다고 보기 어렵다. 따라서 absent 제외 raw 최고점은 기록하되, 검증 근거가 더 강한 Model326+Model270을 보수적인 현재 후보로 둔다.

## 결론과 현재 사용 권고

1. empty-union=0을 목표 판정 기준으로 두면 현재 최고는 **Model326+Model270의 79.1937%**로, 80%까지 `0.8063%p` 부족하다.
2. absent-class를 제외하는 evaluator라면 Model331의 **80.0359%**가 raw validation 최고다. 다만 grouped CI가 0을 포함하므로 robust 개선 또는 독립 test 성능이라고 말할 수 없다.
3. 대회 evaluator의 empty-class 정책이 확정되기 전에는 두 점수를 같이 기록한다. 제출/운영 후보로는 grouped CI가 양수인 Model326+Model270을 우선하고, Model331은 후보 성능으로 보존한다.
4. 두 결과 모두 `valid.zip` 점수다. `test.zip` 정답은 없으므로 test mIoU는 산출하지 않았다.

## 재현 자료

- Model326 학습·OOF: [`model326_hybrid_fullstats/KoMaP/train.py`](model326_hybrid_fullstats/KoMaP/train.py), [`run_model326_hybrid_fullstats_group_oof.sh`](run_model326_hybrid_fullstats_group_oof.sh)
- 확률 route·grouped CV 코드: [`model323_model270_branch_route_probe.py`](model323_model270_branch_route_probe.py)
- 상세 결과 원장: [`model_results.md`](model_results.md)
- 서버 JSON: `outputs/model326_model270_fusion/route_probe.json`, `outputs/model327_b4_classwise_route/route_probe.json`, `outputs/model331_b4_fine_gamma_route/route_probe.json`
