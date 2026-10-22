# KoMaP 최고 성능 모델 (mIoU 기준별)

2026-10-04 재감사: P100 원본 [Model364 결과 JSON](experiments/results/model364_model361_entropy_into_360.json)에서 raw valid 최고 `79.9429727%`를 확인했다 (SHA-256 `d93e47139046567909d738014a95ebb1aae33781a1f0061c8a9ffc38afbe1ae1`). 아래 원래 요약의 작성 기준일은 Model364 실험 전이므로, 최신 순위 및 해석은 아래 갱신 표를 우선한다.

2026-10-05 추가: 독립 검토 후 고정 50:50 B4/HRNet 4-fold 후보 Model400을 valid 20장으로 단회 평가했다. `79.3227251%`로 Model364보다 낮아 채택하지 않았고, current raw 최고는 그대로 Model364다. [Model400 상세·결과](experiments/model400_model396_d2_valid_ensemble_preregistration.md).

2026-10-05 추가: Model397 고정 0.50 B4 + 0.25/0.25 HRNet seed1/seed2 blend(Model401)를 valid 20장으로 한 번 평가했다. 공식 image-wise empty=0 점수는 `79.4181709%`로 Model364보다 `−0.5248018%p` 낮아 채택하지 않았다. test는 사용하지 않았고 raw valid 최고는 Model364 그대로다. [Model401 상세·결과](model_results.md#model401-model397-고정-blend-단발-valid-평가-2026-10-05).

작성 기준: 2026-10-01 갱신. 최신 route 결과는 모두 `valid.zip` 20장 기준이다. `test.zip`에는 정답 mask가 없어 test mIoU는 계산하지 않았다.

## 한눈에 보기

| 최적화·평가 기준 | 최고 route | mIoU | 비고 |
|---|---|---:|---|
| Image-wise, empty-union=0 (공식 raw 최고 관측값) | **Model364** | **79.9430%** | Model360 대비 +0.00131%p; group-held-out 변화 −0.00439%p, 채택 근거 없음 |
| Image-wise, empty-union=0 (보수적 기준 route) | **Model360** | **79.9417%** | Model364와 사실상 동률; 일반화 개선은 미확인 |
| Image-wise, empty-union=0 (선행 route) | Model349 | **79.9095%** | Model346 대비 +0.0155%p, grouped route-LOSO는 음수 |
| Image-wise, empty-union 제외 (참고) | Model341 (직전 재계산값) | 80.3812% | Model349의 해당 산식은 저장된 결과에 없어 미재계산 |
| Global, 전체 픽셀 pooled (참고) | Model341 (직전 재계산값) | 82.1747% | Model349의 pooled 값은 미재계산; image-wise 대회 기준이 아님 |
| 팀 B4 recipe 미사용 자체 route, image-wise empty-union=0 | **Model270** | **78.7209%** | Model195/20-step TENT/Model269 class-wise fusion |
| 팀 B4 recipe 미사용 자체 route, image-wise empty-union 제외 | **Model270** | **79.4978%** | 같은 route, empty class 제외 |

대회의 공식 empty-class 기준은 **empty-union=0**이므로 이 점수를 순위와 80% 목표의 판정 기준으로 사용한다. 저장 결과 JSON의 per-image 벡터를 재평균한 raw valid 최고는 Model364 **79.9429727%**로 목표보다 `0.0570273%p` 낮다. Model360 `79.9416620%`보다 `+0.0013108%p` 높지만 group-held-out 변화가 `−0.0043870%p`라 일반화 우위는 확인되지 않아 Model360을 보수적 기준으로 유지한다. Model349 `79.9095189%`는 낮은 선행 route이며, 그 Model346 대비 grouped route-LOSO 변화도 `−0.0864%p` (10/20장 개선, 95% CI `[−0.2356,+0.0220]%p`)다. 점수는 반복 사용된 같은 valid 20장 기반이므로 독립 test 성능으로 해석하지 않는다. NaN-skip과 pooled global은 Model349에 저장되지 않아 직전 Model341 값을 참고로만 남겼다. 아래 세부 탐색 이력은 2026-09-30 스냅샷이다.

### 이전 상세 상태 스냅샷 (2026-09-30; 최신 결과로 superseded)

Model342는 valid `78.8936%`, Model344는 `77.9186%`로 Model341 최고를 넘지 못했다. Model343은 Model341에 Model342를 추가해도 개선되지 않았다. Model333 Lovasz는 채널 순서를 바로잡은 뒤에도 grouped-CV 변화 `+0.0040%p` (95% CI `[-0.0084,+0.0171]%p`)여서 채택하지 않았다. Model345는 Model344 expert를 Model341 route에 추가하는 CPU 전수 탐색 중이고, Model346은 완성될 Model336 HRNet expert를 Model341에 직접 결합할 후속 탐색으로 대기 중이다.

Model338 strict-best route에서도 strict와 NaN-skip 값이 달라지는 이미지는 A71 한 장뿐이다. A71 정답 mask에는 Primary Si가 없고 해당 route도 이 상을 예측하지 않아 union이 0이다. 이 route의 A71 점수는 strict `49.6248%`, 제외 방식 `66.1664%`다. 따라서 empty=0 목표를 올릴 때는 A71의 나머지 세 상과 다른 이미지의 분할 개선이 중요하다.

Global mIoU는 모든 이미지의 confusion matrix를 합친 뒤 계산하므로 대회 image-wise 방식과 다른 지표다. 큰 이미지·큰 상의 영향이 더 크다.

## 팀 B4 학습 recipe를 쓰지 않은 우리 방식 최고 route

팀의 EfficientNet-B4 branch/16k-step B4 recipe 없이 만든 자체 branch 조합 중, validation에서 관측된 최고 image-wise route는 **Model270**이다. 다만 Model195의 선행 학습에는 팀 보고서에서 착안한 512→768→native 다중 해상도 일정이 쓰였다. 따라서 “팀 방식 미사용”은 팀 B4 모델·학습 recipe를 쓰지 않았다는 뜻이지, 팀에서 공유한 모든 아이디어와 완전히 무관하다는 뜻은 아니다. Model270은 단일 checkpoint가 아니라 ResNet34 계열 확률, 이미지별 적응, class-wise 혼합과 후처리를 결합한 route다.

| 상 | 확률원 | 혼합 비율·설정 |
|---|---|---|
| Al3Ni | Model220의 Model195 기반, 이미지별 BN-affine 20-step TENT 확률 | `0.40` |
| Primary Si | Model195 native-stage ResNet34 U-Net++/Residual ASPP supervised 확률 | `0.75` |
| Al | Model270의 기본 확률 및 train-only prior 보정 | B4 branch 없음 |
| Eutectic Si | Model269 full-image-stat ResNet34 continuation 확률 | `0.50` |

나머지 고정 설정은 prior scale `1.25`, adaptive beta scale `0.80`, Al delta `0`, Eutectic delta `+0.15`, Gaussian blend `0.50`, Eutectic edge blend `0.25`, connected-component cleanup `128 px`다. Model195는 raw grayscale/z-score/CDF 3채널 ResNet34 U-Net++/Residual ASPP(dilation `1,2,4,8`)이고, 512→768→native 일정 뒤 1024 crop으로 재개 학습됐다. Model217에서 검증한 per-image BN-affine TENT 계열을 Model220에서 20-step까지 확장했고, Model269는 Model257 ResNet34 checkpoint의 full-image-stat continuation이다. Model322/326 EfficientNet-B4 팀 branch는 이 route에 들어 있지 않다.

| 지표 | Model270 raw 최고 | Model225 보수 기준 |
|---|---:|---:|
| Image-wise mIoU, empty-union=0 | **78.7209%** | 78.6921% |
| Image-wise mIoU, empty-union 제외 | **79.4978%** | 79.4693% |
| Global mIoU | 81.3874% | 81.3495% |

Model270은 Model225보다 raw validation에서 `+0.0289%p` 높지만, 19개 validation specimen-group held-out route selection의 변화가 `-0.1320%p`, group bootstrap 95% CI `[-0.3548,+0.0574]%p`여서 안정적 개선으로 확인되지 않았다. 따라서 자체 방식의 raw 최고는 Model270으로 기록하되, grouped-CV 근거를 우선하는 보수 운용 기준은 Model225로 구분한다. 두 점수 모두 같은 valid 20장으로 만든 추론 후보 점수이며 독립 test 성능은 아니다.

## Model335: 2,401개 CPU exact route 전수 채점

Model270 추론 route의 class-wise 확률원을 Model322, Model326, 두 확률의 50:50 평균 중에서 바꾸어 exact spatial refinement까지 적용해 전부 채점했다. 각 4개 상마다 7개 선택지(기존 Model270 유지, 세 B4 source 각각의 gamma `0.50` 또는 `1.00`)를 두어 총 `7^4 = 2,401`개 route다. Eutectic Si도 탐색 대상이며, 기존 Model270 선택은 Model269 확률 gamma `0.50`이다. CPU-only 16 worker로 valid 20장을 모두 채점했다. 이 grid 밖의 다른 모델(예: Model328) 조합이나 연속 gamma는 포함하지 않았다.

| 기준 | Al3Ni | Primary Si | Al | Eutectic Si | empty=0 | empty 제외 | Global |
|---|---|---|---|---|---:|---:|---:|
| Image-wise strict 최고 | Model322/326 평균 `0.50` | Model326 `1.00` | Model322 `0.50` | Model270 유지 (Model269 `0.50`) | **79.2903%** | 80.1248% | 81.8794% |
| NaN-skip 최고 / Global 최고 | Model322/326 평균 `0.50` | Model326 `1.00` | Model322 `1.00` | Model270 유지 (Model269 `0.50`) | 79.2898% | **80.1316%** | **81.9359%** |

Strict 최고 route는 기존 Model334의 Al 평균 branch `0.50` 대신 Model322 단독을 `0.50` 섞어 empty=0 점수를 `+0.0549%p` 높였다. NaN-skip/global 최고는 Al에 Model322를 `1.00`으로 바꾸며, strict 최고와 점수 차이는 작다. Model334 대비 최고 점수 증가는 strict `+0.0549%p`, NaN-skip `+0.0561%p`, 별도 global 최고 `+0.0682%p`다. Model270 단독 route 대비 strict는 `+0.5693%p`, NaN-skip 최고는 `+0.6338%p` 상승했다.

| Group-held-out route 선택 기준 | Model270 대비 평균 변화 | 개선 이미지 | specimen groups | 그룹 bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Image-wise empty=0 | **+0.4564%p** | 14/20 | 19 | `[+0.0456,+0.8978]%p` |
| Image-wise empty 제외 | **+0.5060%p** | 15/20 | 19 | `[+0.0524,+1.0264]%p` |
| Global pooled | **+0.5484%p** | — | 19 | `[+0.2796,+0.8593]%p` |

이 CV는 19개 validation specimen group 중 하나를 hold out해 나머지 그룹에서 route를 고르고 held-out group으로 평가한 것이다. 방향성은 Model270 대비 개선을 지지하지만, 동일 validation pool 안에서 2,401개 후보를 탐색한 결과라 독립 test 성능의 대체값은 아니다. Model329의 새 OOF 결과가 완료되면 해당 branch를 포함한 재탐색은 별도다.

## Model334: 이전 최고 route (Model335로 갱신)

Model334는 새 신경망 하나를 학습한 이름이 아니다. Model322와 Model326의 B4 확률을 평균한 보조 branch를 만들고, 기존 Model270 추론 route에 클래스별로 결합한 **확률 앙상블 및 후처리 구성**이다. Model334 전용 단일 `best.pt`는 없다.

### 구성 branch

| 이름 | 구조·역할 | 입력 및 학습 핵심 |
|---|---|---|
| Model322 | EfficientNet-B4 encoder + U-Net decoder, 팀 B4 레시피 재현 | grayscale 3× 확대, crop 1536, batch 2, AdamW `3e-4`, unweighted CE + Dice |
| Model326 | EfficientNet-B4 encoder + U-Net decoder, full-image-stat 입력 | `[raw grayscale, full-image z-score, full-image CDF]`, 3× 확대, crop 1536, batch 2, CE + Dice |
| Model270 | 기존 추론 route 기반 | Model195 Primary branch, Model220 20-step TENT Al3Ni branch, Model269 Eutectic branch 및 train-only prior·spatial refinement |

Model322와 Model326은 각각 4-fold specimen-group 방식으로 학습했다. fold마다 고정된 16,000 updates를 사용했고 validation mask로 checkpoint를 고르지 않았다. 이 update 수는 두 레시피 간 비교를 맞추기 위해 상속한 budget이며 최적이라고 검증된 값은 아니다.

Model322와 Model326의 70장 train grouped-OOF 확률을 50:50 평균했을 때, strict image-wise mIoU는 Model322 단독 약 78.5346%에서 78.7843%로 상승했다. Paired 변화는 **+0.2496%p**, specimen-group bootstrap 95% CI **[+0.0965,+0.4085]%p**, 개선 이미지는 55/70장이었다. 이 결과는 두 B4 branch의 상보성을 뒷받침한다.

## Model334 image-wise 최고 route 상세 (이전 최고)

### 확률 혼합

Model322와 Model326의 valid 확률을 class channel 순서를 맞춘 뒤 각각 50% 평균해 B4 결합 branch를 만든다. 팀 B4 순서 `[Al, Al3Ni, Primary Si, Eutectic Si]`를 내부 route 순서 `[Al3Ni, Primary Si, Al, Eutectic Si]`로 재배열한다.

Image-wise 기준 Model334 최고 설정:

| 상 | 확률원 | 혼합 비율 |
|---|---|---:|
| Al3Ni | Model270 기존 확률 + Model322/326 평균 B4 branch | B4 `0.50` |
| Primary Si | Model322/326 평균 B4 branch로 교체 | B4 `1.00` |
| Al | Model270 기존 확률 + Model322/326 평균 B4 branch | B4 `0.50` |
| Eutectic Si | Model269 branch | Model270 설정 `0.50` 유지 |

Al3Ni·Primary Si·Al의 B4 혼합률은 `{0, 0.5, 1.0}`에서 탐색했다. Eutectic Si는 B4 branch로 바꾸지 않고 Model269를 유지했다. 선정 뒤에도 Model270의 기존 calibration과 spatial refinement를 적용했다.

### 고정된 Model270 설정

- Primary 기준 source: Model195, gamma `0.75` (Model334에서는 B4 결합 branch가 gamma 1.0으로 대체)
- Al3Ni 기준 source: Model220 20-step TENT, gamma `0.40` (Model334에서는 B4 결합 branch를 gamma 0.50 혼합)
- Eutectic Si: Model269, gamma `0.50`
- prior scale `1.25`, adaptive beta scale `0.80`
- Al delta `0.0`, Eutectic delta `+0.15`
- Gaussian blend `0.50`, Eutectic edge blend `0.25`
- connected-component cleanup threshold `128 px`

확률 혼합 뒤 class probability 합을 재정규화하며, 이후 고정 route의 calibration·공간 refinement·argmax 순서로 최종 mask를 만든다.

### 점수

| 지표 | Model334 image-wise 최고 | Model270 기준 | 차이 |
|---|---:|---:|---:|
| Image-wise mIoU, empty-union=0 | **79.2353%** | 78.7209% | +0.5144%p raw valid |
| Image-wise mIoU, empty-union 제외 | **80.0687%** | 79.4978% | +0.5709%p raw valid |
| Global mIoU | 81.8222% | 81.3874% | image-wise 최고 route 기준 |

Image-wise 최고점은 59개 설정 중 같은 valid 20장 점수로 고른 raw 결과이므로 선택 편향을 포함할 수 있다.

Model270 대비 validation specimen-group leave-one-group-out route 선택은 **+0.4642%p**, 15/20장 개선, 19 groups, group-bootstrap 95% CI **[+0.0433,+0.9137]%p**였다. 이는 같은 valid 세트 안에서 group holdout으로 route를 선택한 보조 일반화 점검이다. 독립 test 결과나 train-only OOF 결과는 아니다.

## Model334 당시 Global mIoU 최고 route (이전 최고)

Global 기준 최고는 Model334와 같은 B4 결합 branch·Model270 기반 설정이지만, Al에서 B4를 50%가 아니라 100% 사용한다.

| 상 | B4 결합 branch 비율 |
|---|---:|
| Al3Ni | 0.50 |
| Primary Si | 1.00 |
| Al | 1.00 |
| Eutectic Si | Model269 0.50 유지 |

| 지표 | 점수 |
|---|---:|
| Global mIoU | **81.8677%** |
| Image-wise empty-union=0 | 79.2251% |
| Image-wise empty-union 제외 | 80.0655% |

이 후보는 59개 route grid 중 global mIoU raw 최고다. Image-wise 최고 설정보다 global score는 높지만, 대회 지표가 image-wise라면 이를 대신 선택할 근거는 없다.

## 해석 및 한계

1. 대회 산식은 이미지별 4-class IoU 평균을 구한 뒤 이미지 수로 평균하는 image-wise 방식이며, 공식 empty-class 기준은 empty-union=0이다. NaN-skip은 참고용으로만 병기하고 공식 성능 순위나 목표 판정에 사용하지 않는다.
2. Model334 raw 최고는 valid 20장과 59개 route 탐색에 의존한다. Grouped-heldout 결과는 보완 근거지만 완전히 독립적인 성능 추정은 아니다.
3. 현재 가장 높은 **관측 validation 후보**는 Model339의 `79.3683%` strict route이며, 이를 test 점수나 공식 대회 empty=0 80% 달성으로 해석하지 않는다. Model338 대비 group-CV 개선 근거는 없다.
4. Model339는 Model338 expert source의 네 gamma를 동시에 바꾼 400-route search다. raw 최고는 소폭 높지만 paired group-CV가 개선을 지지하지 않아 채택 보류했다. Model329의 fold1–2 train-only OOF에서는 Model322 대비 strict `+1.0898%p`(36/70장, CI `[+0.4014,+1.8531]`)의 interim 상승이지만 valid 점수가 아니며 나머지 OOF folds와 fusion은 진행 중이다. HRNet 실험도 아직 포함되지 않았다.

## 재현 자료

- Route 탐색 코드: [model323_model270_branch_route_probe.py](model323_model270_branch_route_probe.py)
- Model335 전체 route 탐색 코드: [model335_exhaustive_b4_route_search.py](model335_exhaustive_b4_route_search.py)
- Model338 혼합률 fine-sweep 코드: [model338_fine_gamma_route_search.py](model338_fine_gamma_route_search.py)
- Model339 네 클래스 joint-gamma 탐색 코드: [model339_joint_gamma_route_search.py](model339_joint_gamma_route_search.py)
- Model339와 Model338 paired group-CV 감사 코드: [model339_grouped_cv_parent_audit.py](model339_grouped_cv_parent_audit.py)
- Model329 partial OOF empty=0 mask audit: [model340_partial_oof_metric_audit.py](model340_partial_oof_metric_audit.py)
- 전체 실험 이력: [model_results.md](model_results.md)
- Model335 2,401-route 결과: `/workspace/komap_model1/outputs/model335_exhaustive_b4_routes/results.json`
- Model338 38-route 결과: `/workspace/komap_model1/outputs/model338_fine_gamma_route_search/results.json`
- Model334 route 결과: `/workspace/komap_model1/outputs/model334_b4_equalblend_route/route_probe.json`
- 평균 확률 archive: `/workspace/komap_model1/outputs/model334_b4_equalblend_route/candidate/probabilities.npz`
- Model322 OOF: `/workspace/komap_model1/outputs/model322_team_b4_group_oof/`
- Model326 OOF: `/workspace/komap_model1/outputs/model326_hybrid_fullstats_group_oof/`

## Model338: Model335 expert 혼합률 fine-sweep

Model335 strict-best route에서 네 class의 확률원은 고정하고 한 class씩 혼합률 `0.1`부터 `1.0`까지 바꿔 exact CPU 평가했다. Model270 reference를 포함해 38개 고유 route를 비교했다. 공식 empty-union=0 최고는 **79.3400%**로 Eutectic Si에 Model269 확률을 `0.90` 혼합한 route다. 다른 구성은 Al3Ni `Model322/326 평균 × 0.50`, Primary Si `Model326 × 1.00`, Al `Model322 × 0.50`이다.

| 기준 | 최고 점수 | route |
|---|---:|---|
| 공식 image-wise, empty=0 | **79.3400%** | Eutectic Si gamma `0.90` |
| Image-wise, empty 제외 (참고) | 80.1685% | Eutectic Si gamma `0.80` |
| Pooled global (참고) | 81.9667% | Eutectic Si gamma `0.90` |

Model335 strict-best 대비 공식 점수 변화는 `+0.0497%p`다. 19-group leave-one-specimen-out route 선택은 Model270 대비 `+0.5884%p` (18/20 images positive, group-bootstrap 95% CI `[+0.2305,+0.9916]%p`)였다. 이 CV 결과는 route 선택의 안정성 근거지만, 같은 valid pool에서 38개 후보를 평가했으므로 독립 test 성능은 아니다. Model329는 이 fine-sweep에는 포함되지 않았다.

## Model339: 네 expert gamma 동시 탐색

Model338의 네 확률원을 고정하고 `5 × 4 × 5 × 4 = 400`개의 혼합률 조합을 CPU exact route scorer로 평가했다. Raw 공식 최고는 **79.3683%**로 Model338보다 `+0.0283%p`다.

| 상 | 확률원 | gamma |
|---|---|---:|
| Al3Ni | Model322/326 평균 | 0.6 |
| Primary Si | Model326 | 1.0 |
| Al | Model322 | 0.7 |
| Eutectic Si | Model269 | 0.8 |

Model338 고정 route 대비 19-group held-out route selection은 `−0.0139%p`, 9/20장 개선, group-bootstrap 95% CI `[-0.0613,+0.0352]%p`다. Raw validation은 약간 올랐지만 일반화 개선은 확인되지 않아 **Model338을 보수적 reference로 유지**하고 Model339는 raw leaderboard 최고로만 기록한다. Model339의 NaN-skip은 `80.2062%`, pooled global은 `82.0064%`이며 둘 다 공식 empty=0 목표 판정에 쓰지 않는다.

재현: [joint-gamma search](model339_joint_gamma_route_search.py), [Model338 paired group-CV audit](model339_grouped_cv_parent_audit.py). 결과 JSON은 `/workspace/komap_model1/outputs/model339_joint_gamma_route_search/`에 있다.
