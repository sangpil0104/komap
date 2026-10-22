# KoMaP 최고 성능 모델 (mIoU 기준별)

상단 순위 표는 2026-10-04 원자료 대조 결과를 반영했다. Model349/360/364 결과의 valid image IDs 20개가 동일하고, Model360/364 per-image 벡터 재평균도 저장 요약과 일치한다. 상세 route 설명은 날짜가 표시된 역사 기록이다.

작성 기준: 2026-10-02. 아래 validation 점수는 `valid.zip` 20장 기준이며, `test.zip`에는 정답 mask가 없어 test mIoU는 계산하지 않았다.

## 한눈에 보기

| 최적화·평가 기준 | 최고 route | mIoU | 비고 |
|---|---|---:|---|
| Image-wise, empty-union=0 (raw valid 최고 관측값) | **Model364** | **79.9430%** | Model360 대비 +0.00131%p; group-held-out 변화 −0.00439%p, 일반화 개선 미확인 |
| Image-wise, empty-union=0 (보수적 기준 route) | **Model360** | **79.9417%** | raw 최고와 사실상 동률; group-held-out 기준 유지 |
| Image-wise, empty-union=0 (선행 route) | Model349 | **79.9095%** | 625개 route 탐색 결과; Model364보다 낮음 |
| Train-only grouped OOF (공식 metric; valid 점수 아님) | Model397 fixed blend | **79.9859%** | Model392 gate 실패로 official-valid 단계 미승인 |
| Image-wise, empty-union 제외 (참고) | Model341* | **80.3812%** | union이 0인 class를 해당 이미지 평균에서 제외; Model346에서 재계산하지 않아 이전 기록 유지 |
| Global, 전체 픽셀 pooled (참고) | Model341 route* | **82.1747%** | Model341 strict-best와 다른 gamma; Model346 pooled 점수는 미계산 |
| 팀 B4 recipe 미사용 자체 route, image-wise empty-union=0 | **Model270** | **78.7209%** | Model195/20-step TENT/Model269 class-wise fusion |
| 팀 B4 recipe 미사용 자체 route, image-wise empty-union 제외 | **Model270** | **79.4978%** | 같은 route, empty class 제외 |

`*` empty-union 제외 및 pooled-global 수치는 기존 Model341 참고 기록이며 최신으로 재감사하지 않았다.

Model364의 stored score vector는 20장 평균으로 JSON score를 재현하며 Model360과 image IDs가 일치한다. 따라서 앞서 표시했던 점수 충돌은 최신 요약의 Model349 항목이 stale였던 것으로 정정했다. 공식 순위에는 empty=0만 사용한다. valid 20장에서 탐색·선택된 값은 독립 test 성능이 아니며, Model397의 79.9859%는 별도 train-only OOF라 valid 최고와 직접 비교하지 않는다. 최신 제한은 [한국어 현황](current_status_ko.md), 재현 방법은 [Model397 guide](model397_reproduction.md)를 참조한다.

## 과거 route 및 모델 상세

아래 문단과 표는 모델 계열의 세부 비교를 보존한 기록이다. 순위·현재 기준은 위 표를 우선한다.

Model329의 4-fold grouped-OOF/valid 평가는 완료했다. 단독 strict valid는 `78.8334%`; train OOF는 `79.1233%`이고 네 class 모두 gamma `0.50`인 OOF route는 `79.3974%`였다. 이 route를 valid에 고정 적용하면 `78.1169%`다. Model329를 Model270 experts와 결합한 route probe는 strict valid `79.2615%`였다. Model341에서 Model339 expert route에 Model329를 더해 `79.5407%`를 기록했고, Model346이 HRNet 확률을 Model341에 class-wise로 추가해 `79.8940%`를 기록했다. Model349는 Model346 위에 Model347 weighted-CE HRNet을 Al gamma `0.25`로 더해 **79.9095%**로 raw 최고를 갱신했으며, 80%까지 `0.0905%p` 남았다. 다만 Model346 대비 grouped-CV delta는 `−0.0864%p` (10/20; CI `[-0.2356,+0.0220]%p`)다. Model352의 D4-HRNet route는 `79.8833%`로 raw 최고를 넘지 못했지만 Model341 대비 grouped-CV `+0.2648%p` (15/20; 95% CI `[-0.0303,+0.4866]%p`)였다. CI가 0을 포함하며 valid route 탐색 결과이므로 개선 근거는 불확실하다. [Model349 결과](../results/outputs/model349_model347_into_346_route/results.json), [Model352 결과](../results/outputs/model352_d4_hrnet_into_341_route/results.json)를 참조한다.

Model342는 Model322와 Model329를 train-only grouped OOF로 class-wise 결합해 valid에서 `78.8936%`를 기록했다. Model322 대비 group-held-out 개선은 `+0.8720%p` (51/70장, 59 groups, bootstrap 95% CI `[+0.4640,+1.2964]%p`)지만 이후 Model341과 Model346이 더 높아 공식 순위를 바꾸지 않는다. 선택 gamma와 상세 수치는 [Model342 기록](model_results.md#모델-342-train-grouped-oof로-선택하는-model322329-class-wise-blend-완료) 참조.

Model333의 Lovasz arm은 채널 순서 교정 후에도 control 대비 grouped-CV 변화가 `+0.0040%p` (95% CI `[-0.0084,+0.0171]%p`)로 사실상 차이가 없어 채택하지 않았다. Model344의 Model322/CE+Dice-control OOF blend는 고정 gamma `[1,1,0,1]`로 valid `77.9186%`다. Model345는 이 expert를 Model341 route에 추가하는 625-setting exact CPU 탐색에서 개선이 없음을 확인했다. Model336 HRNet-W48/U-Net++ 단독 valid는 `78.1836%`; Model337의 Model270+HRNet route는 `79.4078%`였다. Model346은 Model341에 HRNet을 class-wise로 결합해 `79.8940%`를 얻었다. Model347의 train-only inverse-frequency CE fine-tune은 strict valid `76.6677%`로 Model336보다 `1.5159%p` 낮았지만, Model348은 그 확률을 Model341에 gamma `[0,0,0.75,0.25]`로 더해 `79.7882%`를 기록했다. Model351은 Model336에 D4 TTA를 적용해 `78.2001%`를 얻었다. flip4 대비 `+0.0165%p`에 그쳤고 paired group-bootstrap 95% CI `[-0.0467,+0.0914]%p`가 0을 포함한다. Model349 raw 최고는 `79.9095%`이나 Model346 대비 group-held-out 변화가 `−0.0864%p` (10/20, CI `[-0.2356,+0.0220]%p`)로 일반화 개선은 불확실하다. Model350 fine-gamma 최고는 `79.9021%`로 Model349를 넘지 못했다. Model352 D4 route search는 완료됐으며 `79.8833%`로 Model349를 대체하지 않는다.

대회의 공식 empty-class 기준은 **empty-union=0**이므로 순위와 80% 목표 판정은 strict 점수로 한다. 보수적 기준 Model360은 **79.9417%**이며 목표까지 `0.0583%p` 남았다. Model355의 공간 설정 위에 HRNet D4 확률을 낮은 엔트로피 픽셀에만 적용한 결과로, 19-group held-out 모두 같은 게이트를 고르지만 paired group-bootstrap 95% CI `[-0.0078,+0.0191]%p`는 0을 포함한다. Model364의 raw `79.9430%`는 0.00131%p 높지만 held-out route 변화가 `−0.00439%p`여서 adoption 근거가 없다. 같은 20장 valid에서 다수 route를 검색한 raw 최고는 validation 선택 편향을 포함하며 독립 test 성능이 아니다. NaN-skip과 pooled global 점수는 참고용이고 official 80% 달성으로 세지 않는다.

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
3. 가장 높은 **관측 raw validation 후보**는 Model364의 `79.9430%` strict route다. 보수적 기준은 Model360 `79.9417%`이며, Model364의 grouped held-out 변화 `−0.00439%p`는 개선 근거가 아니다. 둘 다 80% 달성이나 test 점수로 볼 수 없다.
4. Model329 grouped OOF/valid, Model341 626-route search, Model342 train-only OOF blend, Model343 625-route probe, Model344 train-OOF blend, Model345 exact route search, Model336 HRNet 학습/valid 평가, Model337/346/348/349/350/352/353 route searches와 Model354–360 후처리·확률 라우팅 점검이 완료됐다. Model355 strict valid는 `79.9352%`, Model356 사후 세밀 탐색은 `79.9358%`, Model360은 `79.9417%`다. Model357 bias, Model358 cleanup, Model359 전체 D4 혼합은 Model355를 개선하지 못했다. Model348과 Model349의 group-held-out CI는 모두 0을 포함한다. Model333 CE+Dice control의 70장 OOF는 parent보다 `+0.2137%p`; Lovasz의 corrected full paired audit는 grouped-CV 유의 개선을 보이지 않았다. Model336 단독은 `78.1836%`, Model337 route는 `79.4078%`, Model347 weighted-CE standalone은 `76.6677%`였다. Model351 D4 TTA valid는 `78.2001%`다.

## 재현 자료

- Route 탐색 코드: [model323_model270_branch_route_probe.py](../experiments/historical_scripts/model323_model270_branch_route_probe.py)
- Model335 전체 route 탐색 코드: [model335_exhaustive_b4_route_search.py](../experiments/historical_scripts/model335_exhaustive_b4_route_search.py)
- Model338 혼합률 fine-sweep 코드: [model338_fine_gamma_route_search.py](../experiments/historical_scripts/model338_fine_gamma_route_search.py)
- Model339 네 클래스 joint-gamma 탐색 코드: [model339_joint_gamma_route_search.py](../experiments/model339_joint_gamma_route_search.py)
- Model339와 Model338 paired group-CV 감사 코드: [model339_grouped_cv_parent_audit.py](../experiments/model339_grouped_cv_parent_audit.py)
- Model329 partial OOF empty=0 mask audit: [model340_partial_oof_metric_audit.py](../experiments/model340_partial_oof_metric_audit.py)
- Model329 full grouped-OOF/valid result: [results.json](../results/outputs/model329_fullstats_width4000_uniform_group_oof/results.json)
- Model329 + Model270 expert route audit: [route_probe.json](../results/outputs/model329_model270_fusion/route_probe.json)
- Model341 626-route results and grouped-CV audit: [results.json](../results/outputs/model341_model329_joint_route_search/results.json)
- 전체 실험 이력: [model_results.md](model_results.md)
- Model335 2,401-route 결과: [results.json](../results/outputs/model335_exhaustive_b4_routes/results.json)
- Model338 38-route 결과: [results.json](../results/outputs/model338_fine_gamma_route_search/results.json)
- Model334 route 결과: `$KOMAP_WORKSPACE/outputs/model334_b4_equalblend_route/route_probe.json`
- 평균 확률 archive: `$KOMAP_WORKSPACE/outputs/model334_b4_equalblend_route/candidate/probabilities.npz`
- Model322 OOF: `$KOMAP_WORKSPACE/outputs/model322_team_b4_group_oof/`
- Model326 OOF: `$KOMAP_WORKSPACE/outputs/model326_hybrid_fullstats_group_oof/`

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

재현: [joint-gamma search](../experiments/model339_joint_gamma_route_search.py), [Model338 paired group-CV audit](../experiments/model339_grouped_cv_parent_audit.py). [route 결과](../results/outputs/model339_joint_gamma_route_search/results.json), [paired CV](../results/outputs/model339_joint_gamma_route_search/cv_vs_model338.json).
