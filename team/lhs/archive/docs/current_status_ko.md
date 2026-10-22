# KoMaP 최근 현황 (2026-10-04)

## 공식 점수와 현재 기준

대회 공식 지표는 이미지마다 4개 상의 IoU를 평균한 뒤 이미지 평균을 내는 image-wise mIoU이며, 해당 이미지에서 정답·예측 모두 비어 있는 상도 IoU `0`으로 계산한다. NaN 제외와 전체 픽셀을 합산하는 pooled mIoU는 참고 지표다.

- raw validation 최고 관측값은 **Model364: 79.9429727%**다. 저장된 20개 per-image 점수를 재평균해 JSON 요약과 일치함을 확인했고, Model360과 같은 valid image IDs 및 empty-union=0 지표를 사용한다. 80%까지 `0.0570273%p` 남았다. 동일 valid에서 여러 route를 탐색한 선택 편향이 포함된다.
- Model360은 `79.9416620%`로 Model364보다 `0.0013108%p` 낮다. Model364의 Model360 대비 group-held-out 변화는 `−0.0043870%p`이므로 raw valid 최고일 뿐 일반화 개선으로 채택하지 않았다. Model349 `79.9095189%`는 그보다 낮다.
- Model397 fixed blend의 **79.9859%는 train-only grouped OOF**다. Model392의 사전등록 2-seed OOF adoption gate가 실패했으므로 Model397 official-valid 단계는 승인되지 않았으며, 79.9859%를 valid/test 또는 독립 일반화 점수로 보면 안 된다.
- `test.zip`에는 정답 mask가 없으므로 test mIoU는 계산할 수 없다. validation 점수를 test 점수로 간주하지 않는다.

세부 순위와 각 지표의 정의는 [최고 모델 요약](best_models_by_miou.md), 전체 연혁은 [모델 결과 기록](model_results.md)을 참고한다.

## valid/test 원본 영상 유사도 감사

대회 ZIP의 원본 영상만 사용했으며 valid/test mask나 test 정답은 읽지 않았다. 재현 코드는 [`valid_test_group_similarity_audit.py`](../experiments/valid_test_group_similarity_audit.py), 산출물은 [`results.json`](../results/outputs/valid_test_group_similarity_audit/results.json)이다.

- 데이터는 train 70장/59 specimen group, valid 20장/19 group, test 10장/10 group이다.
- valid와 test 사이 exact-image hash 중복은 없다. 다만 specimen prefix 기준 A63 group은 valid/test에 모두 있으며, train/test 사이에도 A51·A87·A167 group이 겹친다. 따라서 표본 독립성은 이미지 단위 exact duplicate 여부와 specimen 단위 분리를 구분해 봐야 한다.
- step 분포는 크게 다르지 않지만 해상도 구성은 다르다. `675×480`은 valid 9/20(45%), test 3/10(30%)이고 `1230×875`는 valid 3/20(15%), test 4/10(40%)다. `1096×780` 영상 한 장은 valid에만 있다.
- A63을 valid/test 양쪽에서 제외한 18개 valid group 대 9개 test group의 저수준 영상 특징 centroid 검정은 permutation `p=0.357`(20,000회)이다. 15개 특징의 specimen-group bootstrap 구간은 모두 0을 포함했고, grayscale 표준편차 차이(test−valid)는 `−10.45`, 95% CI `[-20.22,+0.19]`였다.

현재 증거로는 전반적인 저수준 feature shift를 뚜렷하게 검출하지 못했지만, 이는 두 split이 동등하다는 증명이 아니다. 표본이 작고 해상도 비율 및 specimen 구성이 완전히 같지 않으므로 valid는 test의 **중간 정도 입력-domain proxy**로만 볼 수 있다. test label 없이 실제 test mIoU나 상별 정답 구성은 추정하지 않는다.

## 최근 OOF 결과와 재현 자료

Model392의 2-seed × 4-fold train-only grouped OOF 평균은 `79.0437%`로 Model382 기준 `78.9395%` 대비 `+0.1042%p`였다. 사전등록 기준의 최소 개선폭, group-bootstrap CI, fold 일관성 조건을 충족하지 못해 primary gate가 실패했다. 따라서 full-train 및 official-valid 단계로 진행하지 않았다.

Model397은 Model329 EfficientNet-B4/U-Net과 Model392 HRNet-W48/U-Net++의 OOF 확률을 `0.50/0.25/0.25`로 고정 평균한 조합이다. Train-only grouped OOF image-wise mIoU(empty=0)는 `79.9859%`였으나, Model392 gate 실패 때문에 별도 valid 평가는 실행·승인되지 않았다. [Model397 재현 안내](model397_reproduction.md)와 [코드 ZIP](../bundles/Model397_train_route_code.zip)은 외부 `train.zip`으로 4개 B4 및 8개 HRNet fold/seed 모델을 학습하고 OOF scorer를 실행한다. valid/test 입력은 사용하지 않는다.

과거 문서에 기록된 376/377 번호 충돌은 별도 모델 계열을 가리키므로, 실험 수치를 인용할 때는 모델 ID뿐 아니라 평가 split과 날짜도 함께 확인한다. test 라벨은 제공되지 않아 test mIoU는 없다.
