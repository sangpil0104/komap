# 자기지도학습 A·B·C 실험 분석

분석일: 2026-09-26. 실행: `suite_20260924T113719_442712Z`, seed 2026, GPU 0·1·2·5·6·7.

**이번 가림 복원 자기지도학습은 기본 지도학습 대비 개선을 보이지 않았다.** A가 image mIoU 77.4258%, B가 77.1467%로 B−A는 −0.2792%p이다. B는 Eutectic Si와 Al3Ni, 경계 및 외곽 오류에서도 A보다 소폭 나빴다. 현재 A를 후속 실험의 기준으로 유지하는 것이 타당하다.

**C의 76.3541%는 A71 이미지 상단의 Primary Si 오검출 46픽셀에 크게 영향을 받았다.** 그 이미지에는 Primary Si가 없어서 해당 상의 IoU가 1에서 0으로 바뀐다. 이 항의 전체 image mIoU 영향은 −1.25%p이다. C는 경계·외곽 픽셀 오류율은 오히려 줄였으므로, 전체 조직 분할이 일괄적으로 악화됐다고 해석해서는 안 된다.

## 1. 실험 조건과 비교 기준

| 항목 | 내용 |
|---|---|
| 백본 | ResNet18, ImageNet IMAGENET1K_V1 초기값 |
| 구조 | 단일 U-Net, 원해상도 skip, encoder BatchNorm 통계 고정, decoder GroupNorm |
| 데이터 | 동일한 train70/valid20/test10. 사전학습에도 train 원본 70장만 사용 |
| 입력 | 원해상도 중심 512×512 + 사방 halo64, 입력 640×640 |
| 지도학습 | CE + Dice, 유효 배치 12, BF16, D4 학습 증강 |
| 자기지도학습 | 16×16 블록 약 50%를 가리고, 실제 중심의 가려진 픽셀에서 L1 복원 손실 계산 |
| 이전하는 가중치 | B는 복원 encoder만 이전. 분할 decoder·head는 A와 같은 초기값 |
| 추론 | 동일한 halo 타일 추론, 50% 겹침과 Gaussian 확률 결합, TTA 없음 |
| checkpoint 선택 | 정해진 update/계산 예산 종료 모델. 검증 점수로 선택하지 않음 |

주 지표는 **각 이미지의 4개 상 IoU 평균 → 검증 20장의 평균**이다. 상별 값도 이미지별 해당 상 IoU의 평균이다. 아래 수치는 모두 %, 차이는 %p이며 global mIoU를 사용하지 않았다.

빈 클래스는 `TP+FP+FN=0`일 때 IoU=1이다. 정답이 없더라도 예측이 존재하면 IoU=0이다. 다른 두 정책은 진단용으로 별도 기록하고, 정식 평가 기준은 유지했다.

## 2. 전체 및 상별 결과

| 모델 | 학습 내용 | image mIoU | Primary Si | Al3Ni | Eutectic Si | Al |
|---|---|---:|---:|---:|---:|---:|
| **A** | 지도학습 4,000 updates | **77.4258** | 89.7174 | 70.6359 | 55.9465 | 93.4036 |
| **B** | 복원 사전학습 2,000 → 지도학습 4,000 | 77.1467 | **90.1241** | 69.9591 | 55.1440 | 93.3595 |
| **C** | A + 지도학습 1,037 updates 추가 | 76.3541 | 84.7592 | **70.9850** | **56.1490** | **93.5231** |

| 차이 | image mIoU | Primary Si | Al3Ni | Eutectic Si | Al |
|---|---:|---:|---:|---:|---:|
| B−A | −0.2792 | +0.4067 | −0.6767 | −0.8025 | −0.0442 |
| C−A | −1.0718 | −4.9582 | +0.3491 | +0.2025 | +0.1195 |

B는 Primary Si에 작은 개선이 있지만, 나머지 세 상의 감소를 상쇄하지 못했다. 개선을 기대했던 **Eutectic Si는 55.9465 → 55.1440%**로 내려갔다. 이번 설정을 근거로 자기지도학습을 채택할 성능·계산량상의 이점은 확인되지 않았다. 이 결론의 범위는 **현재의 CNN 가림 복원 설정과 seed 한 번**이며, 자기지도학습 전체가 효과 없다는 의미로 확대하면 안 된다.

![전체 점수와 상별 변화](figures/summary.png)

## 3. 테두리 문제가 개선됐는가

| 지표 | A | B | C | B−A | C−A |
|---|---:|---:|---:|---:|---:|
| 조직 경계 1px 오류율 ↓ | 16.4188 | 16.4491 | 16.2060 | +0.0304 | −0.2127 |
| 조직 경계 2px 오류율 ↓ | 13.3657 | 13.4080 | 13.1996 | +0.0424 | −0.1660 |
| 조직 경계 4px 오류율 ↓ | 10.5103 | 10.5823 | 10.3839 | +0.0721 | −0.1264 |
| 이미지 외곽 16px 오류율 ↓ | 8.9827 | 9.0112 | 8.8320 | +0.0285 | −0.1507 |
| 이미지 외곽 32px 오류율 ↓ | 8.6829 | 8.7621 | 8.5334 | +0.0791 | −0.1495 |
| 이미지 외곽 64px 오류율 ↓ | 8.4132 | 8.5134 | 8.3055 | +0.1002 | −0.1077 |
| 이미지 내부(외곽 32px 제외) 오류율 ↓ | 7.8627 | 7.9726 | 7.7726 | +0.1099 | −0.0901 |
| Boundary IoU 2px ↑ | 56.8157 | 56.6711 | 56.6360 | −0.1446 | −0.1797 |

조직 경계 오류는 정답 상 경계의 지정한 거리 이내에서 계산하고, 외곽 오류는 영상 프레임 주변 띠에서 계산한다. 모두 이미지별 값을 평균했다.

- **B는 조직 경계와 이미지 외곽 모두 개선되지 않았다.** 차이는 작지만 확인한 모든 거리의 오류율이 A보다 높다.
- C는 두 종류의 픽셀 오류율이 소폭 낮다. 그러나 A71의 작은 Primary Si 오검출 때문에 주 mIoU와 일부 클래스 기반 보조 지표는 내려간다.
- Boundary IoU는 클래스 내부 경계 띠(영상 프레임 포함)를 사용하며 양쪽에 없는 클래스는 제외한다. A71에서 A/B는 3개 상, C는 오검출된 Primary Si까지 4개 상이 평균에 포함된다. A71의 Boundary IoU 2px가 A 45.0547% → C 33.8167%로 내려가는 데에도 이 변화가 영향을 준다. 픽셀 오류율 개선과 Boundary IoU 감소를 구분해 읽어야 한다.

## 4. C의 점수 하락: A71 상단의 46픽셀

A71은 675×480 이미지이며 정답 Primary Si 픽셀이 0개이다. A와 B는 Primary Si를 예측하지 않았고, C는 **x=561~568, y=0~8**(0부터 시작하는 좌표)에 46픽셀을 예측했다. 이 픽셀의 실제 상은 Eutectic Si 45픽셀, Al 1픽셀이다. 원본 전체 픽셀의 약 0.0142%이다.

![A71 영상 상단 확대](figures/A71_top_edge.png)

이 그림은 동일 위치의 진단용 확대이다. Original / GT / A / B / C를 나란히 놓았으며, 전체 원해상도 2×2 비교 사진은 아래 링크에 별도로 보존했다.

| A71 지표 | A | B | C |
|---|---:|---:|---:|
| Primary Si 예측 픽셀 | 0 | 0 | 46 |
| Primary Si IoU (%) | 100 | 100 | 0 |
| image mIoU (%) | 64.7551 | 64.5386 | 39.6570 |

현재 평가 정의에서 이 변화는 다음 크기로 반영된다.

- A71 이미지 점수: `100 / 4 = 25.00%p`.
- 전체 20장 평균: `25 / 20 = 1.25%p`.
- 실제 C−A 전체 차이 −1.0718%p는 **−1.2500%p(빈 클래스 항) + 0.1782%p(나머지 항의 합)**로 분해된다.

따라서 C의 Primary Si 평균 하락을 넓은 Primary Si 영역의 전반적 분할 악화로 해석할 수 없다. Primary Si가 존재하는 19장의 평균은 A 89.1762%, B 89.6043%, C 89.2202%로 A와 C가 가깝다.

**진단용 계산:** C에서 A71의 Primary Si IoU 항만 0에서 1로 바꾼다고 가정하면 전체는 77.6041%가 된다. 이는 항 하나의 영향을 설명하기 위한 가정이며, 실제 모델 점수나 후처리 성능이 아니다. 정식 C 점수는 **76.3541%**로 유지한다. 예측 마스크를 수정하지 않았다.

또한 A71을 제외한 나머지 19장의 image mIoU 평균은 A 78.0927%, B 77.8103%, C 78.2855%이다. 이 역시 진단용 부분집합 결과이며 정식 20장 평가를 대체하지 않는다. B가 C보다 전체 점수에서 +0.7926%p 높다는 사실만으로 자기지도학습의 우월성을 주장할 수 없는 이유다.

오검출은 실제 최상단에 집중되어 있지만, 이것만으로 padding·학습 패치·문맥 부족 중 어느 것이 원인인지 확정할 수는 없다. 그리고 작은 오검출 외에도 A71의 Al3Ni IoU가 모든 모델에서 약 28%, Eutectic Si가 약 39%에 머무르는 큰 분류 문제가 남아 있다.

## 5. 이미지별 변화: B는 어디서 나빠졌는가

| 비교 | 좋아진 이미지 | 나빠진 이미지 | 평균 차이 (%p) | 중앙값 차이 (%p) |
|---|---:|---:|---:|---:|
| B−A | 7/20 | 13/20 | −0.2792 | −0.1701 |
| C−A | 12/20 | 8/20 | −1.0718 | +0.2161 |
| B−C | 8/20 | 12/20 | +0.7926 | −0.2118 |

![20장 전체 점수 변화](figures/per_image_deltas.png)

B−A 하락이 가장 큰 두 이미지는 다음과 같다.

| 이미지 | A mIoU | B mIoU | B−A | 크게 감소한 상 |
|---|---:|---:|---:|---|
| A63_step1_sample4 | 80.9432 | 76.2949 | −4.6482 | Eutectic Si 56.0207 → 41.5349 (−14.4858%p) |
| A67_step1_sample1 | 72.8662 | 68.0906 | −4.7756 | Al3Ni 54.9724 → 43.7845 (−11.1879%p), Eutectic Si −5.7324%p |

이 두 이미지의 변화만 합쳐 전체 평균에 −0.4712%p를 기여한다. 나머지 18장의 평균 B−A는 +0.2134%p이다. 즉 B가 모든 조직에서 일관되게 나빠진 양상은 아니지만, 일부 이미지에서 큰 손실이 발생했고 전체 20장 중 13장에서 점수가 낮아졌다.

반대로 A72_step4_sample4는 B−A +2.3227%p로 개선됐다. Primary Si +5.2506%p, Al3Ni +3.2186%p가 주된 증가 항이다. 결과가 상과 이미지에 따라 달라, 복원 사전학습이 일관되게 필요한 표현을 학습했다고 결론내리기 어렵다.

## 6. 학습·계산 예산 확인

| 단계 | updates | 계산 구간 합계 (s) | 단계 경과 시간 (s) |
|---|---:|---:|---:|
| A 지도학습 | 4,000 | 486.1869 | 504.6620 |
| B 복원 사전학습 | 2,000 | 146.9336 | 150.7998 |
| B 분할 학습 | 4,000 | 463.6432 | 482.1466 |
| C 추가 지도학습 | 1,037 | 124.4490 | 129.3865 |

B의 합계 계산 시간은 610.5768초, C의 A 포함 합계는 610.6359초로 차이가 **0.0591초**이다. 6 GPU 수가 같으므로 측정한 계산 예산 대조는 의도대로 이루어졌다. A 대비 B는 계산량이 **25.58% 증가**했다. 전체 A/B/C suite는 검증·저장 등을 포함해 약 **22분 50.7초** 걸렸다. C는 A의 학습 구간을 공유하므로 실제 실행에서 A를 다시 학습하지 않았다.

이 계산 시간은 가장 느린 rank의 update 내 입력 전송·forward·backward·optimizer 구간 합이다. FLOPs 일치나 학습 경로 일치를 뜻하지 않는다. C는 A의 optimizer를 이어 받되 추가 구간의 LR을 다시 시작했다. 따라서 C 비교는 **추가 지도학습 계산 예산의 대조 실험**으로 해석한다.

![학습 손실 추이](figures/training_curves.png)

손실 로그 처음 20개 기록과 마지막 20개 기록의 평균은 B 복원 L1이 0.125885 → 0.099257로 감소했다. 이는 동일한 고정 이미지 세트의 복원 평가가 아닌, 학습 중 서로 다른 패치에서 기록한 손실 추이이다. B 분할의 마지막 20개 기록 평균은 0.310387로 A의 0.313724보다 조금 낮지만, 검증 점수는 높지 않았다. **복원 및 지도학습 손실 감소가 이번 검증 분할 개선으로 이어지지 않았다.** 이 로그만으로 과적합이나 특정 구조의 실패 원인을 확정하지 않는다.

## 7. 이후 실험 방향

1. **A를 기준 모델로 유지한다.** 이번 세 모델 중 주 지표가 가장 높고 계산 비용도 낮다. B의 작은 차이에 대한 안정성을 판단하려면 A/B를 같은 추가 seed로 반복해야 한다.
2. **외곽 오검출을 따로 다룬다.** A71에서 확인된 최상단 오류를 기준으로, train 내부 검증에서 외곽 패치 노출과 추론 padding/halo 조건을 한 번에 하나씩 비교한다. 정답을 보고 이 이미지의 픽셀만 지우는 수정은 실제 모델 개선으로 취급하지 않는다.
3. **Eutectic Si·Al3Ni의 혼동을 우선 확인한다.** 특히 A63·A67에서 복원 사전학습 후 큰 손실이 생겼다. 밝기 복원 중심의 목표가 상 구분에 필요한 정보와 맞지 않을 가능성은 가설이며 추가 실험이 필요하다. 같은 복원 설정을 더 오래 실행하는 것의 이득은 현재 결과로 뒷받침되지 않는다.

평가 공식을 바꾸거나 C의 정식 점수를 보정하지 않았다. 한 seed, 동일한 검증 20장에 대한 결과이므로 통계적 우월성·다른 시편에 대한 일반화까지 주장하지 않는다.

## 8. 검증과 산출물

- 실제 원본 100장·정답 90장의 파일 해시, 학습 시점 데이터 목록, 저장된 학습·평가 소스 사본의 해시를 확인했다.
- A/B/C 예측 마스크 60장에서 confusion matrix와 주 mIoU·상별 IoU·빈 클래스 3개 정책의 전체 mIoU를 다시 계산해 저장된 점수와 일치함을 확인했다.
- 비교 그림 60장의 240개 패널을 원본/오류/예측/정답 배열과 픽셀 단위로 대조했다. 우측 상단은 정답 픽셀=흰색, 오답 픽셀=정답 상의 색이다.
- 기존 그림은 아래에 바이트가 동일한 사본으로 보존했다. 기존 모델·학습 소스·이전 결과는 수정하지 않았고, 재학습·재추론하지 않았다.

수치: [summary.csv](summary.csv), [이미지별 비교 CSV](per_image_comparison.csv), [분석 JSON](analysis.json), [검증 JSON](verification.json), [출처](provenance.json), [분석 코드](analyze.py).

원본 보고서: [A · 0924_5](../model_results_0924_5/model_results_0924_5.md), [B · 0924_6](../model_results_0924_6/model_results_0924_6.md), [C · 0924_7](../model_results_0924_7/model_results_0924_7.md), [실행 직후 종합 · 0924_8](../model_results_0924_8/model_results_0924_8.md).

## 9. 검증 20장 비교 사진

각 링크의 배치는 좌상단 원본 / 우상단 오류 / 좌하단 예측 / 우하단 정답이다. 우측 상단의 회색은 Eutectic Si의 정답 색 `(204,204,204)`이며, 맞은 픽셀의 흰색 `(255,255,255)`과 다르다.

| 이미지 | A | B | C |
|---|---|---|---|
| A117_step4_sample2 | [비교](valid_mask_comparison/A/A117_step4_sample2.png) | [비교](valid_mask_comparison/B/A117_step4_sample2.png) | [비교](valid_mask_comparison/C/A117_step4_sample2.png) |
| A11_step1_sample4 | [비교](valid_mask_comparison/A/A11_step1_sample4.png) | [비교](valid_mask_comparison/B/A11_step1_sample4.png) | [비교](valid_mask_comparison/C/A11_step1_sample4.png) |
| A11_step2_sample4 | [비교](valid_mask_comparison/A/A11_step2_sample4.png) | [비교](valid_mask_comparison/B/A11_step2_sample4.png) | [비교](valid_mask_comparison/C/A11_step2_sample4.png) |
| A125_step1_sample4 | [비교](valid_mask_comparison/A/A125_step1_sample4.png) | [비교](valid_mask_comparison/B/A125_step1_sample4.png) | [비교](valid_mask_comparison/C/A125_step1_sample4.png) |
| A153_step3_sample3 | [비교](valid_mask_comparison/A/A153_step3_sample3.png) | [비교](valid_mask_comparison/B/A153_step3_sample3.png) | [비교](valid_mask_comparison/C/A153_step3_sample3.png) |
| A170_step4_sample3 | [비교](valid_mask_comparison/A/A170_step4_sample3.png) | [비교](valid_mask_comparison/B/A170_step4_sample3.png) | [비교](valid_mask_comparison/C/A170_step4_sample3.png) |
| A181_step1_sample1 | [비교](valid_mask_comparison/A/A181_step1_sample1.png) | [비교](valid_mask_comparison/B/A181_step1_sample1.png) | [비교](valid_mask_comparison/C/A181_step1_sample1.png) |
| A186_step2_sample2 | [비교](valid_mask_comparison/A/A186_step2_sample2.png) | [비교](valid_mask_comparison/B/A186_step2_sample2.png) | [비교](valid_mask_comparison/C/A186_step2_sample2.png) |
| A191_step4_sample3 | [비교](valid_mask_comparison/A/A191_step4_sample3.png) | [비교](valid_mask_comparison/B/A191_step4_sample3.png) | [비교](valid_mask_comparison/C/A191_step4_sample3.png) |
| A40_step3_sample3 | [비교](valid_mask_comparison/A/A40_step3_sample3.png) | [비교](valid_mask_comparison/B/A40_step3_sample3.png) | [비교](valid_mask_comparison/C/A40_step3_sample3.png) |
| A42_step2_sample4 | [비교](valid_mask_comparison/A/A42_step2_sample4.png) | [비교](valid_mask_comparison/B/A42_step2_sample4.png) | [비교](valid_mask_comparison/C/A42_step2_sample4.png) |
| A44_step4_sample3 | [비교](valid_mask_comparison/A/A44_step4_sample3.png) | [비교](valid_mask_comparison/B/A44_step4_sample3.png) | [비교](valid_mask_comparison/C/A44_step4_sample3.png) |
| A55_step2_sample1 | [비교](valid_mask_comparison/A/A55_step2_sample1.png) | [비교](valid_mask_comparison/B/A55_step2_sample1.png) | [비교](valid_mask_comparison/C/A55_step2_sample1.png) |
| A59_step3_sample1 | [비교](valid_mask_comparison/A/A59_step3_sample1.png) | [비교](valid_mask_comparison/B/A59_step3_sample1.png) | [비교](valid_mask_comparison/C/A59_step3_sample1.png) |
| A63_step1_sample4 | [비교](valid_mask_comparison/A/A63_step1_sample4.png) | [비교](valid_mask_comparison/B/A63_step1_sample4.png) | [비교](valid_mask_comparison/C/A63_step1_sample4.png) |
| A67_step1_sample1 | [비교](valid_mask_comparison/A/A67_step1_sample1.png) | [비교](valid_mask_comparison/B/A67_step1_sample1.png) | [비교](valid_mask_comparison/C/A67_step1_sample1.png) |
| A71_step2_sample1 | [비교](valid_mask_comparison/A/A71_step2_sample1.png) | [비교](valid_mask_comparison/B/A71_step2_sample1.png) | [비교](valid_mask_comparison/C/A71_step2_sample1.png) |
| A72_step4_sample4 | [비교](valid_mask_comparison/A/A72_step4_sample4.png) | [비교](valid_mask_comparison/B/A72_step4_sample4.png) | [비교](valid_mask_comparison/C/A72_step4_sample4.png) |
| A74_step3_sample4 | [비교](valid_mask_comparison/A/A74_step3_sample4.png) | [비교](valid_mask_comparison/B/A74_step3_sample4.png) | [비교](valid_mask_comparison/C/A74_step3_sample4.png) |
| A7_step1_sample4 | [비교](valid_mask_comparison/A/A7_step1_sample4.png) | [비교](valid_mask_comparison/B/A7_step1_sample4.png) | [비교](valid_mask_comparison/C/A7_step1_sample4.png) |
