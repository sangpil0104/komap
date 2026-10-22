# Model 349 추론 소스 축소 진행 기록

## 범위와 기준

- 공식 지표: valid 20장의 이미지별 4-class mIoU 평균, 결측 클래스 IoU=0.
- 보관된 P100 환경의 고정 Model349 경로를 재현한 기준은 **79.9095189%**다.
- 다른 실행 환경에서 생성된 최종 마스크의 80.068799%와 직접 섞어 비교하지 않는다. 그 환경의 하위 확률 출력은 이번 제거 실험에 없다.
- 아래 결과는 이미 경로 선택에 사용된 valid 20장에 대한 **고정 경로 진단**이다. 새 후보를 선택해 독립 성능을 주장하는 검증은 아니다.

## 실제로 제거 가능한 소스 확인

원래 Model349 경로 재현 스크립트는 확률 파일 8개를 읽는다. 그러나 고정된 경로 설정에서 195와 217은 로드만 하고 확률 결합에 사용하지 않는다. 명시적으로 결합되는 소스는 269, 322, 326, 329, 336, 347의 6개다. 별도로 `Model183` 캐시 기초 확률을 생성하는 상위 모델들이 있으므로 **6개가 전체 추론 체크포인트 수라는 뜻은 아니다**.

322·326·329 확률 파일은 각각 4-fold B4 체크포인트 출력을 합친 것이다. 336과 347은 서로 다른 HRNet 학습 단계의 출력이다. 따라서 322와 347을 생략하면 해당 B4 4개와 HRNet 1개의 추론을 생략할 수 있지만, 336이나 기초 캐시 생성에 필요한 모델은 그대로 남는다.

## 고정 경로 제거 실험

| 설정 | 활성 명시 소스 | valid image mIoU | 349 대비 |
|---|---:|---:|---:|
| 349 원형 | 6 | 79.9095189% | 기준 |
| 322 제거, 326으로 대체 | 5 | 79.8954105% | -0.0141%p |
| 347 제거 | 5 | 79.8940137% | -0.0155%p |
| 322와 347 제거 | 4 | 79.8748846% | -0.0346%p |
| 329 제거 | 5 | 79.7516167% | -0.1579%p |
| 336 제거 | 5 | 79.6774619% | -0.2321%p |
| 269 제거 | 5 | 79.7702405% | -0.1393%p |

322 제거는 기존 Al 채널의 소스를 326으로 바꾸고, 322·326의 1:1 B4 평균도 326 단독으로 바꾼 **고정 계수 대체**다. 347 제거는 해당 클래스 혼합 계수를 0으로 둔다. 두 제거를 결합한 경량 경로를 `--compact` 모드로 실행하여 269·326·329·336 네 확률 파일만 실제로 읽었고, 전체 소스를 읽는 진단의 79.8748846%를 재현했다.

## 아직 해결되지 않은 부분

1. **정확한 점수 보존은 아직 달성하지 못했다.** 4소스 경로는 valid에서 0.0346%p 낮다. 이 차이가 독립 데이터에서 유지·확대·역전되는지도 알 수 없다.
2. `Model183` 기초 확률을 만드는 여러 모델이 그대로 필요하다. 최종 배포 체크포인트 수를 크게 줄이려면 이 부분까지 포함한 증류나 다른 단일 모델 학습이 필요하다.
3. test 정답이 없어 test 성능 유지 여부를 검증할 수 없다. valid에서 제거 조합을 다시 최적화하면 반복 선택 편향이 커진다.

## 다음 검증 기준

train specimen-group OOF에서 원형·경량 경로의 동일 이미지 점수와 추론 시간·최대 메모리를 비교한다. 채택 기준은 **원형 대비 점수 하락이 없다는 독립 근거**와 실제 체크포인트/시간 절감이다. OOF 기초 확률을 재현할 수 없다면, 경량 경로를 근거 없는 동등 모델로 선언하지 않는다. 이후 기초 캐시를 포함한 앙상블 전체를 teacher로 하여 작은 학생 모델을 학습하되, 이전 단순 증류의 낮은 성능을 출발점으로 삼아 성능을 별도로 검증한다.

재현 코드: [`model349_source_prune_audit.py`](model349_source_prune_audit.py). 원형 및 제거 실험의 이미지별 값: [`model349_source_prune_audit.json`](results/model349_source_prune_audit.json). 4소스만 로드한 확인값: [`model349_compact_4sources_valid.json`](results/model349_compact_4sources_valid.json).

## 326·329 B4 체크포인트 가중치 평균 시험

두 계열은 같은 EfficientNet-B4 U-Net state_dict 키 766개와 텐서 크기를 공유한다. 각 fold의 체크포인트를 0.5:0.5로 평균하고 329의 입력 설정(`image_global`/`zcdf3`/`target_width=4000`) 및 D4 TTA로 4개 fold의 held-out train 이미지 70장(59 specimen groups)을 평가했다. fold 1 결과만 먼저 본 뒤 남은 fold의 방법과 채택 기준을 [사전 고정 문서](model349_b4_soup_protocol.md)에 기록했다.

| 70장 OOF | 공식 image mIoU |
|---|---:|
| 326 B4 4-fold | 78.7792% |
| 329 B4 4-fold | **79.3804%** |
| 326·329 fold별 가중치 평균 4개 | 79.1329% |

평균 모델은 329보다 **-0.2475%p** 낮았다. fold별 변화는 `+0.4393/-0.5402/-0.1969/-0.7154%p`로 1/4 fold만 개선됐으며 specimen-group bootstrap 95% CI는 `[-1.0159,+0.3894]%p`다. 사전 기준을 통과하지 못해 **채택하지 않고 valid 결합 평가도 하지 않았다**. 한 fold의 개선만 보고 모델을 합치면 안 된다는 반례다. [가중치 평균 생성 코드](model349_b4_weight_soup_probe.py), [paired OOF 집계 코드](model349_b4_soup_oof_audit.py), [OOF 결과 JSON](results/model349_b4_soup_oof_audit.json).

## 322 확률을 326·329에서 회귀 복원한 시험

70장 train OOF에서 326·329의 여덟 확률 채널로 322의 네 채널을 예측하는 ridge 선형 회귀를 네 fold로 교차적합했다. 정답 마스크는 적합에 쓰지 않았고, 적합 전 [검증 기준](model349_surrogate322_protocol.md)을 고정했다. held-out 확률 MAE는 fold 1-4 모두 326 확률을 그대로 322 대신 쓰는 것보다 컸다. 따라서 이 근사기는 **기각했고 valid 자료는 읽지 않았다**. [코드](model349_surrogate322_train.py), [OOF 결과](results/model349_surrogate322_linear_oof.json).

별도로 공유된 HRNet 단일 모델의 과거 `81.55%`는 사용자 확인에 따라 mIoU 측정 오류가 있는 수치다. 349의 성능을 유지하는 단일 모델 근거로 사용하지 않는다.

## 보조 표현 모델의 상쇄와 기초 분기 제거

역사적 Model183 기초 캐시에서는 별도 representation-ASPP 모델을 Primary Si 채널에 `gamma=0.40`으로 섞지만, 최종 Model349 경로는 `undo_model181_primary_blend`로 그 혼합을 즉시 역산한다. 따라서 원본 영상 처리에서 해당 보조 모델을 생략하고 **혼합 전 확률을 직접 만드는 경로**를 구현했다. 기존 캐시를 보조 모델 출력 없이 필요한 필드만 남긴 형태로 바꿔 채점하면 20장 각각의 점수가 원본과 정확히 같았다. 원본 영상 20장을 보조 모델 로드·추론 0회로 다시 처리한 결과는 `79.9094187%`로 원본 `79.9095189%`와 `-0.0001002%p` 차이였고, 최종 마스크 차이는 전체 15,757,260픽셀 중 18픽셀이다. 수치 차이는 새 GPU 재실행의 부동소수점 변동 범위로 보이지만, 같은 실행 조건에서의 원인 확정은 아직 하지 않았다. 캐시 크기는 2.6GB에서 481MB로 줄었다. [직접 생성 코드](model349_rep_free_raw_builder.py), [확률 재생성 비교](results/model349_rep_free_raw_valid_parity.json), [점수](results/model349_rep_free_raw_valid_score.json), [마스크 차이](results/model349_rep_free_raw_mask_parity.json).

기초 확률의 세 이전 분기와 pair 분기를 제거하는 고정 경로도 CPU에서 시험했다. 기존 경로에서 각 분기 기여도를 빼고 네 채널을 정규화한 **고해상도 단독 기초 확률**은 같은 valid 20장에서 `79.9473641%`로 원형 대비 `+0.0378452%p`였다. 원본 영상 20장을 해당 경로로 재추론한 고정 349 결합은 `79.9474070%`, 명시 소스 322·347까지 제거하고 실제로 269·326·329·336 네 소스만 읽은 경로는 `79.9257956%`였다. 이는 보관된 P100 원형 `79.9095189%`보다 `+0.0162767%p`지만, 이미 반복 사용한 valid의 진단 결과일 뿐 독립 개선 근거가 아니다. [고정 분기 제거 코드](model349_base_component_ablation.py), [원본 재추론 결과](results/model349_high_only_raw_valid_score.json), [4소스 로드 확인](results/model349_high_only_compact4_valid.json).

## 별도 D3 경량판 및 표현 모델 제거 검증

다른 실행 패키지의 D3는 B4 4-fold 두 세트, HRNet336, native269, 고해상도/TENT 기초 모델을 유지하고, 명시 소스322·347, 기존 기초 분기3개·독립 pair 예측, 1024 타일, B4/HRNet 다중 TTA를 생략한다. 해당 패키지의 **같은 환경**에서 valid20 원형 `80.0683465%`, D3 `80.0642960%`이며 P100 1장 벽시계 시간은 `1895.94초 → 407.52초`다. 이는 위 보관 P100 `79.9095%` 경로와 입력·확률·실행환경이 다르므로 서로의 절대점수를 직접 비교하지 않는다.

D3는 원래 보조 표현 모델181을 여전히 실행한다. 하지만 Model349 결합이 그 출력을 역산하므로 표현 모델 없이 pre-blend 기초 확률을 직접 넘기는 [수정 추론 코드](model349_compression/infer_lite.py)를 만들었다. 기존 릴리스 안에서 같은 valid20 원본 영상을 재추론한 점수는 `80.0644719%`로 D3 대비 `+0.0001758%p`, 원본 D3 마스크와 다른 픽셀은 `31/15,757,260`이다. 벽시계 시간은 `409.68초`로 기존 D3 `407.52초`보다 길어 **표현 모델 제거 단독의 속도 개선은 확인되지 않았다**.

[묶음 생성기](model349_compression/build_rep_free_bundle.py)로 독립 경량 패키지 `/workspace/model349_compress/model349_lite_rep_free_15weights_v2`를 만들었다. 여기에는 체크포인트 **15개**와 B4 fold별 `args.json` 8개가 있다. [무결성 검사기](model349_compression/verify_rep_free_bundle.py)로 체크포인트 15개 SHA-256을 모두 확인했다. 원래 23개 가중치 릴리스는 수정하지 않았다.

독립 15가중치 패키지에서 **valid20 전체를 원본 영상부터 재추론**한 점수는 `80.0643596%`로 같은 환경의 원형 `80.0683465%` 대비 `-0.0039868%p`, D3 `80.0642960%` 대비 `+0.0000636%p`다. D3 마스크와 다른 픽셀은 `22/15,757,260`이다. 벽시계 시간은 `366.56초`; 원형 `1895.94초`의 약 `5.17배`지만 단회 측정이며, 같은 코드의 이전 `409.68초` 실행보다도 빨라 실행 변동이 있다. 따라서 표현 모델 제거로 추가 속도가 생겼다고 단정하지 않는다. [독립 패키지 valid 점수](model349_compression/results/bundle15_valid20_metrics.json), [실행 시간](model349_compression/results/bundle15_valid20_completion.json).

독립 패키지의 test10 출력은 기존 D3 test 마스크와 `9,074,964`픽셀 중 **14픽셀**만 달라 일치율 `99.99985%`다. 시험 정답 점수는 알 수 없다. [test 실행 완료 기록](model349_compression/results/bundle15_test10_completion.json).

이 D3 관련 선택도 이미 여러 번 본 valid20에서 이루어졌으며 독립 일반화 주장은 아니다. 시험 데이터에는 정답이 없어 시험 점수 확인은 불가능하다. 위 test 마스크 일치율은 **원형 D3 출력을 얼마나 보존했는지**만 보여 준다.

## 고해상도 선택 경로의 추가 축소 탐색

기초 고해상도 경로는 이미지 평균 밝기에 따라 `restart51`, `model109`, `seed456`, `pairwise`, `ema`의 다섯 체크포인트 중 하나를 선택한다. valid20과 test10을 합치면 다섯 후보가 모두 실제 사용된다. [결과를 보기 전 판정 기준](model349_compression/high_source_prune_protocol.md)에 따라 먼저 모두 `restart51`로 대체했다. 11가중치 후보의 valid20 점수는 `80.0188978%`로 15개판보다 `-0.0454619%p` 낮고, A71은 `-0.3759%p`, A11은 `-0.3300%p`였다. 평균 손실 기준 `0.02%p`를 넘으므로 **11개 일괄 축소는 기각**한다. [원본 재추론 점수](model349_compression/results/force_restart51_valid20_metrics.json).

두 전체 재추론의 이미지별 독립성을 이용해 밝기별 소스 제거 효과를 분리했다. `ema`만 `restart51`로 바꾸는 14개 후보는 valid 이미지 2장에만 영향을 주며, 후처리까지 적용한 가상 조합은 15개판 대비 `+0.0005813%p`였다. `ema+seed456`을 대체하는 13개 후보는 `-0.0071256%p`; 가장 나빠진 영상은 `-0.1640%p`였다. 이는 **반복 사용된 valid를 본 후 선택한 탐색 결과**다. [이미지별 분해 결과](model349_compression/results/high_source_prune_valid20.json).

14개 체크포인트만 포함한 독립 묶음 `/workspace/model349_compress/model349_lite14_ema_pruned`를 만들고 SHA-256/메타데이터를 확인했다. valid20을 원본 영상에서 재추론하면 `80.0650399%`로 15개판 `80.0643596%` 대비 `+0.0006802%p`, 원형 23개판 `80.0683465%` 대비 `-0.0033066%p`였다. 가장 나빠진 valid 영상은 `-0.01608%p`, 15개판과 다른 마스크 픽셀은 `1,464/15,757,260`이다. 단회 벽시계 시간은 `417.09초`로 15개판의 `366.56초`보다 길었으므로 **추가 속도 향상은 확인되지 않았다**. test10 원본 영상 재추론 마스크는 15개판과 `11/9,074,964`픽셀만 다르다. 다만 test10에는 EMA 밝기 구간 영상이 하나도 없으므로 그 구간의 신규 자료에 대한 근거는 valid 2장뿐이다. [14개 추론 코드](model349_compression/infer_lite_14.py), [14개 묶음 생성기](model349_compression/build_lite14_bundle.py), [valid 점수](model349_compression/results/lite14_valid20_metrics.json). 14개판은 반복 사용한 valid에서 후보를 선택한 **탐색판**으로 보관하고, 독립 자료가 없으면 15개판을 보수적인 기본 선택으로 둔다.

EMA와 seed456을 모두 `restart51`로 대체한 **13가중치 독립 묶음** `/workspace/model349_compress/model349_lite13_ema_seed_pruned`도 SHA-256 검사를 통과했고 원본 valid20을 재추론했다. 공식 점수 `80.0572455%`는 15개판보다 `-0.0071141%p`, 14개판보다 `-0.0077943%p` 낮다. A74 한 장은 `-0.1639561%p`로 국소 악화가 커진다. valid 마스크 차이는 15개판 대비 `3,963/15,757,260`픽셀이고, test10 마스크 차이는 `13/9,074,964`픽셀이다. test10에는 EMA·seed456 밝기 구간 영상이 전혀 없으므로 이 일치율은 제거 경로의 안전성을 검증하지 못한다. [13개 추론 코드](model349_compression/infer_lite_13.py), [valid 점수](model349_compression/results/lite13_valid20_metrics.json). **13개는 더 작은 탐색 후보**지만, 영향받는 영상의 손실과 선택 편향 때문에 기본판으로 권장하지 않는다.


## 2026-10-08 FP16·TENT·축소·병목 재개 감사

GPU 사용자 매뉴얼을 읽고 실제 서버 상태를 확인했다. GPU6에는 다른 컨테이너 작업이 있어 사용하지 않았고, 비어 있던 P100 GPU7 한 장에서 순차 실행했다. 원래 중단 FP16 5-step은 모든 component 확률이 완성돼 별도 CPU fusion으로 복구했다(80.0655029%, 전체 시간 불명). TENT3/12는 부분 결과가 없어 신규 경로에 재실행했다. 기존 릴리스·가중치·미완료 산출물은 보존했고 기존 실험 파일 63개 해시와 원래 14·15개 가중치의 conversion source SHA를 재확인했다.

공식 지표는 valid20 이미지별 네 클래스 IoU 평균의 20장 평균이며 없는 클래스는 0이다. PNG와 GT에서 별도 재채점해 모든 metrics와 1e-12 이내 일치를 확인했다.

| 원본 valid20 재추론 | 체크포인트 용량 | image mIoU | 이번 lite15 대비 최악 이미지 | 시간 | 판정 |
|---|---:|---:|---:|---:|---|
| 14 FP16/TENT5 | 0.765131 GB | 80.0656365% | -0.0155649%p | 409.98초, 단회 | 저장 용량·품질 통과, 속도 개선 미확인 |
| 14 FP16/TENT3 | 0.765131 GB | 80.0486797% | -0.2007161%p | 430.11초, 단회 | 평균·이미지 손실 기준 실패 |
| 12 FP16/fold326 1·3 | 0.683321 GB | 80.1255610% | -0.0930182%p | 364.28초, 단회 | 품질 한도 통과, fold 선택 편향으로 기본판 미채택 |
| 10 FP16/326 제거·Al3Ni=0.8 | 0.601511 GB | 80.1188236% | -0.2232411%p | 345.08초, 단회 | 이미지 손실 기준 실패 |

단회 시간으로 개선을 확정하지 않았다. 이후 EMA 포함 15개 모델·TENT5·raw/style·모든 fold·269·계수를 보존한 FP16 저장+OpenCV2+D4 view batch4를 같은 GPU에서 A-B 각 3회 비교했다. 기준 시간 451.06/395.71/373.88초, 후보 422.88/384.15/395.72초로 중앙값은 395.71→395.72초이며 속도 개선이 없었다. 조건부 HRNet CUDA FP16 autocast+tile batch2→8도 새 A-B 각 3회 비교했다. 기준 795.25/408.67/408.58초, 후보 395.82/395.52/419.88초로 중앙값 408.67→395.82초(3.14% 감소)였으나 사전 10% 기준에는 미달했다. 품질은 두 후보 각각 3회 모두 통과했으며 15개 FP16 용량은 1,641,328,608→821,762,423 bytes(49.93% 감소)다. 기존 기본판은 교체하지 않았다.

사용자 요청에 따라 test10까지 마치고 추가 GPU 실험은 종료했다. 기존 lite15 대비 D4 마스크 변화는 462/9,074,964픽셀(99.99490907% 일치), HRNet 추가 후보는 537픽셀(99.99408262%)이다. 정답이 없어 test 점수는 주장하지 않는다. GPU7은 작업 종료 뒤 유휴 상태를 확인했다.

원본349 대비 평균 점수는 거의 유지되지만 기존 lite15에도 A125 한 장에서 -0.2282%p 차이가 있다. 이번 신규 후보 판정은 사전 프로토콜의 lite15 기준이며 모든 이미지에서 원본349 성능 보존을 주장하지 않는다. 반복 사용한 valid에서의 선택·재현이지 독립 일반화 검증이 아니다. 증류 예상 77.5~78.5%는 미측정 가설로 기록했으며 train70 성적만으로 교사의 암기 여부를 단정하지 않았다. GT+soft target 학습은 가능하지만 이번에는 학생을 훈련하지 않았다.

[한국어 상세 보고서](model349_compression/audit_20261008/report.md), [재현 코드·설정 안내](model349_compression/audit_20261008/README.md), [이미지·클래스별 결과](model349_compression/audit_20261008/comparison.json), [추가 의견·증류 검토](model349_compression/audit_20261008/review_notes.md). 서버 신규 결과는 `/workspace/model349_compress/audit_20261008`에 보관한다.


## 12개 FP16판 일반화 해석 보완 및 외부 검토 준비

12개판의 valid 평균 상승은 재현됐으나 같은 valid에서 fold를 골랐으므로 일반화 개선은 미확인이다. 같은 FP16/TENT5인 14개판 대비 평균 +0.0599245%p, 11장 개선·9장 악화, 최악 -0.0922867%p였다. 표본·specimen 상관·후보 선택 편향·비교 조건 혼재·단회 시간 및 12개판 test10 미실행을 [한계 기록](model349_compression/audit_20261008/lite12_generalization_notes.md)에 남겼다. [다른 AI 검토 프롬프트](model349_compression/audit_20261008/review_lite12_prompt.md)를 작성했다. 실제 외부 전송이나 추가 서버/GPU 실험은 하지 않았다. 이 두 문서는 감사 종료 후 추가했으며 기존 결과 아카이브에는 포함되지 않는다.


## 2026-10-08 12개·10개판 일반화 가능성 검증

사용자는 train70/valid20/test10 합계 100장뿐이며 추가 정답이 없다고 확인했다. 후보와 계수를 동결하고 데이터·학습 이력 감사, specimen bootstrap 20,000회(seed20261008), 한 specimen 제외 민감도, 두 후보의 원본 test10 재추론을 수행했다. 독립 정답 holdout이 없어 **독립 일반화 개선은 두 후보 모두 미확인**이다.

train-valid에는 specimen 7개(A125/A153/A181/A191/A40/A55/A7)가 중복된다. train70과 specimen이 겹치지 않는 valid는 13장/12그룹이나 이미 후보 선택에 쓰여 독립 holdout이 아니다. 동일 디코딩 이미지의 split 간 중복은 없었다. test 중 train/valid와 specimen이 겹치지 않는 것은 6장이다.

주 비교는 같은 FP16/TENT5/OpenCV2의 14개판이다. 12개판 평균 +0.0599245%p, 조건부 cluster-bootstrap 95% 구간 [-0.0090074,+0.1662496]%p; 10개판 +0.0531871%p, [-0.0464607,+0.2103558]%p로 둘 다 0을 포함한다. 이 구간은 선택 편향을 교정하지 않은 동일 valid의 표본 변동 진단이다. 한 그룹 제외 시 12개판 변화는 [+0.0113918,+0.0679356]%p로 양수였지만, 10개판은 A71을 제외하면 -0.0155908%p로 상승이 사라졌다. 14개 FP16 대비 최악 이미지 손실은 각각 -0.0922867/-0.2297681%p다. 12개판은 탐색 후보로 유지하고 10개판의 기존 국소 손실 기각을 유지한다.

원본 test10을 두 후보 각각 재추론했다. 기존 lite15 마스크 대비 12개판은 10,313/9,074,964픽셀 변화(99.8863577% 일치), 10개판은 24,250픽셀(99.7327813% 일치)였다. 정답이 없어 어느 모델이 더 정확한지 판정하거나 test mIoU를 주장하지 않는다. 전체 시간은 429.04/332.79초 단회이며 동시기 기준 반복이 없어 속도 개선 확정 근거가 아니다.

현재 서버의 326·329 각 네 fold 및 HRNet336 학습 출력은 동결된 배포판과 파일 SHA뿐 아니라 실제 저장 dtype으로 변환한 텐서도 다르다(각 B4 655키, HRNet1719키 불일치). 동결 bundle의 B4 args는 추론 설정만 포함해 training split의 완전한 실행 이력을 인증할 수 없다. 따라서 현재 OOF 캐시를 배포 12·10개판의 검증으로 가져오지 않았다. 기록된 원래 CV 분할을 적용해도 329 네 fold를 함께 사용하는 후보는 train 각 이미지를 다른 세 멤버가 학습했으므로 전체 후보의 strict OOF가 되지 않는다. 이력 불일치 원인은 미확인이며 후보 손상 판정이 아니다. 후보 가중치·코드·설정은 실행 전후 해시 일치를 확인했다.

[한국어 검증 보고서](model349_compression/generalization_20261008/report.md), [조건부 안정성 결과](model349_compression/generalization_20261008/valid_conditional_stability.json), [데이터·이력 감사](model349_compression/generalization_20261008/data_exposure_audit.json), [test 출력 비교](model349_compression/generalization_20261008/test_mask_stability.json), [사전 고정 프로토콜](model349_compression/generalization_20261008/protocol.md). 서버 신규 결과는 `/workspace/model349_compress/generalization_20261008`이다. 추가 학습·계수 탐색·릴리스 교체는 하지 않았고 GPU7은 종료 후 6MiB/0%로 유휴 상태를 확인했다.


## 2026-10-08 사용자 지정: 12개 FP16판을 후속 base로 전환

사용자 요청으로 `cpu2_fp16_12`(326 fold 1·3, 329 전체 4-fold, TENT5, OpenCV2)를 새 실험 base로 지정했다. 이전의 보수적 lite15 기본판 유지 권고와 별개인 명시적 사용자 결정이다. 원본 릴리스·가중치를 교체하지 않는다. `Model349_lite12_base/`에 한국어 설명서·레시피·검증된 원본 추론 코드·설정·12개 SHA manifest·valid/test/일반화 결과를 준비했다. base 포인터는 `experiments/model349_compression/model349_base.json`이다. 비공개 GitHub 저장소 `hyundoria/KoMaP-model349-lite12`에 코드·문서를 게시하며 체크포인트·원본 이미지·GPU 매뉴얼·서버 로그는 제외한다. valid 80.1255610%, 683,320,900 B; 시간은 단회이며 일반화 개선 미확인 판정은 유지한다.

게시 완료: https://github.com/hyundoria/KoMaP-model349-lite12 (private), 릴리스 `v1.0.0`, commit `e98ee1f4879c9acb3e22f54dfd43991dde9703fc`. 원격 54개 blob의 Git 해시를 로컬과 전부 대조해 일치 확인했다. 기존 작업 폴더의 변경/미추적 파일을 일괄 커밋하지 않고 새 하위 저장소만 게시했다. 실제 추론을 다시 실행한 게시 작업은 아니며 검증 완료된 코드·결과를 동결한 릴리스다.


### 업로드 대상 정정 완료
사용자 지정 대상은 `CNUKoMaP/lhs`였다. 해당 저장소 main의 `Model349_lite12_base/`에 동일한 54개 파일을 게시하고 루트 README에 안내를 추가했다. commit `f6456f54e5fc6517a89a3fdb679645a0520bbf61`; 기존 파일 삭제/교체 없이 README 끝에 설명만 추가했다. 원격 54개 blob 해시 일치를 검증했다. base 포인터의 GitHub URL을 정정했다. 잘못 생성한 `hyundoria/KoMaP-model349-lite12`는 삭제하지 않고 남겨두었다.


## 2026-10-08 12개 base 후속 탐색 시작
새 폴더 `base12_followup_20261008`에서 후보/새 base gate를 결과 전에 고정했다. 11개 seed456→pairwise, 12개 gamma329 Al3Ni 0.6/0.7, 12개 TENT4를 분리한다. gamma 저장 확률 탐색은 baseline 재현 일치했고 평균은 증가했으나 A153 국소 손실 −0.164744/−0.316721%p로 모두 기각했다. GPU7 순차 원본 재추론 진행 중이며 기존 audit GPU7 lock을 함께 확보했다. 다른 GPU 작업 및 다수 기존 CPU `exhaustive.py` 작업은 보존했다. CPU load 약62로 단회 이전 시간과 직접 비교하지 않는다. 기존 base 변경 없이 후보 별도 폴더·출력으로 보존한다.


### 12개 base 첫 후속 탐색 완료
11개 seed456→pairwise는 valid 80.1139142%, base 평균 −0.0116468%p·A74 최악 −0.2598343%p로 기각했다. 626,688,625바이트(8.29% 감소)지만 품질 기준을 넘었다. TENT4는 80.1199428%, base 평균 −0.0056183%p·최악 −0.0869077%p로 base 조건은 통과했다. 그러나 기존 lite15 대비 A74 −0.1177126%p로 실행 전 선택 코드의 이중 조건을 넘어서 반복 시간/test를 진행하지 않았다. 프로토콜 본문의 이중 조건 표현이 불완전했던 점을 report에 명시했다.

gamma0.6/0.7은 저장 확률 탐색 80.1397563/80.1562526%지만 최악 −0.164744/−0.316721%p로 기각했고 원본 재추론하지 않았다. 평균 증가를 성능 개선으로 채택하지 않는다. Al3Ni 변경 후 4클래스 재정규화 때문에 다른 클래스도 영향을 받는다. report의 마지막 일반 문구에서 계수 후보의 원본 재추론을 언급한 것은 부정확하며, 실제 수행한 원본 valid20 후보는 11개/TENT4 둘뿐이다.

[첫 후속 탐색 결과](model349_compression/base12_followup_20261008/report.md), [상세 비교](model349_compression/base12_followup_20261008/comparison.json). 서버 결과를 수집했고 아카이브 SHA256 213c53d5741d1d8d3c1db0c8b68f885805631f8ad5a87c630cc104f222c27304, 232파일/6,590,448바이트를 검증했다. 기존 로컬 1,169파일과 서버 원본 가중치/설정 보존 확인. 이후 B4 입력 재사용 반복 비교를 진행하며 고해상도 대칭 4방향 예측 후보는 그 종료 후 잠금·유휴 확인을 거쳐 단일 실행하도록 예약했다.


### B4 입력 재사용 구현 검증 완료
기존 12개 base의 fold별 중복 전처리를 같은 영상/동일 설정에서 재사용했다. 326·329 단일 GPU tensor와 valid20 전체 326/329/336/269 확률 모두 원본과 정확히 같았다. 전체 raw valid mIoU 80.1255844%, base 평균 +0.0000233%p·최악 −0.0010261%p로 미세 변동이며 성능 개선은 아니다.

동일 조건 교차 순서 각3회 기준 시간 [419.6481,356.8511,345.3724]초, 재사용판 [431.3948,422.6172,343.2912]초. 중앙값 356.8511→422.6172초(18.4296% 증가), 마지막 회차만 0.6026% 감소했다. 모든 반복 품질 조건은 통과했지만 시간 10% 감소 조건은 실패했다. 공유 CPU 부하가 감소하며 양쪽 시간도 달라졌다. 한 장 old-first probe의 속도 차이는 초기화/순서 영향을 포함하므로 개선 근거로 쓰지 않는다. GPU7 복수 compute 감지 없음. 체크포인트 수/용량 12개/683,320,900바이트 동일. 새 base로 채택하지 않는다.

test10은 base와 9,074,964픽셀 중 12픽셀 차이(99.9998678% 일치), 원본 재추론 208.3617초 단회이며 test IoU/정확도가 아니다. PNG/GT 독립 재채점 완료. [상세 보고서](model349_compression/base12_inputcache_20261008/report.md), [공식 비교/이미지별 손실](model349_compression/base12_inputcache_20261008/comparison.json). 469파일/24,780,276바이트 아카이브 수집 및 SHA c8ef295659ebcedabe292758fae4ce9276b7ec4ba215e7be434a3ca9d5927f12 검증. 고해상도 flip4 후보는 이 단계 종료 뒤 별도 실행하며 그 초기 단회 추론 중 독립 CPU 재채점 작업을 수행했으므로 초기 단회 시간을 개선 판정에 사용하지 않는다.


### 고해상도 4방향 예측 및 후속 탐색 종료
TENT5·원본/밝기 보정 입력·12가중치·계수·후처리를 유지하고 high D4 8방향만 대칭 flip4 4방향으로 줄였다. valid20 원본 재추론 및 PNG 독립 재채점 80.1141264%, base 평균 −0.0114346%p·A11 최악 −0.2031486%p로 둘 다 한도를 넘어 기각했다. lite15 대비도 A74 −0.1387592%p로 기각이다. 단회 321.8916초는 확정 속도 개선이 아니며 품질 실패로 반복 시간/test를 실행하지 않았다. source SHA와 모든 변화는 별도 bundle에 고정했다.

[고해상도 결과](model349_compression/base12_highflip4_20261008/report.md), [이번 전체 결과/판정](model349_compression/base12_followup_summary_20261008.md). 고해상도 아카이브 142파일/3,304,316바이트, SHA764d7ab2b466982354de7444f271e5cf1d46139b0b78d370877e098d5525b507 검증 완료. 기존 로컬 1,169파일과 서버 원본 가중치/설정 보존 확인, GPU7 종료 후 6MiB/0%, 우리 controller/inference 프로세스 없음. 타 사용자 작업은 보존했다. 이번 후보 중 새 채택안은 없어 사용자 지정 12개 base를 그대로 유지한다. 동일 valid 반복 탐색·일반화 미확인·test 점수 없음 판정도 유지한다.


## 2026-10-08 사용자 요청: 추가 탐색 시작
새 폴더 `base12_resident_packed_20261008`에 실행 전 프로토콜과 코드를 고정했다. GPU 매뉴얼 재확인, GPU7 6MiB/0%·기존 추론 없음·CPU load0.19 확인 후 기존 네 단계 GPU7 flock을 모두 확보하고 단일 순차 실행한다. 다른 GPU 작업은 유지한다. 현재 파일 2,024개의 보존용 해시를 기록했다.

고정 후보는 seed456/pairwise 0.5:0.5 가중치 평균 11개, TENT 객체 재사용+매 이미지 전체 원본 복원 12개, HRNet autocast FP16+tile batch8 12개다. CPU에서 두 high 모델 stem/encoder/context/decoder/head cosine 각각 0.9999993/0.9999559/1.0/0.9999670/0.9999178로 가중치 정렬이 가까웠으나 출력 유지의 근거로 삼지 않는다. 비율/fold/계수/TENT 단계는 후속 결과로 재조정하지 않는다. resident와 HRAMP 각각 이중 품질 조건을 통과하면 고정 결합판+전체 체크포인트 ZIP DEFLATE6 무손실 저장을 한 번 검증한다. soup도 통과하면 11개 결합, 아니면12개 결합이다. 모든 ZIP entry와 실제 모델 tensor 일치를 확인해야 full inference를 실행한다. 사전 우선순위 하나를 각3회 교차 시간 비교하고 test10을 확인한다. 프로토콜은 [추가 탐색 프로토콜](model349_compression/base12_resident_packed_20261008/protocol.md). 모델 가중치의 합성과 ZIP 압축을 구분하며 일반화 미확인 판정을 유지한다.


## 2026-10-08 base12와 80.2 후보의 고정 1:1 확률 평균
사용자 요청으로 공통 후처리 직전 최종 4클래스 융합 확률을 1:1 평균했다. 결과 전에 비율/위치/판정 기준을 고정했으며 비율 탐색은 하지 않았다. 기존 저장 확률 재생 및 GT 재채점으로 base12 80.1251258%, best14 80.2309769%, 평균판 80.1712491%였다. 단독 양 끝 모델 20장 마스크와 모든 클래스 IoU가 이전 재생 결과와 정확히 같음을 확인했다.

base12 대비 평균 +0.0461233%p, 최악 A181 −0.1742671%p, A11_step1_sample4 −0.1417468%p, A153 −0.1021124%p로 최악 0.10%p 한도를 초과했다. 평균판은 12장 상승/8장 악화이며 고유 checkpoint 16개가 필요하다. 평균 mIoU의 산술 평균이나 checkpoint tensor 평균이 아니다. 원본 재추론·실제 시간/파일 크기·test10·독립 일반화 성적을 주장하지 않는다. 기존 base/릴리스 유지. [코드·사전 프로토콜·전체 이미지별 결과](model349_compression/base12_best14_mean_20261008/report.md), [검증 결과 JSON](model349_compression/base12_best14_mean_20261008/result.json).


### 상태 복원·HRNet AMP·무손실 저장 반복 검증 완료
후속 resident/HRAMP/ZIP 결합12개판은 valid 원본 재추론 및 PNG 재채점80.1251872%, 모든3회 반복 이중 품질 기준 통과. 기존 기준 [347.63134,347.93431,347.51782]초, 후보 [331.39949,331.27345,331.15790]초; 중앙값347.63134→331.27345초, 4.70553% 단축으로 사전10% 시간 조건은 실패했다. 저장683,320,900→628,669,975바이트(7.99784% 감소), 원본12파일의 모든ZIP entry/tensor 동일성 직접 확인. AMP를 포함한 전체 결합은 예측 함수 완전 동일 보증과 구분한다. test10 일치99.9972452%, 250/9074964픽셀 변화이며 test점수 없음. GPU7 복수 compute 감지 없음, 서버 원본 SHA 보존 확인.

압축 감사 초기 KeyError 후 이종 checkpoint 키를 재귀 tensor 검사로 처리해 완료 추론을 반복하지 않고 재개했다. canonical 직접 원본 대조를 저장량 기준으로 사용한다. 실패 이력을 유지한다. 결과746파일/34,605,364바이트 수집, SHA e68cc2724fdf1b2837717503e2168761ba836a4f0693973a0b6111f07f7cfa91 검증; 기존 로컬2024파일 보존 체크 변경/누락 0개. [결과](model349_compression/base12_resident_packed_20261008/report.md), [일반화 개선 경로 제안](model349_compression/generalization_improvement_options_20261008.md).

조건부 singleworker12는 초기 GPU7 memory6MiB/utilization5% 유휴 검사 실패로 worker 시작 전 중단했다. 새 추론 결과나 속도 개선이 없으며 failure.json/preceding_gate.json을 수집해 보존했다. 타 사용자 프로세스 종료나 자동 릴리스 교체를 하지 않았다.


## 2026-10-08 사용자 지정: 80.2% 최고 조합 exh14로 후속 base 전환
사용자가 명시적으로 base전환과 탐색시작을 요청했다. 기존 Model349_exh14의14개 원본FP32 가중치(1,755,428,956바이트), 322f3·4/326f1·2/329f1·3·4/HR336·347/269/high4개를 동결한다. 이미 다른 실행에서 완료된 valid20/test10을 중복실행하지 않고 수집·PNG/GT 독립재채점했다. 공식 image mIoU80.2312313%, base12최악−0.2654099%p라는 이전채택한도초과 사실과 같은valid선택편향은 유지한다. 사용자지정비교기준 전환이며 일반화우위 인증이 아니다. test10 base12일치99.6205164%, 정답점수 없음. valid단회260.51초는GPU3,4 두개이므로 기존한GPU시간과비교하지 않는다.

현재 model349_base.json을 새base로 지정하고 변경전원문을 exh14_base_search_20261008/previous_base_pointer.json에 보존했다. 기존base12 문서/릴리스/가중치 유지. base12단일worker는 raw80.1253464%/302.47초 후 반복중이었고, 사용자기준전환에 따라 정확히우리PGID2131668/controller및두하위process만SIGTERM해 부분결과를보존했다. 타사용자나이미완료된14판프로세스에신호를보내지 않았다. base12증류pilot은 아직학습이시작되지않아 cancelled_before_launch.json으로보존했다.

새ROOT /workspace/model349_compress/exh14_base_search_20261008, PID2134675. 모든14원본가중치SHA/크기와실행entrySHA를검사한 frozen_before.json 완료, GPU7유휴·잠금·중복검사통과. 현재control14_valid20 한GPU기준을원본영상부터실행중이며failure없음. 다음FP16저장+ZIP14 → 조건부resident/HRAMP336·347/단일worker fast14 → 각3회교차시간 → test10을 결과전에고정했다. 새base14평균손실≤0.01%p/최악≤0.10%p, 단축10%gate. base12등비교는보조진단이다. 신규후보성과/실제속도를미리주장하지 않는다. [사전프로토콜](model349_compression/exh14_base_search_20261008/protocol.md), [원본base감사](model349_compression/exh14_base_search_20261008/base14_audit.json). 후속부분증류는3293→1/14→12로다시정의하며아직새학습미실행이다.


### exh14 저장판 완료 및 검사 경로 오류 후 안전 재개
서버현재파일재확인: control14_valid20 완료80.2312297%,450.2783초단회. FP16+ZIP14 완료80.2309419%,base14평균−0.0002878%p/최악A55−0.0108329%p로새품질조건통과. 저장808,883,605바이트(원본1,755,428,956 대비53.9210% 감소),14파일유지. FP16저장판단회833.7183초는기준단회보다느리며속도개선주장없음. 공유서버부하와packed로드비용을반복/경로별측정에서구분해야한다.

fast14본추론전복원probe가이전control12경로를참조하는FileNotFoundError로실패했다. run_search/probe 초기코드및failure_probe_path_initial.json을보존하고control14로수정했다. PID2137506/resume_search.py로잠금·GPU7유휴·원본SHA와준비된checkpointSHA를다시확인하여probe부터재개; 완료control/FP16추론은반복하지않는다. 현재새failure없고복원probe/fast14valid후각3회시간/test10은미완료다. 모든가중치는14개이며부분증류학습은아직시작하지않았다.

재개 최종상태 확인: RESIDENT_RESET_PROBE_PASSED, fast14_valid20 실제추론 START, GPU7 11,568MiB/100%, PID2137506 정상동작. 평균/최악손실과반복시간결과는아직없다.


## 2026-10-08 81% 정확도 목표로 전환: 전역 보정120개 완료 / TTA 재추론 시작

이전 대화 응답은 실험 진전이 없었던 턴으로 분류하고 현재 상태부터 재감사했다. 서버 GPU 매뉴얼을 다시 확인하고 타사용자 작업/공유스토리지와 원본 파일을 보존한다. 사용자 확인 대회는 KoMaP2026. 공식 process1/award_info/apply_info 페이지에서 데이터70/20/10, 이미지별4클래스 mIoU 평균, 제공데이터 augmentation 허용, mIoU50점 및test10 마스크/재현코드 제출을 확인했다. 외부데이터/valid정답학습 허용 여부는 공개페이지로 확인되지 않아 사용하지 않는다.

기존 fast14 raw valid 완료80.2311362%,607.2732초 단회; 저장판80.2309419%,833.7183초, 원본control80.2312297%,450.2783초. 속도 비교 반복은 진행중이었으나 정확도 목표에 맞춰 우리PGID2137506의controller/child argv를 모두 검증 후SIGTERM했다. cancelled_accuracy_priority.json에 이유와프로세스목록을보존했다. 타사용자프로세스에신호하지않았다. 단회시간은 속도개선 근거가 아니다.

새 accuracy81_20261008/protocol.md에서 결과전고정한120전역보정 후보(prior_scale5×al_delta3×eut_delta8)를 Mac CPU에서 모두채점했다. 기존14fold/라우팅/융합/TENT/필터는 유지했다. baseline20마스크/80클래스IoU가 기존replay와 정확히 같았고 선정후보 원래predict함수 재검증도 정확히 일치했다. 최고80.2309769%는 원래계수1.25/0/0.15로 **개선없음**. 고정grid specimen제외선택 진단은−0.2211463%p이며 독립일반화 검증이 아니다. result.json/grid.npz/selected_exact.npz/report.md/코드/프로토콜을보존한다.

다음은 정확도용대칭TTA이며 tta_protocol.md에 HR336/347 flip4→B4 322/326/329 D4→결합 순서를 결과전에고정했다. 이전YAML에tta라벨이있어도실제enabled=False였으므로 별도entry에서 실제True전달을구현했다. original14weights에symlink하고 원본entry/source/config/14체크포인트SHA를동결한다. GPU7lock9개+GPU유휴3회/compute없음 통과, PID2141791로단일순차실행. 현재hrflip4_14 rawvalid20 실제worker322 PID2141849 실행, GPU7 11568MiB/97%,failure없음 확인. 결과 미완료. 서버ROOT /workspace/model349_compress/accuracy81_20261008, local同名experiments폴더. 81%목표미달성/active유지, 일반화보장·test점수주장 없음.


### 81% 목표 후속: 전역융합48개 완료, TTA live 재확인

직전목표턴은전역보정완료/TTA시작으로progress분류. 서버현재 PID2141791/controller,원본infer child,실제GPU7worker를확인했고터미널/실패상태없음. 재실행하지않았다. HRflip4실험의기존분기322/326/329/base가완료되고현재336은6/20장까지진행, GPU7 1574MiB/47%,controller06:13동작 확인.

Mac에서결과전fusion_protocol.md로고정한3×4×4=48개Al3Ni326/329·Eutectic336전역융합을완료했다. 기존baseline20개IoU정확재현. 최고는326Al3Ni0.5,329Al3Ni0.5,336Eutectic0.75, 캐시80.2328978%(기준80.2309769 대비+0.0019208%p). 클래스변화Al3Ni−0.0346495/PrimarySi−0.0005858/Al−0.0028360/Eutectic+0.0457546%p. 최악A71−0.2878223%p; 조건부그룹bootstrap95%[−0.0382387,+0.0359856]%p,固定grid시편제외선택평균−0.1229598%p. 반복사용valid조건부이며독립일반화아님. 매우작은평균변화에비해국소손실과선택불안정성이커채택/추가raw추론하지않음. 탐색YAML과decision/report/grid/result동결,기존base유지.

완료TTA산출물만SHA검증수집하고PNG/GT독립채점하는collect_tta.py를추가준비했다. 실행중우리소스/로그를덮어쓰지않는timestamp snapshot을쓴다. TTA점수는아직미완료;81%미달성/goalactive유지.


### 81%목표: TTA 외부GPU점유 중단/부분진단/추가학습 준비완료

직전턴progress로분류,현재상태를재감사했다. 원래TTAcontroller2141791은hr336완료후다음347시작전GPU7 7526MiB/100%외부compute PID1159786 점유검사에서터미널실패했다. 원래controller/worker는없음을직접확인했다. 다른GPU3/4/6에도별도작업이있어건드리지않았다. 실패traceback/log/완료322/326/329/base/336각20장receipt보존. failure_gpu_occupied_initial.json으로이력을보존했다.

원본sourceSHA/완료5분기파일구조/유한probability/해시를검증하고347/269만재개하는resume_tta.py를준비했다. 실제PID2146242가30초마다GPU7점유를관측하며GPU유휴3회확인을기다린다. GPU를예약/사용하거나타프로세스를중지하지않는다. 이후B4/결합TTA는run_remaining.py로기존고정순서를따른다. 재개completion의시간은불연속resume구간이며속도비교에사용하지않는다.

완료HR336 flip4확률20파일을Mac에수집/전부서버SHA일치확인. 기존확률의336만치환한부분진단은캐시80.2370908%,base80.2309769 대비+0.0061138%p지만최악A181−0.4351261%p. 전체원본14모델/HR336+347flip4의확정점수가아니며일반화성능주장/채택하지않는다. 코드는diagnose_hr336_partial.py,결과/이미지별손실은hr336_partial_diagnostic.json/report.md.

새accuracy81_hrlovasz_20261008에결과전프로토콜/독립Lovasz손실/pairedtrain코드/sequence를준비했다. 기존HR336동결checkpoint공통초기화,train70만4epoch,effectivebatch8,encoderLR5e-6/decoder2e-5,BN통계고정,Control0.5CE+0.5Dice vs Treatment0.5CE+0.25Dice+0.25Lovasz. 새사전학습/외부데이터/valid학습없음. epoch/index별동일crop/augmentation/forwardseed,최종epoch4만원본valid20평가,원본13개가중치와기존융합유지. arm당40분cap/다른GPUcompute출현시우리학습만중단/소스불변검사.

첫CPU smoke는inference에잘려나간rgb_to_class import가없어실패했다. 별도training_palette.py/엄격4색디코딩을분리해수정했으며소스에기존영향없음. 분석적으로알려진Lovasz0.3625/완전예측0,유한gradient,pairedaugmentation,4색classorder,원본HR336로드/CPU128forward검사를모두통과했고P100마지막smoke.log와source SHA lock을수집/로컬동일성확인했다. 실제학습optimizerstep0. queue PID2148796가검증된TTA재개PID2146242종료를기다리는것을현재ps/queue_status로확인했다. TTA이미81이상이면추가학습skip하고독립재채점으로진행하도록한다. 현재81미달/goalactive유지.


### 81%목표: 검증된GPU대기 및Lovasz선행결과 원본대조

직전턴은부분확률검증/학습준비로progress,이번턴은라이브PID2146242(TTA재개)·2148796(학습대기)를두차례ps/새관측으로확인한verifiedwait다. GPU7은여전히타작업PID1159786/7526MiB/약100%이므로추가GPU시작/타작업종료/중복재추론하지않았다. 워크스테이션은MPS지원하지만RAM16GB/로컬HR336가중치·train70없음을직접확인해현재동결P100실험을별도백엔드/레시피로임의대체하지않았다.

이전Lovasz기록을model_results.md만으로신뢰하지않고서버model333_b4_imagewise_lovasz_group_oof/paired_control_vs_lovasz_orderfix.json 원본을수집했다. train70/59그룹/emptyunion0,control단독78.7483624%,Lovasz단독78.6379184%,직접−0.1104440%p이며Eutectic변화−0.3294446%p였다. 이전요약+0.0039924%p/CI[−0.0083531,+0.0171417]는선택된혼합route결과로단독Lovasz향상이아니다. artifactSHA와이구분을prior_evidence_note.json에명시했다. HR336의all-four-class Lovasz/Dice치환은B4의present-class추가와달라대조실험을유지하지만효과를예상/보장하지않으며boundedpaired검증이필요하다. 학습은아직0step,81%미달/goalactive유지.


### 81% 목표: 공간 후처리 100개 탐색 완료

직전 턴은 검증된 프로세스 대기와 선행 결과 대조였고, 이번 턴은 Mac CPU 탐색을 완료해 진전으로 분류한다. 결과 전에 spatial_protocol.md에서 고정한 Gaussian 5개 × Eutectic edge 5개 × cleanup 4개를 채점했다. 기준 모델의 20개 마스크와 80개 클래스 IoU가 기존 재생과 정확히 같았으며, 최고 후보도 원래 predict 함수로 재계산해 IoU 일치를 확인했다.

최고 후보는 Gaussian 0.5 / Eutectic edge 0.125 / cleanup 128이다. 캐시 mIoU 80.2354351%로 기준 80.2309769% 대비 +0.0044582%p였다. 최악 A71은 −0.0819971%p, 조건부 그룹 bootstrap 95% 구간은 [−0.0150950,+0.0253183]%p, 고정 grid의 시편 제외 선택 진단은 −0.0225282%p였다. 독립 일반화나 전체 원본 재추론 성적으로 주장하지 않는다. 개선이 매우 작아 채택하거나 추가 GPU 추론을 예약하지 않고 YAML/grid/result/report를 탐색 기록으로 보존한다.

TTA 재개 PID2146242와 후속 학습 PID2148796는 실제 ps로 모두 살아 있음을 확인했다. GPU7은 다른 작업이 7526MiB/100%로 점유 중이며, 우리 GPU 실험은 대기 상태다. 다른 GPU/프로세스를 변경하거나 중복 실행하지 않았다. 전역 보정 120개, 융합 48개, 공간 후처리 100개를 합한 268개 탐색에서 아직 81% 근거는 없다. 실제 TTA와 대조 학습의 검증을 이어가며 목표는 active로 유지한다.


### 81% 목표: GPU 대기 해소, TTA 재개 및 GPU3 대조 학습 실제 시작

직전 턴은 공간 후처리 탐색 완료로 진전이었다. 이번 턴 처음에는 GPU7 타 작업을 확인해 대기했으나, 새 자원 감사에서 GPU3/4/6의 점유가 없어지고 TTA 재개가 실제로 시작된 것을 확인했다. TTA PID2146242/worker2157310은 GPU7에서 남은347을 실행한다. 완료된5분기는 반복하지 않는다.

학습 결과를 보기 전에 실행 GPU/대기 순서만 변경했다. 원래 학습 대기 PGID2148796은 child와 training 폴더가 없는 것을 확인한 뒤 우리 대기 프로세스만 종료했다. 원래 소스/예약은 보존하고 cancelled_for_idle_gpu3.json에 기록했다. 새 accuracy81_hrlovasz_gpu3_20261008 루트의 scheduling_addendum.md는 GPU3 병행 실행을 명시한다. train70, 부모 HR336, seed/손실/LR/4epochs/paired crop·augmentation/최종checkpoint만 평가하는 정책은 그대로다. TTA 결과에 맞춘 레시피 변경은 없다.

새 GPU3 전용 lock/유휴3회/compute없음 검사를 통과했다. CPU smoke를 새 루트에서 다시 통과하고 source SHA lock과 환경(Python3.10.12, torch2.6.0+cu118, SMP0.5.0, timm1.0.29, albumentations2.0.8)을 수집해 로컬 소스와 전부 대조했다. 실제 새 controller PID2157571, train child2157582가 control 학습을 시작했다. 최신 확인은 epoch1/minibatch120 of420/optimizer30 updates, loss0.1768222, GPU3 4840MiB/75%로 OOM/실패 없음. GPU7 HR347은12/20까지 진행됐다. loss는학습신호이며 valid mIoU가 아니다.

추가학습/TTA 모두 결과 미완료, 새로운81% 성능 주장은 없다. 다른 사용자 프로세스/원본14 가중치/기존 릴리스는 유지하며 2개의 서로 다른 실험을 GPU3과7에서 진행한다. goal active 유지.


### 81% 목표: HRNet flip4 원본 valid20 완료 및 독립 재채점

직전 턴은 TTA 재개/실제 대조 학습 시작으로 진전이었다. 이번 턴은 완료한 HRNet flip4를 수집해 독립 PNG/GT 재채점했으므로 진전이다. hrflip4_14_valid20은 80.2400395%, 기준 원본80.2312313% 대비+0.0088082%p지만 A181_step1_sample1은−0.4375342%p다. source 불변/완료5분기 확률 불변 receipt와 20개 마스크/80 IoU metrics 일치를 확인했다. 아카이브 SHA를 검증하고 timestamp snapshot으로 수집해 기존 파일을 덮어쓰지 않았다. hrflip4_report.md에 클래스/이미지별 손실과 사후 조건부 그룹 bootstrap을 기록했다. 평균 상승이 작고 국소 손실이 커 기준 모델을 교체하지 않는다. 일반화 개선/81%/test점수를 주장하지 않는다.

중단 후 재개 completion의457.99초는 재개 구간만이며 전체 시간으로 잘못 보고하지 않도록 collector 표기를 수정했다. 기존3개 TTA 순서에 따라 GPU7의 B4 D4 실제 worker가 시작됐다. GPU3 control 추가 학습은 epoch2/minibatch40/optimizer115 업데이트까지 확인했으며 loss는 valid 성능이 아니다. 원본 가중치/릴리스/다른 사용자 작업은 보존하고 목표 active를 유지한다.


### 81% 목표: B4 D4 및 HR336 대조 학습 완료, 독립 PNG 재채점

이번 턴은 원본 재추론 두 실험을 수집·독립 재채점한 진전이다. timestamp archive SHA를 검증하고 기존 파일을 덮어쓰지 않는 snapshot으로 수집했다. 각 20개 마스크/80 클래스 IoU가 서버 metrics와 정확히 일치한다. 빈 union=0인 이미지별 4클래스 IoU 평균 후20장평균을 적용했다.

B4 322/326/329 D4는80.2457089%, 기준 원본80.2312313% 대비+0.0144776%p, 최악 A74_step3_sample4−0.2043264%p였다. 클래스 변화 Al3Ni−0.0147450 / PrimarySi+0.0235565 / Al+0.0007543 / EutecticSi+0.0483445%p. 단회 전체 추론897.0472초(GPU7)다. 조건부19그룹 bootstrap95%[−0.0290002,+0.0633737]%p로 0을 포함한다. 이는 반복 사용한 valid의 사후 진단으로 독립 일반화 증거가 아니다. b4d4_report.md/review.json에 이미지별 표와 판단을 보존하고, 결합 TTA·대조 학습 비교 전 base를 변경하지 않는다.

train70만 HR336 CE/Dice 추가 학습한 control 최종 epoch4는420 optimizer updates/899.4879초 학습 완료. valid/test를 학습 중 읽지 않았고 원본 부모 SHA가 불변이다. 원본13개+새HR336 구성의 전체 valid 원본 추론은80.1856429%, 기준 대비−0.0455884%p, 최악 A74_step3_sample4−1.1201246%p다. EutecticSi 평균−0.4504688%p가 주요 악화다. 14개 checkpoint bytes1,755,406,329, 단회 추론452.2102초(GPU3). 학습 자체가 개선을 보이지 않아 base 교체를 기각한다. control_report.md/review.json/learning_comparison.json에 학습receipt/부모SHA/이미지·클래스별 변화를 기록했다.

최신 live 관측: controller2157571은 GPU3에서 Lovász 처리군을 동일 부모/seed/train70/paired augmentation/4epoch 정책으로 시작했다. loss는 valid 성능이 아니다. TTA 재개2146242/remaining2158061은 GPU7 결합 TTA에서322·326 완료,329 시작을 확인했다. 두 실험을 중복 실행하거나 다른 사용자 작업을 변경하지 않았다. 처리군은 matched control 및 원래 base 양쪽과 비교해야 한다. 기존 릴리스/원본14가중치 보존, 81% 아직 미달성/goal active 유지, test 점수 및 일반화 개선 주장 없음.


### 81% 목표: 기존 HR392 보완성 7개 고정 후보 탐색 완료

직전 턴은 B4 D4/control raw PNG 독립 검증으로 진전이었다. 이번 턴은 GPU3 Lovász controller2157571과 GPU7 결합 TTA controller2146242/remaining2158061의 라이브 상태를 확인한 뒤, 새 CPU 보완성 탐색을 완료했다. 외부 사용자의 점유/원본/기존 진행 실험은 변경하지 않았다.

기존 Model401에서 원본 valid20 추론을 완료한 HR392 seed1·2의 전체4fold 평균 확률40개를 수집했다. 모든 server manifest SHA 및 복사 후 SHA가 정확히 일치했고 native 크기/20 ID/유한 값/정규화/team-to-main 순서를 검증했다. 처음 수집 검사는 manifest status를 generic COMPLETE로 잘못 예상해 파일 쓰기 전에 실패했다. 실제 producer의 INFERENCE_COMPLETE_NOT_SCORED와20 ID/test exclusion을 검사하도록 고쳐 원본 자료를 변경하지 않고 완료했다. 이미 과거 Model401 valid 결과를 읽은 뒤 설계한 탐색임을 protocol에 명시했다.

결과 전 hr392_complement_protocol.md에서 강도0.25/0.5/1 × 전체 또는 Eutectic-only 혼합+기준0, 총7개를 고정했다. 두 seed/8개fold를 모두 동일 비율로 사용해 개별fold를 고르지 않았다. 기준 cached20의80 IoU가 기존 best14 replay와 정확히 같고 최고도 원래 predict로 재실행해 정확 일치했다. 최고는 전체 혼합0.25, cached80.2380231%, 기준80.2309769 대비+0.0070461%p, 최악 A186−0.1324724%p다. Eutectic 평균−0.0375362%p, 조건부 group bootstrap95%[−0.0259319,+0.0407002]%p, 고정7개 범위 시편 제외 선택 진단−0.0197856%p. 독립 일반화/전체 후보 원본 재추론 성적이 아니다.

최고는 새8checkpoint 추가로22개가 되며 매우 작은 평균 개선에 비해 비용 증가/클래스 악화/선택 불안정성이 있어 채택 및 새 GPU 추론을 예약하지 않는다. protocol/search/result/grid/report/decision/40입력mapSHA를 보존했다. 기존 base14 유지, GPU3 paired Lovász 및 GPU7 combined TTA 실행 중,81% 미달/goal active 유지,test점수 없음.


### 81% 목표: 결합 TTA 원본 재추론/독립 재채점 완료 및 HR336 배율 실험 시작

직전 턴은 HR392 보완성탐색 완료로진전이었다. 이번 턴은새고해상도softensemble 탐색과결합TTA 독립재채점을완료하고 다음배율실험을 실제시작한진전이다. 처음라이브controller2146242/2158061/2157571을확인했고 중복실행하지않았다.

결합TTA14는원본valid20의B4 D4 + HR336/347 flip4 전체재추론을완료했다. 타임스탬프archive SHA대조/PNG·GT독립재채점에서서버metrics의80 IoU와정확히같다. image mIoU80.2548841%,기준80.2312313 대비+0.0236528%p. 최악 A186_step2_sample2−0.5006887%p, Eutectic 평균−0.0133253%p. 조건부19그룹bootstrap95%[−0.0493973,+0.0843454]%p로0을포함한다. 전체단회시간1350.2989초(GPU7),원본14가중치/bytes유지. 현재raw최고관측후보지만81미달/독립일반화증거아님. combined_tta_report.md/review.json/tta_comparison.json에이미지·클래스표와검증보존,현재base는교체하지않는다. 원본소스불변receipt와TTAsequencecompletion을확인했고controller2146242/2158061은정상종료했다.

Mac CPU에서결과전 high_ensemble_protocol.md에 고해상도4경로/EMA포함5경로 ×산술/기하평균 ×혼합0.5/1.0+기준 총9개를고정했다. 다른fold/분기/계수/후처리유지. 기준80 IoU와최고원래predict 재실행정확일치. 변경8개는모두80.18805~80.21551%로기준캐시80.2309769보다낮아기각하며GPU추론예약없음. protocol/search/grid/result/report/decision과입력SHA를보존. 이것은캐시탐색성적이며whole-raw/일반화성적아니다.

다음 accuracy81_hrscale_20261008/protocol.md에 HR336 동일 가중치의 scale1/3 치환 및기존2·신규1·3의고정1/3평균,총3후보를결과전에고정했다. 13개기존가중치/다른분기/계수/TENT5불변,새학습/외부데이터없음. 실제inference에서overlap이native px임을읽어scale3에170을유지하면native171tile/stride1 계산폭증임을실행전에발견했다. 기존overlap170/256비율을유지하도록scale1 tile512/overlap340/stride172,scale3 tile171/overlap114/stride57로확정했다. 실제_tile_starts의20장tilecount281/1540/3924를CPU확인했고local/server protocol·run_scale SHAexact를geometry_preflight.json으로보존했다.

새queue2162847은기존TTA가종료/완료되기전GPU를사용하지않았고,종료후9개기존GPU7lock+전용lock/유휴3회/compute없음통과후원본source·14checkpoint·raw확률SHA동결/입력파일SHA·calibration검증을수행했다. 바뀌지않는6분기는기존control14의raw확률을읽기전용symlink 재사용한다. baseline을재조립해80 IoUexact를확인했다. 새실험은rawHRcomponent+immutable분기replay이며전체fresh재추론/전체추론시간으로보고하지않는다. CPUbaseline/신규HRcomponent/평균시간의범위를completion에명시하고81관측시전체fresh확인이필요하다.

최신실제PID2162847은GPU7 scale1의원본HR336 추론을완료하고scale3을실행중이며 baseline_replay exact80 IoU를확인했다. scale1서버점수78.0854409%는독립PNG수집전이므로확정보고를보류한다. GPU3 Lovász는4epoch학습후rawvalid20의336까지진행중인controller2157571을확인했다. 다른사용자GPU/원본/기존릴리스보존. 목표81%미달/active유지,test점수주장없음.


### 81% 목표: Lovász/HR 배율 기각, 경계 보조 손실 복원 학습 시작

이번 턴 후반에 두 실험의 완료 파일과 controller 종료를 직접 확인했다. 원본 SHA 불변 및 원본 확률 불변 receipt도 확인했다. completed PNG/metadata를 타임스탬프 snapshot으로 수집하고 archive SHA를 대조했다. 독립 PNG/GT 공식 산식 재채점에서 각 20장/80 IoU가 실행 metrics와 정확히 일치한다.

Lovász 최종 epoch4(420 optimizer updates)는80.1806589%, 원래base 대비−0.0505724%p, matched CE/Dice control 대비−0.0049840%p였다. 원래base 대비 Eutectic 평균−0.4464457%p, 최악 A74−0.9763981%p다. 같은 train140파일/부모SHA/seed를 검사했다. 양쪽 모두 원래base보다 낮고 Lovász도control보다 낮아 채택하지 않는다. learning_comparison.json, lovasz_report.md/review.json에 matched 차이와 이미지·클래스별 표를 기록했다. test 추론/점수 주장을 추가하지 않는다.

HR336 입력배율 실험은 동일 원본 가중치로 scale1/3을 원본 valid20부터 생성하고 SHA로 동결한 기존6분기를 replay했다. baseline 조립은 기존control의80 IoU exact이며 같은 replay 대조군80.2312297%다. scale1 78.0854409%, scale3 78.8111609%, 고정1·2·3 평균79.9252238%로 모든 후보를 기각한다. scale_report.md/scale_decision.json에 같은 replay baseline 비교와 원본 canonical2GPU 결과의 미세차이, 시간범위(component/CPU만)를 명시했다. 이는 전체fresh 재추론/독립일반화 실험이 아니다. 기존 배율과 다른 입력의 실패이며 배율별 재학습 효과를 평가한 것은 아니다. controller2162847과paired controller2157571은 모두 종료/정상 completion을 확인했다.

후속 accuracy81_hrboundary_20261008의 protocol을 새 결과 전에 고정했다. 원래 HR336의 학습에 존재한 boundary auxiliary objective를 직전 CE/Dice/Lovász에서는 공통 제외했으므로, 같은 부모·train70·seed81336·data/augmentation·4epoch·batch2×accum4·LR5e-6/2e-5·BN통계고정 조건에 복원한다. segmentation0.5CE+0.5Dice + 원래 boundary BCE/Dice1.0, boundary head도 gradient를 받는다. 다른13가중치/fold/라우팅/융합/TENT5 불변이고 최종epoch4만 rawvalid20 평가한다. 이미 완료한 matched control을 재학습하지 않는다. 기존 부모가 train70으로 학습되어 있으므로 이 추가학습은 새 grouped-OOF/일반화인증이 아니다.

첫 CPU smoke는 릴리스용 model.py에 forward_with_boundary가 문서 이름만 남고 실제 함수는 없어 import 단계에서 실패했다. GPU step0/학습폴더없음을 확인하고 실패 log를 별도 보존했다. 원래 training source의 helper와 BoundaryHeadLoss/GT band를 training_boundary.py에 독립 복원했다. 서버 원본 함수 AST exact, step GT 경계band256px, 빈 경계 유한 loss/gradient, 부모HR336 CPU128 forward+boundary backward/all gradients finite, 기존 paired synthetic augmentation/palette/strict parent load를 모두 통과했다. 기존 source/weights는 변경하지 않았다. 준비된 모든py/md의 local/server SHA exact와 원래 paired loss/dataset/palette 파일 SHA 동일을 확인해 prepared_source_lock.json과 smoke logs를 수집했다.

새 controller2165362는 기존 paired 완료/sourceunchanged/PID종료를 확인하고 GPU3 기존학습lock+전용lock/유휴3회/compute없음 검사를 통과했다. 실제 train child2165446은 boundary arm epoch1/minibatch80/optimizer20 updates까지 진행해 loss/gradient 유한, 실패 없음이다. 외부 compute 등장 시 우리학습만 중단하고40분cap/부분checkpoint보존을 적용한다. 수집·PNG독립채점·같은stream control 비교용 collect_boundary.py를 준비했다. 학습 loss는 valid점수로 부르지 않는다. 원본14/릴리스/타GPU작업/기존미추적 변경 보존, 목표81%미달/active 유지.


### 81% 목표: 추가 후처리 기각 및 작은 잔차 보정 pipeline 실제 시작

직전 턴은 Lovász/배율 독립 채점과 boundary 학습 시작으로 진전이었다. 이번 턴은 boundary 원본valid 독립채점, CPU 고정77후보 검토, 신규 잔차pipeline 구현·검사·실제 실행으로 진전이다. 최신 프로세스와 GPU점유를 확인하고 타사용자작업/원본/기존dirty tree를 보존했다.

경계손실복원 boundary arm은최종epoch4/420 optimizer updates의전체원본valid20을완료했다. PNG/GT독립재채점에서80 IoU exact, 80.1662518%, 원래base 대비−0.0649795%p, 동일stream CE/Dice control 대비−0.0193911%p다. Eutectic mean−0.4258651%p, 최악A74−1.0055761%p. 동일train140파일/부모SHA/seed/segmentationloss·dataset·paletteSHA를확인했다. boundary 복원으로도개선되지않아채택하지않으며,손실제외가하락의원인이라고단정하지않는다. boundary_final_report.md/decision.json/comparison.json을보존하고controller2165362 종료/정상completion/sourceunchanged를확인했다.

원논문 Guided Image Filtering(ECCV2010; 저자MIT PDF의수식5·6·8)을확인하고원본native밝기guidance로4logit을보정하는guidedfilter를독립구현했다. radius[1,2,4]×epsilon[1e-4,1e-3,1e-2]×blend[.25,.5,1]+기준의28개를결과전에고정했다. radius0identity/constantfield/constantguide/finite synthetic smoke와원본영상20개SHA/canonical manifest일치를확인했다. 기존base20마스크/80 IoU 및최고별도재계산이exact다. 최고는기존그대로80.2309769%이며변경27개는모두낮아기각했다. guided_protocol/filter/search/grid/result/report/decision을보존한다. 저장확률탐색/반복valid이며wholefresh/독립일반화성적아니다.

그동안독립조정하지않은Al3Ni·Primary Si 상대logit bias를각[-.2,-.1,-.05,0,.05,.1,.2]의49개로사전고정했다. Gaussian/bilateral 후/cleanup 전에모든이미지동일계수를적용했고다른fold/계수/후처리불변. 기준마스크/80 IoU와최고재계산exact. 최고다시기준00 80.2309769%,고정49범위시편제외선택진단−0.0819895%p. 변경후보채택/추가GPU추론없음. phase_bias_protocol/search/grid/result/report/decision을보존한다. 독립일반화나test점수주장없음.

새 accuracy81_residual_20261008는원래parent14를보존하고precleanup refined logits에작은native CNN 잔차를더하는방법이다. input4logits+rawgray1,Conv3×3 5→32→32→4/SiLU/정규화·dropout없음,마지막conv zero-init,11,876parameters,delta=tanh(output) bounded1이다. 최초출력은부모와같고이후원래cleanup/presence규칙을그대로쓴다. train70 GT만seed81349/8epoch/70×16native256crops/batch8/AdamW1e-3·wd1e-4/cosine8/FP32/clip1/CE.5+Dice.5+.05delta²로학습하며최종epoch8/1120updates만평가한다. 50%train GT Al3Ni/Eutectic positive crops와동일geometric D4 augmentation만사용한다. valid/test를trainer에전달하지않고중간epoch를valid로고르지않는다. 외부데이터/새pretrained다운로드없음. parent가학습한train예측을학습자료로쓰므로held-out/strictOOF/일반화인증이아님을명시했다. pipeline은parenttrain70→fields exact검증→잔차학습→parent전체freshvalid20→잔차적용이며재현/원본불변/test10까지확인전81달성으로부르지않는다.

기존 /runs/lite14_train_gpu6가완료70장/82.9767928%fit/5분기cache로존재함을파일에서확인했다. 현재exh14는322·347 및fold구성이달라이캐시를현재parent 학습자료로재사용하지않는다. 현재 원본14 recipe로teachertrain70을다시생성한다. CPU smoke에서synthetic zeroidentity/유한headgradient/잔차상한/paramcount,원래predict AST로추출한실제parent precleanup fields의마스크exact를확인했다. local/server모든py/md SHA exact와preflight_source_lock/smoke.log를수집했다.

새 controller2167874/teacher child2167948이GPU7에서실제로시작됐다. 기존GPU7 controller종료·배율completion/sourceunchanged 확인,기존10개lock+새lock/유휴3회/compute없음검사를통과했다. 모든14 checkpoint/source/config SHA를parent_frozen.json으로동결했다. 매10초 foreigncompute 확인 시우리child group만종료하고부분산출물을보존한다. 학습cap20분/비유한값/OOM은terminalfailure로기록한다. 최신ps에서둘다live이며B4 322/326/329가완료되고고해상도base52/70까지진행했다. 잔차optimizerstep은아직0,새valid성적없음이다. 최종수집용collect_residual.py는freshparent와의paired PNG/GT채점,checkpoint SHA/actual parent14 bytecounts,zero20-maskexact,train-onlyreceipt/preflight-sourceexact/wholecompletion을검사하도록준비했다. 현재최고raw80.2548841%,81미달/goalactive유지,test mIoU주장없음.


### 81% 목표: 잔차 pipeline 라이브 대기 및 실제 crop/objective 합성 검사

직전 턴은 boundary독립채점/77후보탐색/잔차pipeline실행으로 진전이었다. 이번 턴은 시작부터 controller2167874/teacher child2167948을ps로확인했고 GPU7 실제HR336 실행/외부GPU0·1·2·5 점유를관측했다. 다른사용자작업/GPU/원본을변경하거나새GPU작업을중복실행하지않았다. observation timeout이나과거state파일을근거로재시작하지않는다.

현재동결된train_residual.Crops/__getitem__와objective를GPU비활성CPU에서직접검사했다. 합성320×340 source에서label값을logit·gray에도encoding하여16 index의crop/H/Vflip/rot90후GT와동일위치인지확인했다. 동일epoch/index의bitwise재현,다른epoch 스트림분리,zero-refineridentity,실제CE/Dice/residual목표의유한loss/gradient를모두통과했다. 실제KoMaP정답이나valid/test를이검사에쓰지않았고실행중소스/레시피는변경하지않았다. dataset_objective_smoke.json에소스SHA/결과를보존하고서버→로컬수집후SHA를직접대조했다.

부모train70생성은계속실행중이며B4세분기와고해상도분기를완료했고HR336을진행한다. 잔차학습optimizerstep은아직0이다. 새성능/81%/독립일반화/test점수주장없음. 현직접라이브handle을기다리며goalactive를유지한다.


### 81% 목표: KoMaP 2026 확인 및 잔차 실험 진행 상태

- 사용자가 대회명을 KoMaP 2026으로 확인했다. 공개 규정 확인 기록은 `accuracy81_20261008/protocol.md`에 있다. 신규 학습은 제공 train70 정답만 사용하고 외부 데이터·valid/test 정답을 학습에 사용하지 않는다.
- 서버 실측 확인: 잔차 controller PID2167874와 teacher PID2167948이 계속 실행 중이며 failure/completion 파일은 아직 없다. 원본 14가중치의 train70 7개 분기 추론이 모두 완료됐고 CPU에서 최종 마스크를 결합 중이다. GPU7은 이 결합 단계에서 6MiB/0%; 다른 GPU의 기존 점유를 변경하지 않았다.
- 이는 완료 성적이 아닌 진행 상태다. 잔차 모델 학습 및 fresh valid20 평가가 아직 남아 있고, 81% 목표 달성이나 일반화 개선을 주장하지 않는다. 기존 controller를 재시작하거나 중복 작업을 실행하지 않았다.


### 81% 목표: train70 입력 완료와 GPU 대기 재개

- 원본 exh14 train70 전체 재추론 완료: 1779.8751초(단회), train fit image mIoU 83.03372494%. 학습 이미지 성적이며 held-out/valid/test 성적으로 사용할 수 없다.
- 학습 입력 70개 모두 생성 완료; manifest의 `parent_masks_exact=true` 확인. 학습용 정답은 train70에만 들어가며 valid/test는 학습에 제공하지 않는다.
- 타사용자의 GPU7 점유(PID1301197) 때문에 학습 시작 전 컨트롤러가 유휴 대기했다. GPU3의 3회 유휴 및 우리 자식 없음 확인 후 우리 대기 PID2167874만 SIGTERM 종료했다. 타사용자 작업은 변경하지 않았다.
- 학습 목표함수/모델/seed/샘플러/8epochs/1120updates는 그대로다. 점유 검사와 결과 GPU 영수증만 선택 GPU 환경변수에 연결했다. 변경 전 코드/프로토콜/해시를 `accuracy81_residual_20261008/migration_snapshot_gpu7/`에 보존하고, 제한된 변경의 정확한 비교와 문법 검사를 `gpu_migration_code_audit.json`에 기록했다. 원본14 가중치 및 원본 추론 코드는 변경하지 않았다.
- `resume_gpu3.py`를 새 컨트롤러 PID2176349로 실행해 완료된 teacher70/feature70의 SHA를 검증하고 재사용한다. 현재 PID2176349가 실제 살아 있으며, GPU3에 다른 작업이 들어와 `waiting_gpu3_idle`로 대기 중이다. 아직 training 디렉터리/학습 자식/실패 기록이 없으며 valid20 점수도 없다. 유휴가 되면 같은 controller가 학습→fresh valid20→보정 평가를 순차 실행한다. 중복 teacher 재추론이나 다른 사용자 작업 중단은 하지 않았다.


### 81% 목표: 잔차 8-epoch 학습 완료, fresh valid20 실행

- 첫 학습 시도는 PyTorch nll_loss2d CUDA가 strict deterministic 모드를 지원하지 않아 첫 optimizer update 전에 실패했다. 실패 로그/receipt를 서버 `attempt1_failed_deterministic_ce/`에, 수정 전 소스를 `deterministic_ce_snapshot/`에 보존했다.
- strict deterministic을 유지하면서 CE 계산만 동치인 −log_softmax(logits).gather(target).mean()으로 수정했다. CPU 값/gradient 차이0, CUDA 값 차이1.1920928955e−7 및 gradient 차이0; 실제 모델·전체 objective strict backward 합성 검사 통과. KoMaP 입력/정답을 검사에 사용하지 않았다. 학습 설정과 seed는 변경하지 않았다.
- 새 controller PID2177147/학습 PID2177183으로 재개해 8epochs/1120updates 정상 완료, 학습 loop37.8637초, epoch8.pt 49,726bytes, SHA256 da59463b38643b91a3baec2a20ec421a82f70d5f24dfe6b7154bd5ad280811c9. valid_read=false/test_read=false를 확인했다. 로컬 `training_receipts/`에 결과를 수집하고 checkpoint SHA/epoch/updates를 독립 확인했다.
- 현재 controller2177147과 fresh 원본 valid20 추론 PID2177394가 실제 살아 있고 GPU3에서 실행 중이다. 완료된 teacher train70/입력70은 SHA 검증하여 재사용했으며 중복 학습/재추론을 실행하지 않았다. 아직 보정 후 valid20 성적은 없어 81% 달성/일반화 개선을 주장하지 않는다.


### 81% 목표: 잔차 모델 valid20 독립 검증 완료 및 기각

- 모든 순차 작업 정상 완료, 원본14 source/weights SHA 불변 확인. 새 fresh parent80.2312280101%, 잔차79.7361955388%; PNG/GT 독립20장/80IoU와실행metrics 정확 일치. 같은 fresh parent 대비−0.4950324713%p.
- Al3Ni−0.8257924,PrimarySi−0.2980392,Al+0.0460634,EutecticSi−0.9023617%p.9장 개선/11장 악화,최악A71_step2_sample1−5.3888960%p. rawgray/logit tiny residual의 train 학습이 이 valid의 개선으로 이어지지 않았다. 원인을 클래스 불균형으로 확정하지 않는다.
- zero-refiner20마스크exact,기존 source/14가중치변경없음. 15개checkpoint총1,755,478,682bytes. 전체fresh valid pipeline521.2025초(원본450.7893초),보정/eval9.8093초,모두단회. 속도개선/독립일반화/test mIoU 주장없음.
- `accuracy81_residual_20261008/residual_comparison.json`, `residual_report.md`, `residual_decision.json`에 archiveSHA·입력/source lock·학습receipt·독립채점·이미지/클래스차이를보존했다. 기각 후보이므로 base/릴리스교체하지 않는다.81%목표미달이며목표를완료처리하지않는다.


### 81% 목표: 균형 CE 후보 기각 및 OOF 입력 감사 완료

- 클래스 균형 잔차 8epochs/1120updates 완료, 학습37.9650초. 원본 예측을 SHA 확인해 재사용한 평가이며 새 전체 fresh pipeline은 아니다. 독립 PNG/GT 채점79.2159837592%, 같은 원본 대비−1.0152442508%p. 네 클래스 모두 악화했고 최악A71_step2_sample1−8.2337733891%p다. zero-refiner20마스크exact 및 원본 source/14weights 불변을 확인했다. 후보를 기각하고 base를 유지했다. 결과와 차이는 accuracy81_residual_balanced_20261008/decision.json 및 residual_report.md에 기록했다.
- 후속 입력으로 기존 Model397 network-group OOF 확률12개를 감사했다. 실제70장/59시편/4fold18,18,17,17이며 학습/제외 시편이 겹치지 않는다. 확률파일SHA/ID, B4훈련split, HR392실제train/valid링크, args/runreceipt SHA, OOFmanifest확률SHA가 모두 일치했다. GPU 및 공식valid/test를 사용하지 않았다. 일반Python의 numpy없음 오류 후 기존venv로 실행해 통과했다.
- accuracy81_oof_refiner_20261009/audit_oof.py, oof_input_audit.json, protocol.md에 입력 감사와 고정 후속 정의를 보존했다. rawOOF 실제 점수 재현과 OOF 입력 기반 잔차 학습이 다음 단계다. 학습 Model397과 배포 parent14의 입력 차이, 반복valid 개발, 전체pipeline 독립OOF/일반화 미확인의 한계를 명시했다. 아직 후속 학습/새valid 점수가 없고81%목표는 active다.


### 81% 목표: network-OOF 입력 잔차 학습·독립 valid20 평가 완료

- Model397 OOF70 실제 확률/GT 재현79.98586200807742% 확인. 처음main순서argmax로0.0000037395%p 차이가나 학습전종료했으며,원래team argmax→mainlabel변환으로수정했다. 동률218pixels의우선순위차이였고엄격한허용오차를늘리지않았다. 실패source/log/receipt를보존했다. 이미생성한70개features를모든배열exact검사로재사용했다.
- 동일unweighted trainer/model/seed/8epochs/1120updates로GPU3학습완료39.9527초,epoch8.pt49,726bytes,SHA9fd39362251f737c092222317b7e775e533f89902135cc4839df739ce24ed63a. train70정답만사용/valid_read=false/test_read=false 확인. 원본valid20fields는SHA검증해읽기전용재사용,새fullfreshpipeline성적이아니다.
- PNG/GT독립채점79.9993891386%,같은부모80.2312280101%대비−0.2318388715%p. Al3Ni−0.6918257,PrimarySi−0.1862216,Al−0.0648791,EutecticSi+0.0155709%p. 최악A71_step2_sample1−2.2621820%p. 초기zero20maskexact/원본source14weights불변. source/입력lock/OOF재현/학습receipt/20PNG/독립차이/기각결정은accuracy81_oof_refiner_20261009에보존했다.
- 앞선in-sample잔차보다손실은작지만부모보다낮아기각했다. OOF모델397raw입력과배포parent14후처리logits차이의영향은분리되지않았으며원인으로단정하지않는다. basepointer와릴리스보존/test mIoU없음/81미달/goalactive.


### 81% 목표: 같은 source와 밝기 입력 세 후보 독립 평가 및 기각

- 같은 network 계열 Model401 raw 확률에 OOF 잔차를 적용했다. 원래 baseline79.4181708851%와 zero-refiner20mask exact를 재현했고 확률40개/B4archiveSHA 및 원본 source/14weights 불변을 확인했다. 독립PNG/GT79.2448654032%, 같은source대비−0.1733054819%p, 최악A74_step3_sample4−1.3974390%p. 기각했다. 입력계열을맞춰도개선되지않았으므로앞선손실원인을입력차이만으로확정할수없다. 결과는accuracy81_oof_refiner_20261009/matched_comparison.json 및 matched_report.md에보존했다.
- 사전고정한밝기zscore/밝기제거를GPU3에서순차학습·평가했다. 각8epochs/1120updates와마지막epoch8만평가했다. train70만학습/정규화합성검사/feature-sourceSHA/zero20identity/원본불변/독립20PNG80IoU채점이통과했다. zscore79.2195996595%,밝기제거79.1731641089%로모두baseline79.4181708851%보다낮아기각했다. 최대손실은A74에서각−1.3196890/−1.4655659%p다. 밝기변환이해결책이라는증거없음.
- 두밝기후보의archiveSHA/source snapshot/학습receipt/최종checkpoint/독립차이는accuracy81_oof_gray_ablation_20261009/comparison.json 및 report.md에보존했다. 모두부모출력재사용이며wholefresh실행/독립일반화/속도개선/test mIoU주장없음. 기존base와릴리스유지/81미달/goalactive.
- 다음실험은전체native이미지의SoftIoU를공식이미지별4class/empty0단위에맞추는두profile로 accuracy81_fullimage_iou_20261009/protocol.md에고정했다. base14의학습이미지예측과Model397networkOOF입력의차이를명시했다.560updates/finalepoch8/합성hardIoU일치검사/원본보존을정의했으며아직후속구현·학습·새성적은없다.


### 81% 목표: 전체 이미지 SoftIoU 두 profile 기각 및 HR336 BN 구조 감사

- 공식hardIoU/emptyclass0 합성검사(단일존재클래스완전예측mIoU=.25)/strictbackward/zeroidentity가통과했다. native원본해상도·batch1·70images×8epochs=560updates를각profile에학습하고최종epoch8만평가했다. train70정답만사용/valid_read=false/test_read=false 확인. 원본feature·sourceSHA 및 원본14weights불변을확인했다.
- base14입력profile 독립PNG/GT79.6194301977%,같은parent80.2312280101% 대비−0.6117978124%p,최악A71_step2_sample1−5.9226730%p. OOF397입력profile 같은source40179.4181708851% 대비79.0873207457%,−0.3308501394%p,최악A74_step3_sample4−1.2026909%p. 둘다기각. 각checkpoint49,726bytes,학습34.7723/35.7656초. 부모출력재사용이라전체fresh시간으로보고하지않는다.
- source snapshot/학습receipt/20PNG80IoU 독립채점/zero20maskexact/원본불변/전체차이를 accuracy81_fullimage_iou_20261009/comparison.json 및 report.md에보존했다. 수집시심볼릭링크기준출력이누락돼후처리만멈췄고,동일한기존수집·검증control출력을사용해수정했다. archiveSHA를서버와다시대조했다. 새학습/추론을중복실행하지않았다. 기존base/릴리스유지/81미달/goalactive/독립일반화·test mIoU주장없음.
- 다음경로로기존ResNet34 AdaBN코드를읽고HR336의새이미지별BNbuffer보간을분리했다. HR336 원본checkpoint를CPU에서만strictload하고BN runningstat유한값·track_running_stats·eval상태·checkpoint SHA불변을검사했다. 이미지/GT/GPU는사용하지않았다. 감사파일/고정alpha=.25,1프로토콜은 accuracy81_hr336_moments_20261009에보존했다. 아직새GPU후보결과없음.

HR336 감사: BN 343개, 전체 BN 채널 49248개.


### 81% 목표: HR336 이미지별 BN 통계 보간 후보 기각

- 합성 hook 평균/분산·alpha0 exact·원본 state 복원 검사가 통과했다. 고정 alpha=.25,1 후보를 GPU3에서 순차 실행했다. 대기 상태에서 GPU 이동을 검토했으나 이미 자식 추론이 시작되어 안전 검사로 이동을 취소했고 기존 컨트롤러를 즉시 재개했다. 중복 실행/외부 작업 종료/소스 변경은 없었다. 컨트롤러2191094 및 추론2191590은 완료 후 종료했다.
- 원래 HR336 확률20장 exact와 baseline PNG/80개 IoU exact를 확인했다. 독립 PNG/GT 채점 기준80.2312280101%, alpha.25는78.3182126181%(−1.9130153920%p), alpha1은70.8233146336%(−9.4079133765%p)다. 두 후보 모두 기각하고 기존 base를 유지했다.
- alpha.25 클래스 변화 Al3Ni−0.627815,PrimarySi−1.140071,Al−0.535829,EutecticSi−5.348347%p, 최악 A71_step2_sample1−12.571585%p. alpha1은 각각−3.003128,−6.385937,−3.103603,−25.138986%p, 최악 A44_step4_sample3−22.828618%p다. 이 방식의 통계 보간은 특히 Eutectic Si에서 크게 악화됐다. 학습 분포 통계를 이미지별 통계로 바꾸는 것이 개선을 보장하지 않음을 기록한다.
- 14개 가중치1,755,428,956bytes 불변. 변경 HR336 원본 이미지 추론 및 통계 수집 전체 GPU단계596.3928초, 통계 수집 경로 합156.6172초, 후보별 predict 합71.7405/71.7561초. 다른 분기 확률 재사용으로 전체 fresh 앙상블 시간이 아니며 단회 속도 개선을 주장하지 않는다.
- archive SHA·고정 source snapshot·원본/재사용 확률 SHA 불변·독립 채점을 accuracy81_hr336_moments_20261009/comparison.json, report.md, decision.json 및 collected에 보존했다. 이미지별 통계 수집/추론에서 GT를 열지 않았고 모델 파라미터와 비BN buffer는 불변이며 이미지마다 원래 BN buffer로 복원했다. test를 사용하지 않았다.
- 사용자가 대회를 KoMaP 2026으로 확인했다. 제공 train70만 감독학습에 사용하고 valid/test 정답은 학습에 넣지 않는 범위를 유지한다. 반복 valid 개발평가의 일반화 한계/외부 데이터 및 추가 사용 규정 미확인 범위를 유지한다. 81% 목표는 미달이며 완료로 처리하지 않는다.


### 81% 목표: HR336 학습 문맥 확대 후보 시작

- 직전 후보 기각·독립 재채점·기록 완료는 progress로 분류한다. 서버 매뉴얼을 다시 읽고 GPU/실제 프로세스 및 lock을 확인했다. 원래 HR336 및 기존 control 학습 기록을 확인한 뒤 새로운 학습 문맥 가설을 고정했다.
- accuracy81_hrcontext_20261009/protocol.md: 원본 HR336 초기화, seed81336, train70만, 기존0.5CE+0.5Dice/4epochs/840samples-per-epoch/420updates/배치2×accum4/학습률/BN eval/증강을 유지한다. patch512→1024(input_scale2 유지), native crop256→512만 계획된 설정 변경이다. 학습 픽셀과 계산량은4배이며 기존control과 동시실행/픽셀별동일증강 비교는 아니다. 마지막epoch4만 고정 평가하고 valid로 epoch/LR를 선택하지 않는다.
- 합성 batch2×1024의 AMP forward/backward/finite gradient/optimizer step/BN buffer 불변/원본checkpoint SHA 불변 검사가 통과했다. peak_cuda_allocated_bytes10,424,971,264, step3.1212초. valid/test 입력을 읽지 않았다. 이 검사가 통과한 뒤 실제 학습을 시작했다.
- 서버 컨트롤러2193241, 실제 학습 자식2193324를 ps로 확인했고 GPU3에서10updates/minibatch40까지 정상 진행했다. 외부 점유가 생기면 우리 생성 process group만 종료하고 부분 결과를 보존한다. 최대3900초 제한이며 한 GPU에서 순차 실행한다. 4epoch 후 별도14가중치 bundle을 만들어 HR336만 교체하고 원본 영상에서 전체 valid20 경로를 새로 실행하도록 구현·예약했다. 아직 신규 성적은 없다.
- source lock 및 원본 frozen SHA를 확인했고 독립 PNG/GT 수집·대조 코드 collect_results.py를 준비했다. base/릴리스/원본 파일은 보존했다. 반복 valid 개발평가·독립 일반화 미확인·test mIoU 없음·81미달/goalactive를 유지한다.

학습 문맥 입력 감사 추가: train70 중17장은675×480으로 nativecrop512에서 기존방식의 높이32 reflectionpadding이 적용된다. train/valid 동일이미지SHA겹침없음, 시편ID7개(A125,A153,A181,A191,A40,A55,A7)겹침을 현재recipe/manifest로 재확인했다. context_data_audit.json 및 context_observations.md를 로컬/서버에 보존했다. 최종 collector에 시편겹침/비겹침 부분의 기술점수 비교를 추가했고, 독립 일반화 평가라고 부르지 않는다. 현재 실행설정/고정소스는 변경하지 않았다.

HRcontext epoch1 완료:105updates의실제checkpoint를CPU에서읽기전용감사했다. BN buffers1029개 및동결boundary tensors2개가원본과exact,전체state유한값,1031tensors실제갱신,원본HR336 SHA불변/후보SHA읽기전후불변을확인했다. epoch1.pt308,912,746bytes,SHA1bfa0f679c9f7bbafd1708e9ba0187baa8b9364238bf6134faab18640d4b1f10. epoch1_invariant_audit.json과학습progress스냅샷을수집했다. 중간epoch를valid로채점하지않았으며최종epoch4고정평가/81미달목표active를유지한다. 컨트롤러2193241과학습자식2193324의실제생존을재확인했다.

학습 대기 중 기존 완료Model400/401/405/406/407/408의 결과파일을읽기전용감사하고source SHA/평가범위/점수/실패gate를 completed_model_inventory.json에기록했다. Model400/401공식valid79.322725/79.418171%, Model405–408은train-groupOOF NO_GO로새81valid증거가없다. OOF와valid점수를혼합하지않고이미실패한branch실험을중복실행하지않는다. HRcontext실제프로세스2193324 및부모2193241생존과새업데이트로그를확인해기존학습을유지했다.

HRcontext verified wait:실제PID2193241/2193324를반복확인했고epoch2까지210updates완료,epoch3진입을로그와progress.json으로확인했다. epoch1/2학습손실0.1526465/0.1490083,각679.7914/684.6779초. 학습loss감소를valid개선증거로보지않으며중간valid채점/재시작/추가GPU중복작업없음. 최종epoch4만평가하도록고정한예약을유지한다.

HRcontext학습완료:4epochs/420updates/2726.7253초,peakallocated11,391,257,600bytes. finalepoch4.pt308,912,746bytes,SHA74f6843c12be6927ac5420c87a9094fd3eae6fd3b4ca8f950e591e3d1bde1c58. train70만사용/valid_read=false/test_read=false/원본checkpoint불변receipt를수집했다. 최종state도CPU읽기전용감사해BN1029buffers·동결boundary2tensors exact,전체finite,1031tensors갱신,읽기전후SHA불변확인. final_invariant_audit.json을collector필수검사에추가했다. 학습자식2193324정상종료/동일부모2193241이새추론자식2200743을시작한실제생존확인. 전체14가중치valid20새추론중이며최종성적은아직없다.


### 81% 목표: HR336 문맥 확대 실험 완료·기각

- 최종epoch4만사용한전체14가중치원본valid20재추론완료. 독립PNG/GT20장80IoU채점과실행metrics일치80.1820844324%. 원본base80.2312313198%대비−0.0491468874%p,기존동일미세조정control80.1856429471%대비−0.0035585147%p. 문맥확대의성능개선을입증하지못해기각했다.
- 원본대비Al3Ni+0.2462308,PrimarySi−0.0935429,Al+0.0680649,EutecticSi−0.4173403%p.9장개선/11장악화,최악A74_step3_sample4−0.7784475%p. 기존control대비11장개선/9장악화,최악A42_step2_sample4−0.3438368%p다. 클래스별대조군차이와전체20장이미지차이는 report.md/comparison.json에보존했다.
- 학습시편겹침valid7장 원본대비−0.0848408%p/control대비−0.0107745%p,비겹침13장각−0.0299271/+0.0003270%p. 모두같은반복valid의기술평가이고독립일반화증거아님.
- 전체freshvalid단회476.6477초,학습2726.7253초.14가중치1,755,406,329bytes이며구조/파라미터수경량화없음. 저장size의미세차이를경량화개선으로보지않는다. source/preflight/archiveSHA/원본14weights불변/최종checkpointSHA/BN1029buffer및boundary2tensor동결/전체finite검사를통과했다. 원본base/기존릴리스보존/test정답·test mIoU없음.
- 컨트롤러2193241과추론2200743정상종료확인. 수집및독립채점완료후 decision.json에기각판정을기록했다.81%목표는미달이며완료처리하지않는다.


### 81% 목표: B4 Model329 현재3fold train70 추가 학습 시작

- 직전HRcontext후보를완료·독립채점·기각한턴은progress다. 원본HR336 args/loss를다시확인해CEclassweights누락가설을지지하지않음을기록했다. Original Model329 실행스크립트는crop1536/batch2/가로4000/fullstats/zcdf3/uniform-image/CE+Dice였다. 이를근거로현재base의fold1,3,4를모두400updates씩train70로추가학습하는고정후보를정의했다. fold제거/valid별fold선택/epoch선택/외부학습자료없음. 추가학습후세모델은모두train70을보므로OOF라고부르지않는다.
- accuracy81_b4all70_20261009/protocol.md 및 source/preflightSHA를동결했다. Original augmentation과image-globalwidth4000 mean/std/CDF,1536crop/batch2,seed20261009/샘플별고정seed,CE+Diceeps1,LR1e-5/wd1e-2/cosine400,BNbuffers고정/affine학습/FP32저장으로정의했다. 같은800samples를세fold에적용하고한GPU3에서순차실행한다. 마지막400updates만전체14가중치rawvalid20으로평가한다. 비교는원본대비추가학습전체효과이며all70효과만분리한compute-matched control은없다.
- CPU전처리감사:70이미지/GTstrictdecode,전체영상stats/CDF유한값·단조CDF,같은crop seed의입력·GTexact재현,계획800samples가70장모두포함을확인했다. valid/test/GPU는사용하지않았다.
- 최초독립random3채널synthetic gate의AMPbackward에서nonfinitegradient발생,step전종료했다. 실제400update학습은시작하지않았다. 컨트롤러2204643/자식2204921terminal을확인하고source/log/preflight/receipt를 attempt1_failed_synthetic에보존했다. 같은1536/batch2/부모fold1/GradScaler1024로전처리된train고정sample0,1을별도진단해finitegradient/optimizerstep/BN불변/원본SHA불변을확인했다. maxallocated13,936,979,968bytes. 이증거로smoke입력만자연전처리traincrop으로고쳤고loss scale/finite판정/학습recipe/valid기준은유지했다. 수정이유를 smoke_input_amendment.md에보존했다. 모든향후crop의안전성을이두입력만으로보장하지않는다.
- 수정source lock을확인하고새컨트롤러2205897을시작했다. 새gate통과후실제fold1학습PID2206057을ps로확인했고step50/400까지정상진행했다. GPU3단일작업/부분상태보존/외부점유·nonfinite·OOM시우리생성자식만중지/자동재학습없음. 다른사용자GPU4/6/7작업은건드리지않았다. 새성적은아직없고base14/기존릴리스유지/81미달목표active다.

B4all70 수집·독립채점 코드 collect_results.py 준비: 완료 후 세 checkpoint의 source/shape/dtype/finite·원본BN exact·실제tensor변경·800samples와3fold 동일순서를서버CPU에서검사하고archiveSHA/안전추출/20PNG80IoU/source lock을검증한다. canonicalbase와기존동일GPU3control 둘다비교하고전체fresh시간/14weightsbytes/클래스·이미지손실을기록하도록했다. 실제컨트롤러2205897과현재학습자식의생존을다시확인했고새학습업데이트로그가증가했다. 아직신규valid성적없음/goalactive.

B4all70 fold1완료:400updates/train70모두노출/607.4832초/peakallocated14,139,793,920bytes. FP32checkpoint81,607,472bytes,SHA6bbfe0115a1926536394cff1d34f5818ae1af0c4bb194296409dc29ff0ac786f. BNbuffers불변/모든parameterfinite/원본불변/valid_read=false/test_read=false. 동일컨트롤러2205897이fold3학습자식2207876을시작했고실제ps로생존확인했다. 부분foldvalid채점없이세fold마지막400updates고정평가를유지한다.

### B4 all70 완료·독립채점·기각 (2026-10-09)

- 서버 실상태를 다시 감사했다. controller2205897 및 B4 자식은 종료, failure.json 없음, fold1/3/4 각각400updates·train70노출·동일800sample순서 완료, 전체원본영상 fresh valid20 완료. GPU3에는 다른 compute 작업이 있어 건드리지 않았다. 기존 학습이나 추론을 재실행하지 않았다.
- 수집 archive SHA256/안전추출/source lock, 서버 동결 원본 소스51개와 원본14checkpoint SHA256, 최종3checkpoint의 키·shape·dtype·finite·BN buffer exact·실제tensor변경을 검증했다. 독립PNG/GT 20장80IoU가 실행 metrics와 일치했다. 결과는 accuracy81_b4all70_20261009/comparison.json, report.md, decision.json 및 collected에 보존했다.
- 새후보80.1189329935%, canonical base80.2312313198% 대비−0.1122983263%p, 기존 동일GPU3 control80.2312280101% 대비−0.1122950166%p. 10장개선/10장악화, 최악A71_step2_sample1−2.4244267818%p, 네 클래스 평균 모두 하락. 전체freshvalid 단회499.3827초, 14가중치1,755,301,831bytes. 기각하고 Model349_exh14 base/릴리스 포인터 유지,81% 미달/목표 active. test정답·test mIoU 주장 없음.
- 팀원 원본 학습 GPU·정확한 환경은 미확인이다. 원본은 P100 추론에서 약80.2312% 재현되며, 이번 P100 추가학습은 LR·업데이트 수·데이터 노출·BN 설정도 달라 GPU 단독 효과 비교가 아니다. train70 노출 효과만 분리한 동일계산량 대조군도 없고 반복valid의 독립 일반화 증거도 없다.

### 사용자 지적: 80.2% 방법을 P100 원본 가중치에 이식하여 비교

- 사용자의 제안에 따라 릴리스 `Model349_exh14` 14개와 서버 `/workspace/komap_model1/outputs`의 대응 P100 checkpoint를 CPU에서 중첩 텐서까지 대조했다. 14개 모두 key/shape/dtype은 호환하지만 각 파일에 실제 값이 다른 텐서가 있다. 따라서 릴리스 80.2312313%를 P100 학습 출력에 같은 80.2 방법을 적용한 결과로 취급할 수 없다.
- `accuracy81_p100_exh14_transfer_20261009`에 릴리스 exh14의 추론 코드·route·fold선택·후처리를 그대로 두고 14가중치만 P100 outputs로 바꾸는 단일 후보를 고정했다. 후보 bundle의 inference SHA와 route가 릴리스와 exact, P100 14가중치 SHA/bytes를 기록했다. 원본 두 묶음은 변경하지 않았다. 과거 P100 Model349 raw valid79.9095188739%의 원본 결과 JSON도 별도로 수집했다. 이 과거 점수와 후보 점수의 차이는 route/구현 차이를 포함하므로 릴리스와 P100 가중치판의 동일방법 비교를 주 비교로 삼는다.
- GPU3에 외부 compute가 있어 실제 controller PID2263413은 lock을 보유한 채 유휴 3회 검사 대기 중이다. 마지막 직접 확인 시 외부 PID1956758, GPU3 7526MiB/100%였고 우리 추론 자식·완료·실패 파일은 없었다. 3시간 대기 상한과 외부 compute 등장 시 우리 자식만 종료하는 규칙을 적용했다. `collect_results.py`에 완료 후 원본영상 valid20 PNG/GT 독립채점·source/weight/archive 검증 및 한국어 보고를 준비했다. 아직 P100 이식 점수는 없다.
- 별도 CPU Model392 seed1 train OOF 픽셀 보정은 fold1~3의53장으로 학습하고 fold4 17장을 고정보류 평가했다. 기준78.2349438%, 최고고정혼합0.5는78.3433003%(+0.1083565%p)로 사전+0.5%p gate 미달, screen fail이다. valid/GPU 후속실행하지 않는다. 첫 실행의 JSON 직렬화 실패 로그를 보존했고 저장만 수정한 두번째 실행의 결과 JSON 17장/3비율 재평균이 정확히 일치했다. 이는 공식 valid점수가 아니다.

P100 matched route 대조군도 고정했다. 같은 P100 outputs의 B4 322/326/329 각4fold, HR336/347, 269, high5(EMA 포함) 총20가중치이며 원래 Model339/341/346/349 route 값만 사용한다. `Model349_exh14` 실행기에 필요한 원래 b4mean=0.5×(322+326) 계산을 한 줄 추가했다. GPU 사용 전 P100 저장 확률과 기존 Model183 base에서 20장 캐시 재생을 실시해 과거 P100 Model349 공식 empty0 점수79.90951887388398%와 **정확히 일치**했다. `matched_control_cached_parity.json` 보존. 대조군 controller PID2273558은 이식 후보 완료를 기다리고 있으며 두 controller 모두 실제 ps로 생존 확인했다. GPU3 외부작업 때문에 새 추론 자식은 아직 없다. 완료 후 수집·독립채점 코드는 collect_results.py/collect_matched_control.py에 준비했다.

P100 가중치에 릴리스 exh14 추론 방식을 이식한 전체 원본영상 valid20 단회 추론 완료. 실행 지표79.7191563997%, 전체 fresh879.7800초, P100 14checkpoint 1,755,417,338bytes. 수집 archive/source/route/14가중치 SHA·bytes와 20입력 SHA 일치를 확인하고 PNG/GT 20장80IoU를 독립 채점해 동일한79.7191563997%를 얻었다. 릴리스 가중치 같은 방식80.2312313198% 대비−0.5120749201%p, 과거 P100 원래방식 캐시79.9095188739% 대비−0.1903624742%p이다. 후자는 구현·route가 달라 잠정 참고값이다. 최악 이미지 A71_step2_sample1 −4.0864196%p, Eutectic Si 평균−1.3624857%p. `accuracy81_p100_exh14_transfer_20261009/comparison.json`/`report.md`/`collected`에 증거를 보존했다. P100 matched fresh control은 controller2273558이 GPU3의 타 사용자 compute PID2022984 종료를 기다리는 중이며 중복 실행하지 않았다. 최종 route효과 판정은 이 대조군의 독립 PNG 채점 후 기록한다. 원본 base·릴리스 유지, 81% 미달, 반복valid/독립 일반화 한계 유지.

P100 matched fresh control도 외부 GPU3 작업이 빠진 뒤 자동 시작해 완료됐다. 원래 route의20개 P100 checkpoint, 같은 실행기, 동일 원본valid20 입력으로1594.8792초 fresh 추론했다. controller2273558과 추론 자식2276877은 정상 종료, failure없음. archive/source/route lock·20가중치 SHA/bytes·PNG20장/GT80IoU 독립 채점·실행 metrics 일치를 확인했다. 같은 P100 가중치에서 원래 route79.8077876074%, exh14 이식79.7191563997%로 이식이−0.0886312077%p 낮다. 이식은 릴리스 가중치 exh14 80.2312313198%보다−0.5120749201%p 낮다. 원래 route 새 대조군이 과거 P100 캐시79.9095188739%보다−0.1017312665%p 달라 그 원인은 미확정이며 순수한 과거 route 효과로 단정하지 않는다. 릴리스와 P100의 차이를 GPU 단독 효과로 해석하지 않는다. `matched_comparison.json`/`matched_report.md`/`decision.json`에 기각 판정, 수집 archive와 한계를 기록했다. base/릴리스 유지, test 미사용, 81% 미달/목표 active.

추가 CPU 보완 검사: 릴리스·P100 이식·P100 원래 route의 독립 검증 PNG를 이용해 두 P100 예측이 일치하고 릴리스와 다른 픽셀만 클래스별 16개 규칙으로 교체했다. 공식 20장/80IoU 점수 그대로 채점했고 릴리스80.2312313% 재현. 전체valid 최고 subset1은80.2497489%(+0.0185176%p)에 그쳤고, 시편 leave-one-out 규칙 선택은80.2107645%(−0.0204668%p)로 악화되어 사전+0.5%p 화면 gate 실패. 이미지별 세 모델 중 최고를 GT로 골라도80.3407204%이므로 세 완성 마스크만 고르는 접근은81%에 부족하다. `accuracy81_p100_mask_consensus_20261009/screen_result.json`, `decision.json`, 재현 스크립트에 기록하고 GPU 후속실행하지 않았다. 반복valid 선택/독립 일반화 한계, 기존base·릴리스 유지,81미달/목표active.

과거 P100 캐시와 새 no-TTA 대조군의−0.1017313%p 차이를 소스 감사했다. 서버 원본 B4 322/326/329 valid 생산 스크립트는 `--no_tta` 없이 predict.py 기본 D4를, HR336/347은 evaluator 기본 flip4를 사용한다. 반면 릴리스 `infer_lite.py`는 config에 D4/flip4 이름이 있어도 실제 B4/HR 함수 호출에 모두 `tta=False`를 전달한다. 따라서 새 대조군은 과거 캐시 파이프라인의 완전 재현이 아니다. `fresh_cache_gap_audit.json`에 서버 원본 코드 SHA와 조건을 기록하고 `matched_report.md`/`decision.json`의 해석을 정정했다. 정확한 TTA 기여량 및 다른 원인 가능성은 미분리다. 동일 no-TTA 실행기의 P100 이식−0.0886312%p 관측은 유효하지만 과거 TTA route의 순수 인과 효과로 주장하지 않는다.

14대응 릴리스/P100 체크포인트의 CPU 읽기 전용 텐서 기하 감사를 완료했다. B4 7개 상대L2 0.448–0.660, HR336/347 0.775/0.816, high4 0.068–0.075, native269 0.242이고 전14개에 실제 float 텐서 변화가 있다. `audit_checkpoint_geometry.py`, `checkpoint_geometry_audit.json`, `checkpoint_geometry_decision.json`에 SHA/전체 결과를 보존했다. 큰 B4/HR 차이로 전체14 가중치의 단순 선형 soup을 GPU 후보로 올릴 근거가 부족하다. 이 수치는 기능·점수 평가가 아니므로 개선 또는 악화를 단정하지 않는다. 추가 GPU 작업·base 변경 없음,81미달/목표active.

완료된 valid20 후보의 실제 PNG/GT/입력SHA·metrics를 CPU 독립 채점하고 동일 마스크를 제거해 69개 독립 후보를 조사했다. 최고 개별 후보는 이전 결합TTA80.2548841%, 정답으로 이미지마다 최적 마스크를 고르는 비실용적 상한도80.9820608%로81에 미달. 이미지·클래스별 IoU 상한81.2306403%는 일관된 분할마스크 점수가 아님. `accuracy81_candidate_oracle_20261009/audit_result.json`에79실행/69중복제외 후보, 스킵없음,20장 원자료를 보존했다.

같은69후보×후보라벨·릴리스라벨 기준30개 우선순위 규칙의 CPU 화면검사에서 Model401 계열 matched_source/control의 Al3Ni·Al 라벨만 릴리스보다 우선하는 규칙을 골랐다. GT 없이 새 PNG20장을 만든 후 기존 독립 scorer로80IoU 채점해80.4746206171%, 릴리스 대비+0.2433892973%p를 재현했다. 시편 leave-one-out에서 후보/규칙 재선택은80.4087871404%(+0.1775558206%p), 최악A71_step2_sample1−0.2300308849%p다. 같은 고정규칙을 P100 이식/원래 fresh 기준에 적용해각+0.2393308/+0.2796967%p의 방향성을 보았다. 모두 반복valid이며69×30 탐색 선택편향과최악이미지guard 실패가 있어 새base로 채택하지 않았다. `accuracy81_candidate_oracle_20261009/report.md`/`decision.json`/`fusion_candidate_valid20`에검증·한계를보존했다. test 미사용, 기존base/릴리스유지,81미달/목표active.

추가 검사: 80.4746206% 마스크에 z-score refiner의 Al 우선 규칙을 더한 valid20 탐색 최고값80.5006891%는 두 번째 규칙 시편제외 선택에서80.3160176%로 악화해 기각했다. 독립 PNG 채점된80.4746%와 구분하며80.5%를 검증된 모델 성능으로 보고하지 않는다. probability_fusion.py는 기존 릴리스/Model401 확률로 CPU9설정 검사 중이다. 첫 Model401 parity 실패는 argmax 동률 클래스 우선순위 차이였고 원래 우선순위를 복원해 부모PNG exact/원래Model40179.4181709% gate 통과를 확인했다. 새 GPU 추론 없이 실행하며 기존base 유지.

사용자 비교 기준 지적 반영: 두 번째 규칙 조건부CV80.3160176%도 릴리스 대비+0.0847863%p다. 첫 단계보다 낮다는 이유를 릴리스 대비 악화로 설명하지 않는다. 후보PNG20장을 GT 없이 생성해 독립PNG/GT80IoU 채점80.5006890946%(릴리스+0.2694578%p)를 확인했다. 최악이미지−0.2122951%p, 반복valid 및 조건부CV의 첫단계 선택편향은 유지. 첫/둘째단계 모두 탐색후보로 보존하며 base 미변경. fusion_stage2_candidate_valid20/비교JSON 및 decision/report 갱신.

Model401확률을 릴리스 route에 class0/2로 넣는 CPU9설정은 정상종료(exit0),20장완료. 최고gamma(0,0)80.2312297%,추가결합개선없음;시편제외선택80.1324606%. 부모PNG exact/Model401원래argmax동률우선순위parity통과. 결과/선택config0 archiveSHA·안전추출 수집,독립채점진행. 마스크결합80.5006891%후보와구분.81미달/목표active.

확률결합 config0 독립PNG/GT 채점80.2312296960%완료, 부모마스크변경0pixels. 두단계 hard fusion의 GT없는 CPU재생 실행기를 작성해 원래Model401확률+train70학습 zscore잔차checkpoint SHA검증 후 실행(session42213). 원본영상 fresh부모추론은 아직아니며CPU/GPU잔차마스크차이를 먼저기록한다.

사용자 승인된 GitHub 업로드 완료: CNUKoMaP/lhs main eee87cf, Model349_fusion805_reproduction_code.zip(163199bytes, SHA256 b486b6b8df2af066886cb71503d9b239795dd3812f1c6959ca776c4466dbcde5), 한국어문서/SHA256/README추가. 동봉 portable실행기 원래부모출력+외부보정가중치로20장픽셀SHA·GT80IoU exact,80.5006891%재현. 원격다운로드ZIP SHA/무결성검증완료. 원본영상부모통합fresh검증은미완료이며 code ZIP의 가중치/데이터 제외 정책 명시. 목표81active/base유지.

81% 추가 검사: Model401의 top1-top2 확률margin으로 Al3Ni/Al 픽셀교체를 제한하는 고정25설정(각0/.025/.05/.1/.2)을 CPU실행(session25782). 부모80.5006891% 마스크exact gate, Model401 B4/HR확률SHA확인, 첫둘째단계고정조건부시편제외선택을기록한다. 기존 작업프로세스가없고완료파일존재를확인후새접근만시작했다. 원본모델/base/GitHubZIP불변.

확신도25설정 CPU완료(exit0): thresholdAl3Ni0/Al0.1 최고80.5123346016%,805대비+0.0116455%p/릴리스+0.2811033%p. archive SHA·안전추출/PNG20장GT80IoU독립채점일치. 앞단계고정조건부시편제외선택80.4760169%,독립일반화아님. confidence_screen/candidate/comparison에보존,기존GitHub805ZIP/base유지.81목표active.

검토피드백반영: 대표후보를F없는1단계80.4746%로정정,2단계F는조건부CV추가손해−0.1586%p로미채택.69실행×30규칙을시편제외안에서모두재선택한CV80.4088%를별도재계산해20선택내역exact확인. A71−0.2300%p/quality_pass false명시. GitHub d4ef4b0문서/ZIP수정및원격SHA검증. 전용stage1fresh controller2290150/자식2290202 GPU3 release_valid원본추론실행중,유휴3회/전체siblinglocks/부모가중치SHA검사통과. 이후Evalid·Rtest·Etest·1단계마스크/독립채점자동순차예정,중복실행없음.

1단계fresh live확인: controller2290150/자식2290202 실제ps생존,release_valid의B4329분기2/20처리,실패없음. 완료후collector준비: terminal완료gate/sourceLock/부모SHA/archive검증/안전추출/PNG20GT80IoU독립채점,canonical·freshR·cached1단계비교 및test10두R대비일치율/클래스면적/한국어보고. Collector는완료전실행하지않음. 기존canonicaltest10입력manifest확인.

1단계fresh 대기 중 추가 감사: 기존canonicaltest10 입력SHA가 현재서버원본10장과전부일치. 기존exh14 예측Al3Ni pooled면적 valid20.8146%/test28.9387%(GT비율아님),test변화량검사필요성을확인하고 test_reference_audit.json에저장. 실제controller2290150 및release_valid자식진행/실패없음,서버전체source_lock이로컬과동일함을재확인. 완료전collector미실행/중복GPU작업없음.

1단계fresh의R만완료:20장1169.363초/exit0. 전체collector는미실행하고완료된R별도archive SHA/안전추출/PNG20GT80IoU독립채점80.2309863330%확인. canonical80.2312313198%와−0.0002449868%p/17pixels차이(입력SHA일치). 미세차이원인은단정하지않으며E결합을freshR와canonical모두에비교한다. Evalid실제자식2292157/컨트롤러2290150생존,GPU3사용11054MiB/73%,실패없음.

E 첫원본영상 A117_step4_sample2 완료: 원래저장Model401확률과max/meanabsdiff0,마스크변경0/1,076,250pixels. 정답없는읽기전용비교이며전체E/1단계점수검증은미완료. 실제자식2292157과컨트롤러2290150생존/GPU3 70%확인,1/20출력후다음영상계산중. expert_progress_audit.json 보존.

새train70-only E 클래스bias CPU검사완료: 27설정, 각groupedOOFfold를제외하고나머지3fold에서bias선택. 기준E79.9858620%parity통과, nested79.9824486%(−0.0034134%p)로사전gate실패. 저장70장IoU재계산일치,원자료SHA불변/valid·test·GPU미사용. accuracy81_E_oof_bias_20261009/result/decision/report에기각,valid후속실행없음. 기존R+E1단계recipe불변,원본Evalid3/20실행중.81목표active.

## 팀 환경 Model349 원본 100개 시드 평균 (2026-10-09)

사용자 보고: 경량화 전 Model349 원본을 팀 GPU 환경에서 재현한 **100개 시드 평균 성능은 80.255658%**다. 단일 재현 기록 80.0683465% 및 exh14 재학습 80.052074%와 구분한다. 이는 시드별 성능의 평균이며, 100개 모델의 예측 앙상블 점수로 해석하지 않는다. 시드별 점수·표준편차·원본 결과 파일·평가 대상의 상세 정보는 아직 전달받지 않아 독립 검증하지 않았다. 구조화 기록: `model349_compression/team_model349_original_seed100_20261009.json`.

1단계 전체 valid 원본 재추론 완료 및 로컬 PNG/GT 독립 채점: fresh R 80.2309863330%, fresh R+E 80.4744236018% (대조 R 대비 +0.2434372688%p). 저장 1단계와 7픽셀 차이, 점수 -0.0001970153%p. completed_valid_comparison.json 및 completed_valid_summary.md 보존. 고정 가중치 추론 검증이며 재학습 재현성 아님. 시험10은 기존 controller에서 진행 중, 전체 완료 미선언. 사용자 재학습 재현성 요구에 따라 같은 10개 시드 paired 비교 계획을 별도 protocol에 기록; 새 학습 미실행.

재학습 실행기 감사: Model349_train_route_code.zip의 run_model349_pipeline.sh는 구성별 --seed 42/123/456/181/2028/239/242/257/269/0/347를 고정하며 전역 seed override가 없다. 마지막 valid route 재탐색도 포함한다. 새 폴더 10회 실행만으로 10개 시드 검증으로 간주하지 않는다. 모든 확률적 구성의 시드 매핑을 기록하고 frozen recipe 비교와 route 재선택을 포함한 절차 재현을 별도 endpoint로 보고하도록 seed10 protocol에 명시. 현재 새 학습 미실행; test E 기존 job 유지.

1단계 whole fresh valid20/test10 완료. root completion의finished/wholefresh/testGTfalse 확인 후 collector 실행. 동결 source/부모 weights SHA불변, 아카이브 SHA/안전추출/RGB팔레트·native크기/PNG20 GT80IoU 독립 채점 통과. valid80.4744236018%, freshR 대비+0.2434372688%p, A71−0.2300308849%p/qualityfalse. test10 freshR 대비161731/9074964픽셀 변경(1.78216685%), canonicalR 일치98.21780009%. testGT없어 성능주장없음. 고정가중치 추론재현이며 재학습 재현성 미확인. report/comparison/collection_receipt 보존, 후보decision의freshpending 상태 갱신,81미달/base불변.

새 E 국소 확률 평활화 CPU 실험 완료·기각: sigma1/2×mix0.1/0.25/0.5+identity 7설정, grouped OOF fold제외 선택. 기준/nested 모두79.9858620081%, delta0, 전체70장도identity선택. 독립 저장IoU 재평균 일치/부모SHA불변/valid·test·GPU미사용/exit0. accuracy81_E_oof_smoothing_20261009 결과·판정 보존,valid후속없음/base불변/81미달.

새 E generalized mean CPU 실험 완료·기각: 산술/기하/root/quadratic/harmonic 5설정, 구성·weights고정. 기준 79.9858620081%, nested 79.9838121297%, delta -0.0020498783%p. 전체train 선택p2이나nested손해,valid후속없음. 저장IoU독립재계산일치/parentSHA불변/valid·test·GPU미사용/exit0. accuracy81_E_oof_powerpool_20261009 보고서 보존.

E train70 OOF 오류진단70장 완료: Eutectic 오류 중23.49%는개별family맞음,70.94%는전family동일오답. 단순평균변경 한계 근거; next학습후보는공통Eutectic오류개선 필요. 진단은GT사후분석/제출model아님/valid·test·GPU미사용. 저장집계독립재합산일치,accuracy81_E_error_diagnostic_20261009/report 보존.

새 Eutectic 31×31 context scalar residual 7713params metaOOF 학습 완료·기각. 4fold×400updates,train70only,lastfixed. 기준 79.9858620081%, 후보 79.9637103823%, delta -0.0221516258%p. 독립 저장IoU재합산일치/featureSHA불변/validtest미사용. 초기CE deterministic 오류→같은수식manualCE, NVML/container PID차이 감시수정, 완료후npboolJSON실패→weights그대로eval-only복구(4SHA불변). 모든attempt보존. accuracy81_E_context_eut_20261009/report/decision,valid후속없음/base불변/81미달.

새 Al/Eut pair softIoU context 실험 완료·기각. same4fold×400updates, rareclassmask불변70장검사. 기준 79.9858620081%, 후보 79.9372465929%, delta -0.0486154152%p. 클래스delta[0.0, 0.0, -0.028632355777738722, -0.1658293048352078]. 저장IoU독립재합산일치,validtest미사용/featureSHA불변/strictnested아님. accuracy81_E_pair_iou_20261009 보고서보존,valid후속없음/base불변/81미달.
