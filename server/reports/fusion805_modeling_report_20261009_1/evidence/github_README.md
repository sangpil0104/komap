# KoMaP 4상 미세조직 분할 실험 기록

KoMaP 미세조직 4상 분할 대회의 실험 코드, 선택된 결과와 재현 문서를 보관하는 비공개 연구 저장소입니다. 대회 데이터 ZIP·이미지·마스크, 학습 체크포인트, 예측 확률맵/마스크, 인증 정보는 저장소에 포함하지 않습니다. 재현 코드를 묶은 소스 ZIP은 예외적으로 배포합니다.

## 공식 평가와 최신 기록 (2026-10-04)

공식 지표는 **이미지별 4-class mIoU**입니다. 각 이미지에서 네 상의 IoU를 평균한 뒤 이미지 수로 평균하며, 정답과 예측 모두 없는 클래스도 IoU `0`으로 계산합니다. NaN 제외 방식과 pooled/global mIoU는 참고 지표입니다.

- raw validation 최고 관측값은 **Model364: 79.9429727%**입니다. 저장된 per-image 점수의 산술평균을 재계산해 일치했고, Model360과 동일한 valid 20장 및 empty-union=0 지표를 사용합니다. 80%까지 `0.0570273%p` 남았습니다. 이 점수는 같은 valid 20장에서 여러 route/scale 설정을 반복 비교한 개발 결과이며, Model360 대비 group-held-out 변화 `−0.004387%p`로 일반화 개선은 확인되지 않았습니다.
- Model392의 사전등록 2-seed train-OOF gate는 실패했고, Model397 고정 B4/HRNet blend의 train-only grouped-OOF는 **79.9859%**였습니다. 이는 validation/test 점수가 아니며 Model397의 공식 valid 평가는 승인되지 않았습니다. 최신 상태와 과거 기록 간 차이는 [한국어 최신 현황](docs/current_status_ko.md) 및 [전체 모델 결과](docs/model_results.md)를 참조하십시오.
- 점수는 `valid.zip` 20장에 대한 것입니다. `test.zip`에는 정답 mask가 없어 test mIoU는 계산하지 않았습니다.

valid/test 원본 영상 감사에서는 exact-image 중복은 없었지만 A63 specimen group이 양쪽에 있었고, 해상도 분포가 달랐습니다. A63을 제외한 specimen-group 분석은 뚜렷한 전반적 입력 feature shift를 찾지 못했으나, 표본이 작으므로 valid/test 동등성을 입증한 것은 아닙니다. 자세한 수치와 OOF gate 상태는 [한국어 최신 현황](docs/current_status_ko.md)에 있습니다.

## 문서와 코드 찾아보기

- [한국어 최신 현황](docs/current_status_ko.md): 공식 최고·보수 기준, valid/test 유사도, 진행 중 작업 및 모델 번호 주의사항.
- [전체 모델 결과 기록](docs/model_results.md): 모델·진단 실험의 방법, 상태, 결과와 한계.
- [최고 모델 및 mIoU 기준별 요약](docs/best_models_by_miou.md): 공식 지표와 참고 지표별 순위.
- [Model349 재현 안내](docs/model349_reproduction.md): 외부 데이터에서 학습·추론·route 구성을 재현하는 절차.
- [Model349 15가중치 경량 추론 코드 ZIP](Model349_lite_reproduction_code.zip): 추론 코드·설정·체크포인트 경로와 검증 기록. 가중치와 대회 데이터는 제외합니다. ZIP 안의 한국어 README를 먼저 확인하세요.
- [Model349 14가중치 탐색판 코드 ZIP](Model349_lite14_exploratory_code.zip): 고해상도 EMA 경로를 대표 모델로 대체한 추가 축소 실험. valid에서 후보를 골랐으므로 기본 배포판이 아닌 탐색판입니다. 가중치와 데이터는 포함하지 않습니다.
- [Model349 13가중치 탐색판 코드 ZIP](Model349_lite13_exploratory_code.zip) 및 [15·14·13·11개 비교](docs/model349_lite_compression_ko.md): EMA·seed456 경로 추가 제거와 영상별 손실을 포함한 상세 검증.
- [Model397 재현 안내](docs/model397_reproduction.md) 및 [학습·OOF 재현 코드 ZIP](Model397_train_route_code.zip): 외부 `train.zip`만으로 B4/HRNet 4-fold OOF 분기를 학습하고 고정 blend를 재현합니다. Model392 gate 실패로 공식 valid/test 평가는 포함하지 않습니다.
- [mIoU 결측 클래스 정책 감사](docs/miou_policy_audit.md): 과거 결과 재산출과 empty-union=0 기준.
- [실험 코드 색인](experiments/SOURCE_INDEX.md) 및 [전체 Python/shell 목록](experiments/SOURCE_MANIFEST.md).
- `experiments/valid_test_group_similarity_audit.py` 및 `results/outputs/valid_test_group_similarity_audit/results.json`: 원본 영상만 이용한 valid/test 입력 유사도 분석과 결과.

저장소에는 Python/shell source 326개(275 Python, 51 shell)가 추적되어 있습니다. 모델 번호는 독립적인 신경망 개수가 아니라 실험 결과와 route 변형의 식별자이며, 여러 번호가 같은 trainer/evaluator를 공유할 수 있습니다.

## 재현 범위와 데이터 취급

2026-10-08에 검증한 Model349 경량 추론판은 **같은 P100 실행 환경**의 원형 23가중치 `80.0683465%` 대비 15가중치 `80.0643596%`(valid 20장, empty-union=0, 차이 `−0.0039868%p`)입니다. 코드 ZIP만으로 즉시 추론할 수는 없으며, ZIP의 manifest와 일치하는 학습된 가중치를 외부에서 준비해야 기록 점수를 재현할 수 있습니다. test 마스크는 기존 D3와 9,074,964픽셀 중 14픽셀만 달랐으나 test 정답 점수는 알 수 없습니다. 이 실행 환경의 점수는 위의 다른 실험 기록과 절대값만으로 직접 순위를 매기지 않습니다.

추가로 고해상도 EMA 체크포인트 하나를 줄인 **14가중치 탐색판**은 같은 valid20에서 `80.0650399%`였고, 15개판 test 마스크와 9,074,964픽셀 중 11픽셀이 달랐습니다. 그러나 test에는 EMA가 선택되는 밝기 구간 영상이 없어, 이 결과는 신규 자료의 성능 보증이 아닙니다. 11가중치 일괄 축소는 valid `80.0188978%`로 손실이 커 기각했습니다.

EMA·seed456을 함께 줄인 13가중치 탐색판은 valid `80.0572455%`였으나 A74 영상에서 `−0.1639561%p` 악화했습니다. 이 역시 test의 해당 밝기 구간 부재 때문에 기본판으로 권장하지 않습니다. 세부 수치는 [경량화 검증 문서](docs/model349_lite_compression_ko.md)를 참고하십시오.

Model397 재현 ZIP은 외부 `train.zip`만 사용하며 `valid.zip`과 `test.zip`을 읽지 않습니다. 기존 Model349 및 validation route 재현에는 설명된 평가 단계에 따라 `valid.zip`이 별도로 필요할 수 있습니다. 일부 학습은 공개 ImageNet 사전학습 초기 가중치를 라이브러리를 통해 내려받습니다. 저장소에는 저희가 학습한 가중치나 예측 산출물이 없으므로 코드만으로 재학습할 수는 있어도 기존 checkpoint의 bit-identical 복원은 보장하지 않습니다. 실험별 경로·의존성·명령을 실행 전에 확인하십시오.

대회 test 정답은 제공되지 않으므로 test mIoU를 보고하지 않습니다. test 영상 유사도 분석은 입력 영상만 대상으로 했으며, 라벨 유출이나 test 점수 추정을 위한 분석이 아닙니다. HRNet 재현의 출처는 [upstream 참고](docs/upstream_references.md)에 기록되어 있습니다.

Model374의 사전등록 세 문서는 실험 전에 고정한 조건의 원문을 보존하기 위해 영어로 유지했습니다. 그 결과와 현재 판단은 위의 한국어 최신 현황 및 모델 결과 기록을 기준으로 확인하십시오.

## Model349 12개 FP16 base (2026-10-08)

사용자 지정으로 후속 경량화 실험의 base를 **12개 FP16 저장판**으로 고정했습니다. 326 fold 1·3, 329 전체 4-fold, HRNet336, native269, 고해상도 후보 4개를 사용하며 TENT5·OpenCV2를 유지합니다. 기존 릴리스와 가중치는 교체하지 않습니다.

- [12개판 버전 설명서](Model349_lite12_base/README.md)
- [환경·가중치 검증·valid/test 실행 레시피](Model349_lite12_base/RECIPE.md)
- [일반화 검증의 한계와 결과](Model349_lite12_base/docs/generalization_report.md), [다른 AI 검토 프롬프트](Model349_lite12_base/docs/review_prompt.md)
- [고정 추론 코드·설정·가중치 해시·결과](Model349_lite12_base/)

같은 FP16/TENT5/OpenCV2 조건에서 valid20 image mIoU는 **80.1255610%**, 체크포인트는 **683,320,900바이트**입니다. 14 FP16 대비 평균 +0.0599245%p, 최악 이미지 −0.0922867%p입니다. valid 시간 364.28초·test 시간 429.04초는 단회이며 속도 개선을 확정하지 않습니다. test10은 정답이 없고 기존 lite15와 마스크 일치율 99.8863577%만 확인했습니다. fold 선택은 기존 valid에 노출됐으므로 독립 일반화 개선은 미확인입니다. 가중치·데이터·예측 이미지 없이 재현 코드와 결과 기록만 포함합니다.

## Model349 exh14: 14가중치 전수 조합 탐색판 (2026-10-08)

Model349 23개 체크포인트의 부분집합 8,093개를 같은 융합·후처리로 채점해 valid20 최고 조합을 고정했습니다. B4 322 fold 3·4, 326 fold 1·2, 329 fold 1·3·4, HRNet336·347, native269, 고해상도 후보 4개(restart51·model109·seed456·pairwise, TENT5)를 사용하며, 가중치는 Model349 릴리스 FP32 원본(SHA256 일치 확인)입니다.

- [재현 코드 ZIP](Model349_exh14_code.zip): 추론 코드·설정·가중치 해시·조합 탐색 스크립트·결과. 가중치와 데이터는 제외합니다. ZIP 안 README를 먼저 확인하세요.
- P100 원본 재추론 valid20 image mIoU **80.2312313%** (원형 80.0683465%, lite12 base 80.1255610%). test10은 정답이 없으며 원형과 마스크 일치율 99.408%입니다.
- valid20으로 고른 조합입니다. 시편 단위 nested 선택 검증에서 lite14 대비 +0.09%p [−0.12, +0.39]로 손해 근거는 없지만 개선이 확정되지 않았고, valid 이득의 상당 부분이 Primary Si가 없는 A71 한 장에서 나왔습니다.

## Model349 R + E 1단계 대표 후보 (2026-10-09 검토 반영)

- 대표 후보는 **F 없는 1단계 R + E**, valid20 **80.4746% (+0.2434%p)**입니다.
- 시편 제외 때마다 **69개 실행 × 30개 규칙 전체**를 다시 선택한 값은 **80.4088% (+0.1776%p)**입니다. 별도 재계산과 선택 내역이 일치합니다. 과거 후보 개발의 valid 선택 편향까지 제거하지는 못합니다.
- 2단계 F는 valid 80.5007%지만 조건부 시편 제외 선택에서 **1단계 대비 −0.1586%p**여서 **미채택**입니다.
- A71은 −0.2300%p로 기존 최악 영상 손실 기준 미통과(`quality_pass: false`). 고정 가중치 원본 영상 valid20·test10 재추론 완료: valid **80.474424%**, 시험 R 대비 **1.7822% 픽셀 변경**. 재학습 성능 재현성과 시험 정확도는 미확인입니다. 기존 base는 유지합니다.
- [코드 ZIP](Model349_fusion805_reproduction_code.zip), [한국어 실행 안내](docs/model349_fusion805_reproduction.md), [SHA256](Model349_fusion805_reproduction_code.zip.sha256). ZIP의 805 이름은 2단계 탐색 보존본이며 대표 후보 점수가 아닙니다. 1단계 실행기도 포함하고 F 없이 사용할 수 있습니다. 가중치·데이터는 제외합니다.

## exh14·fusion805 재학습 재현 설계 추가 (2026-10-09)

두 ZIP에 fold × 시드 OOF 재학습 코드를 추가했습니다. 공개 가중치로 추론하는 기존 경로는 그대로입니다.

- [exh14 ZIP](Model349_exh14_code.zip)의 `Model349_exh14/oof_design/`
  - 모든 구성 요소를 시편 단위 4-fold × 시드로 학습합니다. 결정적 시드와 cuDNN 결정 모드를 씁니다.
  - OOF 이미지의 정답은 학습, 체크포인트 선택, 사전확률 계산에 쓰지 않습니다.
  - 구성·밝기 배정·결합·후처리는 OOF에서 고정 규칙으로 고르고, 시험에서는 모든 fold × 시드를 평균합니다.
- [fusion805 ZIP](Model349_fusion805_reproduction_code.zip)의 `oof_expert/`: E 앙상블을 같은 설계로 학습하고, 1단계 덮어쓰기 규칙을 OOF로 고릅니다.
  - 교차검증 결과에 따라 **1단계만 권장**합니다. 원본부터 재추론한 valid 점수는 80.4744%이고, 2단계는 1단계보다 시편 제외 선택에서 −0.159%p였습니다.
  - 기존 추론기에 `--stage1-only` 옵션을 추가했습니다.
- 기본은 **A 방식**입니다. train 70장으로 학습·선택하고, valid 20장은 마지막에 한 번만 채점합니다. 대회 규칙이 허용하면 B 방식(train + valid 90장)도 옵션으로 쓸 수 있습니다.
- 보고하는 점수는 세 가지입니다. OOF 개발 점수, 바깥 fold 추정치(선택 과정 포함), 시드별 점수의 평균·표준편차.
- **현재는 학습량을 최소로 줄인 실행(smoke)으로 코드 경로를 검증하는 중**입니다. 진행 상황은 ZIP 안 `oof_design/SMOKE_RESULT.md`에 있고, 전체 규모의 학습 결과는 아직 없습니다.
