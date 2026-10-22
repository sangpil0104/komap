# AlPhase B4 개선 프로젝트 — Codex 구현 명세서

> **대상:** 서버의 기존 EfficientNet-B4 + U-Net 학습·추론 코드를 수정할 Codex 또는 개발자.
> **목표:** 작은 미세조직, 복잡한 혼합조직, 특히 Al₃Ni ↔ Eutectic Si 혼동을 줄여 **원본 이미지별 평균 validation mIoU**를 높인다.
> **버전:** 1.0 / 2026-09-27.
> **문서 성격:** 구현 요구사항, 실험 프로토콜, 테스트 기준이다. 새 모델의 성능이 이미 검증되었다는 뜻이 아니다.

이 문서는 이전의 범용 DINOv3 설계서가 아니라 **현재 운용 중인 B4 U-Net을 개선하는 별도의 명세서**다. DINOv3로 전체를 교체하거나 여러 보조 모듈을 동시에 켜는 작업으로 시작하지 않는다. 먼저 기존 모델을 재현하고, 기존 100개 seed 모델의 확률 앙상블을 기준선으로 확보한 뒤 변경사항을 분리해 검증한다.

문서의 `필수`는 구현 계약, 수치가 붙은 `시작값`은 실험 후보다. 원본 코드와 체크포인트가 이 대화에 없으므로 라이브러리, 디코더 채널, 보간 방식 등은 **서버 저장소를 검사해서 확정**해야 한다. 모르는 값을 그럴듯하게 채워 넣지 않는다.

---

## 0. Codex가 먼저 따라야 할 실행 원칙

1. 기존 저장소의 적용 가능한 `AGENTS.md`, README, 실행 스크립트와 현재 git 상태를 먼저 읽는다. 기존 지침을 이 파일로 덮어쓰지 않는다. Codex의 프로젝트 지침 탐색은 공식 안내 [S1]을 참고한다.
2. 기존 코드·데이터·100개 체크포인트·완료된 결과는 보존한다. 신규 실험은 별도 디렉터리에 기록한다. 다른 사용자의 GPU 작업을 종료하거나 시스템 패키지를 일괄 업그레이드하지 않는다.
3. **코드 구현 → 단위 테스트 → 실제 1개 체크포인트 회귀 검사 → 짧은 학습 → 전체 검증** 순으로 진행한다. 함수 이름만 만들거나 `pass`/가짜 출력으로 기능을 완료 처리하지 않는다.
4. 긴 학습과 100모델 전체 추론은 먼저 실행 계획·메모리/저장공간 요구를 출력한다. 기본은 dry-run이며, 명시적인 실행 옵션으로 시작하도록 만든다. 사용자가 실행을 요청하면 승인된 서버 자원 범위에서 실행한다.
5. 서버에서 확인할 수 있는 경로·버전·모델 정의는 먼저 탐색한다. 탐색해도 확인할 수 없는 필수 항목만 질문한다. 데이터/가중치가 없으면 해당 통합 테스트를 `BLOCKED`로 남기고 나머지 독립 모듈 구현은 계속한다.
6. **기존 모델 완전 재현 모드와 신규 개선 모드를 구분**한다. 기존 코드의 보간·정규화·학습 손실을 묵시적으로 바꾸지 않는다.
7. 모든 변경은 설정으로 독립적으로 켜고 끌 수 있어야 한다. 꺼진 경우 legacy 실행 경로를 그대로 사용한다.
8. 결과에는 실제 실행한 명령, seed, split, checkpoint SHA-256, 설정, 추론 방식과 테스트 결과를 기록한다. 계획한 실험과 실제 완료한 실험을 구분한다.

### 첫 작업에서 만들 파일

```text
reports/repo_audit.md              # 코드/환경/가중치/미확인 항목
reports/implementation_plan.md     # 실제 기존 파일에 연결한 수정 계획
reports/progress.md                # 단계별 상태, 실행 명령, 차단 사유
configs/resolved_legacy.yaml       # 코드에서 확인한 기존 설정
manifests/dataset.jsonl            # 이미지와 정답의 정확한 대응
manifests/checkpoints.jsonl        # seed 0–99 체크포인트 목록과 메타데이터
```

---

## 1. 기존 모델 계약: 먼저 보존하고 재현할 것

아래는 사용자 제공 설정 [U1]이다. 서버 코드와 다르면 차이를 보고하고, 실제 재현 모드에서는 **해당 체크포인트가 생성된 설정**을 사용한다.

| 항목 | 현재 설명 |
|---|---|
| 구조 | ImageNet 사전학습 EfficientNet-B4 인코더 + U-Net 디코더 |
| 입력 | 흑백을 동일한 3채널로 복제하고 정규화 |
| 확대 | 원본 종횡비를 유지하며 가로폭 4,000px |
| 학습 패치 | 확대 영상의 1,536 × 1,536 랜덤 패치 |
| 데이터 | train 70장 / valid 20장 / test 10장 |
| 독립 실험 | 같은 설정에서 seed 0–99 |
| 업데이트 | seed당 16,000 optimizer update |
| 배치 크기 | 2 |
| 손실 | Cross-Entropy + Dice; 세부 구현/가중치는 원본 코드 확인 |
| optimizer | AdamW, 초기 학습률 0.0003 |
| scheduler | 300 update warmup 후 cosine 감소 |
| 증강 | 스케일, 회전·반전, 밝기·대비·감마, 블러·노이즈; 범위는 코드 확인 |
| 기존 체크포인트 선택 | `last.pt` |
| 추론 | 폭 4,000px 전체 영상 + D4 8가지 변환, 확률 평균, 원본 크기로 복원 후 argmax |
| 주지표 | 이미지마다 4클래스 IoU 평균 → valid 20장 평균; union=0 클래스 IoU=1 |
| 보고된 최고값 | seed 32: 80.2265%; 사용자 보고값이며 서버에서 재현 확인할 것 |

### 확인 없이 가정하면 안 되는 항목

- EfficientNet의 구현체 및 정확한 인코더 이름. `efficientnet-b4`, `timm-...` 등은 서로 대체 가능한 문자열이 아니다.
- SMP 사용 여부, SMP 버전, decoder channel/depth, BN/attention 사용 여부.
- 픽셀 범위, RGB 평균·표준편차, dtype, grayscale 디코딩 방식.
- 확대 이미지 보간, 정답 마스크 보간, 확률 축소 보간, `align_corners`, antialias, 높이 반올림 규칙.
- D4의 적용 시점. **4000px 확대 전에 90도 회전한 뒤 새 가로폭으로 재정규화하면 기존 추론과 달라진다.**
- AMP, weight decay, Dice의 batch/class reduction, absent class 처리, gradient clipping.
- checkpoint dict의 key, optimizer/EMA 포함 여부, 학습 이미지 ID 기록 여부.
- 실제 추론이 full-frame인지 sliding-window인지. 사용자 설명은 full-frame이며, 코드를 확인한다.

---

## 2. 개선의 근거와 해석 범위

기존 진단 기록 [D1]에 따르면 `A71_step2_sample1`은 원본 675 × 480이고, 첨부 비교 그림에서 복원한 seed 32 예측의 혼동 행렬은 다음과 같다. 행은 GT, 열은 예측이다.

```text
class order: Al, Al3Ni, PrimarySi, EutecticSi

[[230681, 11160,     0,  2907],
 [  5437, 29653,     0, 19939],
 [     0,     0,     0,     0],
 [   912,   587,     0, 22724]]
```

| 진단값 | 값 |
|---|---:|
| image mIoU, union=0 → 1 | 71.13850263% |
| 픽셀 오류율 | 12.63641975% |
| Al₃Ni IoU | 44.4067% |
| Eutectic Si IoU | 48.2781% |
| Eutectic Si precision | 49.8661% |
| Eutectic Si recall | 93.8117% |
| 전체 오류 중 Al₃Ni → Eutectic Si 비율 | 48.7006% |

A71의 Al₃Ni, Eutectic Si **클래스별 평균 원본 밝기**는 각각 약 81.87, 30.56이고, 학습 70장의 이미지별 클래스 평균 최솟값은 각각 102.53, 58.67이다 [D1]. 이는 증강 전 원본 통계이며, 밝기가 유일한 오류 원인이라는 뜻이 아니다.

이 사례의 가설은 **공정 Si를 더 많이 검출해야 한다**가 아니라, **어두운 Al₃Ni와 공정 Si를 형태·문맥·상대 밝기로 더 정확히 구분해야 한다**이다. 이를 valid 20장 전체에서 검증한다. A71 하나에 맞춘 임계값/클래스 비율/분기문은 구현하지 않는다.

학습 목표의 우선순위:

```text
기존 100개 모델을 확률로 결합하고 공통 오류를 확인
→ 원본 밝기 + 상대 밝기 특징
→ 확대된 세부 영역 + 넓은 조직 문맥
→ Al3Ni–EutecticSi 혼동 패치 샘플링
→ 원본 좌표계 손실
→ OOF 기반 4클래스 보정기
```

---

## 3. 단계와 완료 조건

| 단계 | 구현물 | 다음 단계로 넘어가는 기준 |
|---|---|---|
| P0 | 코드 감사, 데이터/평가 계약, 기존 재현 adapter | 팔레트·지표 단위 테스트 통과, 실제 기존 추론 회귀 확인 |
| P1 | 확률 cache, 100-seed 스트리밍 앙상블, 오류 진단 | K=1 원본 결과 일치, D4/채널/좌표 복원 검증 |
| P2 | native ROI 기하, geometry-only 대조군, 상대 밝기 경로 | 좌표 테스트와 gate=0 회귀 통과, 짧은 실제 학습 |
| P3 | 넓은 context 경로와 좌표 정렬 fusion | 문맥 정렬·gradient·padding 테스트, 단독 ablation |
| P4 | 혼합조직/오류 샘플러, native loss | train-only 접근 보장, 원본 크기 loss gradient 검증 |
| P5 | best checkpoint, EMA, 짧은 미세조정 및 scratch 레시피 | resume와 평가 프로토콜 테스트, 여러 seed 비교 |
| P6 | train70 OOF 생성, residual refiner | OOF lineage 검사와 valid20 독립 평가 |
| P7 | 효과 있는 설정 조합, 최종 평가·제출 파이프라인 | 재현 가능한 report와 원본 크기 PNG 10장 |

P5의 체크포인트 저장/모니터링 인프라는 P2 학습 전부터 구현해도 된다. **학습 방식 자체의 변경**은 별도 실험으로 기록한다. 여러 기능을 동시에 켠 결과만으로 각 기능의 효과를 주장하지 않는다.

---

## 4. 데이터·클래스·지표 계약

### 4.1 클래스 순서는 이 프로젝트에서 고정한다

| ID | name | RGB |
|---:|---|---|
| 0 | `Al` | `[153, 127, 76]` |
| 1 | `Al3Ni` | `[25, 76, 153]` |
| 2 | `PrimarySi` | `[76, 178, 76]` |
| 3 | `EutecticSi` | `[204, 204, 204]` |

`IGNORE_INDEX=255`. 입력 tensor는 float, 정답은 `torch.long [B,H,W]`, 출력 logits는 `[B,4,H,W]`이다. 모델 클래스 ID=0이 배경/무시 클래스라는 가정을 넣지 않는다.

이전 자료에 나온 `PrimarySi, Al3Ni, EutecticSi, Al` 순서를 현재 모델 출력 순서로 사용하지 않는다. 외부 cache/모델의 순서가 다르면 이름에 따른 **명시적 permutation** 후 사용하고 기록한다. metadata가 없어서 알 수 없으면 결합을 중단한다.

### 4.2 데이터 검사

원본 ZIP의 구조는 다음과 같다 [D1, D2]. 서버 경로는 다를 수 있으므로 manifest에서 절대/상대 경로를 해결한다.

```text
DATA_ROOT/
  train/images/*_image.png
  train/masks/*_mask.png
  valid/images/*_image.png
  valid/masks/*_mask.png
  test/images/*_image.png
```

검사 항목:

- 접미사 기반 동일 stem으로 image–mask pair를 만들고 누락·중복·다중매칭은 오류 처리.
- 이전 분석의 70/20/10 수량, 크기 일치, RGBA의 R=G=B와 alpha=255를 실제 파일에서 다시 확인.
- unknown mask color는 nearest palette로 임의 보정하지 말고 파일명/색/픽셀 수를 출력하고 중단.
- 변환 전후의 label set은 `{0,1,2,3,255}` 안에 있어야 한다.
- image hash, mask hash, 원래 split, H/W, A-prefix를 기록. A-prefix가 같다고 동일 시편으로 확정하지 않는다.
- 파일별 mask 클래스 비율은 진단/훈련 샘플링에는 사용 가능하지만 추론 feature 또는 router 입력으로 사용 금지.
- `DATA_ROOT`가 없거나 파일 구조가 다르면 조회 결과를 남긴다. 비교 그림 PNG를 원본 학습 이미지로 오인하지 않는다.

### 4.3 주지표: 반드시 Image mIoU

```text
CM_i[a,b] = image i에서 GT=a, prediction=b인 픽셀 수
TP_i,c = CM_i[c,c]
U_i,c  = row_sum_i,c + column_sum_i,c - TP_i,c
IoU_i,c = TP_i,c/U_i,c                  if U_i,c > 0
IoU_i,c = 1                            if U_i,c == 0
image_miou_i = mean_c(IoU_i,c)          # 항상 네 클래스
val_image_miou = mean_i(image_miou_i)   # 이미지마다 동일 비중
```

필수 구분:

- `GT에 없다`만으로 IoU=1을 주지 않는다. GT에는 없지만 한 픽셀이라도 해당 클래스로 예측하면 union>0, IoU=0이다.
- class presence를 정답에서 알아낸 뒤 예측에서 그 클래스를 지우면 안 된다.
- `val_global_miou`는 모든 confusion matrix를 합쳐 계산하는 **보조 지표**다.
- 모든 점수는 내부/JSON에서 0–1, 표시는 ×100으로 통일. `%`와 `%p`를 혼동하지 않는다.
- 분산 평가에서는 이미지 ID 중복을 제거하고 이미지별 점수의 합/이미지 수로 평균. batch 평균이나 GPU별 평균을 단순평균하지 않는다.
- 원본 이미지 한 장의 전체 픽셀을 평가한다. 검증 random crop 점수는 주지표가 아니다.
- all-ignore 이미지/평가 영역은 유효하지 않은 입력으로 처리한다. 완벽한 점수 1을 반환하지 않는다.

### 4.4 보고할 보조값

`per_image.csv`에는 아래를 저장한다.

```text
image_id, source_split, H, W,
image_miou, pixel_error_rate,
iou_Al, iou_Al3Ni, iou_PrimarySi, iou_EutecticSi,
precision_* , recall_* , gt_fraction_* , pred_fraction_*,
Al3Ni_to_EutecticSi_count, EutecticSi_to_Al3Ni_count,
Al3Ni_to_EutecticSi_rate_of_gt_Al3Ni,
Al3Ni_to_EutecticSi_fraction_of_all_errors,
present_class_miou_diagnostic,
checkpoint_sha256, inference_config_hash
```

분모 0인 precision/recall/오류 비율은 `null`과 이유를 기록한다. mIoU의 empty 정책을 precision에 무조건 적용하지 않는다.

경계 진단은 원본 mask의 인접한 label 차이로 만든 band에서 수행한다. `boundary_band_error_rate_px2`, `interior_error_rate_px2`처럼 명확히 이름을 붙인다. **GT 경계 band 안의 오류율을 논문상의 Boundary IoU라고 부르지 않는다.** class ID 숫자에 직접 dilation−erosion을 적용해 유사 연속값으로 취급하지 않는다.

---

## 5. 기존 코드 감사와 adapter 설계

### 5.1 실제 코드를 읽고 기록할 것

훈련 entry point, 모델 factory, Dataset, augmentation, validation, inference, scheduler, checkpoint loader를 찾아 실제 파일/함수/설정 key를 표로 정리한다. 설치된 패키지 버전, CUDA/torch, GPU 모델·VRAM, 저장 여유 공간도 기록한다.

SMP를 사용한다면 설치된 decoder 구현을 읽는다. 공식 U-Net 구현에는 upsampling 후 convolution이 있으며, raw input 해상도의 첫 encoder feature를 제외하는 구현이 존재한다 [S4]. 따라서 **현재 모델을 저해상도 logits 단순 확대 구조라고 단정하지 않는다.** 실제 decoder의 tensor shape를 forward hook 또는 명시적 반환으로 기록한다.

### 5.2 legacy adapter 계약

저장소 convention을 따르되 다음 기능이 필요하다.

```python
class LegacyModelAdapter:
    # 아래는 구현할 interface 계약이다. 가짜 tensor 반환으로 대체하지 않는다.
    def load_checkpoint(self, path): ...
    def forward_logits(self, normalized_rgb): ...
    def forward_features(self, normalized_rgb): ...
    def legacy_predict_native_probs(self, grayscale_native): ...
```

- `forward_features`는 실제 기존 encoder/decoder를 호출해 최종 decoder feature와 필요한 중간 feature를 반환한다.
- batch/channel/spatial shape를 테스트한다. encoder channel 수를 다른 EfficientNet 구현의 기억으로 하드코딩하지 않는다.
- 원래 checkpoint는 strict load. 확장 모델로 전이할 때만 **신규 모듈에 한정한 missing-key allowlist** 사용. 원래 encoder/decoder weight가 빠졌는데 `strict=False`로 무시하지 않는다.
- 이미지별 확률 비교용 `legacy_predict_native_probs`는 기존 D4/resize/softmax 순서 그대로 재현한다.
- 체크포인트를 읽을 때 지원 환경에서는 안전한 weights-only 로딩을 사용한다. 임의 외부 pickle을 실행하기 위해 자동으로 안전 옵션을 해제하지 않는다.

### 5.3 회귀 기준

지표 계산은 [D1]의 A71 confusion matrix로 정확히 재현한다. 실제 seed32 `last.pt`가 있으면 기존 코드와 adapter를 같은 입력/정밀도에서 비교한다.

```text
지표 fixture: image_miou 오차 < 1e-8
legacy adapter: 원본 코드와 확률 max_abs 차이 및 argmax 불일치 수 보고
동일 경로/정밀도라면 동일 예측을 목표로 함
full valid20: 사용자 보고 80.2265%와 차이를 보고하되 숫자를 맞추는 보정 금지
```

하드웨어/AMP/원래 라이브러리 차이로 불일치하면 재현 실패 사유를 분리한다. tolerance를 계속 느슨하게 해서 통과시키지 않는다.

---

## 6. P1: 100개 모델 확률 앙상블과 공통 오류 진단

### 6.1 정확한 연산 순서

Legacy 앙상블에서는 각 seed의 **기존 전체 추론 결과**를 그대로 사용한다.

```text
native OM
→ 기존 4000px 확대
→ D4 8개 transform
→ 각 transform의 softmax probability
→ transform 역변환
→ 기존 규칙으로 평균/원본 크기 복원
→ P_seed_native [4,H,W]
→ seed 간 probability 평균
→ 마지막 한 번 argmax
```

D4는 회전 4개와 반전 여부 2개의 고유한 8개다. 예를 들어 `rot90(k)` 뒤 horizontal flip을 적용하면 역변환은 flip을 먼저 취소하고 `rot90(-k)`한다. 기존 구현이 다른 순서면 그대로 기록하고 동일하게 사용한다.

**매 seed의 P를 먼저 원본 좌표계에 맞춘다.** 서로 다른 크기의 확률, logits와 probability, 다른 클래스 순서를 섞지 않는다. argmax mask 평균이나 다수결은 probability ensemble의 구현으로 인정하지 않는다.

초기 weight는 `w_m=1/K`. temperature=1, class bias=0. 복잡한 validation weight 최적화는 후속 실험으로 분리한다.

### 6.2 메모리와 cache

100개 모델을 GPU에 동시에 올리지 않는다. 모델/이미지를 순차 처리하고 원본 크기 확률을 캐시한다. 첫 구현은 float32 cache와 float64 또는 충분히 검증된 float32 누적을 사용한다. float16 저장은 별도 오차 비교 후 허용한다.

```text
outputs/probabilities/<run_id>/<image_id>.npy   # [4,H,W], float32
outputs/probabilities/<run_id>/manifest.jsonl
```

manifest의 필수 필드:

```json
{
  "schema_version": 1,
  "image_id": "A71_step2_sample1",
  "source_split": "valid",
  "native_hw": [480, 675],
  "class_names": ["Al", "Al3Ni", "PrimarySi", "EutecticSi"],
  "space": "native",
  "representation": "probability",
  "dtype": "float32",
  "checkpoint_sha256": "RESOLVED_FROM_FILE",
  "image_sha256": "RESOLVED_FROM_FILE",
  "inference_config_hash": "RESOLVED_FROM_CONFIG",
  "training_ids_hash": "RESOLVED_FROM_MANIFEST",
  "checkpoint_selection_ids_hash": "RESOLVED_OR_MARKED_UNKNOWN",
  "probability_path": "RELATIVE_PATH",
  "completed": true
}
```

불완전 cache는 `.tmp`에 쓰고 검증 후 atomic rename한다. 확률 범위, finite 여부, channel sum≈1, shape, ID, hash를 검사한다. 데이터/설정/가중치가 달라진 cache는 재사용하지 않는다. metadata가 없으면 `UNKNOWN`을 조용히 신뢰하지 말고 해당 작업의 사용 가능 범위를 제한한다.

### 6.3 진단 출력

- seed32 단독 vs 100개 동일가중치 평균.
- 미리 정한 seed 목록의 K별 곡선. top-K를 valid로 고르면 `validation-selected`로 표시.
- 픽셀별 모델간 vote agreement와 probability variance.
- `GT=Al3Ni` 영역에서 ensemble의 EutecticSi 확률과 model agreement.
- `Al3Ni→EutecticSi`, 역방향, `Al→Al3Ni` 등 방향별 오류.
- 회색 EutecticSi 오류가 흰 correct 배경에 묻히지 않도록 오류 클래스 범례/대조 배경을 선택.

동일한 70장으로 학습한 seed 100개는 **OOF 모델 100개가 아니다.** 이들의 train70 예측을 refiner용 OOF라고 표시하지 않는다.

---

## 7. P2/P3의 모델 구조: 기존 B4를 보존한 잔차 확장

### 7.1 목표 구조

```text
Detail image: [B,3,1536,1536] (기존 정규화)
       │
       ▼
기존 EfficientNet-B4 encoder → 기존 U-Net decoder → F_detail → 기존 4ch head → Z_base
                                                        │
Native grayscale → relative maps → detail ROI → TextureStem ──┐
                                                              │
Wider context ROI → small ContextCNN → ROI-aligned features ───┤
                                                              ▼
                                                ResidualFusion / 4ch head
                                                              │
                                                      Delta_Z [B,4,P,P]
                                                              │
                                                Z_final = Z_base + Delta_Z
```

`TextureStem`, `ContextCNN`, `Delta_Z`는 새 모듈이다. **모든 feature를 한꺼번에 추가하지 않고 각각 on/off ablation**을 지원한다. context가 꺼졌다면 context tensor를 만들거나 CNN을 실행할 필요가 없다. `residual_head=true, texture=false, context=false`인 detail-only 잔차 대조군도 지원한다.

### 7.2 모듈별 최초 설계값

| 모듈 | 입력 | 내부 설계 시작값 | 출력 |
|---|---|---|---|
| Base | 정규화 RGB detail | 실제 기존 B4/U-Net | 원래 decoder feature, 4ch logits |
| TextureStem | raw, relative4, relative16의 3ch | Conv3×3 → GN → SiLU, full-res 16ch; 1/2 32ch; 1/4 64ch | texture pyramid |
| ContextCNN | grayscale context 1ch | 1/2 32ch, 1/4 64ch, 1/8 128ch, 1/16 256ch | context features |
| ContextProjection | context의 1/8, 1/16 | 각 1×1 Conv → 32ch, detail feature 좌표로 sampling | 정렬된 문맥 특징 |
| ResidualFusion | 기존 feature projection + texture + context | 1/4에서 64ch, 1/2에서 32ch, 원해상도에서 16ch | 16ch fine feature |
| ResidualHead | fine feature | Conv1×1 → 4ch | Delta_Z |

이는 시작값이다. 실제 B4 decoder feature의 C/H/W를 확인한 뒤 projection 입력 채널을 연결한다. 신규 context의 downsampling은 초기 구현에서 `AvgPool2d(2)` + stride1 conv로 명확히 정의해 grid를 단순하게 유지한다. padding은 same-size, 크기는 입력에서 결정한다. GN group 수는 채널을 나누는 수로 고정한다(예: 8). 모든 신규 경로의 normalization은 별도 설정으로 관리한다 [S8].

### 7.3 기존 checkpoint를 안전하게 활용하는 초기화

최종 `ResidualHead`의 weight/bias만 0으로 초기화해 `Z_final == Z_base`로 시작한다. 앞단 feature conv까지 모두 0으로 만들지 않는다. `Delta_Z=0`이면 신규 모듈을 켜도 eval logits가 원래와 일치해야 한다.

최종 head를 0으로 초기화하면 첫 backward에서 더 앞선 신규 feature 경로의 gradient가 0일 수 있다. 첫 optimizer step으로 head가 업데이트된 다음 두 번째 backward까지 검사한다. 이 정상적인 현상을 branch가 끊겼다고 오판하지 않는다.

신규 모듈에 gate를 추가한다면 gate와 최종 head를 모두 0으로 해서 영구적으로 gradient가 막히지 않도록 설계한다. 기본 구현은 별도 gate 없이 zero-init 마지막 head 하나다.

이 수정은 기존 학습 가중치를 활용하기 위한 **잔차 확장 설계**이지, 특정 공개 논문 모델을 그대로 구현한다는 뜻이 아니다.

---

## 8. 기하 계약: 확대, native ROI, 문맥 정렬

이 프로젝트의 가장 위험한 구현 오류는 **모양이 맞아 보이지만 한두 픽셀 어긋난 학습/추론**이다. texture 성능이나 backbone을 평가하기 전에 이 절의 테스트를 통과해야 한다.

### 8.1 두 geometry profile을 분리한다

**`legacy_scaled_crop`:** 기존 코드의 4000px resize와 그 위의 랜덤 crop을 그대로 사용한다. P0/P1과 원본 재현 전용이다.

**`native_roi_v1`:** 원본에서 정수 좌표 detail ROI를 먼저 고른 뒤 모델 입력 크기로 확대한다. 신규 native loss와 dual-FOV의 기본이다. 반올림/보간 순서가 legacy와 완전히 같지는 않으므로, **기능을 모두 끈 geometry-only 대조군을 반드시 학습/평가**한다.

```text
P_detail = 1536
s_nominal = target_width / W_original
n_detail = floor(P_detail / s_nominal + 0.5)
D = [x0, y0, x0+n_detail, y0+n_detail)    # 원본 정수 좌표
```

A71에서 `n_detail=259`가 시작값이다. legacy의 259.2px 범위를 임의로 동일하다고 표시하지 않는다. 물리 배율 메타데이터가 없으므로 이 값을 μm로 해석하지 않는다 [D1].

### 8.2 detail/context 규칙

- detail은 원본의 정수 ROI D를 crop하고 `1536×1536`으로 변환한다.
- context는 D와 **동일한 중심**, 변 길이 `context_factor × n_detail`인 정사각형 C를 사용한다. 시작 `context_factor=2.0`, 입력 `768×768`.
- C의 시작 좌표는 반정수일 수 있다. 조용히 반올림하거나 영상 안으로 이동시키지 않는다. 기록된 연속 좌표로 sampling한다.
- 영상 밖은 입력에 한해 일관된 reflection padding을 사용한다. 외삽/반사된 위치는 정답 손실의 유효 영역으로 간주하지 않는다.
- `reflect`가 허용하지 않는 극소 크기 입력은 명시적으로 replicate fallback을 정의하고 테스트한다.
- mask/validity는 nearest 계열, intensity/probability는 각 profile에 정의된 보간을 사용한다. 새 profile의 기본은 bilinear, `align_corners=False`; downsampling antialias 여부를 config에 명시한다 [S2].
- mask를 확대하고 다시 축소해서 native GT로 사용하지 않는다. 원본 mask의 D를 별도로 보관한다.

### 8.3 좌표 정의와 context feature sampling

원본 pixel의 중심은 `(x+0.5, y+0.5)`, 영상 경계는 `[0,W]×[0,H]`로 정의한다. feature를 해당 ROI를 균등 분할한 grid로 취급하는 coordinate contract를 사용한다.

D에서 정렬 대상 feature 크기가 `(H_t,W_t)`일 때:

```text
x_native = D.x0 + (j + 0.5) * D.width  / W_t
y_native = D.y0 + (i + 0.5) * D.height / H_t

gx = 2 * (x_native - C.x0) / C.width  - 1
gy = 2 * (y_native - C.y0) / C.height - 1

grid[...,0] = gx
grid[...,1] = gy
aligned = grid_sample(context_feature, grid, align_corners=False)
```

이 식은 context가 ROI 전체를 비율 보존 변환한 경우의 계약이다. letterbox/pad가 추가되면 해당 affine 변환을 포함해야 한다. shape만 맞추는 중앙 crop으로 대체하지 않는다. `grid_sample`과 resize에서 `align_corners` 정책을 섞지 않는다 [S2, S3].

**주의:** 의미 feature의 grid 정렬 계약과 convolution의 유효 수용영역 중심은 같은 개념이 아니다. 실제 decoder의 feature 해상도·padding과 정렬 convention을 문서화하고, 합성 ramp/랜드마크 테스트로 구현된 좌표를 검증한다.

### 8.4 증강 순서

첫 신규 구현은 D4와 공유 photometric 증강만 사용한다. 임의 각도 회전/elastic transform은 좌표 테스트 후 별도 추가한다.

```text
원본 grayscale + 원본 mask 로드
→ 원본 영상에 photometric transform 1회 적용 (image-only)
→ 변환된 원본에서 상대 밝기 map 계산
→ 동일한 D,C로 detail/context/texture/target/validity 준비
→ 동일한 D4 변환을 모든 해당 tensor에 적용
→ 모델
→ native loss에서는 예측 D4를 역변환
→ 원래 D 크기로 probability 복원
→ 원본 D 정답과 비교
```

- detail/context에 서로 다른 gamma, flip, rotate를 적용하지 않는다.
- relative map을 변환 전 영상에서 만들고 raw에는 변환 후 영상을 넣는 불일치를 만들지 않는다.
- 원본 방향의 `W_original`을 먼저 고정한다. D4마다 가로폭을 재정의해서 확대율을 바꾸지 않는다.
- 새로운 scale jitter는 원본 D의 크기를 바꾸는 방식으로 구현하고 시작 ablation에서는 끈다. 켤 경우 D/C/label/native target에 같은 geometry를 사용한다.

### 8.5 새 모델의 inference

dual-FOV에서는 `native_roi_v1`의 **detail tile + 더 넓은 context** 단위로 추론한다. 이는 기존 full-frame 추론과 다른 설정이므로 geometry-only/tile-only 대조군과 함께 보고한다.

- detail tile 크기는 해당 원본의 n_detail, stride는 시작값 `max(1, floor(n_detail/2))`.
- 첫/마지막 tile을 포함해 원본 모든 픽셀이 덮이도록 위치 목록 생성.
- D4는 준비된 detail/context/texture에 동일하게 적용하고 역변환 후 probability를 native D에 복원.
- native D 확률을 전체 canvas에 가중합. 양수 weight floor가 있는 Gaussian 또는 taper window를 사용.
- 픽셀별 weight sum으로 나눈 뒤 channel을 정규화한다. 분모 0인 원본 픽셀이 하나라도 있으면 오류.
- 이미지 밖 padding을 제거하고 원본 H/W로 출력한다. 전체 native probability가 완성된 다음 argmax.
- tile 중앙을 신뢰하는 blending은 후보이며, tile을 작게 바꿨다는 사실만으로 동일 추론이라고 주장하지 않는다.

넓은 저해상도 시야와 작은 고해상도 시야 결합은 HRDA에서 참고할 수 있다. 본 구현은 HRDA의 도메인 적응 학습 전체를 재현하는 것이 아니라 시야 결합 원리를 적용하는 것이다 [S5].

---

## 9. P2: 상대 밝기·텍스처 경로

### 9.1 입력 feature 정의

원본 grayscale을 `I∈[0,1]`로 변환한다. GT 없이 아래 특징을 만든다.

```text
mu_sigma = Gaussian(I, sigma)
var_sigma = max(Gaussian(I*I, sigma) - mu_sigma*mu_sigma, 0)
relative_sigma = (I - mu_sigma) / sqrt(var_sigma + eps_std**2)
relative_sigma = clip(relative_sigma, -clip_z, clip_z) / clip_z

texture_native = stack([I, relative_4, relative_16])
```

시작값: `sigma_native_px=[4.0,16.0]`, `eps_std=0.01`, `clip_z=5.0`, Gaussian truncate=3.0. sigma는 **원본 px**다. kernel 반경은 `ceil(truncate*sigma)`로 명시한다. separable Gaussian kernel을 정규화해 사용하며 padding convention을 train/inference에서 일치시킨다.

이 세 채널을 native D 좌표에서 detail input 크기로 변환해 TextureStem에 전달한다. raw 채널이 기존 RGB 경로에도 있어도 무방하다. 기존 B4 first conv의 입력 채널 수를 바꿔 사전학습 가중치를 억지로 이식하지 않는다.

### 9.2 invariants

- 상수 영상: relative 채널 ≈0, NaN/Inf 없음.
- finite 입력이면 모든 출력 finite, relative∈[-1,1].
- 클래스별 GT 평균을 계산해 정규화하는 기능은 금지.
- histogram matching 대상에 validation GT를 사용하지 않는다.
- 원본을 CLAHE 하나로 대체하지 않는다. CLAHE 등은 나중의 독립 옵션이며 기본은 꺼짐.
- native feature cache는 image hash, photometric transform, sigma, dtype, padding에 종속된다. stochastic augmentation에 고정된 stale cache를 재사용하지 않는다.

### 9.3 증강 진단

원본/증강 후 영상 예시와 클래스별 밝기 통계를 **훈련 이미지에서** 저장해 현재 augmentation이 어느 범위를 보이는지 확인한다. validation 통계는 분석 전용으로 분리한다. `A71 수준 밝기로 무조건 맞춘다`는 규칙은 구현하지 않는다.

블러/노이즈 약화 실험은 한 번에 하나의 강도를 바꾸고 원래 설정을 대조군으로 유지한다. 증강의 scale은 원본 px 기준인지 확대 px 기준인지 기록한다.

---

## 10. P4-A: 혼합조직·혼동 패치 샘플링

### 10.1 분할 원칙

먼저 train 이미지 ID를 균등하게 고른다. 그 이미지 안에서 sampler mode와 중심점을 선택한다. 큰 이미지가 픽셀 수만큼 더 자주 선택되지 않도록 한다.

시작 비율:

```text
random:  0.50
mixed:   0.30
hard:    0.20
```

확률의 합=1을 검사한다. 특정 이미지에서 mixed/hard 후보가 없으면 그 이미지의 random으로 fallback하고 횟수를 로그에 남긴다. 다른 이미지를 반복 선택해 이미지 균형을 깨뜨리지 않는다.

### 10.2 mixed score

원본 mask의 one-hot와 validity로 각 위치의 `41×41` 국소 클래스 비율 `p_c`를 만든다. padding/ignore를 분모에서 제외한다.

```text
H_local = -sum_c(p_c * log(max(p_c, eps))) / log(4)
B = 이웃 유효 label이 다르면 양쪽 pixel을 1로 표시한 경계 map
D_boundary = local_mean(B, window=41, valid-normalized)
S_mixed = min(p_Al3Ni, p_EutecticSi) * H_local * D_boundary
```

상 두 개가 커다랗게 반씩 나뉜 패치의 entropy도 높기 때문에 경계 밀도를 같이 사용한다. score>0인 위치를 score 비례로 sampling하는 구현부터 시작한다. 넓이가 다른 영상에서 threshold를 임의 적용하지 않는다.

이 지도는 **훈련용 sampler metadata**다. 추론 네트워크 입력, validation crop 선택, validation 지표 가중치로 사용하지 않는다.

### 10.3 hard score

가능하면 train70의 OOF probability를 사용한다. 아직 OOF가 없으면 train70에 대한 현재 모델의 예측을 sampler 용도로만 사용할 수 있으나 metadata에 `in_sample_train_mining`을 표시한다. 이를 refiner의 OOF 데이터로 재사용하지 않는다.

시작 예시:

```text
S_hard = min(-log(max(P[GT], 1e-6)), training_image_95th_percentile)
S_hard *= 1 + 1 * indicator(GT=Al3Ni and argmax(P)=EutecticSi)
```

방향성 boost는 1개의 시작값이며, 단방향 hard mining이 공정 Si를 놓치게 하는지도 확인한다. validation mask 또는 A71의 오류 위치는 train sampler에 들어갈 수 없다.

경계·mixed·hard는 중첩될 수 있으므로 각 sample이 어떤 mode로 선택됐는지와 ROI 내 GT 분포를 기록한다. 입력 image만으로부터 생긴 edge를 곧바로 상의 GT 경계로 취급하지 않는다.

---

## 11. P4-B: 원본 좌표계의 추가 손실

### 11.1 baseline loss 보존

`L_base = 기존 코드의 CE + Dice`를 그대로 감싼다. logits/probability, 클래스 처리, reduction이 다른 구현으로 조용히 교체하지 않는다. 새 loss가 꺼지면 legacy loss와 같은 값을 반환한다.

### 11.2 native loss의 입력

```text
Z_detail [B,4,P,P]
→ softmax(float32)
→ D4 역변환
→ 각 sample의 native D 크기 (n_i,n_i)로 resize
→ P_native_i [4,n_i,n_i]
→ 원래 Y_native_i와 valid_native_i로 loss
```

native 크기가 sample마다 다를 수 있으므로 list로 유지하거나 pad+ignore를 사용한다. 배치 tensor 모양을 맞추려고 원본 정답들을 같은 크기로 강제 resize하지 않는다.

원본 NLL:

```text
l_i(x) = -log(max(P_native_i[GT_i(x),x], 1e-6))
L_native_NLL = mean_i( sum_valid_x(l_i(x)) / count_valid_i )
```

시작 조합:

```text
L_total = L_base
        + lambda_native * (L_native_NLL + 0.5 * L_native_Lovasz)
lambda_native = 0.5      # 독립 실험의 시작값
```

Lovász-Softmax는 저자의 구현 또는 출처를 보존한 검증된 구현을 사용한다. 시작 `per_image=True`, `classes='present'`, `ignore=255` [S6, S7]. NLL/기존 CE는 absent class로의 잘못된 예측도 억제한다. 이 Lovász 설정이 union=0→1인 hard mIoU와 완전히 같은 수학적 함수라고 주장하지 않는다.

**중요:** softmax 확률을 일반적인 logits용 CrossEntropy에 다시 넣지 않는다. native NLL은 `log(P)`에 NLLLoss를 적용하거나 gather로 직접 계산한다. AMP 환경에서도 probability 합산, log, Dice/Lovász는 float32로 수행한다.

### 11.3 원본 이미지 단위와 ROI 단위를 구분

native loss를 ROI에 적용해도 **full-image mIoU를 직접 최적화한 것은 아니다.** 좌표와 샘플 간격을 평가에 맞춘 보조 목적함수다. 모델 선택은 full-image validation으로 한다.

나중에 full-image 미세조정을 구현한다면 differentiable tile blending 또는 실제 full-frame forward를 사용하고, 중간 probability를 detach하지 않은 채 한 원본 이미지 단위로 loss를 계산해야 한다. 계산이 부담된다고 random crop 평균으로 대체하면서 full-image loss라고 이름 붙이지 않는다. 초기 구현의 필수 범위는 native ROI loss까지다.

### 11.4 경계 loss는 기본적으로 꺼둔다

첫 버전에서는 EutecticSi class weight 확대, 경계 CE ×3, boundary auxiliary head를 동시에 켜지 않는다. 필요하면 별도 ablation으로 추가한다.

weighted CE를 추가할 경우 `sum(w*l)/sum(w)`처럼 분모를 정규화해 weight 증가가 전체 loss scale만 바꾸지 않게 한다. GT boundary는 native label의 인접 차이로 계산한다. 이 프로젝트의 주된 오류가 단순 위치 어긋남인지 클래스 혼동인지 진단부터 수행한다.

---

## 12. P5: 학습 레시피, checkpoint, EMA, 재현성

### 12.1 두 레시피를 구분한다

**`scratch_comparable`:** 기존 16,000-update 레시피에서 시작하고 변경하려는 변수만 바꾼다. 새 모델도 동일한 ImageNet 인코더 초기화 정책을 사용한다. seed `[0,1,2]`를 미리 고정해 비교하고, 좋은 결과만 골라서 보고하지 않는다.

**`finetune_existing`:** 대응 seed의 기존 `last.pt`를 읽고 새 모듈을 추가한다. 같은 checkpoint에서 같은 학습 길이로 추가 학습한 **모듈 없는 미세조정 대조군**을 반드시 둔다. seed32 하나에서 모든 변형을 시작했다면 이는 독립 3-seed scratch 실험과 다름을 명시한다.

미세조정 시작값:

| 구간 | 길이 | base encoder/decoder | 새 모듈 |
|---|---:|---|---|
| A | 500 optimizer updates | 동결, BN running stats도 고정 | lr=3e-4 |
| B | 3,500 optimizer updates | encoder lr=1e-5, 기존 decoder/head lr=3e-5 | lr=1e-4 |

구간 B는 새 optimizer 또는 parameter-group별 새 scheduler를 명확히 시작한다. 이전 16,000-step cosine의 마지막 learning rate를 그대로 resume해 사실상 학습하지 않는 상태를 만들지 않는다. weight decay 등 미지정 값은 기존 설정을 감사한 뒤 유지한다.

새 모듈이 전혀 없는 plain finetune 대조군은 A구간을 생략하고 B의 base 학습률로 4,000 update를 수행한다. 빈 parameter list로 optimizer를 만들거나 동결된 모델에 backward를 강제하지 않는다. 구조의 추가 정보 효과를 더 정확히 분리하기 위해 **detail feature만 받는 동일 잔차 head 대조군(E03b)**도 두고, 이 모델에는 동일한 A/B 스케줄을 적용한다.

이 4,000 update는 최적 학습 길이가 아니다. 이후 2,000/4,000/8,000 등 미리 정한 비교를 할 수 있다. 비교군의 실제 update 수·학습된 원본 이미지/ROI 수·벽시계 시간을 함께 기록한다.

### 12.2 checkpoint 종류와 선택 지표

```text
last.pt                    # 완전 resume용, 마지막 상태
best_image_miou.pt          # 정해진 full validation 프로토콜의 최고값
best_ema_image_miou.pt      # EMA를 켰을 때 별도 저장
history.jsonl              # step별 loss/LR/validation 점수
```

주 checkpoint 선택 지표는 `val_image_miou`다. loss, global mIoU, A71 점수로 대체하지 않는다. seed32의 historical best와 새 설정의 best-of-many를 같은 독립 비교처럼 표현하지 않는다.

검증 프로토콜은 모든 비교군에 고정한다. 시작 예시는 1,000 optimizer update마다 valid20 전체/D4 평가다. 빠른 no-TTA 평가도 별도 이름으로 기록할 수 있지만, 후보를 no-TTA로 고른 후 그 값을 D4-best라고 부르면 안 된다. 평가 횟수와 방식은 설정에 남긴다.

### 12.3 EMA

EMA는 별도 ablation으로 켠다. 시작 decay=0.999. **실제 optimizer update 이후**에만 갱신한다. gradient accumulation microstep마다 갱신하거나 AMP overflow로 optimizer가 skip됐는데 갱신하지 않는다.

PyTorch의 `AveragedModel`과 EMA averaging utility를 사용할 수 있지만, 설치 버전에 맞는 API를 확인한다 [S9]. BN buffer 처리는 별도 정책으로 고정한다.

권장 첫 정책:

```text
학습 parameter: EMA
BN running stats/counters: base에서 동기화하는 명시적 정책
최종 후보: 필요하면 train-only 데이터로 BN 통계 재보정 후 별도 결과 기록
```

EMA의 BN 통계 보정에 valid/test 영상을 기본적으로 사용하지 않는다. train-only 보정도 데이터 순서/증강/eval protocol을 고정한다. `model.train()` 호출로 frozen encoder의 BN이 다시 업데이트되지 않는지 테스트한다.

### 12.4 작은 batch의 normalization

기존 EfficientNet의 pretrained BN을 전체 GN으로 일괄 치환하지 않는다. 기존 BN 유지, BN 통계 동결 등을 독립 비교한다. 신규 TextureStem/ContextCNN/Refiner는 GN을 시작값으로 쓴다. GroupNorm은 batch 통계에 의존하지 않는 정규화로 제안되었다 [S8]. 이 사실이 해당 데이터에서 성능 향상을 보장하지는 않는다.

### 12.5 재개 가능한 실행

checkpoint에는 model/optimizer/scheduler/AMP scaler/EMA, global optimizer step, seed, Python/NumPy/torch CPU/CUDA RNG, sampler 상태, resolved config, split/hash를 저장한다. `resume`와 `initialize_from`을 다른 옵션으로 만든다.

완전 동일한 resume를 보장하려면 worker 난수를 단순히 대충 초기화하지 않는다. 권장: 각 sample의 이미지·ROI·증강 난수를 `(run_seed, optimizer_step, microstep, rank, sample_index)`에서 결정하거나 sampler 상태를 저장한다. 실제 저장소 방식과 호환시키고 재개 시 ROI 선택 재현 테스트를 수행한다.

DDP에서 validation loader가 중복 padding한 image ID를 두 번 점수에 넣지 않는다. 서로 다른 GPU가 같은 output/cache 파일을 덮어쓰지 않도록 atomic write와 rank ownership을 사용한다.

---

## 13. P6: OOF 기반 원본 해상도 4클래스 보정기

### 13.1 역할

보정기는 단순 경계 이진 분류기가 아니다. 원본 image와 4클래스 확률을 보고 **Al₃Ni ↔ Eutectic Si를 포함한 4클래스 점수 수정량**을 예측한다.

```text
I_native                         1 channel
P_base_native                    4 channels
relative4, relative16            2 channels
entropy(P_base)/log(4)           1 channel
------------------------------------------
기본 입력                         8 channels
```

optional gradient magnitude는 별도 설정으로 추가할 수 있으나 첫 버전에서는 입력을 8채널로 고정한다. input channel order를 checkpoint metadata에 저장한다.

시작 구조: small residual U-Net, channels `[32,64,128,256]`, 원본 해상도 shallow skip, GN, SiLU. 출력은 4채널 `delta_logits`다.

```text
P_safe = clamp_min(P_base, 1e-6)
P_safe = P_safe / sum_channels(P_safe)
Z_final = log(P_safe) + R(I_native, P_base, relative_maps, entropy)
P_final = softmax(Z_final)
```

최종 1×1 conv를 zero-init. 초기 `P_final`은 clamp/renormalization 오차 범위에서 `P_base`와 같아야 한다. native 512px crop 등을 시작값으로 학습할 수 있고, 추론은 native 전체 또는 overlap tile을 쓰되 그 설정을 맞춰 검증한다. uncertainty가 높은 픽셀에서만 실행하도록 강제하지 않는다. 높은 확신으로 틀린 픽셀도 보정 대상이다.

### 13.2 train70과 valid20을 유지하는 OOF 생성

`T=train70`, `V=valid20`, `F1…F5`는 T의 비중복 fold다.

```text
for k in 1..5:
    teacher를 T\Fk의 이미지로만 학습
    Fk는 teacher optimizer/BN 통계/early stopping/학습형 후처리에 사용하지 않음
    teacher가 Fk에 대해 원본 확률 P_OOF 생성

모든 Fk cache를 합침 → T의 이미지마다 OOF probability 1개
refiner는 (T 원본, P_OOF, T 정답)으로 학습
T 전체로 학습된 teacher가 V의 확률 생성
refiner가 V 확률을 보정 → V의 full-image score 측정
```

fold가 5개면 teacher는 보통 56장으로 학습한다. teacher checkpoint 선택은 고정 update 또는 teacher 학습 집합 내부의 분할에서 수행한다. 해당 Fk 정답으로 best teacher를 고르면 엄밀한 OOF가 아니다.

반드시 지킬 것:

- 기존 seed0–99 teacher는 전부 T 전체를 봤으므로 T에 대한 OOF 생성에 사용 불가. **이 checkpoint에서 초기화한 뒤 T\Fk로만 미세조정하는 것도 금지**다. Fk 정보가 이미 가중치에 들어 있다. OOF teacher는 ImageNet 초기화 또는 Fk와 모든 상위 학습 이력이 분리된 checkpoint에서 시작한다.
- `A71`이 속한 V는 refiner의 training example에 포함하지 않는다.
- base architecture, preprocessing, 클래스 순서, TTA, teacher ensemble 수와 cache dtype을 OOF와 deployment에서 맞춘다.
- teacher가 OOF 시점에는 단일 모델인데 추론에서는 100모델 평균이면 입력 분포가 달라진다. 반드시 그 차이를 별도 실험으로 기록한다.
- 초기 teacher 구성은 동일한 3-seed 평균 또는 동일한 단일 seed로 고정한다. 계산 여유가 있으면 모든 fold에서 같은 수의 seed를 사용한다.
- fold teacher마다 56장, 최종 teacher는 70장인 차이는 구조상 남는다. 이 차이를 숨기지 말고 실제 V 보정 이득으로 검증한다.

### 13.3 leakage lineage

각 OOF cache에는 적어도 다음을 저장한다.

```text
predicted_image_id
fold_id
teacher_weight_training_ids
teacher_ancestor_checkpoint_training_ids
teacher_checkpoint_selection_ids
teacher_bn_calibration_ids
teacher_postprocessor_training_ids
teacher_config_hash
teacher_checkpoint_hashes
teacher_ensemble_definition
```

`predicted_image_id`가 현재 및 상위 checkpoint의 어느 training/selection/calibration 목록에도 없어야 한다. lineage를 알 수 없는 warm-start 가중치는 strict OOF에 사용하지 않는다. 단순히 `is_oof=true` 문자열만 믿지 않는다. 원본 image group overlap도 별도 진단한다.

### 13.4 공정한 보정기 검증

V를 반복적으로 보며 refiner/앙상블 가중치를 튜닝한 점수는 개발용 validation 결과다. 독립적인 최종 성능 주장에는 outer fold 평가가 필요하다 [S10].

엄격한 outer fold에서는 outer holdout을 완전히 제외한 데이터 안에서 inner OOF teacher → refiner 학습을 다시 수행하고 outer holdout을 예측한다. **90장 전체의 OOF 확률 하나를 먼저 만들고 outer fold만 나눠 refiner를 평가하면 teacher 경로를 통한 누수가 생길 수 있다.** outer fold 밖 정보가 inner teacher의 weight/selection/normalization에 들어가지 않아야 한다.

같은 20장으로 class bias/ensemble weight/refiner를 학습한 뒤 같은 20장 점수를 독립 성능처럼 쓰지 않는다.

---

## 14. 검증 설계와 실험 우선순위

### 14.1 처음 수행할 실험 표

| ID | 기준 | 변경점 | 목적 |
|---|---|---|---|
| E00 | 현재 코드 | seed32 last + 기존 D4 | 원본 재현 |
| E01 | E00 모델군 | seed0–99 probability 평균 | 추가 학습 없는 ensemble 기준선 |
| E02 | legacy training | native_roi_v1 + 새 tile inference, 새 branch 모두 off | geometry/inference 변경 효과 분리 |
| E03a | E02 | 같은 checkpoint에서 4,000-step plain finetune | 추가 학습 자체의 효과 |
| E03b | E03a | detail-only 잔차 head + A/B 스케줄; texture/context off | 새 잔차 경로의 용량·스케줄 대조군 |
| E04 | E03b | relative texture branch만 on | 상대 밝기 정보의 효과 |
| E05 | E03b | context branch만 on | 더 넓은 시야의 효과 |
| E06 | E03a | mixed/hard sampler만 on | 샘플 선택의 효과 |
| E07 | E03a | native loss만 on | 원본 좌표계 추가 손실 효과 |
| E08 | E03a | EMA 또는 BN 통계 고정 중 하나 | 학습 안정화 효과 |
| E09 | 검증된 단독 변경 | 유효한 변경만 조합 | 상호작용 확인 |
| E10 | E09 | strict OOF refiner | 추가 보정 이득 |
| E11 | 유효 모델들 | 이종/다중 시야 probability ensemble | 최종 결합 |

표의 `기준`은 비교할 설정을 뜻하며 이전 실험의 best checkpoint를 자동으로 이어받으라는 뜻이 아니다. paired 실험은 대응 seed의 **동일한 원본 parent checkpoint**에서 독립적으로 시작한다. parent를 바꾸면 다른 실험으로 기록한다.

이 표는 run 수를 고정하는 뜻이 아니다. 한 실험에서 여러 요소가 바뀌었다면 각 요소의 독립 기여를 보고하지 않는다. seed32를 시작점으로 하는 빠른 개발과 `[0,1,2]` paired-seed 비교를 구분한다.

별도 scale 실험으로 `target_width=2500` 모델을 **그 설정으로 학습**해 4000 모델과 비교/결합할 수 있다. 4000으로만 학습한 모델의 추론 해상도만 바꾼 결과는 동일 실험이 아니다.

### 14.2 채택 기준

- 1순위: valid20 전체의 `val_image_miou` 개선.
- 함께 보고: paired seed별 차이, 이미지별 변화, Al₃Ni/Sie IoU·precision/recall, 오류 방향, 최악 사례.
- A71만 오르고 전체 점수가 하락하면 기본 채택하지 않는다.
- 점수 차이가 매우 작으면 이미지 단위 paired bootstrap 또는 적절한 group 단위 bootstrap으로 불확실성을 확인한다. 동일 이미지 내 픽셀을 독립 표본처럼 bootstrap하지 않는다.
- 성능 보장 수치를 미리 약속하지 않는다. 음의 결과도 report에 남긴다.
- image mIoU가 선택 기준이며, 물리적으로 보기 좋은 경계나 오류율 최소화로 대체하지 않는다.

### 14.3 seed ensemble selection의 주의

100개 seed 중 best를 고르고, 같은 V에서 ensemble weight/temperature/class bias를 또 조절하면 selection이 누적된다. 초기에는 동일가중치/temperature1/bias0을 사용한다. 후속 학습형 calibration은 OOF 또는 inner split에서 fit하고 외부 평가한다 [S10].

Al₃Ni 확률에 상수를 더하거나 EutecticSi 면적을 강제로 줄이는 규칙을 A71에 맞춰 넣지 않는다. PrimarySi가 없다는 V 정답 정보로 해당 클래스 출력을 제거하지 않는다.

---

## 15. 권장 코드 구조와 입출력 interface

기존 저장소 구조를 우선한다. 아래는 신규 기능의 **논리적 분리**를 위한 예시이며 전체 repository를 이 구조로 옮기라는 요구가 아니다.

```text
alphase/
  contracts.py                 # class order, palette, dataclasses, config validation
  data/
    manifest.py
    legacy_adapter.py
    native_roi.py
    relative_features.py
    samplers.py
    transforms.py
  models/
    legacy_b4.py
    enhanced_b4.py
    texture_stem.py
    context_encoder.py
    residual_refiner.py
  geometry/
    coordinates.py
    d4.py
    native_restore.py
  losses/
    legacy_loss.py
    native_loss.py
  evaluation/
    confusion.py
    metrics.py
    diagnostics.py
  inference/
    legacy.py
    dual_fov_tiles.py
    cache.py
    ensemble.py
  training/
    loop.py
    checkpoints.py
    ema.py
    oof.py
  cli.py
configs/
  resolved_legacy.yaml
  examples/enhanced_candidate.yaml
  experiments/*.yaml
manifests/
tests/
reports/
outputs/                        # git-ignore
```

### 15.1 sample contract

```text
sample:
  image_id: str
  detail_rgb: float32 [3,P,P]
  texture_detail: float32 [3,P,P] | absent when disabled
  context_gray: float32 [1,C,C], range [0,1] | absent when disabled
  target_detail: int64 [P,P]
  target_native: int64 [n,n]
  valid_native: bool [n,n]
  geometry:
    native_hw: [H,W]
    detail_rect_xywh: [x0,y0,w,h]
    context_rect_xywh: [x0c,y0c,wc,hc] | null
    d4_k: int
    d4_flip: bool
    profile: legacy_scaled_crop | native_roi_v1
    original_width_reference: int
```

`target_native`는 augmented D4 좌표가 아니라 **D4 역변환된 prediction이 비교할 원래 D 좌표**로 통일한다. collator가 이 규칙을 바꾸지 않는다. geometry 구조체/serializer의 field 이름을 일관되게 유지한다.

### 15.2 model output contract

```text
output:
  logits_detail: float [B,4,P,P]
  base_logits_detail: float [B,4,P,P] | diagnostic
  delta_logits_detail: float [B,4,P,P] | diagnostic
  aux: dict
```

`model.train()/eval()`에 따라 클래스 채널/좌표를 다르게 반환하지 않는다. logits key와 probability key를 구분한다. optional branch가 꺼졌을 때도 기본 output은 유효해야 한다.

### 15.3 loss contract

```text
losses = {
  "total": differentiable scalar,
  "base": scalar,
  "native_nll": scalar or disabled,
  "native_lovasz": scalar or disabled
}
```

metric 계산만 `no_grad`. native probability를 NumPy로 변환해서 loss를 계산하거나 `.detach()`해 학습 신호를 끊지 않는다.

---

## 16. 설정 파일 명세

### 16.1 설정 관리 규칙

사용 중인 config framework가 있으면 활용한다. 없다면 YAML + 명시적 validation으로 구현한다. 아래 YAML은 **구현할 schema의 예시**이며, 아직 구현되지 않은 CLI가 이미 존재한다는 뜻이 아니다.

- `inherit`는 단일 parent YAML을 먼저 읽고 dictionary를 재귀 merge한다. list/scalar는 전체 교체한다.
- inherit cycle, unknown key, class mismatch는 error.
- `null` 필수값이 남으면 train/infer를 시작하지 않는다. audit/dry-run은 unresolved 목록을 출력할 수 있다.
- CLI `--set key=value`를 지원한다면 type checking 후 override한다.
- 최종 merged/resolved YAML을 run directory에 저장하고 hash를 기록한다.
- baseline mode에서는 새 defaults를 적용해 기존 설정을 바꾸지 않는다.
- 평가와 추론 양쪽에 D4 옵션이 있으면 반드시 일치해야 한다. recipe별로 사용하지 않는 legacy warmup 등은 무시되는 이유를 resolved config에 남기고, 충돌한 스케줄을 중복 적용하지 않는다.

### 16.2 서버 감사 후 생성할 legacy profile

```yaml
schema_version: 1
run:
  name: legacy_reproduction
  seed: 32
  output_root: outputs
paths:
  dataset_root: null             # 서버 검사 후 채움
  checkpoint_manifest: manifests/checkpoints.jsonl
  legacy_config: null            # 원본 설정 파일 또는 추출된 설정
  legacy_checkpoint: null
classes:
  names: [Al, Al3Ni, PrimarySi, EutecticSi]
  rgb: [[153,127,76], [25,76,153], [76,178,76], [204,204,204]]
  ignore_index: 255
model:
  family: legacy_b4_unet
  implementation: null          # 실제 factory/module 경로
  encoder_name: null            # 실제 기존 이름
  new_branches:
    texture: false
    context: false
    residual_head: false
geometry:
  profile: legacy_scaled_crop
  target_width: 4000
  detail_input_size: 1536
  legacy_resize_policy: null    # 실제 보간/높이 반올림/TTA 순서 포함
training:
  recipe: scratch_comparable
  batch_size: 2
  total_optimizer_updates: 16000
  warmup_updates: 300
  optimizer: AdamW
  learning_rate: 0.0003
  weight_decay: null
  augmentation_config: null
  legacy_loss_config: null
  ema_enabled: false
loss:
  native_enabled: false
  boundary_enabled: false
evaluation:
  primary_metric: image_miou
  empty_union_policy: one
  class_reduction: macro_four
  image_reduction: equal_mean
  native_resolution: true
  d4: true
inference:
  mode: legacy_full_frame
  probability_cache_dtype: float32
  ensemble_accumulation_dtype: float64
  temperature: 1.0
  class_bias: [0.0, 0.0, 0.0, 0.0]
execution:
  dry_run: true
  overwrite_existing_outputs: false
```

### 16.3 모든 신규 기능을 기술한 후보 profile

아래는 설정 공간을 보여주기 위한 **조합 후보**다. 첫 학습은 여기서 옵션을 모두 켜는 것이 아니라 E02–E08의 single-change config들로 생성한다.

```yaml
inherit: ../resolved_legacy.yaml
run:
  name: enhanced_candidate
  seed: 0
model:
  family: enhanced_b4_unet
  new_branches:
    texture: true
    context: true
    residual_head: true
  residual_last_layer_zero_init: true
  texture_channels: [16, 32, 64]
  context_channels: [32, 64, 128, 256]
  fusion_channels: [64, 32, 16]
  new_module_norm: groupnorm
  new_module_gn_groups: 8
geometry:
  profile: native_roi_v1
  target_width: 4000
  detail_input_size: 1536
  context_input_size: 768
  context_factor: 2.0
  intensity_interpolation: bilinear
  probability_interpolation: bilinear
  align_corners: false
  antialias: false
  mask_interpolation: nearest-exact
  padding: reflect
  native_side_rounding: half_up
  scale_jitter_enabled: false
relative_features:
  sigma_native_px: [4.0, 16.0]
  eps_std: 0.01
  clip_z: 5.0
  gaussian_truncate: 3.0
  image_photometric_before_feature: true
sampling:
  image_selection: uniform
  random: 0.50
  mixed: 0.30
  hard: 0.20
  local_window_native_px: 41
  hard_cache: null              # hard>0이면 존재 여부와 사용 정책을 검사
  allow_train_in_sample_mining: true
  allow_validation_mining: false
training:
  recipe: finetune_existing
  initialize_from: null        # 대응 seed의 기존 checkpoint를 확인해 설정
  total_optimizer_updates: 4000
  stage_a_updates: 500
  stage_b_updates: 3500
  stage_a_new_lr: 0.0003
  stage_b_encoder_lr: 0.00001
  stage_b_decoder_lr: 0.00003
  stage_b_new_lr: 0.0001
  stage_b_warmup_updates: 100
  frozen_bn_during_stage_a: true
  validation_every_updates: 1000
  ema_enabled: false
  ema_decay: 0.999
loss:
  native_enabled: true
  native_weight: 0.5
  native_lovasz_weight: 0.5
  native_lovasz_per_image: true
  native_lovasz_classes: present
  probability_eps: 0.000001
  boundary_enabled: false
inference:
  mode: dual_fov_native_tiles
  native_overlap_fraction: 0.5
  blend: gaussian
  blend_weight_floor: 0.001
  d4: true
execution:
  dry_run: true
  overwrite_existing_outputs: false
```

metadata의 명시적 주석과 config 자체의 language는 읽기 쉽게 통일한다. 위 값의 `null`은 실제 코드에서 해결해야 할 항목이다. **hard cache가 없을 때에는 warmup random/mixed fallback 여부를 분명히 기록하고 cache가 생긴 뒤 전환**한다. 존재하지 않는 파일을 불러오는 dummy 경로를 만들지 않는다.

### 16.4 앙상블/refiner 설정에서 추가할 항목

```text
ensemble:
  checkpoint_manifest, fixed_member_ids, weights, all_members_required,
  class_order, space, cache_dtype, accumulation_dtype

oof:
  source_split=train, n_folds=5, fold_manifest,
  grouping_policy=image (초기값), fold_seed=2026, teacher_recipe, teacher_fixed_updates,
  teacher_seeds, teacher_inference_profile,
  strict_lineage=true

refiner:
  input_channels=8, channel_names,
  channels=[32,64,128,256], norm=groupnorm,
  zero_init_last=true, teacher_ensemble_hash,
  probability_eps=1e-6, native_crop_size=512,
  batch_size, learning_rate, optimizer_updates,
  training_image_manifest, validation_image_manifest
```

refiner 시작 레시피 후보는 batch2, AdamW lr=1e-4, 4,000 update, 원본 native crop512와 zero-init head다. 변경 없는 teacher 확률을 대조군으로 저장하고 early stopping/EMA는 다른 모델과 동일한 검증 원칙을 따른다.

---

## 17. 구현할 CLI와 명령 계약

기존 CLI가 있으면 이에 대응하는 하위 명령을 추가해도 된다. 명령 이름을 바꾸면 `reports/implementation_plan.md`에 매핑표를 기록한다. 아래 명령은 **Codex가 구현 후 작동하게 만들어야 할 요구사항**이다.

```bash
# 저장소/환경/데이터/체크포인트를 확인하고 unresolved 항목 보고
python -m alphase.cli audit --config configs/resolved_legacy.yaml

# 명세 및 cache/팔레트/shape 검사. GPU 학습을 시작하지 않음
python -m alphase.cli validate-config --config configs/experiments/E04.yaml

# 실제 1개 checkpoint와 지정 이미지로 기존 추론 회귀
python -m alphase.cli reproduce --config configs/resolved_legacy.yaml \
  --image-id A71_step2_sample1

# 소형 CPU 수학 테스트 + 선택적 실제 GPU forward/backward
python -m pytest tests -q
python -m alphase.cli smoke --config configs/experiments/E04.yaml --steps 4

# 기존 checkpoint 전체 inference/cache 계획, --execute 없으면 dry-run
python -m alphase.cli cache-probs --config configs/resolved_legacy.yaml \
  --split valid --members all
python -m alphase.cli cache-probs --config configs/resolved_legacy.yaml \
  --split valid --members all --execute

# 확률 앙상블 및 분석
python -m alphase.cli ensemble --config configs/ensemble_uniform.yaml --execute
python -m alphase.cli diagnose --prediction-manifest outputs/ensemble/manifest.jsonl

# 신규 학습. 실행 권한/자원 정책은 기존 서버 규칙을 준수
python -m alphase.cli train --config configs/experiments/E04.yaml --seed 0 --execute
python -m alphase.cli evaluate --run-dir outputs/E04_seed0 --checkpoint best

# OOF 먼저 생성/검사한 뒤 별도 보정기 학습
python -m alphase.cli build-oof --config configs/oof_train70.yaml --execute
python -m alphase.cli validate-oof --config configs/oof_train70.yaml
python -m alphase.cli train-refiner --config configs/refiner.yaml --execute

# 최종 설정으로 test 원본 크기의 PNG 생성 + 제출 파일 검사
python -m alphase.cli predict --config configs/final.yaml --split test --execute
python -m alphase.cli validate-submission --config configs/final.yaml
```

필수 동작:

- 존재하지 않는 필수 경로는 시작 전에 알려준다. 긴 job을 시작한 뒤 뒤늦게 fail하지 않도록 사전검사.
- `--execute`는 run-level `dry_run`을 명시적으로 해제한다. 모델 선택/데이터 누수 검사를 우회하지 않는다.
- 에러 exit code와 stderr를 실제 원인에 맞게 반환. broad exception을 잡아 성공처럼 종료하지 않는다.
- 완료한 cache/checkpoint는 manifest 검증 후 resume 가능. 다른 설정으로 덮어쓰기 금지.
- 지원하지 않는 feature 설정은 error. `context=true`인데 실제로 사용하지 않는 silent no-op 금지.

---

## 18. 테스트와 acceptance criteria

### 18.1 CPU에서 먼저 통과해야 할 테스트

| 테스트 | 기대 결과 |
|---|---|
| RGB→ID→RGB | 네 색상 왕복 완전 일치 |
| Unknown color | 명시적 error |
| ID/채널 순서 오류 | 확인되지 않은 cache 결합 거부 |
| A71 CM fixture | image mIoU=0.7113850263… |
| Empty union | GT/Pred 모두 없으면 1 |
| GT absent + false positive | 그 클래스 IoU=0 |
| All-ignore | 유효 평가로 처리하지 않음 |
| Image vs Global | 합산 순서가 다른 synthetic case에서 다른 값 |
| Uneven batch/DDP | image ID 수 기준 결과 일치 |
| D4 8개 | non-square tensor도 순방향/역방향 완전 복원 |
| Context alignment | native 좌표 ramp를 이용해 정의한 D grid와 수치 일치 |
| Native restore | output shape/확률합 보존, gradient finite |
| Constant relative features | 상대 밝기 ≈0, NaN 없음 |
| Non-divisible size | 홀수 H/W와 오른쪽/아래 padding 후 원래 크기 복원 |
| Tile coverage | 모든 원본 픽셀의 weight sum>0 |
| K=1 ensemble | 단일 모델 probability와 일치 |
| Duplicate identical members | 평균이 원래 prediction과 일치 |
| Corrupted cache | shape/hash/확률이 잘못되면 거부 |
| Sampler isolation | train manifest에 valid/test가 있으면 차단 |
| Mixed sampler fallback | 후보가 없는 이미지에서도 종료/정상 random 선택 |
| OOF contamination | predicted ID가 teacher train/selection/BN 목록에 있으면 거부 |
| Config validation | null 필수값/unknown key/inherit cycle 차단 |

### 18.2 실제 모델 통합 테스트

- `branches off`가 legacy 경로와 동일한 logits/probability를 생성.
- 새 head zero-init으로 base와 동일한 logits에서 시작.
- 첫 두 optimizer step에서 새 head와 앞선 branch에 finite gradient가 도달.
- encoder 동결 중 parameter와 BN running stats가 변하지 않음.
- native loss를 on하면 base/new module로 gradient가 전달되고 optimizer step이 실제 값을 바꿈.
- 작은 실제 train 패치 1–2개에 대한 overfit smoke test에서 loss가 충분히 줄어드는지 확인. 그 점수를 validation 성능으로 보고하지 않음.
- resume 전후 실제 update 수, LR, RNG/샘플 선택이 설계한 범위에서 재현.
- GPU OOM이면 조용히 입력 해상도를 축소하지 않음. batch/accumulation/checkpointing의 변경을 명시적으로 config에 기록.

### 18.3 평가·보정기 회귀 테스트

- 기존 이미지 fixture의 GT 패널/예측 패널은 분석 재현에만 사용. 새로운 추론 결과를 그 마스크로 대체하지 않음.
- OOF validator는 seed만 다른 in-sample 예측과 all-train checkpoint에서 warm-start한 가짜 OOF 예측을 거부.
- refiner의 학습/선택에 사용한 ID와 평가 ID를 artifact에 기록.
- refiner zero residual은 base probability와 clamp 오차 이내 일치.
- inference 전체 과정에서 validation GT가 없더라도 같은 probability를 생성.
- summary JSON은 NaN/Infinity를 쓰지 않고 undefined 진단값을 null로 기록.

### 18.4 제공된 reference code의 범위

이 handoff의 `reference_code/`는 **지표, D4, 좌표 정렬, native 확률 복원**의 작은 계약 검사용이다. 실제 B4 loader, 데이터셋, 학습 루프, 전체 추론기는 아니다. 서버 구현에 옮길 때 기존 코드와 함께 통합 테스트를 해야 한다.

참고 코드 테스트 통과를 실제 EfficientNet 학습/새 모델 성능 검증 통과로 바꾸어 보고하지 않는다.

---

## 19. 최종 산출물과 보고 형식

각 단계가 끝나면 다음을 갱신한다.

```text
reports/progress.md
  status: NOT_STARTED | IMPLEMENTED | TESTED | RUN_COMPLETED | BLOCKED
  changed_files
  executed_commands
  test_results
  training/evaluation_run_ids
  known_limitations
  next_action
```

최종 프로젝트 산출물:

```text
1. 기존 checkpoint를 읽는 실제 adapter와 회귀 테스트
2. 데이터/클래스/지표 검증기
3. seed ensemble cache와 스트리밍 결합기
4. native ROI/공유 D4/좌표 정렬 구현
5. 상대 밝기·context·잔차 head의 독립 on/off 모델
6. train-only mixed/hard sampler
7. gradient가 연결된 native loss
8. best/last/EMA/resume 지원 학습 루프
9. strict OOF 생성기/lineage 검사와 refiner
10. single-change config들과 실행 가능한 CLI
11. per_image.csv, per_class.csv, confusion matrices, experiments.csv
12. 원본 H/W와 네 RGB를 보존한 최종 PNG 및 제출 검사기
```

`experiments.csv`에는 최소 다음을 남긴다.

```text
experiment_id, seed, parent_checkpoint, training_recipe,
train_ids_hash, valid_ids_hash, geometry_profile, branch_flags,
sampler_config_hash, loss_config_hash, optimizer_updates,
selection_metric, selection_protocol_hash, selected_step,
checkpoint_hash, inference_config_hash,
val_image_miou, val_global_miou,
mean_iou_Al3Ni, mean_iou_EutecticSi,
A71_image_miou_diagnostic, status
```

### 19.1 최종 90장 재학습

개발이 끝나 설정이 고정되면 train+valid=90장으로 최종 teacher를 다시 학습할 수 있다. 이때 기존 valid20 점수를 독립 validation 점수라고 재사용하지 않는다.

refiner를 포함한 최종 모델이면 **90장에 대해서도 OOF teacher probability를 다시 만들고 refiner를 학습**한다. 90장을 전부 본 teacher의 90장 in-sample 확률로 refiner 학습을 대신하지 않는다. 이후 90장 teacher가 test10 확률을 만들고 동일한 refiner가 이를 보정한다.

최종 score가 없는 test10에 대해 mIoU를 임의 계산/추정해 실측값으로 기록하지 않는다.

### 19.2 제출 검증

최종 색상 PNG는 원본 test와 동일한 H/W, 정해진 4색 팔레트만 포함해야 한다. 제출 파일명/압축 규칙은 실제 대회/기존 서버 코드의 확정된 규칙을 사용한다. 색상 PNG를 JPEG로 저장하거나 최종 argmax mask에 bilinear resize를 적용하지 않는다. 확률 cache는 내부 산출물이며 제출 요구가 없는 한 제출 mask와 구분한다.

---

## 20. 이번 범위에서 하지 않을 것

- 처음부터 B4를 DINOv3/초대형 backbone으로 대체.
- A71의 정답 비율을 모든 이미지에 강제하거나 A71 전용 분기 추가.
- 공정 Si 가중치와 경계 손실을 진단 없이 대폭 증가.
- morphology opening/closing으로 작은 상을 일괄 삭제.
- 복잡한 영역을 GT entropy만 보고 inference에서 선택.
- validation/test GT를 학습 feature, sampler, normalization, refiner input으로 사용.
- 100개 seed를 OOF 100개 fold라고 취급.
- score를 올리기 위해 absent class 정의, 이미지 평균 방식, 원본 크기를 변경.
- 새 모듈을 추가했으므로 성능이 올랐다고 실험 없이 보고.

추후 유효한 기준선을 확보한 뒤 DINOv3, 추가 texture operator, local-composition head, specialist expert 등은 별도 연구 실험으로 확장할 수 있다. 이번 구현의 완료 조건으로 삼지 않는다.

---

## 21. Codex에 전달할 시작 지시문

아래 문장을 프로젝트 경로에서 실행 중인 Codex에 전달한다. 전체 명세를 `AGENTS.md`로 덮어쓰지 말고 일반 문서로 두고 읽게 한다. 기존 `AGENTS.md`가 있으면 유지한다 [S1]. 별도 파일 `CODEX_START_PROMPT.md`에도 같은 목적의 지시문을 제공한다.

```text
현재 저장소의 AGENTS.md와 기존 EfficientNet-B4 U-Net 학습/추론 코드를 먼저 읽고,
현재 저장소에서 alphase_b4_codex_spec.md를 찾아 읽고 그 명세에 따라 구현해줘.

기존 100개 seed checkpoint와 기존 결과를 보존해.
먼저 repo_audit, dataset/checkpoint manifest, resolved legacy config를 만들고
현재 Image mIoU 평가와 seed32 기존 추론을 재현해.
이어서 확률 앙상블, native ROI 기하, 상대 밝기 경로, 넓은 context 경로,
train-only 혼동 패치 sampler, native loss, best/EMA, strict OOF refiner를
명세서 단계별로 구현하고 테스트해.

핵심 클래스 순서는 Al/Al3Ni/PrimarySi/EutecticSi이고,
주지표는 원본 이미지별 4클래스 mIoU 평균이며 union=0이면 1이야.
A71 사례는 Al3Ni를 EutecticSi로 과하게 예측한 오류가 크므로,
공정 Si 가중치를 무조건 키우는 해결책으로 바꾸지 마.

새로운 geometry는 기존 4000px crop과 동일하다고 가정하지 말고 대조군을 둬.
기존 동일 train70 seed 모델의 train 예측을 OOF라고 사용하지 마.
원본 코드에서 확인 가능한 것은 먼저 확인하고, 알 수 없는 필수값만 질문해.
구현만 계획하고 끝내지 말고 실제 코드와 테스트를 작성하되,
긴 전체 sweep은 dry-run/실행 옵션을 분리하고 기존 서버 자원 규칙을 따라줘.
각 단계마다 progress.md에 변경 파일, 실행 명령, 결과와 차단 사유를 기록해.
```

---

## 22. 근거 자료와 참고 구현

### 사용자 자료

- **[U1] 현재 모델 설명:** 사용자 제공 EfficientNet-B4 U-Net, train70/valid20, 4000px 확대, 1536 crop, 16,000 updates, 100 seeds, D4, seed32 80.2265%. 서버 코드 검증 전까지 사용자 보고로 취급한다.
- **[D1] `references/a71_error_diagnosis.md`:** 첨부 그림/원본 mask에서 얻은 혼동 행렬, precision/recall, 밝기, 확대 시야 진단. 신규 학습 결과가 아니다.
- **[D2] `references/phase_analysis_final.md`:** 데이터 수량/팔레트/크기, 과거 진단 분류기 분석. 현재 B4 성능이나 신규 branch 효과를 증명하는 자료가 아니다.

단일 MD만 전달해도 핵심 계약/수치는 이 문서에 포함되어 있다. handoff ZIP에는 D1/D2와 수학 회귀 fixture를 함께 넣었다.

### 일차 출처

아래 링크는 구현 개념/API를 확인하기 위한 일차 자료다. 사용 중인 서버 버전/commit을 별도로 고정한다. 링크가 최신 main을 가리킨다고 서버 설치본과 같다고 가정하지 않는다.

```text
[S1] OpenAI, Custom instructions with AGENTS.md
https://developers.openai.com/codex/guides/agents-md

[S2] PyTorch functional.py — interpolate 관련 구현/문서
https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py

[S3] PyTorch functional.py — grid_sample 관련 구현/문서
https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py

[S4] segmentation_models.pytorch — U-Net decoder
https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/decoders/unet/decoder.py

[S5] Hoyer et al., HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation
https://arxiv.org/abs/2204.13132

[S6] Berman et al., The Lovasz-Softmax loss
https://arxiv.org/abs/1705.08790

[S7] Lovasz-Softmax 저자 구현
https://github.com/bermanmaxim/LovaszSoftmax

[S8] Wu and He, Group Normalization
https://arxiv.org/abs/1803.08494

[S9] PyTorch swa_utils.py — AveragedModel / EMA / BN update
https://github.com/pytorch/pytorch/blob/main/torch/optim/swa_utils.py

[S10] Cawley and Talbot, On Over-fitting in Model Selection and Subsequent Selection Bias
https://www.jmlr.org/papers/v11/cawley10a.html
```

**완료 정의:** 작동하는 코드와 재현 가능한 평가 경로를 구축하고, 개선 여부는 실제 full-image validation 실험으로 결정한다. 이 명세서는 성능 향상을 보장하거나 미실행 실험을 완료로 대신하지 않는다.
