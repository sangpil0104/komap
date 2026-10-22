# mIoU 기준별 최고 성능 모델

## 평가 기준

- 대상: validation 20장
- 내부 클래스 순서: `[Al3Ni, Primary Si, Al, Eutectic Si]`
- `global mIoU`: validation 전체 픽셀의 confusion matrix를 합산한 뒤 클래스별 IoU를 평균
- `image-wise 평균 mIoU`: 각 이미지에서 4개 클래스 IoU를 먼저 평균한 뒤 20장 평균
- 아래 순위는 **validation 정답을 사용하지 않는 실제 추론 route**만 포함한다. 정답으로
  이미지별 선택을 한 oracle 결과는 별도로 표시한다.

## 공통 모델 구조와 표기

최고 성능 route의 핵심 네트워크는 모두 `ResNet34 encoder + U-Net++ decoder + residual ASPP`
구조다. 입력은 RGB로 읽지만 이 데이터에서는 세 채널의 현미경 명암 정보가 사실상 같은
정보를 가지므로, 회색조 통계는 별도의 intensity prior 계산에도 사용한다.

### ASPP-U-Net++ 구조

1. ResNet34 encoder가 `stem → enc1 → enc2 → enc3 → enc4`로 feature pyramid를 만든다.
2. 가장 깊은 512-channel feature에 residual ASPP bottleneck을 넣는다. ASPP는
   `1×1`, dilation `6/12/18`인 3×3 convolution, global image-pooling branch의
   5개 출력을 concat하고 1×1 projection으로 되돌린다.
3. ASPP 출력은 U-Net++의 nested skip decoder로 복원한다. decoder는 같은 해상도에서
   이전 decoder 출력과 encoder skip을 누적해 `x0_1 → x0_2 → x0_3`로 dense하게 결합한다.
4. 마지막에는 최상위 decoder feature와 stem feature를 concat해 4-class logit을 만든다.
   ASPP residual scale은 0에서 시작해 기존 ResNet34-U-Net++ checkpoint와의 호환성을
   유지한다.

### 앙상블 branch 표기

아래에서 `P0, P1, P2`는 기존 native-patch branch, `Q`는 pairwise-margin 계열 ASPP
branch, `H`는 high-resolution ASPP branch다.

| branch | 기반 checkpoint/역할 | patch / stride | 비고 |
|---|---|---:|---|
| `P0` | model98 계열 hard-negative fine-tune | 512 / 384 | 희소·어려운 클래스에 대한 supervised 보강 |
| `P1` | model99 계열 boundary-refined fine-tune | 512 / 384 | Al 경계 보존을 위한 boundary 보강 |
| `P2` | model101 계열 고해상도 branch | 640 / 512 | 작은 상·하위상 및 texture 보완 |
| `Q` | pairwise-margin을 적용한 ASPP checkpoint | 768 / 576 | 경계·클래스 간 경쟁 보완용 고정 branch |
| `H768` | source-routed ASPP checkpoint | 768 / 576 | 현재 기본 high-resolution context |
| `H1024` | 같은 source/모델의 큰 문맥 추론 | 1024 / 768 | 큰 영역 문맥을 추가하는 보조 branch |

각 patch 추론은 8개 D4 변환(회전 0/90/180/270°와 horizontal flip)을 평균하고,
overlap 영역은 Gaussian window로 가중 평균한다. 이후 원본 해상도로 bilinear 복원하고
클래스 확률의 합을 다시 1로 정규화한다.

## 1. Global mIoU 최고: 모델 178

| 항목 | 내용 |
|---|---|
| 모델 | **178번 style-TENT route** |
| 구성 | 모델 173의 전체 route 고정 + `H`의 style-tile TENT + `H768/H1024` 혼합 |
| 선택 기준 | global mIoU |
| `H1024` blend | `α=0.20` (`H = 0.80·H768 + 0.20·H1024`) |
| Global mIoU | **80.9780%** |
| Image-wise 평균 mIoU | 78.1895% |
| 결과 파일 | `/workspace/komap_model1/outputs/model178_tta_modes/results.json` |

### 모델 178의 상세 처리 순서

모델 178은 새로운 supervised checkpoint를 학습한 모델이 아니라, 모델 173의 inference
route 중 **high branch의 무라벨 test-time adaptation 입력만 바꾼 비교 모델**이다.

1. `P0/P1/P2`, `Q`, source routing, intensity prior, bias, Primary cleanup은 모델
   173과 완전히 동일하게 고정한다.
2. validation/test 한 장을 train set의 평균 명암과 표준편차로 standardize해 style image를
   만든다.
3. high branch checkpoint의 BN layer에서 affine parameter(`weight`, `bias`)만 trainable로
   열고 convolution weight는 고정한다. style 768px corner tiles의 예측 entropy를
   최소화하는 TENT를 5 step, Adam `lr=1e-5`로 적용한다. ASPP image-pooling의 1×1 BN은
   불안정한 통계 갱신을 막기 위해 eval 상태로 둔다.
4. 적응된 모델로 raw/style 입력을 각각 768px와 1024px로 D4-TTA 추론하고,
   `α=0.20`으로 두 spatial context를 혼합한다.
5. 최종 확률은 아래 route로 결합한다.

   ```text
   base_c = Σ_m (M[c,m] · (1 - high_weight - pair_weight)) Pm[c]
   mixed_c = base_c + high_weight · H[c] + pair_weight · Q[c]
   ```

   여기서 class-conditional matrix `M`은 다음과 같다.

   ```text
   M = [[0.35, 0.40, 0.25],   # Al3Ni
        [0.25, 0.35, 0.40],   # Primary Si
        [0.50, 0.30, 0.20],   # Al
        [0.25, 0.55, 0.20]]   # Eutectic Si
   ```

6. 최종 logit에는 train-only intensity prior와 class bias를 더한다. style-TENT 실험에서
   global mIoU는 상승했지만 이미지별 평균은 모델 173보다 낮아, 대회 지표가 image-wise
   평균이라면 모델 178을 그대로 채택하지 않는다.

### 모델 178에서 비교한 TENT 입력

| 적응 입력 | 최적 `α` | Global mIoU | Image-wise 평균 | 판정 |
|---|---:|---:|---:|---|
| raw tiles | 0.20 | 80.8393% | 78.2317% | image-wise는 173 유지 |
| **style tiles** | **0.20** | **80.9780%** | 78.1895% | global 기준 최고 |
| raw + style joint | 0.20 | 80.5985% | 77.9475% | 미채택 |

## 2. Image-wise 평균 mIoU 최고: 모델 173

| 항목 | 내용 |
|---|---|
| 모델 | **173번 high-resolution blend route** |
| 구성 | 모델 171 hard-source routing + `H768/H1024` context blend |
| 선택 기준 | image-wise 평균 mIoU |
| `H1024` blend | `α=0.20` (`H = 0.80·H768 + 0.20·H1024`) |
| Global mIoU | 80.8435% |
| Image-wise 평균 mIoU | **78.2344%** |
| 결과 파일 | `/workspace/komap_model1/outputs/model173_highres1024/results.json` |

### 모델 173의 상세 처리 순서

모델 173은 현재 대회 지표를 image-wise 평균 mIoU로 해석할 때의 기준 제출 route다.
핵심은 하나의 checkpoint만 사용하는 대신, 서로 다른 patch scale·학습 목적·문맥을 가진
ResNet34-ASPP branch를 확률 공간에서 결합하는 것이다.

#### 1) high source의 밝기 조건부 선택

입력 원본의 grayscale 평균 `mean`을 이용해 `H`의 checkpoint를 고른다. 이는 validation
정답이나 test 정답을 사용하지 않는 입력 기반 routing이다.

| grayscale mean 구간 | high source |
|---|---|
| `[159.0, 161.0)` | pairwise source |
| `[161.3, 162.0)` | EMA source |
| `[174.6, 175.1)` | model109 native ASPP source |
| `[152.3, 152.7)` | seed456 ASPP source |
| `[155.2, 155.7)` | EMA source |
| 그 외 | restart51 source |

각 source는 동일한 ASPP-U-Net++ 계열이지만 학습 seed, hard-case fine-tune, EMA 여부가
달라 특정 명암/시편 조건에서 서로 다른 오류를 낸다. source를 선택한 뒤에는 그 source의
high branch를 raw corner tile 대상으로 5-step TENT(`lr=1e-5`)로 적응한다.

#### 2) raw/style spatial-context branch

- `H768`: patch 768, stride 576, D4-TTA + Gaussian overlap blending
- `H1024`: patch 1024, stride 768, D4-TTA + Gaussian overlap blending
- 두 branch는 같은 적응 checkpoint를 사용하며 patch 크기만 다르다.
- grayscale 표준편차가 `50` 이상인 이미지에서는 raw high prediction과 train-stat
  standardize 입력 prediction을 `0.25 : 0.75`로 섞는다. 그 외에는 raw prediction을
  사용한다.
- 최종 high prediction은 `0.80·H768 + 0.20·H1024`다.

#### 3) brightness-conditioned ensemble calibration

입력 평균 명암에 따라 high branch의 영향력을 조절하고, pairwise branch를 항상 일부
남긴다.

- 기본 high weight: mean `<165`이면 `0.85`, 그 외 `0.55`
- 기본 pairwise weight: `0.10`
- 예외적인 `[159,161)` 구간: high `0.70`, pairwise `0.05`, style weight `1.0`
- Al logit bias: `+0.45`
- Eutectic Si bias: 명암 구간에 따라 `-2.0`, `-0.75`, `+0.3`, `-1.0`, `-0.3` 중
  하나를 적용

Al3Ni는 학습 데이터에서 구한 class-conditioned brightness와 현재 이미지의 예측
brightness 차이를 이용해 adaptive prior를 적용한다. prior는 grayscale z-score
Gaussian prior `85%`와 percentile prior `15%`를 섞고, 예측 분포로 다시 추정한
adaptive Al3Ni prior 차이에 brightness-conditioned beta를 곱한다. 이 prior를
`0.09`로 logit에 더한다. beta는 일반적으로 `7`, 저명도 구간 `3`, 일부 narrow
hard-case 구간 `15`다.

#### 4) 안정화 후처리

- argmax 전에 확률을 logit으로 변환해 prior/bias를 적용한다.
- Primary Si(class 1)의 connected component가 `128 px 이하`이면 해당 픽셀을
  다음으로 높은 class로 재할당한다.
- 최종 Primary Si 면적 비율이 `1% 이하`이면 남은 Primary Si도 다른 class 중
  최대 logit class로 재할당한다. 이는 Primary가 없는 A71-like image의 작은
  false-positive를 줄이기 위한 공통 정책이다.

## 두 최고 모델의 선택 기준

| 사용 목적 | 권장 route | 이유 |
|---|---|---|
| 대회 점수가 image-wise 평균 mIoU | **모델 173** | image-wise 평균 78.2344%로 현재 최고 |
| 전체 픽셀을 합친 global mIoU | **모델 178 style-TENT** | global mIoU 80.9780%로 현재 최고 |

두 수치는 같은 mIoU라는 이름을 쓰지만 집계 순서가 달라 최적 route가 다르다. 특히
global mIoU는 픽셀이 많은 큰 영역(주로 Al)의 영향이 커지고, image-wise 평균은 작은
영역과 어려운 이미지도 동일한 이미지 비중으로 반영한다.

## 참고: 실제 모델이 아닌 oracle 진단

172번의 이미지별 class-bias oracle은 validation 정답으로 이미지별 bias를 고른
상한 진단이다.

- Global mIoU: 81.3526%
- Image-wise 평균 mIoU: 78.6988%
- validation 정답을 사용하므로 제출 모델 성능으로 기록하지 않음

179번 train-only specimen-aware prior는 `rho=0`이 최적이어서 모델 173과 동일한
80.8435% / 78.2344%였다. 따라서 specimen prior를 추가했다고 성능이 향상된 것으로
기록하지 않는다.

## 재현 및 해석 주의

- 결과 파일의 validation 점수는 현재 validation 20장에 대한 수치다.
- 166~173번의 일부 narrow brightness route는 동일 validation을 기준으로 선택했으므로
  독립적인 일반화 성능의 증거가 아니다.
- test mask에는 정답이 없으므로 test에서는 점수를 직접 계산할 수 없다. 제출용 mask를
  만들 때는 위 route의 checkpoint, branch weight, TTA, 후처리를 모두 동일하게 유지해야
  한다.
