# Model394: train-only apparent-scale association screen

## 목적과 해석

Model393의 train-mask 탐색에서 Eutectic-Si 및 Al3Ni의 pixel-width proxy와 원본 래스터 대각선 사이에 group-level 상관이 관찰됐다. 이를 보고 나서 세운 **사후 탐색 가설**이다. 단일 label-free 이미지 스케일 추정량이 raster bin과 phase fraction을 통제한 뒤에도 train-mask width proxy와 연관되는지 CPU로 검사한다. 이 결과는 물리 배율 복원, segmentation 성능, annotation uncertainty, 또는 augmentation 효과를 입증하지 않는다.

분석은 `train.zip`의 train images만 pixel 입력으로 사용하고, 마스크 기하·phase fraction은 이미 잠긴 Model393 JSON에서 가져온다. valid/test 이미지·라벨, OOF 확률, checkpoint, 모델 점수는 읽지 않는다. 학습을 실행하거나 자동 승인하지 않는다. 이 후속 분석은 Model393 전체 train-mask 결과를 본 뒤 설계했으므로 어떤 p-value도 confirmatory 증거가 아니다.

## 고정 입력 스케일 추정기

1. PNG를 RGB에서 고정 luminance 식으로 grayscale 변환한다. 이미지 resize/normalization은 하지 않는다.
2. 각 이미지에서 320×320 크기의 네 모서리 및 중앙 crop을 사용한다(중복 crop은 허용). 각 crop에서 Gaussian blur(σ=16 px)를 빼고 separable Hann window를 적용한다.
3. 2D FFT power를 radial frequency bin으로 평균한다. wavelength `2.5–64 px` 범위에서 power가 최대인 bin의 역수를 해당 crop의 apparent wavelength로 둔다.
4. 다섯 crop wavelength의 중앙값을 이미지별 단일 scale feature로 사용한다. 모수·대역·crop 위치는 결과를 보고 변경하지 않는다.

구현 타당성은 train 데이터와 무관한 주기 16 px 합성 sinusoid 및 0.5×/2× raster 변환 smoke로 확인한다. 원본 16 px 패턴의 추정 오차와 변환 후 wavelength 비례성 허용치는 각각 10%, 15%로 고정한다. 이 smoke는 estimator 구현만 확인하고 미세조직 추정의 타당성을 증명하지 않는다.

## 사전 지정 분석

- image-level 값은 specimen group 안에서 평균해 59개의 독립 group row를 만든다. 검증된 각 group은 단 하나의 원본 래스터 크기에 속한다.
- primary target은 Eutectic-Si의 Model393 per-image doubled-distance-transform q50 width proxy를 group 평균한 값이다. Al3Ni는 secondary descriptive target이다.
- 각 target에 대해 feature와 width를 각각 rank-transform한다. nuisance design은 intercept, 원본 raster-size bin indicator, 해당 phase의 group-mean pixel fraction rank다. nuisance를 제거한 두 rank residual 사이 Pearson correlation을 partial-rank association으로 보고한다.
- 한쪽 대립가설은 apparent wavelength가 클수록 mask width proxy도 크다는 것이다. Freedman–Lane 방식으로 reduced model residual을 raster bin 안에서 group 단위 permutation하여 10,000회 one-sided p-value를 계산한다. Group bootstrap 5,000회 percentile 95% interval도 보조로 보고한다. Seed는 각각 39410, 39420으로 고정한다.
- primary Eutectic-Si 결과만 사전 지정 대상이다. Al3Ni는 기전 비교용 보조 분석이며 별도 유의성 주장은 하지 않는다. 효과 크기와 bin별 산점도를 함께 보고하며 이 audit에서 학습 GO를 내리지 않는다.
- 이미지 scale feature와 mask width 사이에서 raster bin 및 phase fraction 통제 후 positive association이 보이지 않으면 이 apparent-scale-normalization 가설은 보류한다. association이 보여도 이는 train 분포 안의 proxy 관계만 뜻한다. 추후 OOF 오류 분석과 별도 사전등록 없이 모델 입력 배율을 바꾸지 않는다.

## 누수·한계

이미지 feature 계산에서 mask를 사용하지 않는다. 단, train-mask geometry를 이미 살펴본 뒤 이 가설을 골랐으므로 결과 선택 편향은 남는다. 네 raster bin의 group 수가 불균형하고, bin과 획득 조건·물리적 시야·시편 형태가 얽힐 수 있다. phase fraction rank 통제는 연결성·형상 등 비선형 morphology 혼란을 제거하지 못한다. 1 px dilation/erosion/shifts는 annotation noise의 추정으로 사용하지 않는다.

## 재현 자산

- 분석: `experiments/model394_image_scale_mask_association.py`
- synthetic-only smoke: `experiments/model394_image_scale_mask_association_smoke.py`
- immutable lock 생성기: `experiments/create_model394_image_scale_mask_association_lock.py`
- lock receipt: `experiments/results/model394_image_scale_mask_association_lock.json`
- 출력: `experiments/results/model394_image_scale_mask_association.json`
