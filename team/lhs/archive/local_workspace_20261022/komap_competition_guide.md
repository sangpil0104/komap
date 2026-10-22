# KoMaP 대회 준비 가이드

## 문서 목적

이 문서는 KoMaP 현미경 이미지 분할 대회를 준비하기 위한 실전용 정리 자료다. 대회의 문제 구조를 이해하고, 작은 데이터셋에서 안정적으로 성능을 올릴 수 있는 모델·학습·검증 전략을 순서대로 적용하는 것을 목표로 한다.

핵심 메시지는 다음과 같다.

> KoMaP에서는 가장 최신인 모델 하나를 고르는 것보다, 작은 데이터에 맞는 사전학습·증강·검증·앙상블 실험 구조를 만드는 것이 더 중요하다.

---

## 1. 대회 개요

KoMaP은 금속 소재의 현미경 이미지를 입력으로 받아 이미지 안의 여러 상(phase)을 픽셀 단위로 분할하는 semantic segmentation 대회다. 각 픽셀에 정답 클래스를 부여하고, 예측 마스크와 정답 마스크가 얼마나 잘 겹치는지를 평가한다.

대회 준비에서 특히 중요한 조건은 다음과 같다.

- 전체 이미지 수가 매우 적은 소규모 데이터셋이다.
- 학습용 이미지가 약 70장 수준이므로 일반적인 딥러닝 기준으로는 과적합 위험이 매우 크다.
- 단순한 이미지 분류가 아니라 픽셀 단위 예측이 필요하다.
- 큰 영역뿐 아니라 상대적으로 작은 상과 경계도 잘 분리해야 한다.
- 모델 구조보다 데이터 분할, 증강, 사전학습, 검증 설계가 결과에 큰 영향을 줄 수 있다.

따라서 처음부터 복잡한 최신 모델을 무작정 학습하기보다는, 재현 가능한 기준선을 만든 뒤 한 가지 요소씩 비교하는 방식이 적합하다.

---

## 2. 문제 정의와 평가 지표

### 2.1 문제 정의

입력은 금속 미세조직을 촬영한 현미경 이미지이고, 출력은 각 픽셀에 대한 4개 클래스의 예측 마스크다.

주요 클래스는 다음과 같이 정리할 수 있다.

- Primary Si
- Al₃Ni
- Eutectic Si
- Al

모델은 각 픽셀마다 네 클래스 중 하나를 예측한다. 일반적인 출력 형태는 `H × W × 4` 로짓 또는 확률 맵이며, 최종 클래스는 채널 방향의 argmax로 결정한다.

### 2.2 평가 지표: IoU와 mIoU

클래스별 IoU는 예측 영역과 정답 영역의 겹침 정도를 나타낸다.

- IoU = 교집합 / 합집합
- mIoU = 각 클래스 IoU의 평균

mIoU를 사용하면 큰 배경 영역만 잘 맞추는 모델보다, 작은 상까지 고르게 맞추는 모델이 유리하다. 따라서 전체 픽셀 정확도만 보고 모델을 선택하면 안 된다.

검증 시에는 최소한 다음을 함께 기록한다.

- 전체 mIoU
- 클래스별 IoU
- 클래스별 Dice 또는 F1
- confusion matrix
- 이미지별 점수 분포

특히 클래스별 IoU가 한 클래스에서만 크게 떨어진다면, 모델 전체를 바꾸기 전에 해당 클래스의 크기·경계·라벨 품질·샘플 부족 문제를 먼저 확인해야 한다.

---

## 3. 데이터 특성과 핵심 난점

### 3.1 매우 적은 학습 이미지

학습 이미지가 약 70장이면 이미지 단위로 볼 때 모델이 기억할 수 있는 정보가 많지 않다. 고성능 backbone을 처음부터 학습하면 학습 데이터에서는 점수가 빠르게 올라가도 검증 성능이 떨어질 수 있다.

가장 중요한 대응은 다음 세 가지다.

1. ImageNet 또는 관련 도메인에서 사전학습된 encoder를 사용한다.
2. 이미지 단위가 아니라 patch 단위로 다양한 학습 샘플을 만든다.
3. 단일 validation split에 의존하지 않고 교차검증으로 일반화 성능을 확인한다.

### 3.2 소재 이미지의 질감과 경계

금속 미세조직은 색상만으로 클래스가 명확히 구분되지 않을 수 있다. 질감, 밝기, 입자 형태, 주변 상과의 관계가 함께 중요하다. 경계가 흐리거나 얇은 구조가 있으면 downsampling 과정에서 정보가 사라질 수 있다.

따라서 다음을 신경 써야 한다.

- decoder의 고해상도 skip connection
- 경계가 잘 보존되는 patch 크기
- 너무 강한 blur나 색상 변형의 제한
- 작은 객체가 포함된 crop의 충분한 확보

### 3.3 클래스 불균형

배경 또는 넓은 영역의 Al 클래스가 대부분을 차지하고, 일부 상은 매우 작은 영역만 차지할 가능성이 있다. 이 경우 일반적인 cross entropy만 사용하면 큰 클래스에 유리한 예측이 될 수 있다.

대응 방법은 class weight, Dice 계열 loss, focal 계열 loss, 작은 상 중심 crop, hard example mining 등이다. 다만 가중치를 과도하게 주면 작은 클래스의 false positive가 늘 수 있으므로 클래스별 IoU를 보면서 조정해야 한다.

### 3.4 검증 분할의 불안정성

이미지가 적으므로 validation에 어떤 이미지가 들어갔는지에 따라 점수가 크게 흔들릴 수 있다. 한 번의 split에서 높은 점수를 얻었다고 해서 실제 테스트 성능이 높다고 단정하면 안 된다.

가능하면 stratified한 기준을 세워 K-fold 교차검증을 수행하고, 최종적으로는 fold별 평균과 표준편차를 함께 본다.

---

## 4. 필요한 배경지식

### 4.1 Semantic segmentation 기본

Semantic segmentation은 이미지의 모든 픽셀에 클래스를 부여하는 문제다. object detection처럼 박스만 예측하는 것이 아니며, instance segmentation처럼 같은 클래스의 개체를 각각 분리하는 것도 아니다.

학습에서 이해해야 할 개념은 다음과 같다.

- encoder: 입력 이미지에서 특징을 추출하고 해상도를 줄이는 부분
- decoder: 특징을 다시 원래 해상도로 복원하는 부분
- skip connection: encoder의 세부 공간 정보를 decoder에 전달하는 연결
- logits: softmax 이전의 클래스별 출력값
- mask: 픽셀별 정답 또는 예측 클래스 지도

### 4.2 U-Net 구조

U-Net은 encoder-decoder 구조와 skip connection을 결합한 대표적인 segmentation 모델이다. 데이터가 적고 공간적 세부 정보가 중요한 현미경 이미지에서 여전히 강력한 기준선이다.

최근에는 U-Net 전체를 새로 학습하기보다, encoder만 ResNet·ConvNeXt 같은 사전학습 backbone으로 교체해 사용하는 경우가 많다.

### 4.3 사전학습과 전이학습

사전학습된 encoder는 이미 엣지, 질감, 형태와 같은 기본 시각 표현을 학습하고 있다. 작은 데이터셋에서는 이 표현을 활용하는 것이 처음부터 모든 가중치를 학습하는 것보다 안정적이다.

초기 실험에서는 encoder를 일부 동결한 뒤 decoder를 학습하고, 이후 learning rate를 낮춰 encoder까지 미세조정하는 방식이 유용하다.

### 4.4 Loss 함수

기본 선택지는 cross entropy다. 여기에 Dice loss 또는 IoU 계열 loss를 더하면 클래스 불균형과 영역 겹침을 직접 반영할 수 있다.

실전에서는 다음 조합부터 시작하는 것이 좋다.

- Cross Entropy
- Cross Entropy + Dice
- Weighted Cross Entropy + Dice
- Focal + Dice: 작은 클래스 또는 어려운 픽셀의 비중을 높이고 싶을 때

Loss 자체의 값보다 validation mIoU와 클래스별 지표를 기준으로 선택해야 한다.

---

## 5. 최근 semantic segmentation 동향

### 5.1 강한 CNN baseline의 지속적인 유효성

U-Net, DeepLabV3+, HRNet, ConvNeXt 기반 모델은 여전히 의료·현미경·소재 이미지에서 강한 기준선으로 사용된다. 데이터가 적을 때는 대형 Transformer보다 CNN 기반 모델이 더 안정적으로 수렴할 가능성이 높다.

### 5.2 Transformer와 hybrid 구조

SegFormer와 Swin 계열은 넓은 문맥과 장거리 관계를 모델링하는 장점이 있다. 다만 데이터가 매우 적은 경우에는 사전학습과 강한 regularization이 중요하며, 모델 크기가 클수록 과적합을 주의해야 한다.

### 5.3 Foundation model과 self-supervised representation

DINOv2, SAM 계열과 같은 foundation model은 많은 이미지에서 학습한 표현을 활용한다는 점에서 매력적이다. 그러나 범용 이미지로 학습된 표현이 금속 미세조직에 바로 최적화된다고 보기는 어렵다.

KoMaP처럼 라벨은 적지만 관련 도메인의 unlabeled 이미지가 더 있다면, domain-specific self-supervised pretraining이나 feature extraction이 큰 도움이 될 수 있다.

### 5.4 Mamba와 state-space 계열

Mamba 기반 segmentation 모델은 긴 문맥을 효율적으로 처리하려는 방향이다. 흥미로운 연구 주제이지만, KoMaP의 첫 실험으로 선택하기에는 구현·재현성·검증 부담이 크다. 안정적인 기준선 이후 2차 실험군으로 두는 것이 적절하다.

### 5.5 최신 모델보다 중요한 것

최근 흐름을 KoMaP에 적용할 때 핵심은 다음과 같다.

- 최신 구조 자체보다 강한 pretrained representation을 우선한다.
- 도메인과 맞지 않는 augmentation은 줄인다.
- 큰 모델을 쓰더라도 cross-validation과 seed 반복으로 안정성을 확인한다.
- 모델 하나의 점수보다 서로 다른 오류를 내는 모델의 조합을 고려한다.

---

## 6. KoMaP에 적합한 모델 5개

아래 다섯 모델은 성능 비교와 실험 목적이 분명해 KoMaP의 1차 후보로 적합하다.

### 6.1 U-Net + ResNet34

가장 먼저 구축할 기준선이다.

```text
입력 이미지
  ↓
사전학습 ResNet34 encoder
  ↓
U-Net decoder
  ↓
4-class mask
```

장점:

- 구현이 쉽고 안정적이다.
- skip connection 덕분에 공간 정보 복원이 쉽다.
- 다른 모델과 비교할 기준점이 된다.
- 작은 데이터셋에서 과도하게 무겁지 않다.

이 모델의 목적은 반드시 최고 점수를 내는 것이 아니라, 데이터와 전처리의 기본 성능을 확인하는 것이다.

### 6.2 U-Net + ConvNeXt-Tiny

ConvNeXt-Tiny를 encoder로 사용하는 현대적인 CNN baseline이다. ResNet보다 풍부한 표현을 기대할 수 있으며, 소재 이미지의 질감과 패턴을 포착하는 데 유리할 가능성이 있다.

주의할 점:

- 모델이 더 무거우므로 learning rate와 weight decay를 조절한다.
- validation 성능이 일찍 떨어지는지 확인한다.
- pretrained 입력 정규화와 입력 채널 구성을 정확히 맞춘다.

### 6.3 SegFormer-B0 또는 B2

SegFormer는 효율적인 Transformer encoder와 segmentation decoder를 결합한 모델이다.

- B0: 가볍고 빠른 비교군
- B2: 더 큰 표현력, 더 높은 과적합 위험

처음에는 B0로 구조를 확인한 뒤, B2가 교차검증 평균을 개선하는지 확인하는 순서를 권장한다. 단일 split에서 B2가 높더라도 fold 간 편차가 크면 최종 모델로는 부적합할 수 있다.

### 6.4 DINOv2 + segmentation decoder

DINOv2 encoder에서 추출한 강한 visual representation 위에 가벼운 segmentation decoder를 붙이는 방식이다. U-Net decoder, linear head, FPN 계열 decoder 등을 비교할 수 있다.

장점:

- 라벨이 적을 때 강한 표현을 활용할 수 있다.
- encoder를 고정하고 decoder만 학습하는 실험이 가능하다.
- backbone fine-tuning 여부를 단계적으로 비교할 수 있다.

주의할 점:

- DINOv2의 feature stride와 decoder 입력 형상을 맞춰야 한다.
- 범용 이미지 표현이 금속 미세조직에 충분히 맞지 않을 수 있다.
- 메모리 사용량과 입력 해상도를 확인해야 한다.

### 6.5 SAM2 계열 adaptation

SAM2 계열의 범용 segmentation 표현을 KoMaP의 4-class semantic segmentation 문제에 맞게 adaptation하는 방향이다. 원래 SAM 계열은 prompt 기반 또는 범용 mask 생성 목적이므로, 단순히 모델을 가져오는 것만으로는 대회 출력 형식과 맞지 않을 수 있다.

가능한 접근:

- image encoder feature를 추출하고 별도 semantic decoder 학습
- prompt 또는 point/box 정보를 활용한 보조 실험
- encoder 일부 동결 후 task-specific head만 학습

우선순위는 ResNet·ConvNeXt·SegFormer보다 낮게 두는 것이 좋다. 구현 복잡도와 domain mismatch를 감안해, 기준선이 확보된 뒤 실험한다.

---

## 7. 2차 후보 모델

### 7.1 U-Net++

U-Net의 skip connection을 더 촘촘하게 구성해 다양한 해상도의 특징을 결합한다. 경계 복원에 도움이 될 수 있지만, 파라미터와 메모리 사용량이 증가한다.

### 7.2 DeepLabV3+

ASPP를 통해 다양한 receptive field의 문맥을 활용한다. 상의 크기가 다양하고 넓은 문맥이 중요한 경우 유용할 수 있다.

### 7.3 HRNet

고해상도 표현을 계속 유지하는 구조다. 얇은 경계와 작은 구조를 보존하는 데 기대할 수 있지만, 메모리와 구현 부담을 고려해야 한다.

### 7.4 Mamba-UNet 계열

state-space 기반으로 장거리 문맥을 처리하려는 모델이다. 최신 연구 흐름을 확인하는 데는 좋지만, 소규모 대회에서 최우선으로 두기보다는 충분한 기준선과 재현 환경을 만든 후 시도한다.

---

## 8. 실전 학습 전략

### 8.1 Augmentation

현미경 이미지에서는 좌우 반전, 상하 반전, 90도 회전이 비교적 안전한 기본 증강이다. 그 외에는 이미지의 물리적 의미를 훼손하지 않도록 조심해야 한다.

권장 후보:

- horizontal flip
- vertical flip
- 90도 단위 회전
- 작은 각도의 rotation
- 약한 scale 또는 crop
- 약한 brightness/contrast 조정
- 제한적인 Gaussian noise
- random resized crop

주의할 증강:

- 과도한 색상 변형
- 지나치게 강한 blur
- 실제 미세조직에서 나타나기 어려운 기하 변형
- mask 경계를 깨뜨리는 interpolation

증강을 많이 넣는 것보다, 각 증강이 실제 테스트 이미지의 변동을 설명하는지 확인하는 것이 중요하다.

### 8.2 Loss와 class imbalance

첫 실험은 `Cross Entropy + Dice` 조합으로 시작한다. 이후 클래스별 IoU가 낮은 경우 다음 순서로 변경한다.

1. 클래스 빈도를 확인한다.
2. weighted cross entropy를 비교한다.
3. Dice 또는 focal 성분의 비중을 조절한다.
4. 작은 클래스가 포함된 crop 비율을 높인다.

가중치는 한 번에 크게 바꾸지 말고, 각 클래스 IoU와 false positive를 함께 확인한다.

### 8.3 Patch training

원본 이미지 전체를 그대로 학습하는 것보다 여러 patch를 잘라 사용하면 실질적인 학습 샘플 수를 늘릴 수 있다.

권장 방법:

- 고정 크기 patch를 겹치게 추출
- foreground 비율이 일정 이상인 patch를 우선 샘플링
- 작은 클래스가 포함된 patch를 일부 oversampling
- validation과 test에서는 동일한 타일링 규칙을 사용

patch가 너무 작으면 전체 조직 문맥을 잃고, 너무 크면 batch 수와 메모리 효율이 나빠진다. 여러 크기를 비교하되, 모델별로 동일한 조건에서 평가한다.

### 8.4 Cross-validation

소규모 데이터에서는 5-fold 또는 4-fold 교차검증이 유용하다. 같은 원본 이미지에서 나온 patch가 train과 validation에 동시에 들어가면 누수가 발생할 수 있으므로, 먼저 원본 이미지 단위로 fold를 나눈 뒤 patch를 생성해야 한다.

기록할 내용:

- fold별 mIoU
- 평균 mIoU와 표준편차
- 클래스별 평균 IoU
- seed별 변동
- 최적 epoch 분포

### 8.5 TTA

Test Time Augmentation은 입력에 flip이나 회전을 적용해 여러 번 예측한 뒤, 예측 확률을 원래 방향으로 되돌려 평균하는 방식이다.

먼저 다음 정도만 시험한다.

- 원본
- 좌우 반전
- 상하 반전
- 90도 회전

TTA는 단독으로 큰 개선을 보장하지 않으므로, validation에서 실제로 개선되는 조합만 사용한다.

### 8.6 Ensemble

앙상블은 서로 다른 모델 또는 fold의 예측을 결합하는 전략이다. 단순히 점수가 가장 높은 모델 여러 개를 섞는 것보다, 오류 패턴이 다른 모델을 조합하는 것이 중요하다.

가능한 조합:

- ResNet34 U-Net + ConvNeXt U-Net
- CNN 모델 + SegFormer
- 여러 cross-validation fold의 softmax 확률 평균
- 모델별 확률 평균 후 argmax

가능하면 hard voting보다 확률을 평균하는 soft ensemble을 우선한다. 모델 간 calibration 차이가 있으면 validation에서 가중치를 조정한다.

---

## 9. 권장 실험 순서

### 1단계: 데이터와 평가 코드 검증

- 이미지와 마스크의 크기·채널을 확인한다.
- 클래스 ID 매핑을 확인한다.
- 마스크 interpolation이 nearest neighbor인지 확인한다.
- IoU와 mIoU 계산 결과를 작은 예제로 검증한다.
- 시각화로 입력·정답·예측을 나란히 확인한다.

### 2단계: 가장 단순한 기준선

- U-Net + ResNet34
- 기본 augmentation
- Cross Entropy 또는 Cross Entropy + Dice
- 고정된 validation split

이 단계에서 데이터 파이프라인, 학습률, batch size, 추론 코드를 안정화한다.

### 3단계: patch와 loss 개선

- patch 크기 비교
- foreground-aware sampling
- class weight 비교
- Dice 비중 조정
- 작은 클래스의 시각적 오류 분석

### 4단계: backbone 비교

동일한 split·augmentation·학습 예산으로 다음을 비교한다.

1. ResNet34
2. ConvNeXt-Tiny
3. SegFormer-B0
4. SegFormer-B2

모델별 최고 점수뿐 아니라 평균 점수, 학습 안정성, 추론 속도, 메모리를 기록한다.

### 5단계: 교차검증과 seed 반복

상위 2~3개 구성에 대해 K-fold를 수행한다. 단일 validation에서만 좋은 구성이 아니라 fold 전체에서 안정적인 구성을 우선한다.

### 6단계: DINOv2와 SAM2 계열

기준선이 충분히 확보된 후 foundation representation을 추가한다. encoder 고정, 부분 미세조정, 전체 미세조정을 단계적으로 비교한다.

### 7단계: TTA와 ensemble

최종 후보들의 validation 예측을 저장하고, TTA와 soft ensemble 조합을 평가한다. 검증에서 개선되지 않는 기법은 최종 제출에 넣지 않는다.

---

## 10. 공부 순서

### 1단계: segmentation 기초

- semantic segmentation과 instance segmentation의 차이
- mask와 label encoding
- IoU, Dice, mIoU
- cross entropy와 softmax

### 2단계: U-Net과 CNN

- encoder-decoder 구조
- skip connection
- receptive field
- ResNet의 residual block
- pretrained model과 fine-tuning

### 3단계: 데이터와 학습 안정화

- augmentation 설계
- class imbalance
- overfitting과 regularization
- learning rate scheduler
- early stopping
- K-fold cross-validation

### 4단계: 최신 segmentation 모델

- DeepLabV3+
- U-Net++
- HRNet
- SegFormer
- Swin 계열

### 5단계: foundation model과 최신 표현 학습

- self-supervised learning
- DINOv2의 feature 활용
- SAM/SAM2의 prompt 기반 segmentation
- domain adaptation
- parameter-efficient fine-tuning

### 6단계: 대회 실전 기법

- patch inference와 overlap blending
- TTA
- fold ensemble
- soft voting
- 실험 관리와 재현성

---

## 11. 핵심 결론

KoMaP 준비에서 가장 현실적인 전략은 다음과 같다.

1. 먼저 U-Net + ResNet34로 데이터와 평가 파이프라인을 검증한다.
2. ConvNeXt-Tiny를 사용해 현대적인 CNN backbone의 효과를 비교한다.
3. SegFormer-B0/B2로 Transformer 계열을 비교하되 과적합을 확인한다.
4. DINOv2와 SAM2 계열은 기준선 이후 foundation representation 실험으로 진행한다.
5. patch training, 적절한 augmentation, Dice 계열 loss, class imbalance 대응을 함께 검증한다.
6. 단일 validation 점수보다 K-fold 평균과 클래스별 IoU를 중시한다.
7. 마지막에는 TTA와 fold/model ensemble로 안정성을 높인다.

최종 우선순위는 다음과 같이 정리할 수 있다.

```text
데이터 파이프라인 검증
→ U-Net + ResNet34 기준선
→ patch/augmentation/loss 개선
→ ConvNeXt와 SegFormer 비교
→ cross-validation
→ DINOv2/SAM2 실험
→ TTA와 ensemble
```

KoMaP에서는 화려한 모델 하나보다, 작은 데이터에서 흔들리지 않는 실험 설계와 오류 분석이 더 큰 차이를 만든다. 따라서 각 실험은 반드시 동일한 검증 조건에서 비교하고, 점수뿐 아니라 어떤 클래스와 어떤 형태의 경계에서 실패하는지까지 확인하는 것을 권장한다.
