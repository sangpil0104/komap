# 입력 표현·전처리·증강

| 모델 | 입력 표현 | 크기 설정 | 학습 통계 범위 | 평가 시 처리 |
| --- | --- | --- | --- | --- |
| Model322 | grayscale 동일 채널 3회 복제 | 3배 확대 | (g−0.449)/0.226 | 전체 영상 forward + D4 |
| Model326 | Raw / Z-score / CDF | 3배 확대 | 확대한 전체 영상의 μ·σ·CDF | 전체 영상 forward + D4 |
| Model329 | Raw / Z-score / CDF | 너비 4000 | 크기를 맞춘 전체 영상의 μ·σ·CDF | 전체 영상 forward + D4 |
| Model269 | Raw / Z-score / rank-CDF | scale 1; 최소 변이 패치보다 작으면 확대 | 전체 영상 표현 | 1024 타일, stride 576, D4 |
| Model336 / 347 | RGB + ImageNet 정규화 | 학습 입력 scale 2 | 고정 채널별 ImageNet 통계 | native 256 타일을 512로 확대; flip4 |

**B4의 Raw/Z-score/CDF 정의**

\(g\in[0,1]\), \(X_{raw}=(g-0.5)/0.25\), \(X_z=\operatorname{clip}((g-\mu)/(\sigma+10^{-6}),-4,4)/3\), \(X_{CDF}=(CDF(g)-0.5)/0.288675\).

Model326·329는 전체 크기 조정 영상의 통계를 미리 구하고, 증강·패치 추출 후 입력 픽셀을 해당 통계로 변환한다. CDF는 256단계 밝기 histogram 누적분포이다. Model269 계열은 rank 기반 CDF 구현을 사용하므로 B4 함수를 그대로 대체하지 않는다.

**B4 학습 증강**

| 연산 | 설정 |
| --- | --- |
| Affine | scale 0.7–1.4, 회전 −180°–180°, p=0.7 |
| Padding / crop | reflect padding 후 1536×1536 random crop |
| D4 | 직각 회전 및 반사 |
| Brightness / contrast | limit 0.3 / 0.4, p=0.8 |
| Gamma | 60–160, p=0.5 |
| Blur 또는 noise | Gaussian blur 3–5 또는 noise std 0.01–0.05, OneOf p=0.3 |

**HRNet 학습 증강**

| 연산 | 설정 |
| --- | --- |
| 수평·수직 반전 / 90° 회전 | 각 p=0.5 |
| Affine | scale 0.8–1.2, p=0.3 |
| Brightness / contrast | 각 limit 0.15, p=0.5 |
| Gamma | 85–115, p=0.3 |
| Elastic 또는 grid distortion | OneOf p=0.3 |
| Noise / blur | noise std 0.02–0.08, p=0.2 / blur 3–5, p=0.15 |

기하 변환은 입력과 정답의 위치가 일치하도록 함께 적용한다. 크기 조정 시 클래스 마스크에는 nearest-neighbor 보간을 사용한다. Validation/test에서는 학습용 확률 증강을 사용하지 않고 지정된 TTA를 적용한다.

그림: [Raw/Z-score/CDF 입력 예시](../figures/07_raw_zscore_cdf.png), [전처리·추론 흐름](../figures/08_preprocessing_flow.png)
