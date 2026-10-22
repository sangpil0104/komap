# 모델 구성과 설계 의도

| 구성 | Encoder / decoder | 특징 | 최종 결합 역할 |
| --- | --- | --- | --- |
| 기본 확률 | ResNet34 / compact U-Net++ 및 ASPP 계열 | 97·98·109, pairwise, 고해상도·스타일·TENT 분기 | 기본 확률 |
| Model269 | ResNet34 / compact U-Net++ | TeamResidualASPP + stride-1 native stem | Eutectic Si |
| Model322 | EfficientNet-B4 / SMP U-Net | gray3, 4-fold | Al |
| Model326 | EfficientNet-B4 / SMP U-Net | Raw/Z/CDF, 4-fold | 초기 Primary Si·Al3Ni |
| Model329 | EfficientNet-B4 / SMP U-Net | 너비 4000, 이미지 균등 샘플링, 4-fold | Primary Si·Al3Ni |
| Model336 | HRNet-W48 / SMP U-Net++ | 학습용 1채널 boundary head | Primary Si·Al·Eutectic Si |
| Model347 | Model336과 동일 | 클래스 가중 CE로 fine-tuning | 마지막 Al 확률 보완 |

모든 전문가가 네 클래스를 출력하며 위 역할은 선택된 결합 채널을 뜻한다. B4 계열은 총 12개 fold checkpoint이다. 기본 확률 자체도 복수 모델·추론 분기를 포함하므로 전체 시스템을 단일 네트워크로 기술하지 않는다.

| 설계 요소 | 구조 또는 의도 | 현재 근거의 범위 |
| --- | --- | --- |
| RGB residual ASPP | 1×1, dilation 6/12/18의 3×3, global pooling; 각128채널 → 640 → 512 | 소스에서 구조 확인 |
| 표현 변환 ASPP | 1×1, dilation 2/4/8의 3×3, global pooling | 소스에서 구조 확인 |
| Residual scale | x + s·projection, s=0 초기화 | 추가 시 기존 경로 출력을 유지하도록 설계 |
| Native stem | 입력 크기 Conv 3→32→32 + zero-init Conv 32→32를 head 내부에 더함 | 독립 재학습 ablation 미수행 |
| Boundary head | decoder 특징에서 3×3 Conv → 1채널 경계; 학습에만 사용 | 독립 재학습 ablation 미수행 |
| 표현 다양성 | 절대 밝기, 상대 밝기, 밝기 순위 정보를 함께 제공 | 입력 예시 확인; 인과적 기여 분리 실험 미수행 |

그림: [Model349 전체 구조](../figures/09_model349_pipeline.png), [ResNet34·ASPP·native stem](../figures/10_resnet_aspp_native.png), [B4·HRNet 전문가 구조](../figures/11_b4_hrnet_architecture.png)
