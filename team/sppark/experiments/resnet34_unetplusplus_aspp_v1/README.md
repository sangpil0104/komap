# ResNet34 U-Net++ + Residual ASPP 실험 (resnet34_unetplusplus_aspp_v1)

`/Data3/sppark/komap/spec.md` 및 `AGENTS.md` 지침을 준수하여, 기존 모델과 소스코드, 이전 실험 결과를 수정하지 않고 독립된 실험 패키지 내에 **ResNet34 U-Net++ + residual ASPP** 구조를 구축했습니다.

---

## 1. 실행 방법

### 한 줄로 전체 파이프라인 실행 (학습 → 검증 → 테스트 마스크 생성 → 리포트 갱신)

```bash
cd /Data3/sppark/komap
conda activate komap

# RTX 4090 8장 사용 시
bash experiments/resnet34_unetplusplus_aspp_v1/run_suite.sh 8

# GPU 1장 사용 시
bash experiments/resnet34_unetplusplus_aspp_v1/run_suite.sh 1
```

특정 GPU를 지정하려면 다음과 같이 실행합니다:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/resnet34_unetplusplus_aspp_v1/run_suite.sh 4
```

### 개별 단계별 실행

```bash
# 1. 학습만 실행
python experiments/resnet34_unetplusplus_aspp_v1/run.py train --gpus 8

# 2. 중단된 학습 재개 (last.pt에서 자동 이어 학습)
python experiments/resnet34_unetplusplus_aspp_v1/run.py train --gpus 8 --resume experiments/resnet34_unetplusplus_aspp_v1/outputs/runs/last.pt

# 3. 최고 검증 모델 평가 (Plain 및 8-way D4 TTA)
python experiments/resnet34_unetplusplus_aspp_v1/run.py evaluate
python experiments/resnet34_unetplusplus_aspp_v1/run.py evaluate --d4

# 4. 테스트 10장 예측 PNG 생성
python experiments/resnet34_unetplusplus_aspp_v1/run.py predict --d4

# 5. 결과 리포트 갱신
python experiments/resnet34_unetplusplus_aspp_v1/run.py report

# 6. 단위 테스트 실행
python experiments/resnet34_unetplusplus_aspp_v1/run.py test
```

---

## 2. 모델 아키텍처 사양

- **백본 인코더**: ImageNet 사전학습 ResNet-34
- **Stride-1 Stem**: 원본 광학현미경 이미지의 초미세 상 경계를 유지하기 위한 32채널 2-layer Conv 블록
- **Residual ASPP 병목 모듈**: Stage 5 (512채널, Stride 32) 출력단에 위치
  - Atrous Conv Branches: Dilation rate $d \in [1, 2, 4, 8]$
  - Global Context Pooling Branch: 적응형 풀링 + 1×1 Conv
  - Residual Connection: $x + \text{ASPP}(x)$ 잔차 연결로 사전학습 가중치 특성 보존
- **Nested Dense Decoder (U-Net++)**: $X^{i, j}$ 노드 간의 밀집 Dense Skip Connection
- **총 파라미터 수**: 27,481,396개 (약 27.48M)
- **정밀도**: BF16 Mixed Precision

---

## 3. 데이터셋 및 클래스 매핑

- **데이터셋 경로**: `/Data3/sppark/komap/dataset` (Train 70장, Valid 20장, Test 10장)
- **입력 전처리**: 3채널 구성 `[raw gray, z-score, empirical CDF]`
- **클래스 정의**:
  - `0 Al` (알루미늄 모상 / 갈색, RGB `153, 127, 76`)
  - `1 Primary Si, Sip` (초정 실리콘 / 초록색, RGB `76, 178, 76`)
  - `2 Al3Ni` (Al3Ni 금속간화합물 / 파란색, RGB `25, 76, 153`)
  - `3 Eutectic Si, Sie` (공정 실리콘 / 흰색, RGB `204, 204, 204`)

---

## 4. 산출물 구조

```text
experiments/resnet34_unetplusplus_aspp_v1/
├── configs/default.yaml               # 하이퍼파라미터 및 단계별 설정
├── model_lab/                         # 모델 및 파이프라인 구현 모듈
│   ├── models.py                      # ResNet34 U-Net++ + Residual ASPP
│   ├── dataset.py                     # 데이터셋 로더, 통계 채널, D4 증강
│   ├── losses.py                      # CE + SoftDice 복합 손실 함수
│   ├── trainer.py                     # DDP, BF16, EMA, Cosine LR 학습기
│   ├── evaluation.py                  # Plain / D4 TTA 검증 평가
│   ├── inference.py                   # 테스트 10장 RGBA PNG 생성
│   ├── report.py                      # model_results_0921_1.md 갱신 도구
│   └── workspace.py                   # 경로 및 설정 헬퍼
├── outputs/
│   ├── runs/best.pt                   # 최고 검증 체크포인트
│   ├── runs/last.pt                   # 재개용 최신 체크포인트
│   ├── evaluations/                   # 검증 메트릭 CSV 및 요약 JSON
│   └── predictions/                   # 테스트 10장 예측 마스크 PNG
├── tests/                             # pytest 단위 테스트 8종
├── run.py                             # 통합 CLI 스크립트
├── run_suite.sh                       # 원클릭 실행 쉘 스크립트
└── BUILD_REPORT.md                    # 검증 보고서
```

