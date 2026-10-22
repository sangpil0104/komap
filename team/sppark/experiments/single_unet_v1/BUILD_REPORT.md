# 구축 및 검증 기록 — 2026-09-24

요청한 실험을 `experiments/single_unet_v1/`에 새로 구현했다. 기존 모델·소스·결과 파일은 수정하지 않았다. **학습 run과 학습 checkpoint는 생성하지 않았으며 optimizer step은 0회다.**

## 구현

- torchvision ImageNet EfficientNet-B4 + 기본 U-Net, 19,412,300 parameters.
- S0/S1/S2: 동일 원본 512px 시야, 입력 배율 1/2/3, 동일 seed의 동일 샘플·초기화.
- S3: 선택 배율에서 GT 원본 2px 경계 띠의 CE 가중치 +0.2. 별도 head 없음.
- CE+Dice, gradient accumulation, 선택한 1/2/4/8 GPU DDP, encoder activation checkpoint.
- 주 평가 image mIoU: 빈 클래스 IoU=1로 고정 4클래스 평균 → 이미지 평균. zero/exclude도 함께 저장.
- 단일 배율 Plain 타일 추론 기본, full-frame 사방 reflect padding/halo 확대/격자 이동/D4 별도 평가.
- checkpoint별 RNG/optimizer/데이터·소스 hash, timestamp별 독립 run, 중단 재개, 완료 run 학습 덮어쓰기 금지.
- 학습 완료 후 검증·원본/오류/예측/GT 비교 PNG·정확한 RGBA 마스크·새 번호 results 보고서 자동 생성.
- test 10장 예측, 별도 group fold 학습, 독립 seed 평균·표본 표준편차 보고 지원.

## 실제 수행한 검사

1. pytest **14개 통과**: 빈 클래스/이미지 평균, halo ignore, 원본 crop 일치, 배율별 라벨 보간, D4 역변환, 타일 격자/확률 복원, 모델 gradient, S3 통제 조건, synthetic 평가→보고서→마스크 저장 및 기존 보고서 보존.
2. 데이터 SHA256 확인: train 70 / valid 20 / test 10. 라벨 90장 shape·4색 팔레트·alpha 검증 통과.
3. RTX 4090 24GB에서 합성 입력 순전파·역전파 통과. optimizer 생성/업데이트 없음.

| 설정 | 실제 입력 | 최고 CUDA 할당 메모리 |
|---|---:|---:|
| S0 | 640×640 | 0.52 GiB |
| S1 | 1280×1280 | 1.63 GiB |
| S2 | 1920×1920 | 3.45 GiB |

실제 학습에는 optimizer 상태, DDP buffer, 데이터 내용에 따른 추가 메모리가 필요하다. 위 값은 전체 학습 시간/전체 학습 메모리 측정값이 아니다.

4. **2 GPU NCCL/DDP** 합성 gradient accumulation 및 rank 간 gradient 완전 일치 확인. 실제 데이터 학습과 optimizer step 없음. 4/8 GPU는 이번 검사에서 실행하지 않았다.
5. 공식 ImageNet 가중치 74.5MiB를 새 실험의 `weights/`에 준비하고 다운로드 hash 검증 통과.
6. Python syntax 및 Bash syntax 검사 통과.

세부 데이터/메모리 검사: [preflight JSON](outputs/verification/preflight_20260924T041918_766150Z.json).

## 실행 범위

실제 학습 수렴·최종 정확도·실제 optimizer 업데이트를 포함한 학습/재개 전체 실행은 수행하지 않았다. `AGENTS.md`의 “학습 시작은 항상 사용자가 한다” 지침에 따라 실행 명령과 준비만 제공한다. 실제 성능 결과는 사용자 학습 실행 이후 생성된다.

실행 방법은 [README.md](README.md)에 있다.
