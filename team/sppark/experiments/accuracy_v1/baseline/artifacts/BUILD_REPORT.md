# 구현 및 실행 검사 기록

2026-09-20, `/Data3/sppark/komap`, conda `komap`.

**상태: CNN 기준 모델을 사용자가 바로 학습할 수 있도록 준비 완료. 본 학습은 시작하지 않음.**

## 데이터

- 학습 이미지/마스크 70쌍, 검증 20쌍, 테스트 입력 10장 확인.
- 100개 입력은 opaque RGBA, R=G=B. 90개 마스크의 정확한 4색 decode/encode 픽셀 왕복 일치.
- ID·파일 바이트·입력 픽셀 중복 없음. 모든 이미지/마스크 크기 일치.
- 실제 해상도: 675×480, 1096×780, 1230×875, 1278×909, 1440×1024.
- manifest, 팔레트, audit, 원본 ID 마스크 캐시, A-prefix group 5-fold 및 분포 요약 생성.
- 공식 train/valid의 A-prefix 겹침 7개는 `audit.json`에 명시. 같은 물리 시편인지는 확인되지 않음.
- 원본 PNG와 split은 그대로 보존. 학습 통계는 이미지별로 계산하며 검증 마스크를 샘플링/학습에 사용하지 않음.

## 구현

- 원해상도 stride-1 경로를 유지한 CNN/U-Net, gray/z-score/CDF 입력, 이미지 균등·공정 Si·경계 중심 샘플링.
- D4와 약한 밝기 증강, halo 밖/원본 밖 255 ignore 마스킹.
- 이미지별 CE + present Dice + Lovasz, 선택적 soft-area auxiliary / 경계 보조 손실.
- BF16, AdamW, warmup/cosine, gradient accumulation/clipping, EMA, DDP.
- raw/EMA 전체 영상 검증, 이미지별 평균 mIoU, empty-union 3개 정책, 단계별 최고 모델과 전체 최고 모델.
- 체크포인트·재개·설정/데이터/GPU 수 검사·환경/ID/hash 추적.
- 전체/타일·D4·다중 배율 추론, 원본 크기 팔레트 PNG, 출력 10장 규격 검사.
- 터미널 tqdm, ETA, TensorBoard, 외부 CDN 없는 실시간 브라우저 화면.

## 검증 결과

| 검사 | 결과 |
|---|---|
| pytest | 23 tests 통과 |
| 패키지 의존성 | `pip check` 통과 |
| PyTorch GPU 연산 | RTX 4090, CUDA 12.8, bf16 지원 확인 |
| GPU 1장 update | 실제 512/768/full-frame 순전파·역전파·optimizer 측정 통과 |
| GPU 8장 DDP | 실제 데이터에서 3개 단계, 총 6 updates 통과 |
| 분산 재개 | update 3에서 정지 후 새 torchrun 프로세스로 update 6까지 재개 통과 |
| 단계 전환 | 각 단계 최고 체크포인트 복원 후 다음 단계 학습 통과 |
| 검증 재현 | 새 프로세스에서 best.pt를 로드한 20장 mIoU와 저장된 최고 점수가 1e-9 이내 일치 |
| 전체 실제 해상도 | 5개 크기 모두 [4,H,W] 출력 및 픽셀별 확률 합=1 검사 통과 |
| 작은 패치 과적합 | 실제 학습 영상의 128×128 4클래스 패치, 250회 update: loss 2.1758 → 0.000623, training-patch mIoU 1.0 |
| 테스트 출력 | smoke 가중치로 PNG 10장 생성 후 ID·크기·팔레트·alpha 검사 통과 |
| 브라우저 | headless Chromium에서 JS 오류 없음, 그래프/진행률/검증 이미지 표시 |
| 모바일 화면 | 폭 390px에서 가로 넘침 없음 |
| DINO 공식 소스 | ViT-S 무작위 초기화, 64×96 입력→출력→backward 연결 확인 |
| H+/L 공식 pretrained | 가중치 미제공으로 검사하지 않음 |

작은 패치 mIoU는 연결 확인용 **자기 학습 패치 점수**입니다. 일반화 성능을 의미하지 않습니다. smoke 단계의 검증 결과 역시 6회 update짜리 실행 확인 결과이며 최종 모델 성능으로 해석하지 않습니다.

## 시간 측정

각 단계에서 2회 워밍업 후 5회 optimizer update를 측정했습니다. 1 update의 effective batch는 GPU 수와 관계없이 8입니다. 검증 20장 처리 시간을 측정해 raw/EMA 2회 비용을 반영했습니다.

| GPU 수 | 512 단계 초/update | 768 단계 초/update | full-frame 초/update | 4,500 updates + 검증 예상 |
|---:|---:|---:|---:|---:|
| 1 | 0.503 | 0.897 | 1.199 | 56.4분 |
| 8 | 0.072 | 0.127 | 0.312 | 9.5분 |

계획 범위: 1장 45~75분, 8장 8~15분. 저장/로딩/동시 작업과 표본 크기에 따라 달라집니다. 통계적 신뢰구간은 아닙니다. 최대 측정 GPU tensor 메모리는 약 7.62GiB/장입니다. DINOv3 시간이나 VRAM 추정이 아닙니다.

근거: `benchmark_cnn_unet_1gpu.json`, `benchmark_cnn_unet_8gpu.json`, `gpu_verification.json`, `integration_verification.json`, `dino_wiring_test.json`, `dashboard-preview.png`, `smoke_*.log`.

## 실행 범위

사용자 선택에 따라 CNN 학습 경로를 완성했습니다. DINOv3 v1 모델 연결·설정은 있으나 H+/L 사전학습 실험은 별도입니다. 원문 설계의 ConvNeXt/HRNet, Adapter v2, OOF/refiner/nested CV/앙상블 탐색/의사라벨 자동화는 구현 범위에 포함하지 않았습니다.

재현 정보: `requirements.lock.txt`, `environment.yaml`, `environment.json`, `dinov3_commit.txt`, `source_fingerprint.json`. 루트 폴더는 기존에 Git 저장소가 아니었으므로 code commit 대신 소스 hash를 기록했습니다. 공식 DINO 소스는 별도의 Git commit으로 고정했습니다.

실행 명령은 [QUICKSTART.md](../QUICKSTART.md)에 정리했습니다.
