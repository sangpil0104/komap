# HRNet_taeho / 0928 통합 검증

- 검증일: 2026-09-28 UTC
- 원본 커밋: `97adcdf337ab78fb7ec1469fe48943ff9a6363ea`
- 원본 18개 파일: clone과 `source/`의 바이트·SHA-256 일치, 원본 Git 작업 트리 변경 없음.
- 이전 실험의 소스·모델·결과 변경 없음. 새 파일은 `experiments/HRNet_taeho/`에 배치.
- **실제 데이터 학습을 시작하지 않았으며 optimizer update는 0회입니다.** 학습 체크포인트와 실제 모델 성능 점수는 아직 없습니다.

## 데이터 및 실행 환경

`./run.sh preflight` 통과. train 70장 / valid 20장 / test 10장, 이미지와 마스크 크기·짝·정확한 4색 팔레트·RGBA 불투명도·분할 간 중복 ID를 검사했습니다. 각 이미지/마스크의 SHA-256 및 크기는 [reports/preflight.json](reports/preflight.json)에 기록했습니다.

설치 환경은 Python 3.11.16, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, SMP 0.5.0, timm 1.0.29, Albumentations 2.0.8입니다. 원본의 Python 3.12 / PyTorch 2.14 환경과 같지 않습니다. 원본 잠금 파일은 수정하지 않았습니다.

`./run.sh prepare`에서 HRNet-W48 ImageNet 초기 가중치를 정상 로딩했습니다. [로그](reports/prepare.log)의 마지막 줄에 완료를 기록했습니다. 이는 대회 데이터로 학습된 segmentation 모델이 아닙니다.

## 실제 모델 순전파·역전파

`./run.sh smoke` 통과. 합성 랜덤 입력을 사용하고 optimizer를 만들거나 가중치를 갱신하지 않았습니다.

| 항목 | 확인 결과 |
|---|---|
| 입력 | [8, 3, 512, 512] |
| Segmentation 출력 | [8, 4, 512, 512] |
| 경계 head 출력 | [8, 1, 512, 512] |
| 파라미터 수 | 76,964,949 |
| 손실 및 gradient | 유한값 |
| GPU | cuda:0 (물리 GPU 0, RTX 4090) |
| 최대 할당 메모리 | 15.7802 GiB |
| optimizer update | 0 |

메모리 수치는 이 합성 순전파·역전파 검사에서 측정했습니다. AdamW optimizer 상태는 포함하지 않습니다. 수치 기록: [reports/smoke.json](reports/smoke.json).

## 지표·결과 출력 테스트

`pytest tests -q`: **7 passed**. [실행 로그](reports/tests.log), [테스트 코드](tests/test_integration.py).

1. 세 클래스 IoU=0.8, 양쪽에서 빠진 Primary Si IoU=1일 때 image mIoU=0.85.
2. GT에 없는 클래스를 예측하면 FP가 있으므로 해당 클래스 IoU=0.
3. 이미지 면적과 관계없이 이미지별 같은 가중치로 평균.
4. 정확한 RGB/RGBA 팔레트 허용, 미지의 색과 투명 마스크 거부.
5. 2×2 패널 배치와 오류 픽셀의 GT 색·정답 픽셀의 흰색 확인.
6. 기존 결과 폴더/파일 보존 및 다음 빈 결과 번호 생성.
7. 합성 데이터와 모의 모델로 원본 타일+flip TTA 추론을 실행하고 원본 크기 PNG, 비교 그림, MD, CSV, JSON 생성 확인. 평가가 입력 체크포인트를 수정하지 않고 기존 출력 폴더 재사용을 거부함을 확인.

테스트용 마스크와 점수는 임시 폴더에서만 생성했고, 실제 `results/` 모델 결과로 등록하지 않았습니다.

쉘 스크립트 구문 검사와 Python AST 검사 통과. `./run.sh train`은 dry-run으로 설정만 출력하고 학습 폴더를 생성하지 않았습니다.

## 실행 범위

- 실제 학습·valid 성능 평가·test 모델 예측은 실행하지 않았습니다.
- 원본 README의 81.77%/81.62%는 global mIoU이며 HRNet_taeho의 image mIoU 측정값이 아닙니다.
- 학습은 사용자가 `./start_nohup.sh` 또는 `./run.sh train --execute`로 시작합니다.
- 원본의 마지막 epoch 선택을 유지하고, 완료 후 프로젝트 방식의 image mIoU 및 비교 그림을 자동 생성합니다.
