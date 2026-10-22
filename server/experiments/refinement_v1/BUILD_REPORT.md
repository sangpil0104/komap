# refinement_v1 구현 및 검증 기록

검증일: 2026-09-20. `komap/spec.md`, `AGENTS.md`에 따라 기존 모델·소스·실험 결과를 보존하고 이 디렉터리에만 새 작업물을 추가했습니다.

## 구현 결과

- 기존 CNN 가중치에서 시작하는 대조군과 9개 변경 후보, 총 10개 실험을 제공합니다.
- 변경 요소: residual ASPP, 학습 이미지의 실제 오분류 위치 중심 패치 추출, 실제 클래스 가중 CE, Dice/Lovasz 비중, 전체 영상 학습 단계 비중.
- 실험별 2,000 updates 추가 학습, plain/D4 평가, 비교표 작성을 한 명령으로 순차 실행합니다.
- 진행률·경과 시간·ETA는 터미널에서 표시합니다. 서버나 웹 대시보드는 실행하지 않습니다.
- 기존 파일 2,361개의 SHA-256을 작업 전후 비교해 모두 동일함을 확인했습니다. 기록: `preservation.json`, `outputs/verification/preservation_final.log`.
- 패키지 설치나 conda 환경 변경은 수행하지 않았습니다.

## 결과 해석의 전제

동료 문서의 최고 Global mIoU 79.48%와 사용자 모델의 이미지별 평균 mIoU 78.7858%는 다른 지표입니다. 현재 기준 CNN+D4는 이미지별 평균 78.6865%, Global 81.4989%입니다. 동료도 같은 이미지를 사용했다고 사용자가 확인했지만, 정확한 검증 목록·클래스 매핑·평가 CSV·가중치는 확인할 수 없습니다. 따라서 동료 모델의 클래스별 logit 보정을 복제하지 않고 유용한 구조·샘플링 아이디어를 현재 파이프라인의 후보로 구현했습니다.

## 확인한 항목

| 검증 | 결과 |
|---|---|
| 단위 테스트 | 16개 통과 |
| 10개 후보의 실제 8 GPU 분산 smoke 학습 | 모두 통과, 후보마다 3단계 × 1 update |
| 후보별 plain/D4 검증 | 모두 검증 이미지 20장 평가 완료 |
| ASPP 초기화 | 기존 모델 출력과 동일, 학습 후 context 가중치 갱신 확인 |
| 기존 방식 대조군 | 세 단계 데이터 출력이 원본과 동일함을 확인 |
| 가중 CE | 클래스 가중치 반영 및 ignore=255 처리 확인 |
| 오분류 지도 | train 70장만 사용, validation 사용 0장, 오류 픽셀 2,790,072개 |
| 지도와 증강 좌표 일치 | 회전/반전 후 강제 hard crop 중심 16건 확인 |
| 중단·재개 | ASPP+hard+weighted 모델이 update 2에서 중단 후 3까지 정상 재개 |
| 테스트 PNG 생성 | 10장 출력, ID·팔레트·원본 크기 검사 통과 |
| 완료 평가 재사용 | 체크포인트·결과 무결성 확인 후 정상 재사용 |
| 본 실험 비교표 분리 | smoke 결과 제외 확인 |
| 진행률 | 터미널 ETA 표시, HTML/localhost 실행 없음 |
| 기존 파일 보존 | 2,361개 모두 변경 없음 |

검증은 기존 `komap` conda 환경의 PyTorch 2.7.1+cu128, RTX 4090 8장에서 수행했습니다. 종료 후 GPU 8장 모두 유휴 상태임을 확인했습니다.

세부 기록은 `outputs/verification/`의 `integration.json`, `mining_check.json`, `resume_predict.json`, `smoke_suite.log`, `resume_and_predict.log`에 있습니다. 재현 가능한 통합 확인 스크립트는 `outputs/verification/check_integration.py`입니다.

## 전체 학습 상태와 실행

**전체 2,000-update 실험 10개는 아직 시작하지 않았습니다.** smoke는 기존 학습 가중치에서 시작한 기능 점검이며 정확도 개선의 증거가 아닙니다. 어느 후보가 우수한지는 전체 학습 후 같은 검증 기준으로 판단해야 합니다.

```bash
cd /Data3/sppark/komap
conda activate komap
bash experiments/refinement_v1/run_suite.sh 8
```

완료 후 `outputs/reports/comparison.csv`와 `comparison.txt`에서 기준 모델 및 추가 학습 대조군 대비 변화를 확인하세요. 이전 최고점과 새 결과는 이미지별 평균/Global, plain/D4를 맞춰 비교해야 합니다. 세부 조건과 개별 명령은 [README.md](README.md)에 있습니다.
