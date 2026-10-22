# 실험 코드 빌드 및 검증

2026-09-20. 작업 범위는 `experiments/accuracy_v1/`의 새 코드·설정·보관 사본·동작 점검 결과입니다.

## 원본 보존

- 기존 파일 **838개**의 경로·크기·SHA-256을 작업 시작 전에 기록했습니다.
- 기존 원본의 소스, 설정, 모델 가중치, 학습 로그, 평가 결과, 테스트 PNG, 데이터, 문서를 수정하거나 이동하지 않았습니다.
- 별도 보관 사본 **376개**를 복사했고, 사본 파일을 읽기 전용으로 설정했습니다.
- 약 969MiB의 `baseline/`에 기존 official run 전체와 소스·설정·환경·테스트 출력 등을 보관했습니다.
- 실험 실행은 고정 사본의 `alphase` 코드를 import합니다. 기존 conda 환경과 editable 설치는 변경하지 않았습니다.
- 새 학습 결과는 `outputs/runs`, 점검용 결과는 `outputs/smoke*` 아래에 저장합니다.
- `outputs/verification/preservation_final.json`에서 원본 838개와 사본 376개 모두 변경 없음이 확인됐습니다.

## 구현한 실험

| 실험 | 변경 | 총 optimizer updates |
|---|---|---:|
| fullframe | 단계 예산 2,000 / 1,000 / 1,500 | 4,500 |
| boundary | 기존 단계 예산 유지, 경계 보조 손실 0.05 | 4,500 |
| seed2027 | seed만 변경 | 4,500 |
| seed2028 | seed만 변경 | 4,500 |
| dino_large | 기존 DINOv3-L hybrid 소스, 고정→마지막 4→8→8 blocks 학습 | 4,500 |

CNN 기본 suite는 네 실험의 학습, 기본/D4 검증, seed 2026(기존 모델) + 2027 + 2028 확률 앙상블, 테스트 PNG 생성, 비교표 생성을 수행합니다. 완료된 실행은 설정과 데이터가 일치하면 재사용하고 중단된 실행은 재개합니다.

별도 구현: 원본 크기의 float32 확률 캐시, 모델·데이터·팔레트·학습 ID·추론 설정 검사, 확률 평균 후 argmax, 이미지별 점수 비교, 전체 suite 진행 화면, 개별 학습 화면 연결. 추론 캐시의 데이터와 hash가 변하면 재사용을 거부합니다.

## 검사 결과

| 검사 | 결과 |
|---|---|
| 새 단위 검사 | **20 tests 통과** |
| 독립 실험 설정 | fullframe은 예산 배분만, boundary는 보조 손실만, seed 실험은 seed만 달라짐 확인 |
| 원본 소스 선택 | editable 원본 대신 고정 사본을 import함을 확인 |
| 경계 head | 유한하고 0이 아닌 gradient 확인 |
| 확률 앙상블 | hard label 투표와 다른 사례에서 올바른 확률 평균 확인 |
| 잘못된 앙상블 입력 | 잘못된 가중치·형상·확률 합과 데이터/클래스/ID/추론 설정 불일치 거부 |
| 평가 누수 차단 | 학습 ID와 검증 ID 겹침, all90의 독립 검증 요청 거부 |
| GPU 8장 boundary | native 512/768/full-frame 각 단계 1 update, 전체 raw/EMA 검증 통과 |
| 분산 재개 | boundary update 2에서 저장·종료 후 새 프로세스로 update 3 완료 |
| GPU 8장 fullframe | 3개 단계의 학습·검증·단계별 최고 모델 저장 통과 |
| 실제 앙상블 | 서로 다른 두 smoke 모델의 검증 20장 확률 평균 통과 |
| 테스트 PNG | 위 앙상블의 원본 크기 RGBA PNG 10장 생성·팔레트/alpha/ID 검사 통과 |
| 캐시 재사용 | 같은 앙상블을 다시 평가할 때 기존 확률·결과를 검증해 재사용 |
| 기존 모델 재현 | **78.47287474071697%**, 보존한 기존 결과와 일치 |
| 기존 D4 재현 | **78.68647961342091%**, 보존한 기존 결과와 일치 |
| 전체/개별 웹 화면 | Chromium에서 JS 오류 없음, 390px 화면 가로 넘침 없음 |
| 비교표 | 필터와 CSV 링크 작동, smoke 결과 제외 |
| DINO 사전 검사 | 가중치 없이 요청하면 GPU worker를 띄우기 전에 명확히 종료 |

근거 파일은 `outputs/pytest.log`, `outputs/smoke_*.log`, `outputs/baseline*_reproduction.log`, `outputs/verification/`에 있습니다. 화면 캡처는 `outputs/verification/suite_dashboard.png`, `comparison.png`입니다.

## 아직 수행하지 않은 작업

- 새 CNN 실험의 실제 4,500회 본 학습과 그 성능 비교.
- 새 seed 2027·2028의 본 학습 모델을 사용하는 최종 3개 seed 앙상블. 실행 코드는 준비했고, 실제 확률 결합은 별도 smoke 모델로 검증했습니다.
- 공식 사전학습 가중치가 필요한 DINOv3-L의 실제 로드·메모리·학습 성능 검증.
- group CV 5개 전체 학습 및 최종 90장 학습.

따라서 새 모델의 정확도가 향상됐다고 보고하지 않습니다. 짧은 smoke 모델의 점수는 파이프라인 동작 확인용이며 성능 비교표에 넣지 않습니다.

실행 안내: [README.md](README.md).
