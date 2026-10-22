# 100개 모델 실험 빌드 검증

검증일: 2026-09-20. 기존 `komap` conda 환경, PyTorch 2.7.1+cu128, RTX 4090 8장에서 확인했습니다. 추가 패키지 설치나 외부 가중치 다운로드는 하지 않았습니다.

## 완성한 항목

- 정확히 100개의 서로 다른 유효 설정: 추천 1개 / 기존 결과 기반 19개 / 새로운 탐색 80개.
- 100개 설정 YAML, 전체 목록과 논문 출처, 실제 파라미터 수 CSV.
- 단일 명령의 순차 학습, 중단·재개, 완료 결과 재사용, 실패 기록 후 다음 모델 실행.
- 모든 모델의 plain/D4 검증, 이미지 평균과 Global mIoU 분리, 기준/초기 모델 대비 변화 기록.
- 학습 전 update 0까지 포함한 `best.pt`와 학습 이후 최고 `best_trained.pt`를 별도로 보존.
- 실제 100개가 모두 완료됐을 때 프로젝트 루트 `model_results_0920.md` 생성. 기존 동명 파일이 있으면 보관 후 갱신.
- 중간 보고서와 smoke 보고서 분리. 실패나 부분 완료를 100개 성공으로 표시하지 않음.
- 진행률·경과 시간·ETA는 터미널만 사용.

## 검증 결과

| 항목 | 결과 |
|---|---|
| 단위/회귀 테스트 | **65개 통과** |
| 유효 설정 중복 검사 | 100개 모두 서로 다름, 1/19/80 개수 확인 |
| 실제 이미지 모델 검사 | **100개 모두 통과**: 각 단계의 입력, 손실, 유한한 기울기, optimizer 갱신 |
| 원본 크기 입력 | 실제 최대 크기 학습 이미지 또는 각 단계의 실제 patch 크기로 확인 |
| 8 GPU 분산 smoke | **대표 후보 14개 모두 통과**, 각 단계 2 updates |
| plain/D4 평가 | 대표 후보 각각 공식 검증 20장 전체 평가 |
| 초기 모델 보호 | `best.pt` 검증 점수가 update 0 점수 이상임을 체크포인트로 확인 |
| 체크포인트·평가 연결 | 소스/설정/체크포인트 SHA-256, CSV 점수와 선택 점수 일치 확인 |
| 중단·재개 | 추가 ASPP 후보가 update 1에서 멈춘 뒤 update 4까지 재개 |
| 완료 결과 재사용 | 동일 명령 재실행 시 학습/평가 재사용, last.pt 해시 동일 |
| 테스트 마스크 | 10장 생성, ID·색상·원본 크기 검사 통과 |
| 100개 순차 제어 테스트 | 모의 실패 1건 이후에도 100개 모두 시도, 실패 상태 기록 확인 |
| 보고서 생성 조건 | 100개 실제 완료 조건에서만 게시, smoke/부분 결과 게시 차단, 동명 파일 보존 확인 |
| 기존 파일 보존 | **3,065개 모두 SHA-256 동일** |
| 학습/웹 실행 상태 | 본 학습 미실행, 서버/브라우저 미실행, 종료 후 GPU 8장 모두 유휴 |

대표 분산 점검 후보는 추천 전체 영상, staged ASPP, hard patch, pyramid pooling, skip gate, Focal, auxiliary, boundary, SGD, erase, 그리고 네 종류의 scratch U-Net입니다. 폭별 다른 구조와 나머지 손실·증강·최적화 조건은 100개 실제 이미지 검사 및 단위 테스트에서 확인했습니다. 100개 전부를 끝까지 학습했다는 의미는 아닙니다.

극단적으로 확신하는 예측에서 fractional Focal Loss의 기울기가 NaN이 되는 경계 사례를 찾아 안정화했고 회귀 테스트를 추가했습니다. 최종 소스로 100개 실제 이미지 검사와 14개 분산 smoke 검사를 통과했습니다. 수정 전의 점검 파일은 `outputs/verification/before_numeric_fix/`에 별도로 보관했습니다.

검증 근거:

- `outputs/verification/unit.log`
- `outputs/verification/all_model_contracts.json`, `all_model_contracts.log`
- `outputs/verification/pipeline_audit.json`, `smoke_suite.log`
- `outputs/verification/resume_audit.json`, `resume_check.log`, `resume.log`
- `outputs/verification/preservation_final.log`

검사 재현 스크립트는 같은 `outputs/verification/` 아래에 있습니다. 순수 단위 테스트는 다음 명령으로 실행할 수 있습니다.

```bash
conda run --no-capture-output -n komap python -B -m pytest \
  -c experiments/hundred_v1/pytest.ini -q experiments/hundred_v1/tests
```

## 본 실험 실행

**148,900 updates의 본 실험은 아직 실행하지 않았습니다.** 정확도 개선은 이후 실제 결과로 판단해야 합니다. smoke의 점수는 모델 선택 근거로 사용하지 않습니다.

```bash
bash /Data3/sppark/komap/experiments/hundred_v1/run_all.sh 8
```

예상 8–14시간은 기존 속도에 근거한 계획 예산이며 100개 전체 실측값은 아닙니다. 상세 조건은 [README.md](README.md), 100개 목록은 [EXPERIMENTS.md](EXPERIMENTS.md)에 있습니다.
