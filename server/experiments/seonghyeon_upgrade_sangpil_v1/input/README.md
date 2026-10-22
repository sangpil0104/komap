# AlPhase B4 Codex handoff

기존 EfficientNet-B4 U-Net과 100개 seed 실험을 보존하면서 개선 기능을 추가하기 위한 전달 묶음이다. **완성된 분할 모델 코드나 학습 가중치가 아니다.**

## 파일

| 파일 | 용도 |
|---|---|
| `alphase_b4_codex_spec.md` | 단계별 구현 명세, 데이터/좌표/평가 계약, 설정, 테스트, 실행 요구사항 |
| `CODEX_START_PROMPT.md` | 서버의 Codex에 전달할 시작 지시문 |
| `references/a71_error_diagnosis.md` | 첨부 A71 결과에 대한 기존 오류 분석 |
| `references/phase_analysis_final.md` | 이전 데이터 검사 기록 |
| `fixtures/a71_confusion.json` | 과거 A71 예측의 정확한 혼동 행렬 회귀 fixture |
| `reference_code/contract_reference.py` | 지표·팔레트·D4·좌표 정렬·native 복원·상대 밝기 참조 함수 |
| `reference_code/test_contract_reference.py` | 작은 수학/기하 계약 테스트 |
| `VALIDATION_REPORT.md` | 이 묶음 작성 시 실행한 검사와 실행하지 않은 항목 |
| `SHA256SUMS.txt` | 전달 파일 무결성 확인용 해시 |

## 전달 방법

기존 프로젝트에서 접근할 수 있는 문서 폴더에 이 묶음을 풀고, `CODEX_START_PROMPT.md`의 지시문으로 명세서를 읽게 한다. 기존 프로젝트 코드를 이 폴더로 옮길 필요는 없다. 데이터 ZIP과 100개 checkpoint는 묶음에 포함하지 않았으며 서버의 기존 경로를 Codex가 확인해야 한다.

참조 코드만 확인하는 명령은 묶음의 최상위 디렉터리에서 다음과 같다. 의존 패키지는 NumPy, PyTorch, pytest이며 서버 환경을 임의로 업그레이드하지 않는다.

```bash
python -m pytest reference_code -q
```

위 테스트는 실제 B4 모델을 로딩하지 않는다. 실제 저장소 수정, GPU 메모리 점검, 기존 추론 재현 및 새 모델의 학습·성능 비교는 명세에 따라 서버에서 수행한다.
