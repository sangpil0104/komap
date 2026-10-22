# 구현 및 검증 기록

작성일: 2026-09-24 (UTC)

이미지별 Multi-Otsu 임계값 계산, train 전용 상 대응 보정, 모델 저장·불러오기, 정답 없는 이미지 추론, 검증 보고서 및 그림 생성, nohup 실행기를 구현했습니다. 실제 데이터 보정은 시작하지 않았으며 실제 성능 수치는 아직 없습니다.

| 검증 | 결과 | 근거 |
|---|---|---|
| 실제 데이터 사전 검사 | 원본 100장·마스크 90장 통과; train70/valid20/test10 | [preflight.json](preflight.json), [실행 로그](preflight_verified.log) |
| 단위·통합 검사 | 9개 통과 | [test_results.log](test_results.log) |
| 합성 데이터 전체 실행 | 임시 데이터에서 모델 보정·저장, 검증 20장 평가·그림 생성, 정답 없는 추론 성공 | `tests/test_threshold.py`의 CLI 통합 검사 |
| 지표 | per-image 평균, 빈 클래스 1/0/제외, 없는 상의 FP 처리 검사 통과 | 단위 검사 |
| 우측 상단 오류 그림 | 오답=정답 상 색, 정답=흰색; 저장 파일의 네 패널을 픽셀 단위 검사 | 단위·통합 검사 |
| Shell 문법 | 두 실행 스크립트의 `bash -n` 통과 | `run_cpu.sh`, `start_nohup.sh` |

합성 데이터의 모델·결과는 임시 폴더에서 생성·삭제했습니다. 실제 `runs/` 및 `results/model_results_MMDD_N/`는 사용자가 실험을 시작할 때 생성됩니다. 기존 모델, 학습 소스, 이전 결과는 수정하지 않았습니다.

사용 방법과 한계는 [README.md](README.md)를 참고하세요.
