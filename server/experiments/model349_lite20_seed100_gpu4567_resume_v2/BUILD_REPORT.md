# 재개 빌드 검증

- 오류 수정: worker.call 명령 인자를 str로 통일한 후 JSON 저장 및 subprocess 실행.
- 원본 frozen 코드에서 학습 소스/recipe/plan 변경 없음. 수정한 기존 파일은 worker.py, start_nohup.sh, README.md뿐입니다.
- CPU 회귀/설계 테스트 12개 통과. 실 subprocess 인자/로그 일치, B4/HR/OOF 명령 직렬화, 원본 보존, 캐시 복사, GPU 매핑 및 완료 작업 생략 확인.
- baseline/data/source SHA256과 데이터/fold 분리 사전 검사 통과.
- 완료 작업 74개, 그중 학습 64개, 완료 아티팩트 2450개 SHA256 검증 통과.
- 남은 학습 E B4 20 + E HR 8 + F 8 = 36개.
- 첫 준비 작업: E_b4_set0, E_b4_04, E_b4_05, E_b4_06.
- 학습은 시작하지 않았습니다. GPU 추론도 수행하지 않았습니다.
- 재개 run: `/Data3/sppark/komap/experiments/model349_lite20_seed100_gpu4567_resume_v2/outputs/runs/seed100_resume_20261012_120527_923014`
- 결과는 전체 완료 후 results/model_results_날짜_번호에 자동 저장됩니다.
- 검증 로그: checks/tests.log, checks/prepare_resume.log, checks/resume_audit.json.
