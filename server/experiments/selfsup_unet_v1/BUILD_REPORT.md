# 구현·검증 기록

2026-09-24 UTC. 실제 데이터 학습은 시작하지 않았습니다. 기존 실험 파일과 결과는 변경하지 않았습니다.

| 확인 항목 | 결과 |
|---|---|
| 데이터 사전 검사 | 원본 100장, 정답 90장. 파일 SHA-256·해상도·팔레트·투명도·split 중복 확인 통과 |
| 초기 가중치 | ResNet18 IMAGENET1K_V1 다운로드와 체크섬 확인 완료 |
| 단위·통합 검사 | **8개 통과** |
| 전체 합성 실험 | A/B/C 학습·검증 각 20장, 종합 보고서, 60장 비교 그림 저장·픽셀 검사 성공. 임시 폴더만 사용 |
| 재개 | 합성 CPU 실험에서 중간 저장 후 재개와 연속 실행의 모델 텐서가 동일. 이 검사는 CPU 단일 스레드로 수행 |
| 실제 지정 GPU 검사 | **0,1,2,5,6,7의 6 GPU**, 가짜 640×640 입력, rank당 2회 누적, BF16, SSL/분할 각각 backward·optimizer update 성공 |
| GPU 동기화 | 서로 다른 rank별 가짜 입력을 사용해 update 후 rank 간 파라미터 검사 통과 |
| GPU 메모리 관측 | 검사에서 rank0 peak 할당량: SSL 약 804 MiB, 분할 약 855 MiB. 전체 실험의 최대 사용량을 보장하는 수치는 아님 |
| 실행 스크립트 | 두 Bash 스크립트 문법 검사 통과 |

근거: [데이터·가중치 준비 로그](verification/prepare.log), [최종 테스트 로그](verification/tests_deterministic_cpu.log), [6 GPU 입력 크기 검사](verification/ddp_native.log), [소스·의존성 해시 기록](verification/build.json).

CPU 다중 스레드의 별도 실행 간 미세한 수치 차이가 첫 재개 검사에 영향을 주어, 정확한 텐서 일치 검사는 단일 스레드로 고정했습니다. 실제 GPU 학습의 bitwise 재현성을 보장하는 것은 아닙니다. 모델·optimizer·난수·데이터 위치는 checkpoint로 보존합니다.

실험을 시작하는 명령과 설정은 [README.md](README.md)에 있습니다. 실행하면 A/B/C의 실제 성능, 경계·외곽 오류 및 계산 예산을 새 `results/model_results_MMDD_N/`들에 저장합니다.
