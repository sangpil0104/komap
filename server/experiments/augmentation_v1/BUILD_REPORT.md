# 구현·검사 기록

- `spec.md`를 읽고 원본 보존 지침을 적용했습니다. 이후 작업에서도 이를 확인하고 터미널 진행률을 사용하도록 프로젝트에 새 `AGENTS.md`를 추가했습니다. 기존 `spec.md` 내용은 수정하지 않았습니다.
- 작업 전 기존 파일 2,094개의 크기·SHA-256을 기록했고, 새 세 후보의 GPU 실행 후 보존 검사에서 모두 일치했습니다. 이전 모델·모델 소스·accuracy_v1 결과를 변경하지 않았습니다.
- 새 데이터 어댑터는 frozen 모델/학습 엔진에 실행 프로세스 안에서만 연결합니다. frozen 소스 파일을 편집하지 않으며 모델 구조·loss·optimizer를 그대로 사용합니다.
- **자동 검사 20개 통과**: 대조군의 원본 입력 tensor 일치, 회전/반전의 이미지·마스크 정렬, nearest 마스크 리사이즈, 노이즈 강도·마스크 불변, 캐시 원본 불변, 인덱스 재현성, 마지막 단계의 동일성, group fold, 출력 경로 보호, HTML 없는 ETA.
- **GPU 8장 실제 점검**: photo_mild/noise_mild/scale_mild 각각 원래 해상도·패치 크기를 유지하고 3개 단계에서 1 update씩 실행. 각 단계 raw/EMA 분산 검증, 종료 후 검증 20장 plain/D4 평가 통과.
- **중단·재개**: control smoke에서 update 2에 중단 후 같은 8 GPU로 update 3까지 정상 재개.
- **평가 재사용**: 동일 photo_mild smoke 체크포인트의 D4 평가 재실행 시 기존 결과 재사용 확인.
- 모든 학습 점검은 `outputs/smoke/`에 저장했습니다. **4,500 updates 본 실험은 시작하지 않았습니다.** 점검 모델의 mIoU로 augmentation 효과를 판단할 수 없습니다.
- `outputs/reports/comparison.csv`에 smoke 결과가 섞이지 않음을 확인했습니다. 현재 본 실험 전이므로 기준 모델 2행만 있습니다.
- tqdm 로그에서 updates, 경과 시간, loss, 최고 mIoU, ETA를 확인했습니다. 기존 엔진의 웹 모니터 안내를 제거하고 HTML 대시보드를 생성하지 않았습니다. 서버나 브라우저를 실행하지 않았습니다.
- conda 환경과 설치 패키지는 바꾸지 않았습니다.

검사 상세는 `outputs/verification/integration.json`, `smoke_suite.log`, `resume.log`, `evaluation_reuse.log`, `preservation.log`에 있습니다. 전체 실행 방법은 `README.md`를 보세요.
