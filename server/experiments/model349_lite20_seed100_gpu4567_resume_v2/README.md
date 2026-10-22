# Model349 lite20 — 중단된 100회 시드 탐색 재개

기존 탐색의 완료된 **64/100개 학습**과 평가 기록을 검증하여 가져온 별도 재개 경로입니다.
원본 run, 원본 학습 소스, 가중치, 결과서는 수정하지 않습니다.
기존 시드 3407~3506의 계획과 학습 설정, 기존 B4 fold 구성, R/E/F 선택 규칙을 그대로 사용합니다.
r70 B4 교체 실험의 가중치를 이 탐색에 넣지 않습니다.

## 사용자 실행

프로젝트 지침에 따라 학습은 사용자가 시작합니다. 다음 명령은 conda `komap`, 물리 GPU **4,5,6,7**, `nohup`으로 준비된 run을 재개합니다.

```bash
cd /Data3/sppark/komap
bash experiments/model349_lite20_seed100_gpu4567_resume_v2/start_nohup.sh
```

실시간 로그:

```bash
tail -F /Data3/sppark/komap/experiments/model349_lite20_seed100_gpu4567_resume_v2/logs/nohup_latest.log
```

완료 수, 경과 시간, 예상 남은 시간과 GPU별 작업 로그가 터미널에 나옵니다. 별도 웹 서버는 사용하지 않습니다.
실행 시 출력되는 PID에 `kill -TERM <PID>`로 중단할 수 있으며, 같은 실행 명령으로 다시 재개합니다.
실행 전 SHA256 검증을 수행하고 완료 작업을 건너뜁니다. 다른 프로세스가 GPU를 사용 중이면 기다립니다.

## 수정과 보존

`E_b4_set0` 평가 시작 시 `PosixPath is not JSON serializable` 오류로 중단되었습니다.
`worker.call()`에서 명령 인자 전체를 문자열로 바꾼 뒤 동일한 인자 배열을 JSON 기록 및 subprocess 실행에 사용합니다.
B4/HR valid 추론과 정수 fold를 사용하는 OOF 추론을 모두 처리합니다.

- 완료된 64개 가중치와 완료 작업 기록은 원본 SHA256을 검증해 재사용합니다.
- 미완료 학습은 36개: E B4 20개, E HR 8개, F 8개입니다. 실패한 `E_b4_set0` 평가도 다시 수행합니다.
- 중단된 `E_b4_04/05/06`에는 완료 체크포인트가 없습니다. 해당 작업은 `attempt_2`에서 처음부터 다시 수행합니다. optimizer 중간 상태에서 재개하는 방식은 아닙니다.
- 기존 시도는 원본 경로를 가리키는 링크로 보존하고, 새 시도는 새 디렉터리에 씁니다.
- 완료된 평가 캐시는 파일을 복사하므로 새 캐시 잠금/재평가가 원본을 수정하지 않습니다.
- `resume_target.json`이 고정된 재개 run을 지정합니다. 실행기를 반복 호출해도 새 100회 탐색을 만들지 않습니다.
- 원본 run과 원본 데이터/가중치를 이동하거나 삭제하면 재개 검증이 실패합니다.

## 결과

전체 완료 후 `results/model_results_날짜_번호/`에 결과서, `leaderboard.csv/json`, 최종 20개 가중치와 추론 코드, test 예측을 저장합니다.
`valid_mask_comparison/`에는 valid20 각 이미지의 2×2 비교(좌상 원본, 우상 오류, 좌하 예측, 우하 정답)를 저장합니다.
맞은 픽셀은 흰색, 오류는 해당 위치의 정답 색이며 image mIoU와 100%-image mIoU를 표시합니다.
지표는 이미지별 4개 클래스 IoU 평균을 valid20에 대해 평균하며 union=0은 0입니다. global mIoU는 계산하지 않습니다.

전체 탐색 설계는 `ORIGINAL_README.md`, 재개 이력은 run의 `resume_provenance.json`, 검증 결과는 `checks/`를 참조하세요.
