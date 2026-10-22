# Model364 재현 — GPU 4,5,6,7

준비일: 2026-10-06 (한국 시간). 학습 시작은 사용자가 합니다.

## 실행

```bash
cd /Data3/sppark/komap/experiments/model364_reproduction_gpu4567_v1
./start_nohup.sh
```

conda `komap`을 자동 활성화해 `nohup`으로 학습합니다. PID, 로그 경로 및 `tail -f` 명령이 출력됩니다. 물리 GPU 4,5,6,7에 독립 모델/fold를 하나씩 배정하며 최대 4개 작업을 동시에 실행합니다. UUID로 GPU를 지정합니다. DDP가 아니며 선행 의존 단계에서는 일부 GPU만 사용합니다. 57단계(52 GPU, 5 CPU)의 진행률, 경과 시간, ETA와 각 작업의 최근 진행 메시지가 터미널 로그에 표시됩니다. 서버나 웹 대시보드는 사용하지 않습니다.

```bash
./run.sh plan                 # 실행 계획만 출력
./setup_environment.sh        # komap 의존성, CUDA, 데이터/경로 검사; 학습 없음
./run.sh train                # dry run; 학습 없음
```

현재 komap의 설치된 의존성을 재사용합니다. 기존 환경의 패키지를 교체하지 않습니다. 버전 기록은 `reports/environment.freeze.txt`와 각 실행의 `environment.freeze.txt`에 저장합니다. 공개 ImageNet 초기 가중치가 캐시에 없다면 첫 모델 생성 때 다운로드가 필요합니다.

중단은 출력된 메인 PID에 `kill PID`를 실행합니다. 실행기는 자식 작업을 종료하고 체크포인트와 로그를 보존합니다. 중복 실행은 lock으로 차단합니다. 재실행은 새 경로에서 처음부터 시작하며 자동 재개하지 않습니다.

## 원본 및 실행 구성

요청한 https://github.com/CNUKoMaP/lhs/blob/main/Model364_reproduction_code.zip 을 인증된 GitHub API로 다운로드했습니다. 원본 ZIP은 `downloads/`, 수정 없는 압축 해제본은 `upstream/Model364_reproduction_bundle/`에 보존합니다. Git blob와 ZIP SHA256은 `provenance.json`에 기록했습니다.

이 번들은 가중치·데이터·전체 선행 자동 실행기를 포함하지 않습니다. 필요한 선행 실행은 기존 `lhs_349_397_gpu4567_v1/model349` 실행기의 별도 사본으로 구성했습니다. 선행 학습 entrypoint는 Model364 번들과 바이트 단위 일치를 확인했습니다. 번들에 없는 Model183 cache 생성기, CV 준비 코드/분할 자료는 해당 실행기 사본에서 보완했습니다. 파일별 원본과 수정 여부는 `reports/source_merge.json`, 로컬 수정은 `reports/local_patches.diff`에 있습니다.

ResNet 계보 및 B4 322/326/329의 4-fold 학습, Model336 학습, Model347 fine-tune, Model351 D4 추론, Model361 scale-3 fine-tune/추론을 실행합니다. Model347과 Model361은 같은 Model336 `last.pt`에서 독립적으로 출발합니다. Model361은 번들의 epoch12, scale3, seed361, batch4/accumulation2 설정을 사용하며 weighted-CE를 적용하지 않습니다.

Model339/341/346/349/355/360의 번들 reference JSON은 **고정 조합 파라미터**로 사용합니다. Model349의 새 route search는 실행하지 않습니다. Model364의 마지막 gate는 margin=0.0, strength=0.25로 고정합니다. 원본 grid와 과거 점수 일치 assertion은 재학습 평가에 사용하지 않습니다. 실제 새 예측으로 점수를 계산하며 과거 79.94297272%는 참고 기록일 뿐 재현 보장값이 아닙니다. 일부 선행 checkpoint 선택 및 원본 조합 개발에 valid가 사용되었으므로 독립 test 점수가 아닙니다.

## 보존 및 결과

기존 모델·소스·결과를 수정하지 않습니다. 매 실행은 `outputs/runs/model364_train_.../`를 새로 만들고 데이터, 소스/adapter/실행기 스냅샷, 의존성 버전, 명령, 체크포인트, 확률 archive 및 로그를 보존합니다.

완료 후 `/Data3/sppark/komap/results/model_results_MMDD_다음번호/`에 같은 이름의 Markdown 보고서, `metrics.json`, 예측 마스크 및 `valid_mask_comparison/` 20장을 생성합니다. 날짜는 한국 시간이며 기존 결과를 덮어쓰지 않습니다.

지표는 이미지마다 4개 클래스 IoU 평균 후 valid20 이미지 평균입니다. 빈 union IoU=0이며 global mIoU를 계산하지 않습니다. 비교 그림은 좌상 Original / 우상 오류 / 좌하 Prediction / 우하 Ground truth입니다. 정답 픽셀은 흰색, 오류 픽셀은 해당 위치 정답 색이며 image mIoU와 `100%−image mIoU`를 표시합니다. 이 차이는 픽셀 오류율이 아닙니다.
