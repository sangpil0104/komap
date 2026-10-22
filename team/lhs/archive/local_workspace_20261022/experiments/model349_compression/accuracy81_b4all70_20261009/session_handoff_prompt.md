# 다른 세션에 전달할 프롬프트

KoMaP 2026 모델 성능 개선 작업을 중단 지점부터 이어서 진행해 줘. 아래 정보는 인계 당시 확인 기록이며 현재 서버 상태를 뜻하지 않는다. 파일과 프로세스로 다시 확인하고, 기존 작업을 중복 실행하지 말아 줘.

## 작업 환경과 안전 조건

- 로컬: `/Users/hyunseok/orca/workspaces/KoMaP/gurnard`
- 서버: `ssh -p 20670 root@168.188.128.36`
- 서버 실험 부모: `/workspace/model349_compress`
- 서버 Python: `/workspace/venv-komap/bin/python`
- 서버를 사용하기 전에 작업 디렉터리에 있는 GPU 사용자 매뉴얼 PDF를 읽고 준수해 줘. 매뉴얼의 비밀번호 등 인증정보는 출력·기록·업로드하지 마.
- 다른 사용자의 GPU 작업을 건드리지 말고 GPU 점유, 실제 프로세스, 컨트롤러와 lock을 함께 확인해 줘. GPU3도 현재 비어 있다고 가정하지 마.
- 작업 트리의 기존 변경·미추적 파일, 원본 가중치, 릴리스와 base 포인터를 보존해 줘. 자동 재시작이나 원본 덮어쓰기는 금지한다.
- 이전 세션은 SSH가 `Operation not permitted`로 차단돼 현재 진행 상태를 확인할 수 없었다. 서버 작업이 종료되었다는 증거는 아니다. 이 세션도 차단되면 우회하지 말고 확인 불가 상태와 인계 내용을 보고해 줘.

## 목적·평가·현재 base

목표는 규정 내에서 정확도를 높이는 것이다. 사용자는 모델 정확도 배점이 큰 점을 들어 성능 손실을 감수한 경량화보다 정확도를 우선했다. 제공 train70만 감독학습에 사용하고 valid20/test10 정답을 학습에 넣지 않는다. 외부 데이터 사용은 허용된 것으로 가정하지 않는다.

공식 지표는 이미지마다 4개 클래스 IoU를 평균한 뒤 valid20장을 평균한 image mIoU다. 없는 클래스의 union이 0이면 IoU는 0이다. PNG와 GT로 20개 이미지·80개 IoU를 독립 채점하고 실행 metrics와 대조한다. test10에는 정답이 없으므로 test mIoU 또는 픽셀 불일치율에 근거한 성능 증감을 주장하지 않는다.

현재 base는 `experiments/model349_compression/model349_base.json`의 **Model349_exh14**다. 서버 `/workspace/model349_compress/Model349_exh14`, 원본 FP32 14가중치 1,755,428,956bytes. canonical valid **80.23123131980379%**. 기존 동일 GPU3 재추론 control **80.23122801008668%**, 단회450.78929118초. canonical과 control의 미세 수치 차이는 기록한다. 반복 사용한 valid 성적은 독립 일반화 보장이 아니다. 81% 목표는 아직 달성하지 못했다.

최고 원본 모델은 팀원 GPU 환경에서 학습되었다는 사용자 설명이 있다. 팀원 GPU 종류와 정확한 소프트웨어 환경은 확인되지 않았다. GPU/수치 환경 차이가 영향을 줄 가능성은 있으나 최근 P100 추가학습은 LR·업데이트 수·데이터 노출·BN 설정도 달라 **GPU만의 성능 비교가 아니다**. 원본 가중치는 P100 추론에서도 약80.2312%를 재현했다. P100으로 학습하면 반드시 나빠진다고 결론내리지 말고 원본을 보존하며 검증된 개선만 후보로 삼아 줘.

## 가장 먼저 조사할 현재 실험

로컬/서버 동일 이름: `accuracy81_b4all70_20261009`

- 로컬: `experiments/model349_compression/accuracy81_b4all70_20261009/`
- 서버: `/workspace/model349_compress/accuracy81_b4all70_20261009/`
- 먼저 `protocol.md`, `smoke_input_amendment.md`, `preflight_source_lock.json`, `controller.json`, `run_sequence.py`, `train.py`, `collect_results.py`를 읽어 줘.
- 인계 전 마지막 서버 확인: controller PID **2205897**, fold3 학습 자식 PID **2207876**, fold3 **50/400** 업데이트. PID는 재사용될 수 있으므로 명령행·시작 시각·부모 관계도 확인한다. 현재 생존 여부는 미확인이다.
- fold1은400updates 완료: train70 모두 노출,607.48315713초,peak CUDA14,139,793,920bytes,FP32 checkpoint81,607,472bytes. SHA256 `6bbfe0115a1926536394cff1d34f5818ae1af0c4bb194296409dc29ff0ac786f`.
- 로컬 `latest_verified_wait.json`은 마지막 직접 확인보다 오래된 스냅샷이다. `training_completion_snapshot.json`도 현재 서버를 대신하지 않는다.

서버에서 전체 controller/process 목록, GPU compute apps, sibling `gpu3.lock`, 위 실험의 로그·각fold 완료파일·루트 `completion.json`·`runs/b4all70_valid20/completion.json`·부분 PNG를 조사해 줘. 서버 PID와 container PID가 다를 수 있으므로 숫자 차이만으로 외부 작업이라고 판단하지 않는다.

컨트롤러가 살아 있으면 그대로 진행을 관찰하고 새 작업을 시작하지 않는다. 컨트롤러가 없으면 완료 여부와 실패 로그를 먼저 구분한다. 중단이라면 checkpoint와 optimizer/RNG 상태를 조사하고 프로토콜이 허용하는 재개만 수행한다. lock 파일만 삭제하거나 학습을 처음부터 재실행하지 않는다.

## 동결된 실험 레시피

base의 Model329 fold1·3·4를 전부 유지하고 원본 각fold를 초기값으로 train70에 각400 optimizer updates를 추가한다. 세fold에 같은800samples를 같은 순서로 적용하며 GPU3에서 순차 실행한다. fold별 valid 선택·중간epoch 평가·fold 제거·가중치 soup를 하지 않는다. 추가학습 뒤 세모델은 모두 train70을 보므로 OOF 모델이라고 부르지 않는다.

가로4000/full-image mean·std·CDF/zcdf3, 원본 증강, crop1536/batch2, seed20261009, CE+Dice eps1, fresh AdamW LR1e-5/wd1e-2/cosine400/clip1, FP16 AMP+FP32 loss/GradScaler1024, BN running buffer 동결·affine 학습, FP32 저장. 세fold 마지막400updates를 모두 바꾼 전체14가중치로 원본 valid20부터 fresh 추론한다. 나머지11가중치·융합·TENT5·실제identity TTA는 유지한다. 원본 fold 학습 범위에 대해 계산량을 맞춘 대조군은 없어 all70만의 효과를 분리할 수 없다.

최초 독립 random3채널 synthetic AMP gate에서 비유한 gradient가 발생해 실제학습 전에 종료되었다. 실패 자료는 `attempt1_failed_synthetic/`에 보존됐다. 같은 설정의 실제 전처리 train 고정crop0·1 진단은 통과했고 smoke 입력만 변경했다. 판정·loss scale은 완화하지 않았다. 현재 소스와 preflight는 이후 동결됐으므로 진행 중 임의 수정하지 않는다. 실제학습 비유한값/OOM/외부GPU점유 시 우리 자식만 안전 종료하고 자동 재학습하지 않는다.

## 완료 후 수집·검증

서버 루트 completion과 세fold receipt 및 whole-fresh 추론 완료를 확인한 뒤 로컬에서 다음을 실행한다. 완료 전에는 collector를 실행하지 않는다.

```sh
cd /Users/hyunseok/orca/workspaces/KoMaP/gurnard
python3 experiments/model349_compression/accuracy81_b4all70_20261009/collect_results.py
```

collector는 서버 CPU에서 세 최종 checkpoint를 원본과 비교하여 키·shape·dtype·finite·BN exact·실제tensor변경·원본SHA·동일800sample순서/70장 노출을 검사한다. archiveSHA와 안전추출 및 source lock을 확인하고 PNG/GT 독립 채점을 수행한다. `comparison.json`과 `report.md`에 canonical base/동일GPU3 control 대비 전체·클래스·20이미지 변화, 체크포인트 bytes, 전체fresh 단회시간을 기록한다. 오류가 있으면 원인을 보존하고 검증 없이 통과로 처리하지 않는다.

collector의 로컬GT 경로가 없어졌다면 기존 GT 위치를 찾아 동일20개/파일SHA를 대조한다. 다른 GT나 평가 기준으로 조용히 교체하지 않는다. 기존 채점 코드는 `experiments/model349_compression/audit_20261008/analyze_runs.py`다.

성능이 떨어지면 기각 판정을 기록하고 원본 base를 유지한다. 개선해도 반복valid 선택·독립 GT 없음·단회시간 한계를 명시한다. 81%를 달성했다면 원본영상 fresh 재현과 독립PNG 검증을 마친 뒤 test10 마스크·원본 대비 일치율·한국어 레시피를 정리한다. 채택이나 base 변경은 근거와 재현 자료를 갖춘 뒤 사용자의 기존 승인 범위를 확인한다.

## 이미 끝난 후보를 중복하지 않기

`experiments/model349_compression_progress.md`를 읽어 전체 경과를 확인해 줘. 최근 HR336 문맥확대1024crop 후보는 raw valid80.1820844324%, 원본 대비−0.0491468874%p로 완료·기각됐다. HR336 BN통계 보간, 최근 HR 미세조정·잔차보정·밝기보정 후보도 기각됐다. B4D4+HRflip4는80.2548841310%였지만 단회1350.30초이며 반복valid의 소폭 탐색 개선으로 81%나 일반화 개선을 입증하지 못했다. 기존 실험 결과를 새 결과로 보고하지 않는다.

먼저 실제 서버 감사 결과와 B4 실험의 현재 단계/완료 여부를 알려 주고, 필요한 수집·검증·기록부터 끝내 줘. 사용자는 단순 계획보다 실제 진행과 확인 가능한 결과를 원한다.
