# 재현과 원본 경로

이 저장소는 코드·실험 기록 보관용입니다. 학습된 체크포인트는 포함하지 않습니다. 추론에는 별도로 보유한 해당 모델의 가중치가 필요하며, 가중치가 없으면 먼저 학습해야 합니다. 학습은 사용자가 명시적으로 실행합니다.

## 최종 제출 코드

```bash
cd team/lhs/final_lite13/1_소스코드_실행파일_개발환경/komap_model
conda env create -f environment.yml
conda activate komap
```

이미 `komap` 환경이 있다면 필요한 패키지를 먼저 비교하고 환경을 중복 생성하지 않습니다. 이 제출용 환경 정의는 팀의 Python 3.10/PyTorch 2.8 구성이고, 서버 루트의 `server/environment.yaml`과 서로 다른 기록입니다.

가중치를 별도로 준비한 뒤 추론합니다. 입력 이미지와 가중치 경로는 실제 위치로 바꿉니다.

```bash
bash run_predict.sh /path/to/weights /path/to/images /path/to/new_output 0
```

실제 제출 파일의 코드·데이터·예측 마스크 ZIP은 가중치를 제외한 상태로 `team/lhs/submission_files/A616277.zip`에 복원할 수 있습니다.

```bash
# 저장소 루트에서 실행
python tools/restore_assets.py --prefix team/lhs/submission_files/
```

전체 재학습은 최종 코드의 [README](../team/lhs/final_lite13/1_소스코드_실행파일_개발환경/komap_model/README.md)에 명시된 train 70장을 준비한 뒤 사용자가 `run_retrain.sh`를 실행합니다. 이 보관 작업에서는 학습이나 추론을 새로 실행하지 않았습니다.

## 서버 실험 기록

원래 프로젝트 루트를 `server/` 아래에 보존했습니다. 예전 코드와 JSON·로그에는 `/Data3/sppark/komap/...` 절대 경로와 물리 GPU 번호가 남아 있습니다. 이 경로들은 당시 실행 근거이며, 복사본 전체가 다른 머신에서 경로 수정 없이 실행된다는 뜻은 아닙니다.

`manifests/server_symlinks.json`은 기존 심볼릭 링크와 대상 경로를 기록합니다. 소스 파일 링크는 가능한 경우 파일 내용으로 복사했고, 데이터·캐시·가중치 링크는 자동 생성하지 않았습니다. 누락된 가중치나 캐시를 가리키는 실행을 자동으로 시작하지 않습니다.

보관된 대용량 그림·데이터·문서는 `tools/restore_assets.py`로 필요한 경로만 복원합니다. 파일명·SHA-256과 원래 상대 경로는 `manifests/assets.json`에 있습니다. 복원 도구는 기존 내용이 다른 파일을 덮어쓰지 않습니다.

## 지표

최종 비교 기준은 image mIoU(4상, 빈 union=0)입니다. 초기 보고서의 빈 union=1 결과는 당시 값 그대로 보존했으며 여기서 재채점하거나 순위를 혼합하지 않았습니다. 공식 test 정답과 점수는 이 기록에 없습니다.
