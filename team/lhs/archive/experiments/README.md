# 실험 코드 안내

이 디렉터리에는 KoMaP 분할 실험의 Python/shell source가 보관되어 있다. 모델 번호는 실험 결과 식별자이며 번호마다 독립된 코드 파일 하나가 대응한다는 뜻은 아니다. 일부 번호는 같은 trainer/scorer를 공유하는 학습 변형이나 앙상블 route다.

Model349 학습 묶음을 재현하려면 외부에서 `train.zip`, `valid.zip`을 준비한 뒤 `KOMAP_DATA_DIR=/path/to/archives bash experiments/run_model349_pipeline.sh train`을 실행한다. 조상 모델과 구성 branch를 재학습하고 Model335→349 route를 다시 탐색한다. 기존 checkpoint, 예측 결과, 대회 영상은 코드 묶음에 포함하지 않는다. 자세한 절차는 [Model349 재현 안내](../docs/model349_reproduction.md)를 참고한다.

## 주요 경로

- `historical_scripts/`: 과거 학습·추론·감사·탐색 코드 210개. 기존 파일명으로 보존했으며 공유 import를 위해 `PYTHONPATH=experiments/historical_scripts`가 필요할 수 있다.
- `model322_team_b4/KoMaP/`: 팀 EfficientNet-B4 학습 recipe와 grouped cross-validation 도구.
- `model326_hybrid_fullstats/KoMaP/`: Model326/329/333에서 사용한 full-image-stat 입력, 학습·추론·grouped-OOF 도구.
- `model336_hrnet_best/`: HRNet-W48/U-Net++ 모델, 데이터 파이프라인, loss, 학습·예측·평가 코드. Model347은 이 trainer에 train-only CE 가중치와 warm start를 사용한다.
- 최상위 `modelNNN_*.py`: 이후 route 검색, OOF 감사 및 지표 재계산 코드. 예를 들어 `model346_model336_into_341_route.py`는 Model346/348/349/352/353의 exact scorer를 공유하고 `model350_model346_fine_gamma.py`는 route 계수를 미세 탐색한다.
- `valid_test_group_similarity_audit.py`: mask나 test label을 읽지 않는 valid/test 원본 영상 유사도 감사.
- `runners/`: P100 실행/queue 명령. 일부 runner는 순차 gate를 사용한다. `runners/legacy/`는 과거 실행 근거 보존용이며 프로세스·GPU 가정이 오래되어 그대로 실행하면 안 된다.
- `../results/outputs/`: 선택된 JSON 설정과 점수 기록. 데이터, checkpoint, 예측 mask, 확률 tensor 등 큰/민감 산출물은 제외한다.

## 기준 문서

- `../docs/model_results.md`: 모델 번호별 방법·상태·결과 연혁.
- `SOURCE_INDEX.md`: 모델군별 코드 위치 안내.
- `SOURCE_MANIFEST.md`: Python/shell source 전체 경로 목록(326개: Python 275개, shell 51개).
- `../docs/best_models_by_miou.md`: 지표 정책별 순위. 공식 점수는 image-wise 4-class mIoU, empty union=0이며 NaN 제외/pooled-global은 참고값이다.

## 재현 범위

저장소에는 source와 일부 metadata/result만 있고, 비공개 대회 데이터·checkpoint·prediction·확률 cache·인증 정보나 완전히 고정된 실행환경은 없다. 따라서 과거의 일부 route 탐색은 단독 실행이 불가능하며 입력 artifact를 별도로 준비해야 한다. 최근 runner는 기록 당시 `/workspace/komap_model1`, `/workspace/venv-komap`을 사용했으므로 승인된 환경에서 경로를 확인·조정한다. test label이 제공되지 않아 test mIoU는 없다.
