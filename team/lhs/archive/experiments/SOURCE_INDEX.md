# 실험 source 색인

`historical_scripts/`에는 과거 실험 작업공간의 최상위 Python source 210개를 보존했다. 모델 번호별 목적·상태·결과의 기준 문서는 [`docs/model_results.md`](../docs/model_results.md)이며, 지표별 최고 선택은 [`docs/best_models_by_miou.md`](../docs/best_models_by_miou.md)에 요약한다.

모델 번호는 실험 결과와 route 변형을 식별하며 서로 다른 구조나 코드 파일의 개수를 의미하지 않는다. 학습 변형은 같은 trainer를 공유할 수 있고, 앙상블/보정 모델은 탐색 코드와 저장된 설정으로 정의될 수 있다. 2026-10-02 기준 이 보관소에는 Python/shell source 326개(275 `.py`, 51 `.sh`)가 추적되어 있다. 전체 경로는 [`SOURCE_MANIFEST.md`](SOURCE_MANIFEST.md)에 있다. 이전 감사에서 빠졌던 4개 launcher는 `runners/legacy/`에 보존했다. 서로 다른 시점의 사본이 있으면 명시적 B4 channel-order 처리 및 Model336 weighted-CE warm start처럼 수정된 archive 버전을 우선하며, 오래된 workspace 사본으로 덮어쓰지 않는다.

## 모델 번호별 주요 코드 위치

| 모델/계열 | 주요 경로 | 내용 |
|---|---|---|
| 1–321 | `historical_scripts/`, `runners/` | 과거 학습·추론·보정·route 검색·감사 코드. 파일명은 원래 이름을 유지했다. |
| 322 | `model322_team_b4/KoMaP/` | 팀 EfficientNet-B4 recipe와 grouped-OOF 도구. |
| 323–335 | `historical_scripts/`, `runners/` | B4 재현, route 결합, grouped-OOF와 지표 감사. 여러 번호가 evaluator를 공유한다. |
| 336–337 | `model336_hrnet_best/`, `historical_scripts/` | HRNet-W48/U-Net++ trainer·추론·route 검색. |
| 338 | `historical_scripts/model338_fine_gamma_route_search.py` | route gamma 세밀 탐색. |
| 339–346 | `experiments/` 최상위 번호 파일 | joint/fine route 탐색, OOF 학습 blend, exact scorer. 자세한 연결은 결과 기록의 코드 링크를 따른다. |
| 347–349 | `model336_hrnet_best/`, `model346_model336_into_341_route.py`, `runners/` | weighted-CE HRNet fine-tune과 공용 scorer를 쓰는 route 변형. |
| 350–353 | `model350_model346_fine_gamma.py`, `model336_hrnet_best/`, `runners/` | gamma 탐색, D4 TTA, D4 route 검색. |
| valid/test 감사 | `valid_test_group_similarity_audit.py` | 원본 영상 특징과 specimen-group 분포 비교. label과 mask는 읽지 않는다. |

## 재현용 주요 구현

- `model322_team_b4/KoMaP/`: 팀 EfficientNet-B4 recipe와 grouped-CV 유틸리티.
- `model326_hybrid_fullstats/KoMaP/`: full-image-stat 입력, 학습·추론·grouped-OOF 집계/평가, per-image Lovasz loss.
- `model336_hrnet_best/`: HRNet-W48/U-Net++ 구조, dataset, loss, 추론, evaluator와 실행 설정.
- `model339_joint_gamma_route_search.py`: CPU 400-setting joint-gamma route 검색. 공용 route helper를 import하려면 `PYTHONPATH=experiments/historical_scripts`를 지정한다.
- `model339_grouped_cv_parent_audit.py`: Model339 route 선택과 Model338 고정 route의 paired leave-one-specimen-group-out 비교를 재계산한다.
- `model340_partial_oof_metric_audit.py`: Model329 OOF 저장 mask로 empty=0 image-wise mIoU를 다시 계산한다.
- `model341_model329_joint_route_search.py`: Model339/329의 626-setting class-wise route 검색.
- `model342_oof_fitted_b4_blend.py`: train-only grouped-OOF로 Model322/329 class-wise 확률 blend를 맞춘 뒤 고정 설정을 validation에서 채점한다.
- `model343_model342_into_341_route.py`: Model342를 Model341에 추가하는 625-setting probe.
- `model333_partial_oof_audit.py`: Model333 CE+Dice/Lovasz arms의 exact empty=0 OOF 감사. 채널 순서를 명시해 class permutation 오류를 막는다.
- `model344_model333_control_blend_search.py`, `model345_model344_into_341_route.py`: train-OOF-fitted blend와 고정 branch route 비교.
- `model346_model336_into_341_route.py`: Model346/348/349/352/353에서 공유하는 CPU exact scorer. class order, probability normalization, validation target, baseline parity를 검사한다.
- `model350_model346_fine_gamma.py`: Model346 Al/Eutectic-Si route 계수를 0.025 간격으로 정밀 탐색하고 grouped-CV를 계산한다.
- `model336_hrnet_best/komap_seg/inference.py`: 기존 flip4를 유지하면서 opt-in 8-way dihedral probability TTA를 제공한다.
- `runners/`: Model325/326/328/329/333/341–353의 P100 실행 명령과 queue.
- `runners/legacy/`: 과거 launcher의 provenance. `/proc` PID/GPU 가정이 오래됐으므로 그대로 실행할 수 있는 명령이 아니다.

## 제한사항

색인에 있다는 사실만으로 모든 과거 실험이 source만으로 재현되는 것은 아니다. 일부 route 검색은 저장소에 없는 확률 cache·checkpoint·metadata를 입력으로 요구한다. 어떤 artifact가 필요한지는 실험 기록을 확인한다.
