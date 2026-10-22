# Model364 재현 범위와 선행 입력

## Route 계보

`Model364 = Model360 fixed route + Model361 entropy-gated expert`

Model360은 Model355 spatial blend와 Model351 D4 probability를 사용합니다. Model355는 Model349 route, Model349는 Model346에 Model347을 추가한 route이며 Model346은 Model341에 Model336을 추가합니다. 그보다 앞의 Model341/339 route는 Model329를 비롯한 여러 예측원을 Model183의 image metadata와 결합합니다.

재현용 기록 파라미터:

| 단계 | 고정 reference |
|---|---|
| Model346 Model336 class gamma | `[0.0, 0.0, 1.0, 0.25]` |
| Model349 Model347 class gamma | `[0.0, 0.0, 0.25, 0.0]` |
| Model355 spatial blend | `[0.75, 0.0]` |
| Model360 Model351 entropy gate | margin `0.0`, strength `1.0` |
| Model364 Model361 entropy gate | margin `0.0`, strength `0.25` |

클래스 gamma의 순서는 `[Al3Ni, Primary Si, Al, Eutectic Si]`입니다. 상세한 Model361 학습 recipe는 `../config/model364_recipe.json`을 참조하세요.

## 평가에 필요한 프로젝트 입력

아래 경로는 `KOMAP_ROOT` 기준입니다. 데이터와 모델 산출물은 외부에서 준비하며 이 번들에 포함되지 않습니다.

```text
data/valid/images/*
data/valid/masks/*_mask.png
outputs/model183_metric_calibration/entries.pkl
outputs/model195_native_resume/valid_probs.npz
outputs/model220_native_tent_extended/valid_probs_06.npz
outputs/model269_fullstats_1x_extended/valid_probs.npz
outputs/model322_team_b4_group_oof/valid_ensemble/probabilities.npz
outputs/model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz
outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz
outputs/model336_hrnet_best/valid_eval/valid_probabilities.npz
outputs/model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz
outputs/model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz
outputs/model361_hrnet_scale3_finetune/valid_eval/valid_probabilities.npz
outputs/model339_joint_gamma_route_search/results.json
outputs/model341_model329_joint_route_search/results.json
outputs/model346_model336_into_341_route/results.json
outputs/model349_model347_into_346_route/results.json
outputs/model355_model349_spatial_audit/results.json
outputs/model360_entropy_gated_d4_into_355/results.json
```

`results/outputs/` 아래의 작은 JSON들은 route parameter/reference 결과입니다. `scripts/install_route_references.sh`는 해당 파일 중 프로젝트 `outputs/`에 없는 것만 설치하고, 기존 파일이 하나라도 있으면 변경하지 않고 종료합니다. 이는 체크포인트나 확률 archive를 만들지 않습니다.

## Trainer 및 외부 초기 가중치

Model336/347/361은 `experiments/model336_hrnet_best/`의 동일한 HRNet-W48 + SMP U-Net++ 구현으로 학습합니다. Model347은 class-weighted CE fine-tune이며, Model361은 unweighted CE 설정에서 scale 3 fine-tune입니다. Model361은 Model336 `last.pt`에서 시작합니다. Model351은 별도 학습이 아니라 Model336 체크포인트의 D4 TTA 추론입니다.

EfficientNet-B4와 이전 ResNet 계열의 trainer/inference source 및 route code는 `experiments/`의 inherited Model349 code에 포함됩니다. 정확한 Model364 결과를 생성하려면 그 계열의 체크포인트와 해당 validation 확률 archive를 각각 먼저 만들어야 합니다. 이 ZIP은 데이터, ImageNet weight cache, 학습된 checkpoint, route metadata pickle, validation probability archive를 포함하지 않고, 모든 upstream lineage를 한 번에 실행하는 자동화 pipeline도 제공하지 않습니다. 기록 점수의 완전 재현에는 나열된 모든 입력이 필요합니다.

## 결과 해시 기준

번들 내 기준 결과는 Model360 `79.9416619634%`, Model364 raw valid `79.9429727229%`, group-held-out 값 `79.9372749517%`입니다. JSON은 원본 실험 파일을 보존합니다. raw-valid 최댓값은 같은 20장으로 route를 선택한 값입니다. Model364의 group-held-out delta가 음수이므로 이 조합을 독립 일반화 개선으로 해석하지 않습니다.
