# model_results_0923_2 — evolve_accuracy_v1: accuracy_v1 baseline + evolve 기법 누적 적용 (8단계)

**상태**: 완료 (8개 모델 학습·Plain/D4 검증 평가 완료)

## 1. 실험 개요

- **부모 모델**: `experiments/accuracy_v1/baseline` (프로젝트 기존 공식 모델, `model_results_0919.md`의 "accuracy_v1" 베이스라인).
- **적용 방법**: `experiments/evolve_v1`이 정의한 8가지 evolve 기법(hires_decoder, crop_mix,
  hires_cnn_branch, boundary_aux_plus, fullres_refine, composition_head, texture_branch,
  complex_expert)을 이 순서대로 **하나씩 누적 추가**하여 총 8개 모델(`e001`~`e008`)을 만들었다.
  `e008`이 8개 기법을 전부 적용한 모델이다. 상세 설계와 코드는
  [experiments/evolve_accuracy_v1/README.md](../experiments/evolve_accuracy_v1/README.md) 참고.
- 각 단계는 직전 단계가 아니라 **매번 accuracy_v1 baseline에서 새로 warm-start**했다(기법이
  "단독으로 누적됐을 때"의 순수 효과를 보기 위함). 새로 추가되는 서브모듈은 0으로
  초기화되므로, 학습 시작 시점의 예측은 항상 baseline과 동일했다.
- 모든 단계를 동일 예산(총 900 optimizer updates, light 기법 레벨)으로 맞춰 예산 차이가
  아니라 기법 자체의 효과를 비교했다.

## 2. 결론 요약 — 베이스라인 대비 개선 없음

8개 모델 전부 accuracy_v1 baseline보다 **image mIoU·global mIoU 모두 낮았다** (D4, Plain 둘
다). 기법을 더 많이 누적할수록 점수가 오르는 경향도 없었고, 오히려 6단계(`composition_head`)
이후로는 소폭 하락했다. 이 조건(light 레벨, 900 updates, 매번 baseline에서 새로 warm-start)
에서는 8개 기법 중 어느 것도, 또 그 조합도 이 baseline 위에서 실질적 개선을 만들지 못했다.

| | 기준 | image mIoU (Plain) | image mIoU (D4) | Global mIoU (D4) |
|---|---|---:|---:|---:|
| **accuracy_v1 baseline** | 기존 공식 모델 | 78.4729% | **78.6865%** | **81.4989%** |
| 8개 중 최고 (`e002`, 누적 2/8) | +hires_decoder+crop_mix | 78.4633% | 78.6607% | 81.4912% |
| 8개 중 최저 (`e008`, 누적 8/8) | 8개 기법 전부 | 78.4215% | 78.6163% | 81.4046% |

D4 기준으로 최고인 `e002`조차 baseline보다 **-0.026%p** 낮다. 8개 모두 baseline과의 차이가
±0.07%p 이내로, evolve_v1이 hundred_v1의 top-10 parent들에서 관찰한 개선 폭보다 훨씬 작고
오히려 방향이 반대다. 원인으로는 (a) 이 baseline이 이미 자체적으로 잘 수렴된 상태라 light
레벨·900 updates·매우 낮은 학습률(1e-6~2e-6)의 짧은 파인튜닝으로는 새 zero-init 모듈이
유의미하게 학습되기 어려웠을 가능성, (b) 매번 baseline에서 새로 시작(체이닝 없음)하다 보니
기법 간 상호작용을 학습할 기회가 없었을 가능성을 들 수 있다. **이 8개 모델은 채택하지 않고
기존 최고 모델(모델 42/50, ResNet34 U-Net++ + Residual ASPP 앙상블, 79.48%)을 그대로
유지할 것을 권장한다.**

## 3. 전체 결과표 (D4)

| 누적 단계 | 실험 | 이번에 추가된 기법 | 이미지 mIoU | Global mIoU | Primary Si IoU | Al3Ni IoU | Eutectic Si IoU | Al IoU |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| — | **accuracy_v1 baseline** | (기준) | **78.6865%** | **81.4989%** | 90.3304% | 79.0646% | 62.1277% | 94.4727% |
| 2 | `e002_c2_add_crop_mix` | +crop_mix | 78.6607% | 81.4912% | 90.3056% | 79.0570% | 62.1246% | 94.4774% |
| 3 | `e003_c3_add_hires_cnn_branch` | +hires_cnn_branch | 78.6600% | 81.4912% | 90.3052% | 79.0581% | 62.1237% | 94.4776% |
| 5 | `e005_c5_add_fullres_refine` | +fullres_refine | 78.6588% | 81.4810% | 90.2954% | 79.0434% | 62.1152% | 94.4700% |
| 4 | `e004_c4_add_boundary_aux_plus` | +boundary_aux_plus | 78.6578% | 81.4801% | 90.2945% | 79.0425% | 62.1134% | 94.4702% |
| 1 | `e001_c1_hires_decoder` | hires_decoder (단독) | 78.6385% | 81.4375% | 90.2758% | 78.9810% | 62.0326% | 94.4608% |
| 7 | `e007_c7_add_texture_branch` | +texture_branch | 78.6209% | 81.4046% | 90.2151% | 78.9312% | 61.9986% | 94.4736% |
| 6 | `e006_c6_add_composition_head` | +composition_head | 78.6194% | 81.4041% | 90.2145% | 78.9317% | 61.9968% | 94.4736% |
| 8 | `e008_c8_all_techniques` | +complex_expert (전부) | 78.6163% | 81.4046% | 90.2155% | 78.9331% | 61.9954% | 94.4745% |

Plain(비-TTA) 결과와 클래스별 전체 수치는
[experiments/evolve_accuracy_v1/outputs/reports/results.md](../experiments/evolve_accuracy_v1/outputs/reports/results.md)
와 `outputs/evaluations/*/{{plain,d4}}/metrics_summary.json`에 보존되어 있다.

## 4. 검증셋 마스크 비교 이미지

`valid_mask_comparison/<experiment>/<image_id>.png` (8개 모델 × 20장 = 160장, D4 예측 기준
원본/오차/예측 마스크/실제 마스크 4패널). `valid_mask_comparison/index.html`을 브라우저로
열면 모델별로 훑어볼 수 있다.

## 5. 실행 방법 (재현)

```bash
cd /Data3/sppark/komap
conda activate komap
python -B experiments/evolve_accuracy_v1/run.py preflight
python -B experiments/evolve_accuracy_v1/run.py suite   # 8개 순차 학습 + 평가 + 리포트
```

산출물 경로:

- 체크포인트: `experiments/evolve_accuracy_v1/outputs/runs/<experiment>/best.pt`
- 검증 지표: `experiments/evolve_accuracy_v1/outputs/evaluations/<experiment>/{plain,d4}/metrics_summary.json`
- 리포트: `experiments/evolve_accuracy_v1/outputs/reports/results.md`, `results.csv`

---
*본 실험은 [spec.md](../../spec.md) 및 [AGENTS.md](../../AGENTS.md) 지침에 따라 기존 모델,
소스코드 및 이전 결과(accuracy_v1, hundred_v1, evolve_v1)를 일절 수정하지 않고 독립 경로
(`experiments/evolve_accuracy_v1/`)에서 구축되었다.*
