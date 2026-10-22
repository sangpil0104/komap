# fusion805 1단계 OOF fold × 시드 재학습 설계 (`oof_expert/`)

R은 `Model349_exh14/oof_design`으로 만든 exh14 계열 앙상블이고, E는 별도 전문가 앙상블입니다. 1단계 규칙은 E가 고른 클래스에 해당하는 픽셀만 E로 덮어씁니다. 같은 fold × 시드 설계로 E를 학습하고 규칙을 OOF로 고르는 코드입니다.

검토 결과에 따라 **2단계 잔차 보정기는 이 설계에 넣지 않았습니다.** 1단계를 고정한 뒤 2단계를 시편 제외로 고르면 1단계보다 −0.16%p였기 때문입니다.

## 구성

- **E = 0.5 × B4-329 (D4 TTA) + 0.5 × 폭 정규화 HRNet392 (flip4)**
  - B4-329는 R 학습(`oof_design`)에서 만든 `b4_329` 체크포인트를 재사용합니다.
  - HRNet392는 `train_expert.py`가 fold × 시드로 학습합니다. 학습 데이터는 R과 같은 fold 분할을 쓰고, `--rng-mode per_item`, 시드 S+1로 학습합니다.
- **규칙 후보:** 네 클래스의 모든 부분집합 16가지입니다. 빈 집합은 R만 쓰는 경우이고, 원래 fusion805 규칙은 {Al₃Ni, Al}입니다. 순서는 집합 크기, 그다음 사전순으로 고정했고, 점수가 같으면 앞의 규칙을 고릅니다.

## 실행 (`oof_design` 1~4단계 이후)

```bash
cd Model349_fusion805/oof_expert
for k in 1 2 3 4; do for s in $(seq 0 99); do python train_expert.py --work WORK --fold $k --seed $s; done; done
for k in 1 2 3 4; do python predict_oof_expert.py --work WORK --fold $k; done
python select_rule.py --work WORK
# valid 20장 1회 채점 (A 방식: valid는 학습·선택에 쓰지 않음)
python ../Model349_exh14/oof_design/infer.py --work WORK --images DATA/valid/images --gt DATA/valid/masks --output R_VALID
python infer_fusion.py --work WORK --images DATA/valid/images --r-labels R_VALID/predicted_labels --gt DATA/valid/masks --output FUSION_VALID
# 제출 마스크
python ../Model349_exh14/oof_design/infer.py --work WORK --images TEST/images --output R_TEST
python infer_fusion.py --work WORK --images TEST/images --r-labels R_TEST/predicted_labels --output FUSION_TEST
```

기본은 A 방식입니다(`oof_design/README.md` 참고). train 70장으로 학습하고 규칙도 OOF70으로 고르며, valid 20장은 마지막에 한 번만 채점합니다.

## 보고하는 점수 (`work/select_rule/report.json`)

| 항목 | 의미 |
|---|---|
| `dev_oof` | 전체 OOF에서 규칙을 고른 뒤 같은 이미지로 잰 점수 |
| `r_only_dev_oof`, `original_fusion805_rule_dev_oof` | R만 쓴 경우와 원래 규칙 {Al₃Ni, Al}의 점수 (비교용) |
| `outer_estimate` | 바깥 fold j마다 R 설정과 규칙을 모두 나머지 fold로 다시 고르고, fold j로 잰 점수를 합산한 값. **선택까지 포함한 추정치** |
| `per_seed.mean` / `per_seed.sd` | 시드 하나의 R과 E에 최종 규칙을 적용한 OOF 점수의 평균과 표준편차 |

정답은 `select_rule.py`가 채점할 때만 읽습니다. 학습과 예측에는 쓰지 않습니다.
