# 실험: E의 B4를 train70 전체로 BN 고정 추가 학습 (팀원 실행용, 미검증)

**아직 실행하지 않은 실험입니다. 점수는 없습니다.**

## 목적
- 현재 후보(lite23 + B4ws, valid20 80.5464%)의 B4 추가 학습은 fold마다 그 fold의 학습 이미지(약 52장)만 사용했습니다.
- 이번에는 레시피는 그대로 두고 데이터만 **train70 전체**(`train_foldK` + `oof_foldK`)로 바꿉니다.
- 데이터가 늘어 좋아질 수 있습니다. 반대로 4개 fold 모델이 같은 70장을 보게 되어 서로 비슷해지면 fold 평균의 앙상블 효과가 줄 수 있습니다. 이 손익을 확인하는 실험입니다.
- valid·test 정답은 학습에 쓰지 않습니다.

## 실행

```sh
# 1) 학습 (fold당 P100 약 34분, 4 fold). 부모는 ASSETS/model329/foldK/last.pt (lite23 원래 B4)
bash full70_experiment/train_b4_full70.sh /path/Model349_fusion805 /path/model397-output/splits/model329_cv /path/assets 0

# 2) valid20 채점 (R은 이전 추론 결과 재사용 가능)
python infer_compact.py --config selected_config_b4full70.json \
  --parent-package /path/Model349_fusion805 --assets /path/assets \
  --input /path/valid/images --release /path/release_valid --output /path/out_full70 --gpus 0
python /path/Model349_fusion805/score_masks.py --labels /path/out_full70/fusion/predicted_labels \
  --gt /path/valid/masks --output /path/out_full70/metrics.json
```

- `selected_assets_b4full70.json`에는 새로 학습할 B4 4개가 들어 있지 않습니다. 이 4개의 SHA는 검사하지 않고 `expert/completion.json`에 기록만 합니다. 나머지 19개는 SHA를 확인합니다.

## 판단 기준 (미리 정함)
- 비교 대상은 lite23 + B4ws 80.5464316%입니다(`results_b4ws/`의 이미지별 점수).
- **채택 조건:** valid20 평균이 이보다 높고, 가장 나빠진 이미지의 손실이 0.1%p 이내일 때만 채택합니다.
- OOF 평가는 할 수 없습니다(70장 전부 학습). 그래서 고정 후보 하나만 채점합니다. 여러 변형을 valid로 골라 채택하지는 않습니다.
- 결과가 좋으면 같은 방식을 HR fold 모델(seed1 fold2·4, seed2 fold1·4)에도 적용해 볼 수 있습니다.
