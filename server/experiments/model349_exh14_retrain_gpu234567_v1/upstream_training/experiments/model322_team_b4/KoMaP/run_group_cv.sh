#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
CODE="$ROOT/model322_team_b4/KoMaP"
PY=/workspace/venv-komap/bin/python
DATA="$ROOT/data"
CV_DATA="$ROOT/data_model322_group_cv"
OUT="$ROOT/outputs/model322_team_b4_group_oof"
PARENT="$ROOT/outputs/model303_boundary_oof_8ep"

if [[ ! -d "$CV_DATA" ]]; then
  "$PY" "$CODE/prepare_group_cv.py" --data "$DATA" \
    --manifest "$PARENT/progress.json" --out "$CV_DATA"
fi
mkdir -p "$OUT"

for fold in 1 2 3 4; do
  run="$OUT/fold$fold"
  if [[ ! -f "$run/last.pt" ]]; then
    mkdir -p "$run"
    echo "=== train grouped fold $fold: fixed 16000 updates ===" | tee -a "$OUT/run.log"
    "$PY" "$CODE/train.py" --data "$CV_DATA" --train_splits "train_fold$fold" \
      --out "$run" --arch unet --encoder efficientnet-b4 --norm imagenet \
      --crop 1536 --scale 3 --bs 2 --iters 16000 --lr 3e-4 --wd 1e-2 \
      --dice_w 1 --cldice_w 0 --seed 0 --workers 4 --no_valid_eval \
      > "$run/train.log" 2>&1
  else
    echo "=== fold $fold checkpoint already exists; keeping it ===" | tee -a "$OUT/run.log"
  fi
  if [[ ! -f "$OUT/fold${fold}_oof/probabilities.npz" ]]; then
    "$PY" "$CODE/predict.py" --data "$CV_DATA" --split "oof_fold$fold" \
      --runs "$run" --out "$OUT/fold${fold}_oof" --save_probs \
      > "$OUT/fold${fold}_oof.log" 2>&1
  fi
done

"$PY" "$CODE/aggregate_oof.py" --runs "$OUT" --folds 4 --expected-images 70
if [[ ! -f "$OUT/valid_ensemble/probabilities.npz" ]]; then
  "$PY" "$CODE/predict.py" --data "$DATA" --split valid \
    --runs "$OUT/fold1" "$OUT/fold2" "$OUT/fold3" "$OUT/fold4" \
    --out "$OUT/valid_ensemble" --save_probs \
    > "$OUT/valid_ensemble.log" 2>&1
fi
"$PY" "$CODE/evaluate_group_cv.py" --data "$DATA" \
  --parent-oof "$PARENT/train_oof_probs.npz" --candidate-oof "$OUT/train_oof_probs.npz" \
  --parent-valid "$PARENT/valid_fold_ensemble_probs.npz" \
  --candidate-valid "$OUT/valid_ensemble/probabilities.npz" \
  --out "$OUT/results.json" > "$OUT/evaluate.log" 2>&1
echo "DONE: $OUT/results.json" | tee -a "$OUT/run.log"
