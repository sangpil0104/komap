#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
CODE="$ROOT/model326_hybrid_fullstats/KoMaP"
PY=/workspace/venv-komap/bin/python
DATA="$ROOT/data"
CV_DATA="$ROOT/data_model322_group_cv"
OUT="$ROOT/outputs/model326_hybrid_fullstats_group_oof"
PARENT="$ROOT/outputs/model303_boundary_oof_8ep"

if [[ ! -d "$CV_DATA" ]]; then
  "$PY" "$CODE/prepare_group_cv.py" --data "$DATA" \
    --manifest "$PARENT/progress.json" --out "$CV_DATA"
fi
mkdir -p "$OUT"

# Before committing to four long folds, verify global-stat normalization on the
# copied EfficientNet-B4 training/inference path with a short smoke run.
SMOKE="$OUT/smoke_global_stats"
if [[ ! -f "$SMOKE/last.pt" ]]; then
  mkdir -p "$SMOKE"
  "$PY" "$CODE/train.py" --data "$CV_DATA" --train_splits train_fold1 \
    --out "$SMOKE" --arch unet --encoder efficientnet-b4 --norm image_global --input zcdf3 \
    --crop 1536 --scale 3 --sample area --bs 2 --iters 100 \
    --lr 3e-4 --wd 1e-2 --dice_w 1 --cldice_w 0 --seed 0 --workers 4 \
    --no_valid_eval > "$SMOKE/train.log" 2>&1
fi
if [[ ! -f "$OUT/smoke_valid/scores.json" ]]; then
  "$PY" "$CODE/predict.py" --data "$DATA" --split valid --runs "$SMOKE" \
    --out "$OUT/smoke_valid" --no_tta > "$OUT/smoke_valid.log" 2>&1
fi

for fold in 1 2 3 4; do
  run="$OUT/fold$fold"
  if [[ ! -f "$run/last.pt" ]]; then
    mkdir -p "$run"
    echo "=== train hybrid fold $fold: team B4 recipe + full-image-stat normalization, 16000 updates ===" \
      | tee -a "$OUT/run.log"
    "$PY" "$CODE/train.py" --data "$CV_DATA" --train_splits "train_fold$fold" \
      --out "$run" --arch unet --encoder efficientnet-b4 --norm image_global --input zcdf3 \
      --crop 1536 --scale 3 --sample area --bs 2 --iters 16000 \
      --lr 3e-4 --wd 1e-2 --dice_w 1 --cldice_w 0 --seed 0 --workers 4 \
      --no_valid_eval > "$run/train.log" 2>&1
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
  --candidate-order team --out "$OUT/results.json" > "$OUT/evaluate.log" 2>&1
echo "DONE: $OUT/results.json" | tee -a "$OUT/run.log"
