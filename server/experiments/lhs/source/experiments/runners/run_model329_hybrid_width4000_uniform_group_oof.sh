#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
CODE="$ROOT/model326_hybrid_fullstats/KoMaP"
PY=/workspace/venv-komap/bin/python
DATA="$ROOT/data"
CV_DATA="$ROOT/data_model322_group_cv"
OUT="$ROOT/outputs/model329_fullstats_width4000_uniform_group_oof"
PARENT="$ROOT/outputs/model303_boundary_oof_8ep"

if [[ ! -d "$CV_DATA" ]]; then
  "$PY" "$CODE/prepare_group_cv.py" --data "$DATA" \
    --manifest "$PARENT/progress.json" --out "$CV_DATA"
fi
mkdir -p "$OUT"

# Combine the image-wise full-statistics input (Model326) with the 4K
# per-image resize and uniform image sampling (Model328). Smoke-test both
# training and full-image inference before four long grouped folds.
SMOKE="$OUT/smoke_fullstats_width4000_uniform"
if [[ ! -f "$SMOKE/last.pt" ]]; then
  mkdir -p "$SMOKE"
  "$PY" "$CODE/train.py" --data "$CV_DATA" --train_splits train_fold1 \
    --out "$SMOKE" --arch unet --encoder efficientnet-b4 \
    --norm image_global --input zcdf3 --crop 1536 --target_width 4000 \
    --sample uniform --bs 2 --iters 100 --lr 3e-4 --wd 1e-2 \
    --dice_w 1 --cldice_w 0 --seed 0 --workers 4 --no_valid_eval \
    > "$SMOKE/train.log" 2>&1
fi
if [[ ! -f "$OUT/smoke_valid/scores.json" ]]; then
  "$PY" "$CODE/predict.py" --data "$DATA" --split valid --runs "$SMOKE" \
    --out "$OUT/smoke_valid" --no_tta > "$OUT/smoke_valid.log" 2>&1
fi

for fold in 1 2 3 4; do
  run="$OUT/fold$fold"
  if [[ ! -f "$run/last.pt" ]]; then
    mkdir -p "$run"
    echo "=== train fullstats/width4000/uniform fold $fold: fixed 16000 updates ===" \
      | tee -a "$OUT/run.log"
    "$PY" "$CODE/train.py" --data "$CV_DATA" --train_splits "train_fold$fold" \
      --out "$run" --arch unet --encoder efficientnet-b4 \
      --norm image_global --input zcdf3 --crop 1536 --target_width 4000 \
      --sample uniform --bs 2 --iters 16000 --lr 3e-4 --wd 1e-2 \
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
    --out "$OUT/valid_ensemble" --save_probs > "$OUT/valid_ensemble.log" 2>&1
fi
"$PY" "$CODE/evaluate_group_cv.py" --data "$DATA" \
  --parent-oof "$PARENT/train_oof_probs.npz" --candidate-oof "$OUT/train_oof_probs.npz" \
  --parent-valid "$PARENT/valid_fold_ensemble_probs.npz" \
  --candidate-valid "$OUT/valid_ensemble/probabilities.npz" \
  --candidate-order team --out "$OUT/results.json" > "$OUT/evaluate.log" 2>&1

# Evaluate this branch against the current best Model270 route, not only the
# weaker Model303 grouped-CV parent used for the standalone report.
FUSION_OUT="$ROOT/outputs/model329_model270_fusion/route_probe.json"
if [[ ! -s "$FUSION_OUT" ]]; then
  mkdir -p "$(dirname "$FUSION_OUT")"
  cd "$ROOT"
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 19 "$PY" model323_model270_branch_route_probe.py \
      --entries "$ROOT/outputs/model183_metric_calibration/entries.pkl" \
      --model195 "$ROOT/outputs/model195_native_resume/valid_probs.npz" \
      --tent20 "$ROOT/outputs/model220_native_tent_extended/valid_probs_06.npz" \
      --model269 "$ROOT/outputs/model269_fullstats_1x_extended/valid_probs.npz" \
      --candidate "$OUT/valid_ensemble/probabilities.npz" \
      --candidate-id b4_model329 --candidate-order team \
      --output "$FUSION_OUT" > "$(dirname "$FUSION_OUT")/route_probe.log" 2>&1
fi
"$PY" - "$FUSION_OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
valid = d["best_image_empty_zero"]["image_mean_miou"]
cv = d["grouped_cv_vs_model270_empty_zero"]
ci = cv["group_bootstrap_95ci_delta_pp"]
print(f"Model329+Model270 best valid empty=0: {valid:.6%}; grouped-CV delta={cv['image_mean_delta_pp']:+.3f}pp CI=[{ci[0]:+.3f},{ci[1]:+.3f}]pp")
PY
echo "DONE: $OUT/results.json; Model270 fusion: $FUSION_OUT" | tee -a "$OUT/run.log"
