#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
CODE="$ROOT/model326_hybrid_fullstats/KoMaP"
DATA="$ROOT/data"
CV_DATA="$ROOT/data_model322_group_cv"
PARENT="$ROOT/outputs/model322_team_b4_group_oof"
OUT="$ROOT/outputs/model333_b4_imagewise_lovasz_group_oof"
ITERS=2000
LOVASZ_W=0.25

mkdir -p "$OUT"
if [[ ! -d "$CV_DATA" ]]; then
  "$PY" "$CODE/prepare_group_cv.py" --data "$DATA" \
    --manifest "$PARENT/progress.json" --out "$CV_DATA"
fi

train_branch() {
  local branch="$1" fold="$2" lovasz_w="$3" iters="$4" checkpoint="$5"
  local run="$OUT/$branch/fold$fold"
  if [[ ! -f "$run/last.pt" ]]; then
    mkdir -p "$run"
    echo "=== Model333 $branch fold $fold: init=Model322 fold$fold, updates=$iters ===" \
      | tee -a "$OUT/run.log"
    "$PY" "$CODE/train_model333.py" \
      --data "$CV_DATA" --train_splits "train_fold$fold" --out "$run" \
      --arch unet --encoder efficientnet-b4 --norm imagenet --input gray3 \
      --crop 1536 --scale 3 --sample area --bs 2 --iters "$iters" \
      --lr 2e-5 --wd 1e-4 --dice_w 1 --lovasz_w "$lovasz_w" \
      --lovasz_max_pixels 131072 --init_checkpoint "$checkpoint" \
      --deterministic_workers --seed "$((333 + fold))" --workers 4 \
      --no_valid_eval > "$run/train.log" 2>&1
  else
    echo "=== Model333 $branch fold $fold checkpoint exists; keeping it ===" \
      | tee -a "$OUT/run.log"
  fi
  if [[ ! -f "$OUT/$branch/fold${fold}_oof/probabilities.npz" ]]; then
    "$PY" "$CODE/predict.py" --data "$CV_DATA" \
      --split "oof_fold$fold" --runs "$run" \
      --out "$OUT/$branch/fold${fold}_oof" --save_probs \
      > "$OUT/$branch/fold${fold}_oof.log" 2>&1
  fi
}

# Matched smoke runs verify strict checkpoint loading and both loss paths before
# the eight paired fold fine-tunes. They do not inspect validation masks.
for branch in control lovasz; do
  smoke="$OUT/smoke_$branch"
  lovasz_w=0
  [[ "$branch" == lovasz ]] && lovasz_w="$LOVASZ_W"
  if [[ ! -f "$smoke/last.pt" ]]; then
    mkdir -p "$smoke"
    "$PY" "$CODE/train_model333.py" \
      --data "$CV_DATA" --train_splits train_fold1 --out "$smoke" \
      --arch unet --encoder efficientnet-b4 --norm imagenet --input gray3 \
      --crop 1536 --scale 3 --sample area --bs 2 --iters 100 \
      --lr 2e-5 --wd 1e-4 --dice_w 1 --lovasz_w "$lovasz_w" \
      --lovasz_max_pixels 131072 \
      --init_checkpoint "$PARENT/fold1/last.pt" \
      --deterministic_workers --seed 333 --workers 4 --no_valid_eval \
      > "$smoke/train.log" 2>&1
  fi
done

for fold in 1 2 3 4; do
  checkpoint="$PARENT/fold$fold/last.pt"
  [[ -s "$checkpoint" ]] || { echo "missing Model322 checkpoint: $checkpoint"; exit 1; }
  train_branch control "$fold" 0 "$ITERS" "$checkpoint"
  train_branch lovasz "$fold" "$LOVASZ_W" "$ITERS" "$checkpoint"
done

for branch in control lovasz; do
  "$PY" "$CODE/aggregate_oof.py" --runs "$OUT/$branch" --folds 4 \
    --expected-images 70
  if [[ ! -f "$OUT/$branch/valid_ensemble/probabilities.npz" ]]; then
    "$PY" "$CODE/predict.py" --data "$DATA" --split valid \
      --runs "$OUT/$branch/fold1" "$OUT/$branch/fold2" \
        "$OUT/$branch/fold3" "$OUT/$branch/fold4" \
      --out "$OUT/$branch/valid_ensemble" --save_probs \
      > "$OUT/$branch/valid_ensemble.log" 2>&1
  fi
done

# Group-held-out route selection compares the Lovasz branch with the matched
# CE+Dice control; valid is reported only after all training/OOF work is fixed.
"$PY" "$CODE/evaluate_group_cv.py" --data "$DATA" \
  --parent-oof "$OUT/control/train_oof_probs.npz" \
  --candidate-oof "$OUT/lovasz/train_oof_probs.npz" \
  --parent-valid "$OUT/control/valid_ensemble/probabilities.npz" \
  --candidate-valid "$OUT/lovasz/valid_ensemble/probabilities.npz" \
  --parent-order team --candidate-order team \
  --out "$OUT/paired_control_vs_lovasz_orderfix.json" \
  > "$OUT/paired_control_vs_lovasz_orderfix.log" 2>&1

# Also test complementarity against the current Model270 route on CPU.
FUSION="$ROOT/outputs/model333_model270_fusion/route_probe.json"
if [[ ! -s "$FUSION" ]]; then
  mkdir -p "$(dirname "$FUSION")"
  cd "$ROOT"
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 19 "$PY" model323_model270_branch_route_probe.py \
      --entries "$ROOT/outputs/model183_metric_calibration/entries.pkl" \
      --model195 "$ROOT/outputs/model195_native_resume/valid_probs.npz" \
      --tent20 "$ROOT/outputs/model220_native_tent_extended/valid_probs_06.npz" \
      --model269 "$ROOT/outputs/model269_fullstats_1x_extended/valid_probs.npz" \
      --candidate "$OUT/lovasz/valid_ensemble/probabilities.npz" \
      --candidate-id b4_model333_lovasz --candidate-order team \
      --output "$FUSION" > "$(dirname "$FUSION")/route_probe.log" 2>&1
fi

"$PY" - "$OUT/paired_control_vs_lovasz.json" "$FUSION" <<'PY'
import json, sys
paired = json.load(open(sys.argv[1]))
fusion = json.load(open(sys.argv[2]))
cv = paired["grouped_cv_route"]
ci = cv["group_bootstrap_95ci_delta_pp_empty_zero"]
score = fusion["best_image_empty_zero"]["image_mean_miou"]
route_cv = fusion["grouped_cv_vs_model270_empty_zero"]
route_ci = route_cv["group_bootstrap_95ci_delta_pp"]
print(f"Model333 paired route-vs-control OOF delta={cv['image_mean_delta_pp_empty_zero']:+.3f}pp CI=[{ci[0]:+.3f},{ci[1]:+.3f}]pp")
print(f"Model333+Model270 valid empty=0={score:.6%}; grouped-CV delta={route_cv['image_mean_delta_pp']:+.3f}pp CI=[{route_ci[0]:+.3f},{route_ci[1]:+.3f}]pp")
print("per-class OOF delta(pp):", [round((b-a)*100, 3) for a,b in zip(
    paired["parent_oof"]["per_class_image_mean_iou_empty_zero"],
    paired["b4_oof_standalone"]["per_class_image_mean_iou_empty_zero"])])
PY
echo "DONE: $OUT/paired_control_vs_lovasz_orderfix.json; Model270 fusion: $FUSION" \
  | tee -a "$OUT/run.log"
