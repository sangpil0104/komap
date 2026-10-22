#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
CODE="$ROOT/model322_team_b4/KoMaP"
PY=/workspace/venv-komap/bin/python
DATA="$ROOT/data"
CV_DATA="$ROOT/data_model322_group_cv"
OUT="$ROOT/outputs/model328_b4_width4000_uniform_group_oof"
PARENT="$ROOT/outputs/model303_boundary_oof_8ep"

if [[ ! -d "$CV_DATA" ]]; then
  "$PY" "$CODE/prepare_group_cv.py" --data "$DATA" \
    --manifest "$PARENT/progress.json" --out "$CV_DATA"
fi
mkdir -p "$OUT"

# Check that the per-image resize path, uniform sampler, and 4K source width
# execute before committing to the four long grouped folds.
SMOKE="$OUT/smoke_width4000_uniform"
if [[ ! -f "$SMOKE/last.pt" ]]; then
  mkdir -p "$SMOKE"
  "$PY" "$CODE/train.py" --data "$CV_DATA" --train_splits train_fold1 \
    --out "$SMOKE" --arch unet --encoder efficientnet-b4 --norm imagenet --input gray3 \
    --crop 1536 --target_width 4000 --sample uniform --bs 2 --iters 100 \
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
    echo "=== train width4000/uniform fold $fold: fixed 16000 updates ===" | tee -a "$OUT/run.log"
    "$PY" "$CODE/train.py" --data "$CV_DATA" --train_splits "train_fold$fold" \
      --out "$run" --arch unet --encoder efficientnet-b4 --norm imagenet --input gray3 \
      --crop 1536 --target_width 4000 --sample uniform --bs 2 --iters 16000 \
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
    --out "$OUT/valid_ensemble" --save_probs > "$OUT/valid_ensemble.log" 2>&1
fi
"$PY" "$CODE/evaluate_group_cv.py" --data "$DATA" \
  --parent-oof "$PARENT/train_oof_probs.npz" --candidate-oof "$OUT/train_oof_probs.npz" \
  --parent-valid "$PARENT/valid_fold_ensemble_probs.npz" \
  --candidate-valid "$OUT/valid_ensemble/probabilities.npz" \
  --candidate-order team --out "$OUT/results.json" > "$OUT/evaluate.log" 2>&1

# The standalone grouped-CV comparison above uses Model303 as its parent.
# Also test whether this new B4 branch complements the current best Model270
# route. Keep the route probe CPU-only so it cannot reserve another GPU.
FUSION_OUT="$ROOT/outputs/model328_model270_fusion/route_probe.json"
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
      --candidate-id b4_model328 --candidate-order team \
      --output "$FUSION_OUT" > "$(dirname "$FUSION_OUT")/route_probe.log" 2>&1
fi
"$PY" - "$FUSION_OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
valid = d["best_image_empty_zero"]["image_mean_miou"]
cv = d["grouped_cv_vs_model270_empty_zero"]
ci = cv["group_bootstrap_95ci_delta_pp"]
print(f"Model328+Model270 best valid empty=0: {valid:.6%}; grouped-CV delta={cv['image_mean_delta_pp']:+.3f}pp CI=[{ci[0]:+.3f},{ci[1]:+.3f}]pp")
PY
echo "Model270 fusion assessment saved: $FUSION_OUT" | tee -a "$OUT/run.log"
if "$PY" - "$FUSION_OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
score = d["best_image_empty_zero"]["image_mean_miou"]
ci = d["grouped_cv_vs_model270_empty_zero"]["group_bootstrap_95ci_delta_pp"]
print(f"Model328 robust-target gate: valid={score:.6%}; grouped-CV 95% CI=[{ci[0]:+.3f},{ci[1]:+.3f}]pp")
if score >= 0.80 and ci[0] > 0:
    print("Robust 80% target reached; skip Model329.")
    sys.exit(10)
PY
then
  gate_status=0
else
  gate_status=$?
fi
if [[ "$gate_status" -eq 10 ]]; then
  echo "DONE: $OUT/results.json; robust Model270 fusion target reached." | tee -a "$OUT/run.log"
  exit 0
elif [[ "$gate_status" -ne 0 ]]; then
  echo "Model328 fusion gate failed with status $gate_status; refusing Model329." | tee -a "$OUT/run.log"
  exit "$gate_status"
fi

echo "Target gate not passed; preparing Model329 (full-stats + width4000 + uniform)." \
  | tee -a "$OUT/run.log"
# The earlier GPU-free check covered Model328 only. Re-check physical P100 6
# after CPU routing, because the shared device may have been acquired meanwhile.
GPU_ID=6
minimum_free=$((14 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))
stable=0
while (( stable < 3 )); do
  free_bytes=$(env -u CUDA_VISIBLE_DEVICES GPU_ID="$GPU_ID" "$PY" -c \
    'import os,torch; print(torch.cuda.mem_get_info(int(os.environ["GPU_ID"]))[0])')
  if (( free_bytes >= minimum_free )); then
    stable=$((stable + 1))
    echo "P100 GPU $GPU_ID headroom check for Model329 $stable/3 passed" | tee -a "$OUT/run.log"
  else
    stable=0
    echo "P100 GPU $GPU_ID busy before Model329; retrying in 60s" | tee -a "$OUT/run.log"
  fi
  if (( stable < 3 )); then sleep 60; fi
done
exec bash "$ROOT/run_model329_hybrid_width4000_uniform_group_oof.sh"
