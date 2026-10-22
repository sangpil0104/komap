#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
MODEL326_RUN="$ROOT/outputs/model326_hybrid_fullstats_group_oof/run.log"
MODEL326_PID="${MODEL326_PID:-714871}"
MODEL326_VALID="$ROOT/outputs/model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz"
FUSION_OUT="$ROOT/outputs/model326_model270_fusion/route_probe.json"
OUT="$ROOT/outputs/model328_b4_width4000_uniform_group_oof"
QUEUE_LOG="$OUT/queue.log"
mkdir -p "$OUT"

echo "Waiting for Model326 grouped-OOF training/evaluation." | tee -a "$QUEUE_LOG"
while ! grep -q '^DONE:' "$MODEL326_RUN" 2>/dev/null; do
  if ! kill -0 "$MODEL326_PID" 2>/dev/null; then
    echo "Model326 exited without DONE marker; refusing to start Model328." | tee -a "$QUEUE_LOG"
    exit 1
  fi
  sleep 60
done
if [[ ! -s "$MODEL326_VALID" ]]; then
  echo "Model326 valid probabilities are missing; refusing fusion probe." | tee -a "$QUEUE_LOG"
  exit 1
fi

# First test whether the just-trained hybrid already reaches the target when
# routed with Model270. This is CPU-only and precedes any new GPU training.
if [[ ! -s "$FUSION_OUT" ]]; then
  mkdir -p "$(dirname "$FUSION_OUT")"
  cd "$ROOT"
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 nice -n 19 "$PY" model323_model270_branch_route_probe.py \
      --entries "$ROOT/outputs/model183_metric_calibration/entries.pkl" \
      --model195 "$ROOT/outputs/model195_native_resume/valid_probs.npz" \
      --tent20 "$ROOT/outputs/model220_native_tent_extended/valid_probs_06.npz" \
      --model269 "$ROOT/outputs/model269_fullstats_1x_extended/valid_probs.npz" \
      --candidate "$MODEL326_VALID" --candidate-id b4_model326 --candidate-order team \
      --output "$FUSION_OUT" > "$(dirname "$FUSION_OUT")/route_probe.log" 2>&1
fi

if "$PY" - "$FUSION_OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
score = d["best_image_empty_zero"]["image_mean_miou"]
ci = d["grouped_cv_vs_model270_empty_zero"]["group_bootstrap_95ci_delta_pp"]
delta = d["grouped_cv_vs_model270_empty_zero"]["image_mean_delta_pp"]
print(f"Model326+Model270 best valid empty=0: {score:.6%}; grouped-CV delta={delta:+.3f}pp CI=[{ci[0]:+.3f},{ci[1]:+.3f}]pp")
if score >= 0.80 and ci[0] > 0:
    print("Robust 80% target gate passed; skip Model328 training.")
    sys.exit(10)
PY
then
  status=0
else
  status=$?
fi
if [[ "$status" -eq 10 ]]; then
  echo "Model270 + Model326 fusion reached the robust target gate; no further training queued." | tee -a "$QUEUE_LOG"
  exit 0
elif [[ "$status" -ne 0 ]]; then
  echo "Model326 fusion assessment failed with status $status; refusing Model328." | tee -a "$QUEUE_LOG"
  exit "$status"
fi

echo "Target gate not passed; preparing Model328 (target_width=4000, uniform sampling)." | tee -a "$QUEUE_LOG"
GPU_ID=6
minimum_free=$((14 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))
stable=0
while (( stable < 3 )); do
  free_bytes=$(GPU_ID="$GPU_ID" "$PY" -c 'import os,torch; print(torch.cuda.mem_get_info(int(os.environ["GPU_ID"]))[0])')
  if (( free_bytes >= minimum_free )); then
    stable=$((stable + 1))
    echo "P100 GPU $GPU_ID headroom check $stable/3 passed" | tee -a "$QUEUE_LOG"
  else
    stable=0
    echo "P100 GPU $GPU_ID busy; retrying in 60s" | tee -a "$QUEUE_LOG"
  fi
  if (( stable < 3 )); then sleep 60; fi
done
export CUDA_VISIBLE_DEVICES="$GPU_ID"
exec bash "$ROOT/run_model328_width4000_uniform_group_oof.sh"
