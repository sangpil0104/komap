#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
PARENT_PID="${PARENT_PID:-716980}"
MODEL341_PID="${MODEL341_PID:-786112}"
OUT="$ROOT/outputs/model342_oof_fitted_b4_blend"
MODEL329_RUN="$ROOT/outputs/model329_fullstats_width4000_uniform_group_oof/run.log"
MODEL329_OOF="$ROOT/outputs/model329_fullstats_width4000_uniform_group_oof/train_oof_probs.npz"
MODEL322_OOF="$ROOT/outputs/model322_team_b4_group_oof/train_oof_probs.npz"
MODEL329_VALID="$ROOT/outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz"
MODEL341_RESULT="$ROOT/outputs/model341_model329_joint_route_search/results.json"
mkdir -p "$OUT"
QUEUE_LOG="$OUT/queue.log"

echo "Waiting for Model329 pipeline PID $PARENT_PID and Model341 PID $MODEL341_PID." \
  | tee -a "$QUEUE_LOG"
while kill -0 "$PARENT_PID" 2>/dev/null || kill -0 "$MODEL341_PID" 2>/dev/null; do
  sleep 60
done

if ! grep -q '^DONE:' "$MODEL329_RUN" 2>/dev/null; then
  echo "Model329 pipeline ended without DONE marker; not starting Model342." \
    | tee -a "$QUEUE_LOG"
  exit 1
fi
for path in "$MODEL329_OOF" "$MODEL322_OOF" "$MODEL329_VALID" "$MODEL341_RESULT"; do
  if [[ ! -s "$path" ]]; then
    echo "Required completed artifact missing ($path); not starting Model342." \
      | tee -a "$QUEUE_LOG"
    exit 1
  fi
done
if [[ -s "$OUT/results.json" ]]; then
  echo "Model342 result already exists; refusing to overwrite it." \
    | tee -a "$QUEUE_LOG"
  exit 1
fi

echo "Starting train-OOF-fitted, CPU-only Model342 class-wise blend search." \
  | tee -a "$QUEUE_LOG"
cd "$ROOT"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 nice -n 19 "$PY" -u model342_oof_fitted_b4_blend.py \
    --bootstrap 20000 --output "$OUT/results.json" \
    2>&1 | tee "$OUT/run.log"

echo "DONE: Model342 results at $OUT/results.json" | tee -a "$QUEUE_LOG"
