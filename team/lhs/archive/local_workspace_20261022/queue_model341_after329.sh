#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
WAIT_PID="${WAIT_PID:-716980}"
OUT="$ROOT/outputs/model341_model329_joint_route_search"
PARENT_RUN="$ROOT/outputs/model329_fullstats_width4000_uniform_group_oof/run.log"
PARENT_PROBS="$ROOT/outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz"
PARENT_ROUTE="$ROOT/outputs/model329_model270_fusion/route_probe.json"
mkdir -p "$OUT"
QUEUE_LOG="$OUT/queue.log"

echo "Waiting for Model329 pipeline PID $WAIT_PID before CPU route search." \
  | tee -a "$QUEUE_LOG"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

if ! grep -q '^DONE:' "$PARENT_RUN" 2>/dev/null; then
  echo "Model329 pipeline ended without DONE marker; not starting Model341." \
    | tee -a "$QUEUE_LOG"
  exit 1
fi
if [[ ! -s "$PARENT_PROBS" || ! -s "$PARENT_ROUTE" ]]; then
  echo "Model329 validation probabilities or route result is missing; not starting Model341." \
    | tee -a "$QUEUE_LOG"
  exit 1
fi
if [[ -s "$OUT/results.json" ]]; then
  echo "Model341 result already exists; refusing to overwrite it." \
    | tee -a "$QUEUE_LOG"
  exit 1
fi

echo "Starting low-priority CPU-only Model341 search (626 settings, one worker)." \
  | tee -a "$QUEUE_LOG"
cd "$ROOT"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 nice -n 19 "$PY" -u model341_model329_joint_route_search.py \
    --workers 1 --output "$OUT/results.json" \
    2>&1 | tee "$OUT/run.log"

echo "DONE: Model341 results at $OUT/results.json" | tee -a "$QUEUE_LOG"
