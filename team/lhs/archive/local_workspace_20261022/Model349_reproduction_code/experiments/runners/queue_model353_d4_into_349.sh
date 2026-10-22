#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
MODEL349_QUEUE="$ROOT/outputs/model349_model347_into_346_route/queue.log"
MODEL351_QUEUE="$ROOT/outputs/model351_model336_d4_tta/queue.log"
MODEL351_PROBS="$ROOT/outputs/model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz"
OUT_BASE="$ROOT/outputs/model353_model351_into_349_route"
ATTEMPT="${MODEL353_ATTEMPT:?Set a unique MODEL353_ATTEMPT number, e.g. 04}"
OUT="$OUT_BASE/attempt_${ATTEMPT}"

if [[ -e "$OUT" ]]; then
  echo "Refusing to reuse existing Model353 artifacts under: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

echo "Waiting for Model349 and Model351 prerequisites." | tee -a "$OUT/queue.log"
while [[ ! -s "$MODEL349_QUEUE" ]] \
   || ! grep -Fq "DONE: Model349 Model347 expert into fixed Model346 route." "$MODEL349_QUEUE" \
   || [[ ! -s "$MODEL351_QUEUE" ]] \
   || ! grep -Fq "DONE: Model351 8-way dihedral TTA valid inference." "$MODEL351_QUEUE"; do
  sleep 60
done

test -s "$MODEL351_PROBS"
cd "$ROOT"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 nice -n 19 "$PY" -u \
    "$ROOT/model346_model336_into_341_route.py" \
    --root "$ROOT" --expert-id 351 --anchor-model349 --experiment-id 353 \
    --expert-probabilities "$MODEL351_PROBS" \
    --workers 4 --bootstrap 20000 --max-routes 2 \
    --output "$OUT/smoke_results.json" \
  2>&1 | tee "$OUT/smoke.log"

env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 nice -n 19 "$PY" -u \
    "$ROOT/model346_model336_into_341_route.py" \
    --root "$ROOT" --expert-id 351 --anchor-model349 --experiment-id 353 \
    --expert-probabilities "$MODEL351_PROBS" \
    --workers 4 --bootstrap 20000 \
    --output "$OUT/results.json" \
  2>&1 | tee "$OUT/run.log"

echo "DONE: Model353 Model351 D4 expert into fixed Model349 route." \
  | tee -a "$OUT/queue.log"
