#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
PREVIOUS_QUEUE="$ROOT/outputs/model348_model347_into_341_route/queue.log"
OUT="$ROOT/outputs/model349_model347_into_346_route"

mkdir -p "$OUT"
if [[ -e "$OUT/results.json" || -e "$OUT/run.log" ]]; then
  echo "Refusing to reuse existing Model349 artifacts under: $OUT" >&2
  exit 1
fi

echo "Waiting for Model348 route search to finish." | tee -a "$OUT/queue.log"
while [[ ! -s "$PREVIOUS_QUEUE" ]] \
   || ! grep -Fq "DONE: Model348 Model347 expert into Model341 route." "$PREVIOUS_QUEUE"; do
  sleep 60
done

cd "$ROOT"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 nice -n 19 "$PY" -u \
    "$ROOT/model346_model336_into_341_route.py" \
    --root "$ROOT" --expert-id 347 --anchor-model346 \
    --workers 4 --bootstrap 20000 \
    --output "$OUT/results.json" \
  2>&1 | tee "$OUT/run.log"
echo "DONE: Model349 Model347 expert into fixed Model346 route." \
  | tee -a "$OUT/queue.log"
