#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
OUT="$ROOT/outputs/model346_model336_into_341_route"
MODEL336_OUT="$ROOT/outputs/model336_hrnet_best"
MODEL345_RESULT="$ROOT/outputs/model345_model344_into_341_route/results.json"

mkdir -p "$OUT"
if [[ -e "$OUT/results.json" || -e "$OUT/run.log" ]]; then
  echo "Refusing to reuse existing Model346 artifacts under: $OUT" >&2
  exit 1
fi

echo "Waiting for completed Model336 inference/Model337 route and Model345 search." \
  | tee -a "$OUT/queue.log"
while [[ ! -s "$MODEL336_OUT/queue.log" ]] \
   || ! grep -q "DONE: Model336 training/validation/test and Model337 HRNet route search complete" "$MODEL336_OUT/queue.log" \
   || [[ ! -s "$MODEL345_RESULT" ]]; do
  sleep 60
done

cd "$ROOT"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 nice -n 19 "$PY" -u \
    "$ROOT/model346_model336_into_341_route.py" \
    --root "$ROOT" --workers 4 --bootstrap 20000 --output "$OUT/results.json" \
  2>&1 | tee "$OUT/run.log"
echo "DONE: Model346 Model336 expert into Model341 route." | tee -a "$OUT/queue.log"
