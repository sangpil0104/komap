#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
OUT="$ROOT/outputs/model345_model344_into_341_route"

mkdir -p "$OUT"
if [[ -e "$OUT/run.log" || -e "$OUT/results.json" ]]; then
  echo "Refusing to reuse existing Model345 run artifacts under: $OUT" >&2
  exit 1
fi

cd "$ROOT"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 nice -n 19 "$PY" \
    "$ROOT/model345_model344_into_341_route.py" \
    --root "$ROOT" --bootstrap 20000 --output "$OUT/results.json" \
  2>&1 | tee "$OUT/run.log"
