#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
OUT="$ROOT/outputs/model344_model333_control_blend"

mkdir -p "$OUT"
if [[ -e "$OUT/run.log" || -e "$OUT/oof_results.json" ]]; then
  echo "Refusing to reuse existing Model344 run artifacts under: $OUT" >&2
  exit 1
fi

env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 nice -n 19 \
  "$PY" "$ROOT/model344_model333_control_blend_search.py" \
    --root "$ROOT" --mode oof --bootstrap 20000 \
    --output "$OUT/oof_results.json" \
  2>&1 | tee "$OUT/run.log"

echo "After OOF selection and once Model333 control valid probabilities exist, run:" >&2
echo "  $PY $ROOT/model344_model333_control_blend_search.py --root $ROOT --mode valid --oof-results $OUT/oof_results.json --output $OUT/valid_results.json" >&2
