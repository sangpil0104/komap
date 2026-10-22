#!/usr/bin/env bash
# Reconstruct Model360 on valid, assert official-score parity, and measure
# GT-oracle gains from correcting only errors inside/outside GT boundary bands.
set -euo pipefail

ROOT="${KOMAP_ROOT:-/workspace/komap_model1}"
PY="${KOMAP_PYTHON:-/workspace/venv-komap/bin/python}"
OUT="$ROOT/outputs/model360_boundary_tolerance_audit/results.json"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
nice -n 19 "$PY" -u "$ROOT/model360_boundary_tolerance_audit.py" \
  --root "$ROOT" --output "$OUT" --radii 1 3 \
  2>&1 | tee "$ROOT/outputs/model360_boundary_tolerance_audit/run.log"
