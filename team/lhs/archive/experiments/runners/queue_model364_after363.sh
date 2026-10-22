#!/usr/bin/env bash
# Run the CPU-only pixel entropy route after Model362/363 CPU searches.
set -euo pipefail

ROOT="${KOMAP_ROOT:-/workspace/komap_model1}"
PY="${KOMAP_PYTHON:-/workspace/venv-komap/bin/python}"
PARENT="$ROOT/outputs/model363_model361_into_360_fixed/results.json"
OUT="$ROOT/outputs/model364_model361_entropy_into_360/results.json"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }

for attempt in $(seq 1 420); do
  if [[ -s "$PARENT" ]]; then
    break
  fi
  if [[ "$attempt" -eq 420 ]]; then
    echo "Model363 result did not appear within 7 hours" >&2
    exit 1
  fi
  sleep 60
done

"$PY" - "$PARENT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["model"] == 363
assert abs(d["baseline_model360"] - 0.7994166196335197) < 1e-10
print("Model363 and fixed Model360 baseline verified", flush=True)
PY

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
nice -n 19 "$PY" -u "$ROOT/model364_model361_entropy_into_360.py" \
  --root "$ROOT" --output "$OUT"
