#!/usr/bin/env bash
# CPU-only fixed Model362-gamma check on top of Model360. Waits for the
# Model362 result instead of starting a competing route search.
set -euo pipefail

ROOT="${KOMAP_ROOT:-/workspace/komap_model1}"
PY="${KOMAP_PYTHON:-/workspace/venv-komap/bin/python}"
PARENT="$ROOT/outputs/model362_model361_into_349_route/results.json"
OUT="$ROOT/outputs/model363_model361_into_360_fixed/results.json"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }

for attempt in $(seq 1 360); do
  if [[ -s "$PARENT" ]]; then
    break
  fi
  if [[ "$attempt" -eq 360 ]]; then
    echo "Model362 result did not appear within 6 hours" >&2
    exit 1
  fi
  sleep 60
done

"$PY" - "$PARENT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["model"] == 362 and d["expert_model"] == 361
assert abs(d["baseline_valid_image_mean_miou_empty0"] - 0.7990951887388398) < 1e-10
print("Model362 expert and Model349 baseline verified", flush=True)
PY

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
nice -n 19 "$PY" -u "$ROOT/model363_model361_into_360_fixed.py" \
  --root "$ROOT" --output "$OUT"
