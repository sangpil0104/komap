#!/usr/bin/env bash
# Evaluate the predeclared antialiased Model372 probabilities in the existing
# Model360 entropy-routing grid, with specimen-group held-out selection.
set -euo pipefail

ROOT="${KOMAP_ROOT:-/workspace/komap_model1}"
PY="${KOMAP_PYTHON:-/workspace/venv-komap/bin/python}"
PARENT="$ROOT/outputs/model372_scale3_antialias_flip4/valid_eval"
PROBS="$PARENT/valid_probabilities.npz"
SUMMARY="$PARENT/valid_summary.json"
OUT="$ROOT/outputs/model373_antialias_into_360_route/results.json"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }

for attempt in $(seq 1 180); do
  if [[ -s "$SUMMARY" && -s "$PROBS" ]]; then
    break
  fi
  if [[ "$attempt" -eq 180 ]]; then
    echo "Model372 antialias evaluation did not finish within 3 hours" >&2
    exit 1
  fi
  sleep 60
done

"$PY" - "$SUMMARY" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["split"] == "valid" and d["images"] == 20
assert d["inference"]["input_scale"] == 3.0
assert d["inference"]["downsample_antialias"] is True
assert d["inference"]["tta"] == "4-way horizontal/vertical flips"
print("Model372 antialias archive verified; running grouped route comparison", flush=True)
PY

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
nice -n 19 "$PY" -u "$ROOT/model364_model361_entropy_into_360.py" \
  --root "$ROOT" --expert-probabilities "$PROBS" --model-id 373 \
  --output "$OUT"
