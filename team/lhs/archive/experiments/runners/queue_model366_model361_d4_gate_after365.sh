#!/usr/bin/env bash
# CPU-only entropy routing of scale-3 D4 Model361 into fixed Model360.
set -euo pipefail

ROOT="${KOMAP_ROOT:-/workspace/komap_model1}"
PY="${KOMAP_PYTHON:-/workspace/venv-komap/bin/python}"
EVAL="$ROOT/outputs/model365_model361_scale3_d4/valid_eval"
PARENT="$ROOT/outputs/model364_model361_entropy_into_360/results.json"
OUT="$ROOT/outputs/model366_model361_d4_entropy_into_360/results.json"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }

for attempt in $(seq 1 180); do
  if [[ -s "$EVAL/valid_summary.json" && -s "$EVAL/valid_probabilities.npz" ]]; then
    break
  fi
  if [[ "$attempt" -eq 180 ]]; then
    echo "Model365 D4 inference did not finish within 3 hours" >&2
    exit 1
  fi
  sleep 60
done

"$PY" - "$EVAL/valid_summary.json" "$PARENT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["split"] == "valid" and d["images"] == 20
assert d["inference"]["input_scale"] == 3.0
assert d["inference"]["overlap"] == 85
assert d["inference"]["tta"] == "8-way dihedral rotations and flips"
route = json.load(open(sys.argv[2], encoding="utf-8"))
assert route["model"] == 364
print("Model365 D4 inference and Model364 completion verified", flush=True)
PY

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
nice -n 19 "$PY" -u "$ROOT/model364_model361_entropy_into_360.py" \
  --root "$ROOT" \
  --expert-probabilities "$EVAL/valid_probabilities.npz" \
  --model-id 366 \
  --output "$OUT"
