#!/usr/bin/env bash
# Wait for Model361's complete validation evaluation, then perform a CPU-only
# exact 625-route search adding its scale-3 HRNet probabilities to Model349.
set -euo pipefail

ROOT="${KOMAP_ROOT:-/workspace/komap_model1}"
PY="${KOMAP_PYTHON:-/workspace/venv-komap/bin/python}"
EVAL="$ROOT/outputs/model361_hrnet_scale3_finetune/valid_eval"
OUT="$ROOT/outputs/model362_model361_into_349_route/results.json"
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }

for attempt in $(seq 1 180); do
  if [[ -s "$EVAL/valid_summary.json" && -s "$EVAL/valid_probabilities.npz" ]]; then
    break
  fi
  if [[ "$attempt" -eq 180 ]]; then
    echo "Model361 valid evaluation did not finish within 3 hours" >&2
    exit 1
  fi
  sleep 60
done

"$PY" - "$EVAL/valid_summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["split"] == "valid" and summary["images"] == 20
assert summary["inference"]["input_scale"] == 3.0
assert summary["inference"]["overlap"] == 85
print("Model361 20-image scale-3 valid inference verified", flush=True)
PY

export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
nice -n 19 "$PY" -u "$ROOT/model346_model336_into_341_route.py" \
  --root "$ROOT" --expert-id 361 --anchor-model349 --experiment-id 362 \
  --workers 4 --bootstrap 20000 --output "$OUT"
