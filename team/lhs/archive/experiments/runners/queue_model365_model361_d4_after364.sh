#!/usr/bin/env bash
# After CPU route searches complete, evaluate D4 TTA for the scale-3 HRNet.
set -euo pipefail

ROOT="${KOMAP_ROOT:-/workspace/komap_model1}"
PY="${KOMAP_PYTHON:-/workspace/venv-komap/bin/python}"
GPU="${KOMAP_GPU:-4}"
PARENT="$ROOT/outputs/model364_model361_entropy_into_360/results.json"
OUT="$ROOT/outputs/model365_model361_scale3_d4/valid_eval"
[[ ! -e "$OUT/valid_summary.json" && ! -e "$OUT/valid_probabilities.npz" ]] || {
  echo "Refusing to overwrite Model365 evaluation artifacts" >&2; exit 2;
}

for attempt in $(seq 1 420); do
  if [[ -s "$PARENT" ]]; then
    break
  fi
  if [[ "$attempt" -eq 420 ]]; then
    echo "Model364 result did not appear within 7 hours" >&2
    exit 1
  fi
  sleep 60
done

"$PY" - "$PARENT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["model"] == 364
assert abs(d["baseline_model360"] - 0.7994166196335197) < 1e-10
print("Model364 complete; Model360 anchor parity confirmed", flush=True)
PY

CHECKPOINT="$ROOT/outputs/model361_hrnet_scale3_finetune/last.pt"
EVAL_SCRIPT="$ROOT/model336_hrnet_best/evaluate_image_mean.py"
[[ -s "$CHECKPOINT" ]] || { echo "Missing Model361 checkpoint" >&2; exit 2; }
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$PY" -u "$EVAL_SCRIPT" --checkpoint "$CHECKPOINT" \
  --data-root "$ROOT/data" --split valid --output-dir "$OUT" --device cuda:0 \
  --patch-size 512 --overlap 85 --input-scale 3.0 --batch-size 2 --tta-mode d4 \
  > "$ROOT/outputs/model365_model361_scale3_d4/eval.log" 2>&1
