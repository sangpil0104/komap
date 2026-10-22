#!/usr/bin/env bash
# One predeclared scale-3 HRNet flip4 inference with antialiased probability
# downsampling, queued after the active Model365/366 D4 comparison.
set -euo pipefail

ROOT="${KOMAP_ROOT:-/workspace/komap_model1}"
PY="${KOMAP_PYTHON:-/workspace/venv-komap/bin/python}"
GPU="${KOMAP_GPU:-4}"
PARENT="$ROOT/outputs/model366_model361_d4_entropy_into_360/results.json"
CHECKPOINT="$ROOT/outputs/model361_hrnet_scale3_finetune/last.pt"
OUT="$ROOT/outputs/model372_scale3_antialias_flip4/valid_eval"

[[ -s "$CHECKPOINT" ]] || { echo "Missing Model361 checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ ! -e "$OUT/valid_summary.json" && ! -e "$OUT/valid_probabilities.npz" ]] || {
  echo "Refusing to overwrite Model372 evaluation artifacts" >&2; exit 2;
}

for attempt in $(seq 1 420); do
  if [[ -s "$PARENT" ]]; then
    break
  fi
  if [[ "$attempt" -eq 420 ]]; then
    echo "Model366 result did not appear within 7 hours" >&2
    exit 1
  fi
  sleep 60
done

"$PY" - "$PARENT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["model"] == 366
print("Model366 complete; starting the single antialiased scale-3 inference", flush=True)
PY

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$PY" -u "$ROOT/model336_hrnet_best/evaluate_image_mean.py" \
  --checkpoint "$CHECKPOINT" --data-root "$ROOT/data" --split valid \
  --output-dir "$OUT" --device cuda:0 --patch-size 512 --overlap 85 \
  --input-scale 3.0 --batch-size 2 --tta-mode flip4 \
  --downsample-antialias \
  > "$OUT/eval.log" 2>&1
