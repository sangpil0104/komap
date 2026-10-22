#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/workspace/komap_model1}"
PYTHON_BIN="${2:-/workspace/venv-komap/bin/python}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd -P)"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/model392_widthnorm"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ -e "$PROJECT_ROOT/outputs/model382_seed_count_control/.model382.lock" ]]; then
  echo "Model382 is still locked on P100; refusing concurrent training." >&2
  exit 3
fi

mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.model392_full_train.lock"
if ! flock -n 9; then
  echo "Another Model392 full-train runner already holds the lock." >&2
  exit 4
fi

export PYTHONPATH="$PROJECT_ROOT/experiments:$PROJECT_ROOT/model336_hrnet_widthnorm_candidate${PYTHONPATH:+:$PYTHONPATH}"

for SEED in 1 2; do
  PREPARE_OUTPUT="$("$PYTHON_BIN" "$PROJECT_ROOT/experiments/model392_prepare_full_train.py" \
    --project-root "$PROJECT_ROOT" --seed "$SEED")"
  printf '%s\n' "$PREPARE_OUTPUT"
  if "$PYTHON_BIN" -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("already_complete") is True else 1)' \
    <<< "$PREPARE_OUTPUT"; then
    echo "Seed $SEED is already complete; skipping it."
    continue
  fi
  "$PYTHON_BIN" "$PROJECT_ROOT/model336_hrnet_widthnorm_candidate/train.py" \
    --data-root "$PROJECT_ROOT/data_model392_full" \
    --output-dir "$OUTPUT_ROOT/full_train/seed$SEED" \
    --device cuda:0 \
    --architecture UnetPlusPlus \
    --encoder-name tu-hrnet_w48 \
    --encoder-weights imagenet \
    --patch-size 512 \
    --input-scale 2.0 \
    --normalize-width \
    --samples-per-image 12 \
    --batch-size 4 \
    --accumulation-steps 2 \
    --num-workers 4 \
    --epochs 70 \
    --lr 3e-4 \
    --encoder-lr-mult 1.0 \
    --weight-decay 1e-4 \
    --boundary-head-weight 1.0 \
    --seed "$SEED" \
    --rng-mode legacy \
    --val-every 0 \
    --resume \
    2>&1 | tee -a "$OUTPUT_ROOT/full_train/seed$SEED/train.log"
done

echo "Both Model392 full-train seeds completed. Do not run validation tuning; use the locked one-shot evaluator once."
