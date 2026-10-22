#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/workspace/komap_model1}"
PYTHON_BIN="${2:-/workspace/venv-komap/bin/python}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd -P)"
PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd -P)/$(basename "$PYTHON_BIN")"
LOCK_TOOL="$PROJECT_ROOT/experiments/create_model392_sequence_closure_lock.py"
SEQUENCE_LOCK="$PROJECT_ROOT/experiments/results/model392_sequence_closure_lock.json"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/model392_widthnorm"
CV_ROOT="$PROJECT_ROOT/data_model322_group_cv"
MODEL382_ROOT="$PROJECT_ROOT/outputs/model382_seed_count_control"
CANDIDATE="$PROJECT_ROOT/model336_hrnet_widthnorm_candidate"
GATE_OUTPUT="$PROJECT_ROOT/experiments/results/model392_widthnorm_two_seed_confirmatory.json"
GATE_CLAIM="$GATE_OUTPUT.claim"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ -L "$PROJECT_ROOT/outputs" || -L "$OUTPUT_ROOT" ]]; then
  echo "Refusing symlinked outputs path: $PROJECT_ROOT/outputs or $OUTPUT_ROOT" >&2
  exit 3
fi
if [[ -e "$MODEL382_ROOT/.model382.lock" ]]; then
  echo "Model382 is still locked on P100; refusing concurrent training." >&2
  exit 3
fi
if [[ ! -f "$SEQUENCE_LOCK" ]]; then
  echo "Missing pre-run Model392 sequence-closure lock: $SEQUENCE_LOCK" >&2
  exit 4
fi

export PYTHONPATH="$PROJECT_ROOT/experiments:$CANDIDATE${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" "$LOCK_TOOL" --project-root "$PROJECT_ROOT" --verify

if [[ -e "$GATE_CLAIM" && ! -e "$GATE_OUTPUT" ]]; then
  echo "The one-shot Model392 analysis was claimed but did not publish a result; refusing rerun." >&2
  exit 5
fi

mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.model392_oof.lock"
if ! flock -n 9; then
  echo "Another Model392 OOF runner already holds the lock." >&2
  exit 6
fi

if [[ ! -e "$GATE_OUTPUT" ]]; then
for SEED in 1 2; do
  for FOLD in 1 2 3 4; do
    RUN_DIR="$OUTPUT_ROOT/seed$SEED/fold$FOLD"
    OOF_DIR="$OUTPUT_ROOT/seed$SEED/fold${FOLD}_oof"
    MANIFEST="$OOF_DIR/model392_manifest.json"
    if [[ -L "$OUTPUT_ROOT/seed$SEED" || -L "$RUN_DIR" || -L "$OOF_DIR" ]]; then
      echo "Refusing symlinked Model392 run/OOF path for seed$SEED/fold$FOLD." >&2
      exit 7
    fi

    if [[ -f "$MANIFEST" ]]; then
      "$PYTHON_BIN" "$LOCK_TOOL" --project-root "$PROJECT_ROOT" \
        --verify-split --seed "$SEED" --fold "$FOLD"
      "$PYTHON_BIN" "$LOCK_TOOL" --project-root "$PROJECT_ROOT" \
        --verify-fold --seed "$SEED" --fold "$FOLD"
      echo "Verified completed seed$SEED/fold$FOLD; did not reopen its OOF scores."
      continue
    fi

    if [[ -e "$OOF_DIR" ]]; then
      if [[ -f "$OOF_DIR/valid_probabilities.npz" \
            && -f "$OOF_DIR/valid_summary.json" \
            && -f "$OOF_DIR/valid_per_image.csv" ]]; then
        STATUS_JSON="$("$PYTHON_BIN" "$LOCK_TOOL" --project-root "$PROJECT_ROOT" \
          --training-status --seed "$SEED" --fold "$FOLD")"
        STATUS="$(printf '%s' "$STATUS_JSON" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["training_status"])')"
        if [[ "$STATUS" != "complete" ]]; then
          echo "Complete OOF artifacts exist without a clean bound epoch-70 run for seed$SEED/fold$FOLD." >&2
          exit 7
        fi
        "$PYTHON_BIN" "$LOCK_TOOL" --project-root "$PROJECT_ROOT" \
          --verify-split --seed "$SEED" --fold "$FOLD"
        "$PYTHON_BIN" "$PROJECT_ROOT/experiments/model392_record_fold.py" \
          --project-root "$PROJECT_ROOT" --seed "$SEED" --fold "$FOLD"
        echo "Recorded existing complete OOF inference for seed$SEED/fold$FOLD without rerunning inference."
        continue
      fi
      echo "Partial OOF directory exists at $OOF_DIR; preserving it and refusing a second inference." >&2
      exit 7
    fi

    STATUS_JSON="$("$PYTHON_BIN" "$LOCK_TOOL" --project-root "$PROJECT_ROOT" \
      --training-status --seed "$SEED" --fold "$FOLD")"
    STATUS="$(printf '%s' "$STATUS_JSON" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["training_status"])')"

    if [[ "$STATUS" != "complete" ]]; then
      "$PYTHON_BIN" "$PROJECT_ROOT/experiments/model392_prepare_run.py" \
        --project-root "$PROJECT_ROOT" --seed "$SEED" --fold "$FOLD"
      "$PYTHON_BIN" "$LOCK_TOOL" --project-root "$PROJECT_ROOT" \
        --bind-run-receipt --seed "$SEED" --fold "$FOLD"
      "$PYTHON_BIN" "$CANDIDATE/train.py" \
        --data-root "$PROJECT_ROOT/data_model382_seed${SEED}_fold${FOLD}" \
        --output-dir "$RUN_DIR" \
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
        2>&1 | tee -a "$RUN_DIR/train.log"
    fi

    STATUS_JSON="$("$PYTHON_BIN" "$LOCK_TOOL" --project-root "$PROJECT_ROOT" \
      --training-status --seed "$SEED" --fold "$FOLD")"
    STATUS="$(printf '%s' "$STATUS_JSON" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["training_status"])')"
    if [[ "$STATUS" != "complete" ]]; then
      echo "Model392 seed$SEED/fold$FOLD did not reach clean epoch-70 completion." >&2
      exit 8
    fi

    DATA_ROOT="$PROJECT_ROOT/data_model382_seed${SEED}_fold${FOLD}"
    "$PYTHON_BIN" "$LOCK_TOOL" --project-root "$PROJECT_ROOT" \
      --verify-split --seed "$SEED" --fold "$FOLD"
    TARGET_WIDTH="$("$PYTHON_BIN" -c 'import sys; from pathlib import Path; from model392_widthnorm_oof import modal_width_from_training_split; print(modal_width_from_training_split(Path(sys.argv[1])))' "$DATA_ROOT/train/images")"
    "$PYTHON_BIN" "$CANDIDATE/evaluate_image_mean.py" \
      --checkpoint "$RUN_DIR/last.pt" \
      --data-root "$DATA_ROOT" \
      --split valid \
      --output-dir "$OOF_DIR" \
      --device cuda:0 \
      --batch-size 2 \
      --patch-size 512 \
      --overlap 170 \
      --input-scale 2.0 \
      --target-width "$TARGET_WIDTH" \
      --tta-mode flip4 \
      > "$RUN_DIR/oof_inference.log" 2>&1

    "$PYTHON_BIN" "$PROJECT_ROOT/experiments/model392_record_fold.py" \
      --project-root "$PROJECT_ROOT" --seed "$SEED" --fold "$FOLD"
    echo "Completed and recorded seed$SEED/fold$FOLD; OOF scores remain unopened."
  done
done
fi

if [[ ! -e "$GATE_OUTPUT" ]]; then
  if [[ -e "$GATE_CLAIM" ]]; then
    echo "The one-shot Model392 analysis was claimed but did not publish a result; refusing rerun." >&2
    exit 9
  fi
  "$PYTHON_BIN" "$PROJECT_ROOT/experiments/model392_widthnorm_oof.py" \
    --cv-root "$CV_ROOT" \
    --model382-root "$MODEL382_ROOT" \
    --model392-root "$OUTPUT_ROOT" \
    --model382-lock "$PROJECT_ROOT/experiments/results/model382_e5_preregistration_lock.json" \
    --model392-lock "$PROJECT_ROOT/experiments/results/model392_widthnorm_runtime_lock.json" \
    --output "$GATE_OUTPUT" \
    > "$OUTPUT_ROOT/two_seed_confirmatory.log" 2>&1
fi

GATE="$("$PYTHON_BIN" -c 'import json,sys; print("pass" if json.load(open(sys.argv[1]))["valid_evaluation_authorized"] else "fail")' "$GATE_OUTPUT")"
echo "Model392 locked two-seed OOF gate: $GATE"
if [[ "$GATE" != "pass" ]]; then
  echo "Gate failed; full-train and official-valid evaluation were not started."
  exit 0
fi

bash "$PROJECT_ROOT/run_model392_full_train_p100.sh" "$PROJECT_ROOT" "$PYTHON_BIN" \
  2>&1 | tee "$OUTPUT_ROOT/full_train_runner.log"
VALID_OUTPUT="$OUTPUT_ROOT/official_valid_once"
VALID_CLAIM="$OUTPUT_ROOT/official_valid_once.claim"
if [[ -d "$VALID_OUTPUT" ]]; then
  if [[ -f "$VALID_OUTPUT/valid_summary.json" && -f "$VALID_OUTPUT/valid_manifest.json" ]]; then
    echo "The authorized one-shot official-valid evaluation is already complete; not rerunning it."
    exit 0
  fi
  echo "Official-valid output directory is incomplete; refusing any second evaluation." >&2
  exit 10
fi
if [[ -e "$VALID_CLAIM" ]]; then
  echo "The official-valid one-shot attempt was already claimed; refusing reevaluation." >&2
  exit 11
fi
"$PYTHON_BIN" "$PROJECT_ROOT/experiments/model392_official_valid_once.py" \
  --project-root "$PROJECT_ROOT" \
  > "$OUTPUT_ROOT/official_valid_once.log" 2>&1
echo "Model392 authorized one-shot official-valid evaluation completed; no test input was used."
