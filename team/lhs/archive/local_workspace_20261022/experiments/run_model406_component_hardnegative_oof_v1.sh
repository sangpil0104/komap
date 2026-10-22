#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PYTHON="/workspace/venv-komap/bin/python"
DEVICE="cuda:0"
DATA_ROOT="$ROOT/data"
OUT_ROOT="$ROOT/outputs/model406_component_hardnegative_v1"
SPLITS="$ROOT/outputs/model403_local_supcon/splits/model392_cv"
BASELINE_ROOT="$ROOT/outputs/model392_widthnorm/seed1"
LOCK_TOOL="$ROOT/experiments/model406_component_hardnegative_lock.py"
TRAINER="$ROOT/experiments/model406_component_hardnegative_train.py"
CALIBRATOR="$ROOT/experiments/model406_component_hardnegative_calibrate.py"
EVALUATOR="$ROOT/model397_reproduction_bundle/model392/evaluate_image_mean.py"
SOURCE_LOCK="$ROOT/experiments/results/model406_component_hardnegative_source_lock_v1.json"
CALIBRATION="$OUT_ROOT/calibration.json"
CALIBRATION_LOCK="$ROOT/experiments/results/model406_component_hardnegative_calibration_lock_v1.json"
OOF_LOCK="$ROOT/experiments/results/model406_component_hardnegative_oof_lock_v1.json"
RESULT="$ROOT/experiments/results/model406_component_hardnegative_oof_v1.json"
CLAIM="$RESULT.claim"

if [[ -L "$ROOT/outputs" || -L "$OUT_ROOT" || -L "$SPLITS" ]]; then
  echo "Refusing symlinked output or split root." >&2
  exit 3
fi
mkdir -p "$OUT_ROOT"
if [[ -L "$OUT_ROOT" ]]; then
  echo "Refusing symlinked Model406 output root." >&2
  exit 3
fi
OUT_ROOT="$(cd "$OUT_ROOT" && pwd -P)"
exec 9>>"$OUT_ROOT/.model406_oof.lock"
if ! flock -n 9; then
  echo "Another Model406 runner already holds the experiment lock." >&2
  exit 6
fi
if [[ -e "$RESULT" || -e "$CLAIM" ]]; then
  echo "A Model406 result/claim already exists; refusing another score." >&2
  exit 8
fi

MODEL403_LOCK="$ROOT/outputs/model403_local_supcon/.model403_oof.lock"
if [[ -e "$MODEL403_LOCK" ]]; then
  exec 8>>"$MODEL403_LOCK"
  if ! flock -n 8; then
    echo "Model403 still holds the P100 experiment lock; refusing concurrent training." >&2
    exit 7
  fi
fi
MODEL404_LOCK="$ROOT/outputs/model404_component_recall_v6/.model404_v6_oof.lock"
if [[ -e "$MODEL404_LOCK" ]]; then
  exec 7>>"$MODEL404_LOCK"
  if ! flock -n 7; then
    echo "Model404 still holds the P100 experiment lock; refusing concurrent training." >&2
    exit 7
  fi
fi

if [[ -s "$SOURCE_LOCK" ]]; then
  "$PYTHON" "$LOCK_TOOL" --verify-source-lock --root "$ROOT"
else
  "$PYTHON" "$LOCK_TOOL" --create-source-lock --root "$ROOT"
fi

if [[ -s "$CALIBRATION" ]]; then
  if [[ -s "$CALIBRATION_LOCK" ]]; then
    "$PYTHON" "$LOCK_TOOL" --verify-calibration-lock --root "$ROOT"
  else
    "$PYTHON" "$LOCK_TOOL" --create-calibration-lock --root "$ROOT"
  fi
else
  if [[ -e "$CALIBRATION_LOCK" ]]; then
    echo "Calibration lock exists but calibration output is missing; refusing overwrite." >&2
    exit 9
  fi
  "$PYTHON" "$CALIBRATOR" --split-root "$SPLITS" \
    --baseline-root "$BASELINE_ROOT" --out "$CALIBRATION" --device "$DEVICE" \
    --probe-batches 8 --probe-seed 406
  "$PYTHON" "$LOCK_TOOL" --create-calibration-lock --root "$ROOT"
fi

for fold in 1 2 3 4; do
  if [[ ! -s "$BASELINE_ROOT/fold$fold/last.pt" || ! -d "$SPLITS/seed1_fold$fold" ]]; then
    echo "Missing Model392 parent checkpoint or fixed train fold $fold." >&2
    exit 4
  fi
done

for fold in 1 2 3 4; do
  data="$SPLITS/seed1_fold$fold"
  init="$BASELINE_ROOT/fold$fold/last.pt"
  for arm in control candidate; do
    run="$OUT_ROOT/$arm/fold$fold"
    oof="$OUT_ROOT/$arm/fold"$fold"_oof"
    weight=0
    if [[ "$arm" == candidate ]]; then
      weight="$("$PYTHON" -c 'import json,math,sys; f=json.load(open(sys.argv[1]))["folds"][sys.argv[2]]; w=float(f["joint_aux_weight"]); assert f["training_split_only"] and f["optimizer_steps"]==0 and not f["valid_read"] and not f["test_read"] and f["nonempty_gradient_batches"]>=6 and math.isfinite(w) and 1e-5<=w<=1.0; print(f"{w:.15g}")' "$CALIBRATION" "$fold")"
    fi
    resume_flag=""
    if [[ -f "$run/train_state.pt" ]]; then
      resume_flag="--resume"
    else
      if [[ -e "$run/last.pt" ]]; then
        echo "Checkpoint exists without resumable state at $run; refusing overwrite." >&2
        exit 8
      fi
      mkdir -p "$run"
    fi
    "$PYTHON" "$TRAINER" --data-root "$data" --init-checkpoint "$init" \
      --output-dir "$run" --arm "$arm" --component-weight "$weight" \
      --device "$DEVICE" --seed 1 --epochs 8 $resume_flag
    if [[ -s "$oof/valid_probabilities.npz" ]]; then
      echo "$arm fold$fold OOF archive exists; preserving it."
    elif [[ -e "$oof" ]]; then
      echo "Incomplete OOF output exists at $oof; refusing overwrite." >&2
      exit 9
    else
      "$PYTHON" "$EVALUATOR" --checkpoint "$run/last.pt" --data-root "$data" \
        --split valid --output-dir "$oof" --device "$DEVICE" --batch-size 2 \
        --patch-size 512 --overlap 170 --input-scale 2.0 --tta-mode flip4
    fi
  done
done

if [[ -s "$OOF_LOCK" ]]; then
  "$PYTHON" "$LOCK_TOOL" --verify-oof-lock --root "$ROOT"
else
  "$PYTHON" "$LOCK_TOOL" --create-oof-lock --root "$ROOT"
fi
"$PYTHON" "$ROOT/experiments/score_model406_component_hardnegative_oof.py" \
  --data-root "$DATA_ROOT" \
  --candidate-root "$OUT_ROOT/candidate" \
  --control-root "$OUT_ROOT/control" \
  --baseline-root "$ROOT/outputs/model392_widthnorm" \
  --out "$RESULT"
