#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PYTHON="/workspace/venv-komap/bin/python"
DEVICE="cuda:0"
DATA_ROOT="$ROOT/data"
OUT_ROOT="$ROOT/outputs/model406_component_hardnegative_v3"
SPLITS="$ROOT/outputs/model403_local_supcon/splits/model392_cv"
BASELINE_ROOT="$ROOT/outputs/model392_widthnorm/seed1"
TRAINER="$ROOT/experiments/model406_component_hardnegative_train.py"
CALIBRATION="$ROOT/outputs/model406_component_hardnegative_v2/calibration.json"
LOCK_TOOL="$ROOT/experiments/model406_component_hardnegative_v3_lock.py"
RESULT="$ROOT/experiments/results/model406_component_hardnegative_oof_v3.json"
CLAIM="$RESULT.claim"

if [[ -L "$ROOT/outputs" || -L "$OUT_ROOT" || -L "$SPLITS" ]]; then
  echo "Refusing symlinked output or split root." >&2
  exit 3
fi
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd -P)"
exec 9>>"$OUT_ROOT/.model406_v3_oof.lock"
if ! flock -n 9; then
  echo "Another Model406 v3 runner already holds the experiment lock." >&2
  exit 6
fi
if [[ -e "$RESULT" || -e "$CLAIM" ]]; then
  echo "A Model406 v3 result/claim already exists; refusing repeat scoring." >&2
  exit 8
fi

for ancestor_lock in \
  "$ROOT/outputs/model403_local_supcon/.model403_oof.lock" \
  "$ROOT/outputs/model404_component_recall_v6/.model404_v6_oof.lock"; do
  if [[ -e "$ancestor_lock" ]]; then
    exec 8>>"$ancestor_lock"
    if ! flock -n 8; then
      echo "An ancestor P100 experiment is active; refusing concurrent training." >&2
      exit 7
    fi
  fi
done

"$PYTHON" "$LOCK_TOOL" --verify-source-lock --root "$ROOT"
"$PYTHON" "$ROOT/experiments/model406_component_hardnegative_v2_lock.py" \
  --verify-calibration-lock --root "$ROOT"

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
    oof="$OUT_ROOT/$arm/fold${fold}_oof"
    weight=0
    if [[ "$arm" == candidate ]]; then
      weight="$("$PYTHON" -c "import json,sys; print(format(json.load(open(sys.argv[1]))['folds'][sys.argv[2]]['joint_aux_weight'], '.15g'))" "$CALIBRATION" "$fold")"
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
      "$PYTHON" "$ROOT/model397_reproduction_bundle/model392/evaluate_image_mean.py" \
        --checkpoint "$run/last.pt" --data-root "$data" --split valid \
        --output-dir "$oof" --device "$DEVICE" --batch-size 2 \
        --patch-size 512 --overlap 170 --input-scale 2.0 --tta-mode flip4
    fi
  done
done

"$PYTHON" "$LOCK_TOOL" --create-oof-lock --root "$ROOT"
"$PYTHON" "$ROOT/experiments/score_model406_component_hardnegative_oof_v3.py" \
  --data-root "$DATA_ROOT" --candidate-root "$OUT_ROOT/candidate" \
  --control-root "$OUT_ROOT/control" --baseline-root "$ROOT/outputs/model392_widthnorm" \
  --out "$RESULT" --root "$ROOT"
