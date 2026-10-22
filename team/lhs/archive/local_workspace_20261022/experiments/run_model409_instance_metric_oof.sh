#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PYTHON="/workspace/venv-komap/bin/python"
DEVICE="cuda:0"
DATA_ROOT="$ROOT/data"
SPLITS="$ROOT/outputs/model403_local_supcon/splits/model392_cv"
BASELINE_ROOT="$ROOT/outputs/model392_widthnorm/seed1"
BASELINE_OOF_ROOT="$ROOT/outputs/model392_widthnorm"
CONTROL_ROOT="$ROOT/outputs/model406_component_hardnegative_v3/control"
OUT_ROOT="$ROOT/outputs/model409_instance_metric_v2"
CALIBRATION="$OUT_ROOT/calibration.json"
LOCK_TOOL="$ROOT/experiments/model409_instance_metric_lock.py"
RESULT="$ROOT/experiments/results/model409_instance_metric_oof_v2.json"
CLAIM="$RESULT.claim"

if [[ -L "$ROOT/outputs" || -L "$OUT_ROOT" || -L "$SPLITS" ]]; then
  echo "Refusing symlinked output or split root." >&2
  exit 3
fi
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd -P)"
exec 9>>"$OUT_ROOT/.model409_instance_metric_oof.lock"
if ! flock -n 9; then
  echo "Another Model409 runner holds the experiment lock." >&2
  exit 6
fi
if [[ -e "$RESULT" || -e "$CLAIM" ]]; then
  echo "A Model409 one-shot score exists; refusing rerun." >&2
  exit 8
fi

if [[ -e "$ROOT/experiments/results/model409_instance_metric_source_lock_v2.json" ]]; then
  "$PYTHON" "$LOCK_TOOL" --verify-source-lock --root "$ROOT"
else
  "$PYTHON" "$LOCK_TOOL" --create-source-lock --root "$ROOT"
fi

if [[ -s "$CALIBRATION" ]]; then
  "$PYTHON" "$LOCK_TOOL" --verify-calibration-lock --root "$ROOT"
else
  "$PYTHON" "$ROOT/experiments/model409_instance_metric_calibrate.py" \
    --split-root "$SPLITS" --baseline-root "$BASELINE_ROOT" \
    --out "$CALIBRATION" --device "$DEVICE" --probe-examples 32 --seed 409
  "$PYTHON" "$LOCK_TOOL" --create-calibration-lock --root "$ROOT"
fi

"$PYTHON" "$ROOT/experiments/model406_component_hardnegative_v3_lock.py" \
  --verify-oof-lock --root "$ROOT"

for fold in 1 2 3 4; do
  if [[ ! -s "$BASELINE_ROOT/fold$fold/last.pt" || ! -d "$SPLITS/seed1_fold$fold" \
     || ! -s "$CONTROL_ROOT/fold${fold}_oof/valid_probabilities.npz" ]]; then
    echo "Missing parent checkpoint, fixed train fold, or locked Model406 control fold $fold." >&2
    exit 4
  fi
done

for fold in 1 2 3 4; do
  data="$SPLITS/seed1_fold$fold"
  init="$BASELINE_ROOT/fold$fold/last.pt"
  run="$OUT_ROOT/candidate/fold$fold"
  oof="$OUT_ROOT/candidate/fold${fold}_oof"
  weight="$("$PYTHON" -c "import json,sys; print(format(json.load(open(sys.argv[1]))['folds'][sys.argv[2]]['metric_weight'], '.15g'))" "$CALIBRATION" "$fold")"
  resume_flag=""
  if [[ -f "$run/train_state.pt" ]]; then
    resume_flag="--resume"
  elif [[ -e "$run/last.pt" ]]; then
    echo "Checkpoint exists without resumable state at $run; refusing overwrite." >&2
    exit 8
  else
    mkdir -p "$run"
  fi
  "$PYTHON" "$ROOT/experiments/model409_instance_metric_train.py" \
    --data-root "$data" --init-checkpoint "$init" --output-dir "$run" \
    --arm candidate --metric-weight "$weight" --device "$DEVICE" \
    --seed 1 --epochs 8 $resume_flag
  if [[ -s "$oof/valid_probabilities.npz" ]]; then
    echo "candidate fold$fold OOF archive exists; preserving it."
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

"$PYTHON" "$LOCK_TOOL" --create-oof-lock --root "$ROOT"
"$PYTHON" "$ROOT/experiments/score_model409_instance_metric_oof.py" \
  --data-root "$DATA_ROOT" --candidate-root "$OUT_ROOT/candidate" \
  --control-root "$CONTROL_ROOT" --baseline-root "$BASELINE_OOF_ROOT" \
  --out "$RESULT" --root "$ROOT"
