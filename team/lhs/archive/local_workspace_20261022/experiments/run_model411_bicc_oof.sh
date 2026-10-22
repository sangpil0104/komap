#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PYTHON="/workspace/venv-komap/bin/python"
DEVICE="cuda:0"
DATA_ROOT="$ROOT/data"
SPLITS="$ROOT/outputs/model403_local_supcon/splits/model392_cv"
PARENTS="$ROOT/outputs/model392_widthnorm/seed1"
OUT_ROOT="$ROOT/outputs/model411_bicc"
CALIBRATION="$OUT_ROOT/calibration.json"
RESULT="$ROOT/experiments/results/model411_bicc_oof.json"
PREDICTOR="$ROOT/experiments/model411_predict_oof_probabilities.py"
LOCK_TOOL="$ROOT/experiments/model411_bicc_lock.py"
M410_RESULT="$ROOT/experiments/results/model410_component_sampler_oof.json"

if [[ -L "$ROOT/outputs" || -L "$OUT_ROOT" || -L "$SPLITS" ]]; then
  echo "Refusing symlinked output or split root." >&2
  exit 3
fi
if [[ ! -s "$M410_RESULT" ]]; then
  echo "Model410 one-shot train-OOF result is not available; Model411 remains unstarted." >&2
  exit 11
fi
if ! "$PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("preregistered_gate",{}).get("pass") is False and r.get("valid_read") is False and r.get("test_read") is False else 1)' "$M410_RESULT"; then
  echo "Model410 passed its gate or has an invalid scope; Model411 will not start." >&2
  exit 12
fi
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd -P)"

GPU_LOCK="$ROOT/outputs/model409_instance_metric_v2/.model409_instance_metric_oof.lock"
mkdir -p "$(dirname "$GPU_LOCK")"
exec 9>>"$GPU_LOCK"
if ! flock -n 9; then
  echo "Another locked P100 experiment is active; Model411 will not start." >&2
  exit 6
fi
exec 8>>"$OUT_ROOT/.model411_bicc.lock"
if ! flock -n 8; then
  echo "Another Model411 runner holds the experiment lock." >&2
  exit 7
fi
if [[ -e "$RESULT" || -e "$RESULT.claim" ]]; then
  echo "A Model411 one-shot result exists; refusing rerun." >&2
  exit 8
fi

"$PYTHON" "$ROOT/experiments/model411_bicc_loss.py"
if [[ -e "$ROOT/experiments/results/model411_bicc_source_lock.json" ]]; then
  "$PYTHON" "$LOCK_TOOL" --verify-source-lock --root "$ROOT"
else
  "$PYTHON" "$LOCK_TOOL" --create-source-lock --root "$ROOT"
fi

if [[ ! -s "$CALIBRATION" ]]; then
  "$PYTHON" "$ROOT/experiments/model411_bicc_calibrate.py" \
    --split-root "$SPLITS" --baseline-root "$PARENTS" \
    --out "$CALIBRATION" --device "$DEVICE" --probe-examples 32
fi
if [[ -e "$ROOT/experiments/results/model411_bicc_calibration_lock.json" ]]; then
  "$PYTHON" "$LOCK_TOOL" --verify-calibration-lock --root "$ROOT"
else
  "$PYTHON" "$LOCK_TOOL" --create-calibration-lock --root "$ROOT"
fi

for fold in 1 2 3 4; do
  data="$SPLITS/seed1_fold$fold"
  init="$PARENTS/fold$fold/last.pt"
  run="$OUT_ROOT/candidate/fold$fold"
  oof="$OUT_ROOT/candidate/fold${fold}_oof"
  weight="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["folds"][sys.argv[2]]["coefficient"])' "$CALIBRATION" "$fold")"
  if [[ ! -s "$init" || ! -d "$data" ]]; then
    echo "Missing Model392 parent or fixed train fold $fold." >&2
    exit 4
  fi

  resume_flag=()
  if [[ -f "$run/train_state.pt" ]]; then
    resume_flag=(--resume)
  elif [[ -e "$run/last.pt" ]]; then
    echo "Checkpoint exists without resumable state at $run; refusing overwrite." >&2
    exit 9
  elif [[ -e "$run" ]] && find "$run" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Partial non-resumable output exists at $run; refusing overwrite." >&2
    exit 9
  fi
  "$PYTHON" "$ROOT/experiments/model411_bicc_train.py" \
    --arm candidate --data-root "$data" --output-dir "$run" \
    --init-checkpoint "$init" --component-weight "$weight" \
    --device "$DEVICE" --seed 1 --epochs 8 --num-workers 4 \
    "${resume_flag[@]}"

  if [[ -s "$oof/valid_probabilities.npz" ]]; then
    echo "Preserving completed held-out-train predictions for fold $fold."
  elif [[ -e "$oof" ]]; then
    echo "Incomplete OOF output exists at $oof; refusing overwrite." >&2
    exit 10
  else
    "$PYTHON" "$PREDICTOR" --checkpoint "$run/last.pt" --data-root "$data" \
      --output-dir "$oof" --device "$DEVICE" --batch-size 2 \
      --patch-size 512 --overlap 170 --input-scale 2.0
  fi
done

"$PYTHON" "$LOCK_TOOL" --create-oof-lock --root "$ROOT"
"$PYTHON" "$ROOT/experiments/score_model411_bicc_oof.py"
