#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 DATA_ROOT" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
DATA_ROOT="$(cd "$1" && pwd -P)"
PYTHON="${PYTHON:-/workspace/venv-komap/bin/python}"
DEVICE="${DEVICE:?Set DEVICE to the assigned P100 CUDA device}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/model404_component_recall_v5}"
SPLITS="$ROOT/outputs/model403_local_supcon/splits/model392_cv"
BASELINE_ROOT="$ROOT/outputs/model392_widthnorm/seed1"
CALIBRATION="$OUT_ROOT/calibration.json"
LOCK="$ROOT/experiments/results/model404_preregistration_lock_v5.json"
TRAINER="$ROOT/experiments/model404_component_recall_train_v4.py"
EVALUATOR="$ROOT/model397_reproduction_bundle/model392/evaluate_image_mean.py"

if [[ -L "$ROOT/outputs" || -L "$OUT_ROOT" || -L "$SPLITS" ]]; then
  echo "Refusing symlinked output/split path." >&2
  exit 3
fi
mkdir -p "$OUT_ROOT"
if [[ -L "$OUT_ROOT" ]]; then
  echo "Refusing symlinked Model404 v5 output root." >&2
  exit 3
fi
OUT_ROOT="$(cd "$OUT_ROOT" && pwd -P)"
exec 9>>"$OUT_ROOT/.model404_v5_oof.lock"
if ! flock -n 9; then
  echo "Another Model404 v5 OOF runner already holds the lock." >&2
  exit 6
fi

MODEL403_LOCK="$ROOT/outputs/model403_local_supcon/.model403_oof.lock"
if [[ -e "$MODEL403_LOCK" ]]; then
  exec 8>>"$MODEL403_LOCK"
  if ! flock -n 8; then
    echo "Model403 still holds the P100 experiment lock; refusing concurrent training." >&2
    exit 7
  fi
fi
if [[ ! -s "$LOCK" || ! -s "$CALIBRATION" || ! -d "$SPLITS" ]]; then
  echo "Missing Model404 v5 lock/calibration or fixed grouped-fold split." >&2
  exit 4
fi
"$PYTHON" "$ROOT/experiments/model404_preregistration_lock_v5.py" --root "$ROOT" --verify
for fold in 1 2 3 4; do
  if [[ ! -s "$BASELINE_ROOT/fold$fold/last.pt" || ! -s "$BASELINE_ROOT/fold${fold}_oof/valid_probabilities.npz" || ! -d "$SPLITS/seed1_fold$fold" ]]; then
    echo "Missing Model392 checkpoint/OOF or fixed split for fold $fold" >&2
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
      weight="$("$PYTHON" -c 'import json,math,sys; f=json.load(open(sys.argv[1]))["folds"][sys.argv[2]]; w=float(f["component_weight"]); assert f["training_split_only"] and f["nonempty_gradient_batches"]>=6 and f["target_auxiliary_gradient_fraction"]==0.10 and math.isfinite(w) and 1e-5<=w<=1.0 and math.isclose(w,0.10/float(f["median_gradient_norm_ratio"]),rel_tol=0,abs_tol=1e-12); print(f"{w:.15g}")' "$CALIBRATION" "$fold")"
    fi
    if [[ -f "$run/train_state.pt" ]]; then
      resume=(--resume)
    else
      if [[ -e "$run/last.pt" ]]; then
        echo "Checkpoint exists without resumable state at $run; refusing overwrite." >&2
        exit 8
      fi
      mkdir -p "$run"
      resume=()
    fi
    "$PYTHON" "$TRAINER" --data-root "$data" --init-checkpoint "$init" --output-dir "$run" --arm "$arm" --component-weight "$weight" --device "$DEVICE" --seed 1 --epochs 8 "${resume[@]}"
    if [[ -s "$oof/valid_probabilities.npz" ]]; then
      echo "$arm fold$fold OOF archive exists; preserving it."
    elif [[ -e "$oof" ]]; then
      echo "Incomplete OOF output exists at $oof; refusing overwrite." >&2
      exit 9
    else
      "$PYTHON" "$EVALUATOR" --checkpoint "$run/last.pt" --data-root "$data" --split valid --output-dir "$oof" --device "$DEVICE" --batch-size 2 --patch-size 512 --overlap 170 --input-scale 2.0 --tta-mode flip4
    fi
  done
done

mkdir -p "$OUT_ROOT/results"
"$PYTHON" "$ROOT/experiments/score_model404_component_recall_oof_v5.py" --data-root "$DATA_ROOT" --candidate-root "$OUT_ROOT/candidate" --control-root "$OUT_ROOT/control" --baseline-root "$ROOT/outputs/model392_widthnorm" --out "$OUT_ROOT/results/model404_component_recall_oof.json"
