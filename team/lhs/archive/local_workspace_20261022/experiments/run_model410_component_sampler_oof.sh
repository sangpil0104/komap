#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
PYTHON="/workspace/venv-komap/bin/python"
DEVICE="cuda:0"
DATA_ROOT="$ROOT/data"
SPLITS="$ROOT/outputs/model403_local_supcon/splits/model392_cv"
PARENTS="$ROOT/outputs/model392_widthnorm/seed1"
OUT_ROOT="$ROOT/outputs/model410_component_sampler"
LOCK_TOOL="$ROOT/experiments/model410_component_sampler_lock.py"
RESULT="$ROOT/experiments/results/model410_component_sampler_oof.json"
CLAIM="$RESULT.claim"
PREDICTOR="$ROOT/experiments/model410_predict_oof_probabilities.py"

if [[ -L "$ROOT/outputs" || -L "$OUT_ROOT" || -L "$SPLITS" ]]; then
  echo "Refusing symlinked output or split root." >&2
  exit 3
fi
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd -P)"

# Share Model409's exclusive P100 lock. It must finish before this run starts.
GPU_LOCK="$ROOT/outputs/model409_instance_metric_v2/.model409_instance_metric_oof.lock"
mkdir -p "$(dirname "$GPU_LOCK")"
exec 9>>"$GPU_LOCK"
if ! flock -n 9; then
  echo "Another locked P100 segmentation experiment is active; Model410 will not start." >&2
  exit 6
fi

exec 8>>"$OUT_ROOT/.model410_component_sampler.lock"
if ! flock -n 8; then
  echo "Another Model410 runner holds the experiment lock." >&2
  exit 7
fi
if [[ -e "$RESULT" || -e "$CLAIM" ]]; then
  echo "A Model410 one-shot score exists; refusing rerun." >&2
  exit 8
fi

"$PYTHON" "$ROOT/experiments/model410_component_sampler_smoke.py"
"$PYTHON" "$ROOT/experiments/model410_scale_aware_bin_audit.py"
"$PYTHON" "$ROOT/experiments/score_model410_component_sampler_oof.py" --self-test --root "$ROOT"
if [[ -e "$ROOT/experiments/results/model410_component_sampler_source_lock.json" ]]; then
  "$PYTHON" "$LOCK_TOOL" --verify-source-lock --root "$ROOT"
else
  "$PYTHON" "$LOCK_TOOL" --create-source-lock --root "$ROOT"
fi

for fold in 1 2 3 4; do
  if [[ ! -s "$PARENTS/fold$fold/last.pt" || ! -d "$SPLITS/seed1_fold$fold" ]]; then
    echo "Missing Model392 parent checkpoint or fixed fold $fold." >&2
    exit 4
  fi
done

# Alternate order to avoid a fixed arm-order effect across folds.
for fold in 1 2 3 4; do
  data="$SPLITS/seed1_fold$fold"
  init="$PARENTS/fold$fold/last.pt"
  if (( fold % 2 == 1 )); then arms=(control candidate); else arms=(candidate control); fi
  for arm in "${arms[@]}"; do
    run="$OUT_ROOT/$arm/fold$fold"
    oof="$OUT_ROOT/$arm/fold${fold}_oof"
    state="$run/train_state.pt"
    resume_flag=()
    if [[ -f "$state" ]]; then
      resume_flag=(--resume)
    elif [[ -e "$run/last.pt" ]]; then
      echo "Checkpoint exists without resumable state at $run; refusing overwrite." >&2
      exit 9
    elif [[ -e "$run" ]] && find "$run" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      arm_receipt="$run/model410_sampler_arm.json"
      if [[ -f "$arm_receipt" ]] && [[ ! -f "$state" ]]; then
        resume_flag=(--resume)
      else
        echo "Partial non-resumable output exists at $run; refusing overwrite." >&2
        exit 9
      fi
    fi

    "$PYTHON" "$ROOT/experiments/model410_component_sampler_train.py" \
      --arm "$arm" --data-root "$data" --output-dir "$run" \
      --init-checkpoint "$init" --device "$DEVICE" --architecture UnetPlusPlus \
      --encoder-name tu-hrnet_w48 --encoder-weights none \
      --patch-size 512 --input-scale 2.0 --normalize-width \
      --samples-per-image 12 --batch-size 4 --accumulation-steps 2 \
      --num-workers 4 --epochs 8 --lr 1e-4 --encoder-lr-mult 0.3 \
      --weight-decay 1e-4 --seed 1 --rng-mode per_item --val-every 0 \
      "${resume_flag[@]}"

    if [[ -s "$oof/valid_probabilities.npz" ]]; then
      echo "Preserving completed held-out-train predictions for $arm fold$fold."
    elif [[ -e "$oof" ]]; then
      echo "Incomplete OOF output exists at $oof; refusing overwrite." >&2
      exit 10
    else
      "$PYTHON" "$PREDICTOR" --checkpoint "$run/last.pt" --data-root "$data" \
        --output-dir "$oof" --device "$DEVICE" --batch-size 2 \
        --patch-size 512 --overlap 170 --input-scale 2.0
    fi
  done
done

"$PYTHON" "$LOCK_TOOL" --create-oof-lock --root "$ROOT"
"$PYTHON" "$ROOT/experiments/score_model410_component_sampler_oof.py" --root "$ROOT"
