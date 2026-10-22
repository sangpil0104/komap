#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DATA_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

BUNDLE_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
DATA_ROOT="$(cd "$1" && pwd -P)"
OUTPUT_ROOT="$2"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd -P)"
PYTHON_BIN="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
WORKERS=4
B4_CODE="$BUNDLE_ROOT/model329"
HR_CODE="$BUNDLE_ROOT/model392"
SPLITS="$OUTPUT_ROOT/splits"
B4_DATA="$SPLITS/model329_cv"
HR_DATA="$SPLITS/model392_cv"
B4_OUT="$OUTPUT_ROOT/model329"
HR_OUT="$OUTPUT_ROOT/model392"

if [[ ! -x "$(command -v "$PYTHON_BIN" || true)" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -d "$DATA_ROOT/train/images" || ! -d "$DATA_ROOT/train/masks" ]]; then
  echo "Expected DATA_ROOT/train/images and DATA_ROOT/train/masks: $DATA_ROOT" >&2
  exit 2
fi
"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/prepare_oof_splits.py" \
  --data-root "$DATA_ROOT" --output-root "$SPLITS"
mkdir -p "$B4_OUT" "$HR_OUT"

for fold in 1 2 3 4; do
  run="$B4_OUT/fold$fold"
  if [[ -f "$run/last.pt" ]]; then
    "$PYTHON_BIN" "$BUNDLE_ROOT/scripts/verify_training_args.py" \
      --family model329 --args-file "$run/args.json" --fold "$fold"
  elif [[ -e "$run" ]]; then
    echo "Incomplete B4 run exists at $run; it has no resumable optimizer state." >&2
    echo "Preserve it and choose a new OUTPUT_ROOT; this script will not delete it." >&2
    exit 3
  else
    mkdir -p "$run"
    "$PYTHON_BIN" "$B4_CODE/train.py" \
      --data "$B4_DATA" --train_splits "train_fold$fold" --out "$run" \
      --arch unet --encoder efficientnet-b4 --norm image_global --input zcdf3 \
      --crop 1536 --target_width 4000 --sample uniform --bs 2 --iters 16000 \
      --lr 3e-4 --wd 1e-2 --dice_w 1 --cldice_w 0 --seed 0 \
      --workers "$WORKERS" --no_valid_eval > "$run/train.log" 2>&1
  fi

  oof="$B4_OUT/fold${fold}_oof"
  if [[ -f "$oof/probabilities.npz" ]]; then
    echo "B4 fold $fold OOF probabilities already exist; keeping them."
  elif [[ -e "$oof" ]]; then
    echo "Incomplete B4 OOF output exists at $oof; refusing to overwrite." >&2
    exit 3
  else
    "$PYTHON_BIN" "$B4_CODE/predict.py" \
      --data "$B4_DATA" --split "oof_fold$fold" --runs "$run" \
      --out "$oof" --save_probs > "$B4_OUT/fold${fold}_oof.log" 2>&1
  fi
  "$PYTHON_BIN" "$BUNDLE_ROOT/scripts/write_oof_manifest.py" \
    --output-root "$OUTPUT_ROOT" --family model329 --fold "$fold"
done

for seed in 1 2; do
  mkdir -p "$HR_OUT/seed$seed"
  for fold in 1 2 3 4; do
    run="$HR_OUT/seed$seed/fold$fold"
    fold_data="$HR_DATA/seed${seed}_fold${fold}"
    if [[ -e "$run" ]]; then
      if [[ ! -f "$run/last.pt" || ! -f "$run/train_state.pt" || ! -f "$run/args.json" ]]; then
        echo "Incomplete HRNet run at $run; preserve it and choose a new OUTPUT_ROOT." >&2
        exit 3
      fi
      "$PYTHON_BIN" "$BUNDLE_ROOT/scripts/verify_training_args.py" \
        --family model392 --args-file "$run/args.json" --seed "$seed" --fold "$fold"
      epoch="$($PYTHON_BIN -c 'import sys,torch; d=torch.load(sys.argv[1],map_location="cpu",weights_only=False); print(int(d.get("epoch",0)))' "$run/train_state.pt")"
      if (( epoch == 70 )); then
        echo "HRNet seed$seed fold$fold reached epoch $epoch; keeping checkpoint."
      elif (( epoch > 0 )); then
        "$PYTHON_BIN" "$HR_CODE/train.py" \
          --data-root "$fold_data" --output-dir "$run" --device "$DEVICE" \
          --architecture UnetPlusPlus --encoder-name tu-hrnet_w48 --encoder-weights imagenet \
          --patch-size 512 --input-scale 2.0 --normalize-width \
          --samples-per-image 12 --batch-size 4 --accumulation-steps 2 \
          --num-workers "$WORKERS" --epochs 70 --lr 3e-4 --encoder-lr-mult 1.0 \
          --weight-decay 1e-4 --boundary-head-weight 1.0 --seed "$seed" \
          --rng-mode legacy --val-every 0 --resume > "$run/train.log" 2>&1
      else
        echo "Invalid HRNet resume epoch at $run: $epoch" >&2
        exit 3
      fi
    else
      mkdir -p "$run"
      "$PYTHON_BIN" "$HR_CODE/train.py" \
        --data-root "$fold_data" --output-dir "$run" --device "$DEVICE" \
        --architecture UnetPlusPlus --encoder-name tu-hrnet_w48 --encoder-weights imagenet \
        --patch-size 512 --input-scale 2.0 --normalize-width \
        --samples-per-image 12 --batch-size 4 --accumulation-steps 2 \
        --num-workers "$WORKERS" --epochs 70 --lr 3e-4 --encoder-lr-mult 1.0 \
        --weight-decay 1e-4 --boundary-head-weight 1.0 --seed "$seed" \
        --rng-mode legacy --val-every 0 --resume > "$run/train.log" 2>&1
    fi
    "$PYTHON_BIN" "$BUNDLE_ROOT/scripts/verify_training_args.py" \
      --family model392 --args-file "$run/args.json" --seed "$seed" --fold "$fold"

    oof="$HR_OUT/seed$seed/fold${fold}_oof"
    if [[ -f "$oof/valid_probabilities.npz" ]]; then
      echo "HRNet seed$seed fold$fold OOF probabilities already exist; keeping them."
    elif [[ -e "$oof" ]]; then
      echo "Incomplete HRNet OOF output exists at $oof; refusing to overwrite." >&2
      exit 3
    else
      "$PYTHON_BIN" "$HR_CODE/evaluate_image_mean.py" \
        --checkpoint "$run/last.pt" --data-root "$fold_data" --split valid \
        --output-dir "$oof" --device "$DEVICE" --batch-size 2 \
        --patch-size 512 --overlap 170 --input-scale 2.0 --tta-mode flip4 \
        > "$HR_OUT/seed$seed/fold${fold}_oof.log" 2>&1
    fi
    "$PYTHON_BIN" "$BUNDLE_ROOT/scripts/write_oof_manifest.py" \
      --output-root "$OUTPUT_ROOT" --family model392 --seed "$seed" --fold "$fold"
  done
done

"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/score_model397_oof.py" \
  --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"
