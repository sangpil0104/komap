#!/usr/bin/env bash
# E B4-329 BN-frozen warm start on ALL train70 images (train_foldK + oof_foldK) — experiment, not validated.
#   bash full70_experiment/train_b4_full70.sh PARENT_PACKAGE SPLITS_ROOT ASSETS GPU
# PARENT_PACKAGE: unzipped Model349_fusion805 (lhs 19d56d4 or later; uses its b4ws/train_bnfreeze.py)
# SPLITS_ROOT   : <output>/splits/model329_cv from model397_reproduction_bundle/scripts/prepare_oof_splits.py (train70 only)
# ASSETS        : parents ASSETS/model329/foldK/last.pt; writes ASSETS/model329_full70/foldK (must not exist)
set -euo pipefail
PKG=$1; DATA=$2; ASSETS=$3; GPU=${4:-0}; PY=${PYTHON:-python}
for k in 1 2 3 4; do
  out=$ASSETS/model329_full70/fold$k
  [ -e "$out/last.pt" ] && { echo "skip fold$k (exists)"; continue; }
  mkdir -p "$out"
  ( cd "$PKG/b4ws" && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$PKG/Model349_exh14/oof_design/runtime" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 \
    "$PY" train_bnfreeze.py --data "$DATA" --train_splits "train_fold$k,oof_fold$k" --out "$out" --arch unet --encoder efficientnet-b4 \
      --norm image_global --input zcdf3 --crop 1536 --scale 1 --target_width 4000 --sample uniform --bs 2 \
      --iters 2000 --lr 3e-5 --wd 1e-2 --dice_w 1 --cldice_w 0 --seed 819349 --workers 4 --no_valid_eval \
      --deterministic_workers --init_checkpoint "$ASSETS/model329/fold$k/last.pt" ) > "$out/train.log" 2>&1
  echo "fold$k done"
done
