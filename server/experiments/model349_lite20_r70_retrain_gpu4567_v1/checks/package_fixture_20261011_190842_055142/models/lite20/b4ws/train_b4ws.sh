#!/usr/bin/env bash
# BN-frozen warm start of the four Model329 B4 folds used by E (Model401).
#   bash b4ws/train_b4ws.sh SPLITS_ROOT ASSETS GPU
# SPLITS_ROOT: folder with train_fold1..4 splits (model397_reproduction_bundle/scripts/prepare_oof_splits.py
#              output, i.e. <output>/splits/model329_cv). Only train70 images; valid/test are never read.
# ASSETS     : the auxiliary asset folder; parents are read from ASSETS/model329/foldK/last.pt and the new
#              folds are written to ASSETS/model329_ws/foldK (must not exist).
set -euo pipefail
DATA=$1; ASSETS=$2; GPU=${3:-0}
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PYTHON:-python}
for k in 1 2 3 4; do
  out=$ASSETS/model329_ws/fold$k
  [ -e "$out/last.pt" ] && { echo "skip fold$k (exists)"; continue; }
  mkdir -p "$out"
  ( cd "$HERE" && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$HERE/../Model349_exh14/oof_design/runtime" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 \
    "$PY" train_bnfreeze.py --data "$DATA" --train_splits train_fold$k --out "$out" --arch unet --encoder efficientnet-b4 \
      --norm image_global --input zcdf3 --crop 1536 --scale 1 --target_width 4000 --sample uniform --bs 2 \
      --iters 2000 --lr 3e-5 --wd 1e-2 --dice_w 1 --cldice_w 0 --seed 819349 --workers 4 --no_valid_eval \
      --deterministic_workers --init_checkpoint "$ASSETS/model329/fold$k/last.pt" ) > "$out/train.log" 2>&1
  echo "fold$k done"
done
