#!/usr/bin/env bash
# BN-frozen warm start of the 4 retrained E B4 folds (same recipe as parent b4ws/train_b4ws.sh; seed 819349 + 1000*SEED).
#   bash retrain/retrain_b4ws.sh SEED PARENT_PACKAGE E_OUT OUT [GPU]
# E_OUT: output of retrain_e.sh (parents E_OUT/model329/foldK/last.pt, split E_OUT/splits/model329_cv); writes OUT/model329_ws/foldK
set -euo pipefail
S=$1; PKG=$(cd "$2" && pwd); E=$(cd "$3" && pwd); OUT=$4; GPU=${5:-0}; PY=${PYTHON:-python}
for k in 1 2 3 4; do
  out=$OUT/model329_ws/fold$k; [[ -e $out/last.pt ]] && continue; mkdir -p "$out"; out=$(cd "$out" && pwd)
  ( cd "$PKG/b4ws" && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$PKG/Model349_exh14/oof_design/runtime" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 \
    "$PY" train_bnfreeze.py --data "$E/splits/model329_cv" --train_splits train_fold$k --out "$out" --arch unet --encoder efficientnet-b4 \
      --norm image_global --input zcdf3 --crop 1536 --scale 1 --target_width 4000 --sample uniform --bs 2 --iters 2000 --lr 3e-5 \
      --wd 1e-2 --dice_w 1 --cldice_w 0 --seed $((819349 + 1000 * S)) --workers 4 --no_valid_eval --deterministic_workers \
      --init_checkpoint "$E/model329/fold$k/last.pt" ) > "$out/train.log" 2>&1
  [[ -f $out/last.pt ]] || { echo "FAIL: missing $out/last.pt"; exit 1; }
done
echo "B4ws DONE seed=$S"
