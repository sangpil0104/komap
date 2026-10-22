#!/usr/bin/env bash
# Retrain E (Model397: B4-329 4 folds + HRNet392 seed1/seed2 x 4 folds) for one seed, with train70 OOF.
#   bash retrain/retrain_e.sh SEED PARENT_PACKAGE DATA_ROOT OUT [GPU]
# PARENT_PACKAGE: unzipped Model349_fusion805 (uses model397_reproduction_bundle/)
# DATA_ROOT     : folder with train/{images,masks} (only the 70 train images are read)
# OUT           : new folder in the model397 layout: OUT/splits, OUT/model329/fold{f}[_oof], OUT/model392/seed{s}/fold{f}[_oof]
# Commands are copied from model397_reproduction_bundle/scripts/run_model397_oof.sh. Only the training seeds change:
# PART=prep|b4|hr1|hr2|all selects a lane. B4 --seed 0 -> 1000*SEED, HRNet --seed s -> s + 1000*SEED (the seed1/seed2 split folders are unchanged).
set -euo pipefail
S=$1; PKG=$(cd "$2" && pwd); DATA=$(cd "$3" && pwd); OUT=$4; GPU=${5:-0}
PY=${PYTHON:-python}; B=$PKG/model397_reproduction_bundle; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
export CUDA_VISIBLE_DEVICES=$GPU; DEVICE=cuda:0
need(){ [[ -f "$1" ]] || { echo "FAIL: missing $1"; exit 1; }; }
prep(){ [[ -d "$OUT/splits/model329_cv" ]] || "$PY" "$B/scripts/prepare_oof_splits.py" --data-root "$DATA" --output-root "$OUT/splits"; }
e_b4(){
for f in 1 2 3 4; do
  run=$OUT/model329/fold$f; mkdir -p "$run"
  [[ -f $run/last.pt ]] || "$PY" "$B/model329/train.py" --data "$OUT/splits/model329_cv" --train_splits "train_fold$f" --out "$run" \
    --arch unet --encoder efficientnet-b4 --norm image_global --input zcdf3 --crop 1536 --target_width 4000 --sample uniform --bs 2 \
    --iters 16000 --lr 3e-4 --wd 1e-2 --dice_w 1 --cldice_w 0 --seed $((1000 * S)) --workers 4 --no_valid_eval > "$run/train.log" 2>&1
  need "$run/last.pt"
  [[ -f $OUT/model329/fold${f}_oof/probabilities.npz ]] || "$PY" "$B/model329/predict.py" --data "$OUT/splits/model329_cv" \
    --split "oof_fold$f" --runs "$run" --out "$OUT/model329/fold${f}_oof" --save_probs > "$OUT/model329/fold${f}_oof.log" 2>&1
  need "$OUT/model329/fold${f}_oof/probabilities.npz"
done
}
e_hr(){ local s=$1 f run data
for f in 1 2 3 4; do
  run=$OUT/model392/seed$s/fold$f; data=$OUT/splits/model392_cv/seed${s}_fold$f; mkdir -p "$run"
  [[ -f $run/last.pt ]] || "$PY" "$B/model392/train.py" --data-root "$data" --output-dir "$run" --device $DEVICE \
    --architecture UnetPlusPlus --encoder-name tu-hrnet_w48 --encoder-weights imagenet --patch-size 512 --input-scale 2.0 --normalize-width \
    --samples-per-image 12 --batch-size 4 --accumulation-steps 2 --num-workers 4 --epochs 70 --lr 3e-4 --encoder-lr-mult 1.0 \
    --weight-decay 1e-4 --boundary-head-weight 1.0 --seed $((s + 1000 * S)) --rng-mode legacy --val-every 0 --resume > "$run/train.log" 2>&1
  need "$run/last.pt"
  [[ -f $OUT/model392/seed$s/fold${f}_oof/valid_probabilities.npz ]] || "$PY" "$B/model392/evaluate_image_mean.py" --checkpoint "$run/last.pt" \
    --data-root "$data" --split valid --output-dir "$OUT/model392/seed$s/fold${f}_oof" --device $DEVICE --batch-size 2 \
    --patch-size 512 --overlap 170 --input-scale 2.0 --tta-mode flip4 > "$OUT/model392/seed$s/fold${f}_oof.log" 2>&1
  need "$OUT/model392/seed$s/fold${f}_oof/valid_probabilities.npz"
done
}
case "${PART:-all}" in prep) prep;; b4) prep; e_b4;; hr1) prep; e_hr 1;; hr2) prep; e_hr 2;; all) prep; e_b4; e_hr 1; e_hr 2;; *) echo "PART must be prep|b4|hr1|hr2|all"; exit 2;; esac
echo "E ${PART:-all} DONE seed=$S -> $OUT"
