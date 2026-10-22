#!/usr/bin/env bash
# Train the 3 R B4 paths of lite20_r70 once each on ALL train70 images (no specimen folds, no fold selection).
#   bash r70/train_r70.sh SEED DATA OUT_WEIGHTS [GPUS e.g. 0,1,2]
# DATA        : DATA/train/{images,masks} (70). valid/test are never read (--no_valid_eval).
# OUT_WEIGHTS : writes OUT_WEIGHTS/<B4 dir>_full70/{last.pt,args.json,train.log}; finished models are skipped.
# Recipe = the fold recipe of retrain/retrain_r.sh (same code, input, scale, crop, lr, seed rule 0 + 1000*SEED); the data
# changes from train_foldK (52.5 images on average) to train (70), and iterations are scaled by the same ratio,
# 16,000 x 70/52.5 ~= 21,000 (ITERS env overrides; fixed before seeing any result), so each image is seen as often as in
# fold training. One job per GPU; with fewer GPUs jobs queue.
set -euo pipefail
S=$1; DATA=$(cd "$2" && pwd); OUT=$3; IFS=, read -ra G <<< "${4:-0}"
ITERS=${ITERS:-21000}; HERE=$(cd "$(dirname "$0")/.." && pwd); TR=$HERE/train_route/experiments; PY=${PYTHON:-python}
[[ $(ls "$DATA/train/images"/*_image.png | wc -l) -eq 70 ]] || { echo "expected 70 images in $DATA/train/images"; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
# name code norm input scale target_width sample
JOBS=("model322_team_b4_group_oof $TR/model322_team_b4/KoMaP imagenet gray3 3 0 area"
      "model326_hybrid_fullstats_group_oof $TR/model326_hybrid_fullstats/KoMaP image_global zcdf3 3 0 area"
      "model329_fullstats_width4000_uniform_group_oof $TR/model326_hybrid_fullstats/KoMaP image_global zcdf3 1 4000 uniform")
todo=0; for j in "${JOBS[@]}"; do set -- $j; [[ -f $OUT/${1}_full70/last.pt ]] || todo=1; done
(( todo )) || { echo "R70 B4 DONE seed=$S -> $OUT (all trained)"; exit 0; }
# start-only idle check: refuse a GPU that is busy now; once started, keep running even if others join
for g in "${G[@]}"; do u=$(nvidia-smi -i "$g" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  IFS=', ' read -r mem util <<< "$u"; (( mem <= 64 && util == 0 )) || { echo "GPU $g busy before start ($u)"; exit 2; }; done
train_one(){ local gpu=$1 name=$2 code=$3 norm=$4 input=$5 scale=$6 tw=$7 sample=$8 run=$OUT/${2}_full70
  [[ -f $run/last.pt ]] && { echo "skip $name (done)"; return; }
  mkdir -p "$run"; echo "[$(date +%T)] start $name on GPU $gpu (iters $ITERS)"
  ( cd "$code" && CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 "$PY" -u train.py --data "$DATA" --train_splits train \
      --out "$run" --arch unet --encoder efficientnet-b4 --norm "$norm" --input "$input" --crop 1536 --scale "$scale" \
      --target_width "$tw" --sample "$sample" --bs 2 --iters "$ITERS" --lr 3e-4 --wd 1e-2 --dice_w 1 --cldice_w 0 \
      --seed $(( 1000 * S )) --workers 4 --no_valid_eval ) > "$run/train.log" 2>&1
  [[ -f $run/last.pt && -f $run/args.json ]] || { echo "FAIL $name (see $run/train.log)"; return 1; }
  echo "[$(date +%T)] done $name"; }
pids=()
for gi in "${!G[@]}"; do
  ( for ji in "${!JOBS[@]}"; do if (( ji % ${#G[@]} == gi )); then train_one "${G[$gi]}" ${JOBS[$ji]} || exit 1; fi; done ) & pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail == 0 )) && echo "R70 B4 DONE seed=$S -> $OUT" || { echo "R70 B4 training failed"; exit 1; }
