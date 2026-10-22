#!/usr/bin/env bash
# Retrain all 23 weights of lite23 + B4ws for one seed, then predict valid (scored) and test with the fixed lite23 config.
#   bash retrain/retrain_all.sh SEED PARENT_PACKAGE TRAIN_ROUTE_ROOT DATA_ROOT OUT [GPUS]
# GPUS: comma-separated GPU ids, e.g. 0,1,2,3 (default 0). Independent lanes run in parallel, one lane chain per GPU.
# PARENT_PACKAGE   : unzipped Model349_fusion805 (lhs, this commit or later)
# TRAIN_ROUTE_ROOT : R training code (Model349_train_route_code.zip contents)
# DATA_ROOT        : train/{images,masks} (70), valid/{images,masks} (20), test/images (10, optional)
# OUT              : run folder (re-running skips finished stages)
# Nothing is re-selected: R uses the exh14 config, E 0.5/0.25/0.25 with HR seed1 folds 2,4 and seed2 folds 1,4,
# rule {Al3Ni, Al}, then F. Valid labels are read only by score_masks.py (and, as in the original R pipeline,
# for best.pt selection of the ResNet/ASPP chain).
# B4-329 is trained once (4 folds, E lane) and shared with R (folds 1,3,4): same recipe, same specimen folds.
set -euo pipefail
S=$1; PKG=$(cd "$2" && pwd); TR=$(cd "$3" && pwd); DATA=$(cd "$4" && pwd); OUT=$5; IFS=, read -ra G <<< "${6:-0}"
HERE=$(cd "$(dirname "$0")/.." && pwd); PY=${PYTHON:-python}; mkdir -p "$OUT/logs"; OUT=$(cd "$OUT" && pwd)
R="bash $HERE/retrain/retrain_r.sh $S $TR $DATA $OUT/R"; E="bash $HERE/retrain/retrain_e.sh $S $PKG $DATA $OUT/E"
PART=prep bash "$HERE/retrain/retrain_e.sh" "$S" "$PKG" "$DATA" "$OUT/E" "${G[0]}"
# lanes (longest first); B4ws needs the shared B4-329, so it follows it in the same lane
LANES=("e_b4329_then_b4ws" "r_b4" "r_resnet" "e_hr1" "e_hr2" "r_hrnet"); HOURS=(21 18 12 9 9 4)   # rough P100 hours
# longest-processing-time assignment of lanes to GPUs
declare -a LOAD ASSIGN; for gi in "${!G[@]}"; do LOAD[gi]=0; done
for li in "${!LANES[@]}"; do best=0; for gi in "${!G[@]}"; do (( LOAD[gi] < LOAD[best] )) && best=$gi; done
  ASSIGN[li]=$best; LOAD[best]=$(( LOAD[best] + HOURS[li] )); done
echo "lane plan: $(for li in "${!LANES[@]}"; do printf '%s->GPU%s ' "${LANES[li]}" "${G[ASSIGN[li]]}"; done)"
run_lane(){ local lane=$1 gpu=$2; echo "[$(date +%T)] start $lane on GPU $gpu"
  case $lane in
    e_b4329_then_b4ws) PART=b4 $E $gpu && bash "$HERE/retrain/retrain_b4ws.sh" "$S" "$PKG" "$OUT/E" "$OUT/assets" "$gpu";;
    r_b4) PART=b4 $R $gpu;; r_resnet) PART=resnet $R $gpu;; r_hrnet) PART=hrnet $R $gpu;;
    e_hr1) PART=hr1 $E $gpu;; e_hr2) PART=hr2 $E $gpu;;
  esac > "$OUT/logs/lane_$lane.log" 2>&1 || { echo "FAIL $lane (see $OUT/logs/lane_$lane.log)"; return 1; }
  echo "[$(date +%T)] done $lane"; }
pids=()
for gi in "${!G[@]}"; do
  ( for li in "${!LANES[@]}"; do if (( ASSIGN[li] == gi )); then run_lane "${LANES[$li]}" "${G[$gi]}" || true; fi; done ) & pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
for l in "${LANES[@]}"; do grep -q "DONE" "$OUT/logs/lane_$l.log" 2>/dev/null || { echo "lane $l did not finish"; fail=1; }; done
(( fail == 0 )) || exit 1
# share B4-329 folds 1,3,4 with R
for k in 1 3 4; do d=$OUT/R/weights/model329_fullstats_width4000_uniform_group_oof/fold$k; mkdir -p "$d"
  ln -sfn "$OUT/E/model329/fold$k/last.pt" "$d/last.pt"; ln -sfn "$OUT/E/model329/fold$k/args.json" "$d/args.json"; done
# F: features from the retrained E's train70 OOF (original B4 recipe, as for the published F), then 8 epochs
[[ -f $OUT/F/features/manifest.json ]] || "$PY" "$HERE/refiner_retrain/build_features.py" --model397-output "$OUT/E" --train "$DATA/train" \
  --palette-config "$PKG/Model349_exh14/configs/model349_lite.yaml" --out "$OUT/F/features"
[[ -f $OUT/F/training/epoch8.pt ]] || CUDA_VISIBLE_DEVICES=${G[0]} "$PY" "$HERE/refiner_retrain/train_refiner.py" --parent-package "$PKG" \
  --features "$OUT/F/features" --out "$OUT/F/training"
# assemble: a code-only copy of the parent whose Model349_exh14/weights points at the retrained R, and the E/F assets
if [[ ! -e $OUT/parent ]]; then
  mkdir -p "$OUT/parent"; (cd "$PKG" && tar --exclude=./Model349_exh14/weights -cf - .) | (cd "$OUT/parent" && tar -xf -)
  ln -s "$OUT/R/weights" "$OUT/parent/Model349_exh14/weights"
fi
mkdir -p "$OUT/assets/refiner"; ln -sfn "$OUT/F/training/epoch8.pt" "$OUT/assets/refiner/epoch8.pt"
for s in 1 2; do for f in 1 2 3 4; do mkdir -p "$OUT/assets/model392/seed$s/fold$f"; ln -sfn "$OUT/E/model392/seed$s/fold$f/last.pt" "$OUT/assets/model392/seed$s/fold$f/last.pt"; done; done
for split in valid test; do
  [[ -d $DATA/$split/images ]] || continue; [[ -e $OUT/pred_$split/completion.json ]] && continue; rm -rf "$OUT/pred_$split"
  CUDA_VISIBLE_DEVICES=${G[0]} "$PY" "$HERE/infer_compact.py" --config "$HERE/selected_config_b4ws.json" --allow-retrained \
    --parent-package "$OUT/parent" --assets "$OUT/assets" --input "$DATA/$split/images" --output "$OUT/pred_$split" --gpus 0
done
[[ -d $DATA/valid/masks ]] && "$PY" "$PKG/score_masks.py" --labels "$OUT/pred_valid/fusion/predicted_labels" --gt "$DATA/valid/masks" \
  --output "$OUT/pred_valid/metrics.json" && "$PY" -c "import json,sys;print('valid20 mIoU', json.load(open(sys.argv[1]))['image_miou']*100)" "$OUT/pred_valid/metrics.json"
echo "ALL DONE seed=$S -> $OUT"
