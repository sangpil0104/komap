#!/usr/bin/env bash
# Open-ended seed search for R's B4-329 fold models in lite19a (folds 1,3,4 = the folds lite19a uses).
# Recipe = retrain/retrain_r.sh `b4 329 ...` (16,000 iters, group-CV train_fold{1,3,4}, --no_valid_eval, last.pt);
# only --seed changes: 1557, 1558, ... (taken in order from a shared counter). Everything else in lite19a is the
# released weights (other R, E/F, ratios). Per seed: train 3 folds on one GPU -> lite19a valid20 (scored) + test10.
# "orig" = released 329 folds through the same pipeline (reference).
#   bash run.sh                 start/resume (GPUs 0,2,3, one seed per GPU at a time)
#   touch STOP                  graceful stop: each worker finishes its current seed, then exits
#   bash stop_now.sh            immediate stop (kills training/inference; the unfinished seed is resumed on next start)
set -uo pipefail
ROOT=/mnt/hdd/lth22/KOMAP2026_runs/model349-lite19a-r70-swa
B=$ROOT/Model349_lite19a_r70_swa_bundle; L=$B/Model349_fusion805_lite23_b4ws
DATA=/mnt/hdd/lth22/KOMAP2026; RW=$ROOT/WEIGHTS/R; AS=$ROOT/WEIGHTS/assets
OUT=$ROOT/r329_fold_seed_search; CV=$OUT/data_group_cv
PY=/mnt/hdd/lth22/KOMAP2026_runs/release_env/venv/bin/python; export PYTHON=$PY
CODE=$B/train_route/experiments/model326_hybrid_fullstats/KoMaP
N329=model329_fullstats_width4000_uniform_group_oof; FOLDS=(1 3 4); GPUS=(0 2 3); FIRST_SEED=1557
mkdir -p "$OUT/logs" "$OUT/seeds"; EV=$OUT/logs/events.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$EV"; }
[[ -f $OUT/next_seed ]] || echo $FIRST_SEED > "$OUT/next_seed"

claim_seed(){ # resume an unfinished seed claimed earlier by this GPU, else take the next counter value
  local gpu=$1 s
  for d in "$OUT"/seeds/s*; do [[ -f $d/claimed_gpu$gpu && ! -f $d/done ]] && { basename "$d" | tr -d s; return; }; done
  exec 9>"$OUT/next_seed.lock"; flock 9; s=$(cat "$OUT/next_seed"); echo $((s + 1)) > "$OUT/next_seed"; flock -u 9
  mkdir -p "$OUT/seeds/s$s"; touch "$OUT/seeds/s$s/claimed_gpu$gpu"; echo "$s"; }

train_fold(){ local gpu=$1 s=$2 f=$3 run=$OUT/seeds/s$2/$N329/fold$3 t0=$SECONDS
  [[ -f $run/last.pt ]] && { log "GPU$gpu seed$s fold$f skip (done)"; return 0; }
  mkdir -p "$run"; log "GPU$gpu seed$s fold$f train start"
  ( cd "$CODE" && CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 "$PY" -u train.py --data "$CV" --train_splits "train_fold$f" \
      --out "$run" --arch unet --encoder efficientnet-b4 --norm image_global --input zcdf3 --crop 1536 --scale 1 \
      --target_width 4000 --sample uniform --bs 2 --iters 16000 --lr 3e-4 --wd 1e-2 --dice_w 1 --cldice_w 0 \
      --seed "$s" --workers 4 --no_valid_eval ) > "$run/train.log" 2>&1
  [[ -f $run/last.pt && -f $run/args.json ]] || { log "GPU$gpu seed$s fold$f FAIL (see $run/train.log)"; return 1; }
  log "GPU$gpu seed$s fold$f train done ($(( (SECONDS - t0) / 60 )) min) | $(tail -1 "$run/train.log" | cut -c1-200)"; }

infer_score(){ local gpu=$1 tag=$2 W=$OUT/seeds/$2 src=$3 t0=$SECONDS
  mkdir -p "$W/weights_R"
  for e in "$RW"/*; do [[ $(basename "$e") == "$N329" ]] || ln -sfn "$e" "$W/weights_R/$(basename "$e")"; done
  ln -sfn "$src" "$W/weights_R/$N329"
  if [[ ! -e $W/parent ]]; then
    mkdir -p "$W/parent"; (cd "$B/Model349_fusion805" && tar --exclude=./Model349_exh14/weights -cf - .) | (cd "$W/parent" && tar -xf -)
    ln -s "$W/weights_R" "$W/parent/Model349_exh14/weights"
  fi
  local extra=(); [[ $tag != orig ]] && extra=(--allow-retrained)
  for split in valid test; do d=$W/lite19a/$split; [[ -e $d/completion.json ]] && continue; rm -rf "$d"; mkdir -p "$W/lite19a"
    log "GPU$gpu $tag infer $split start"
    "$PY" "$L/infer_compact.py" --config "$L/selected_config_lite19a.json" --parent-package "$W/parent" --assets "$AS" \
      --input "$DATA/$split/images" --output "$d" --gpus "$gpu" "${extra[@]}" > "$W/infer_$split.log" 2>&1 \
      || { log "GPU$gpu $tag infer $split FAIL (see $W/infer_$split.log)"; return 1; }
  done
  [[ -e $W/valid_metrics.json ]] || bash "$B/score.sh" "$W/lite19a/valid/fusion/predicted_labels" "$DATA/valid/masks" "$W/valid_metrics.json" > /dev/null \
    || { log "GPU$gpu $tag score FAIL"; return 1; }
  local m; m=$("$PY" -c "import json,sys;print('%.6f'%(json.load(open(sys.argv[1]))['image_miou']*100))" "$W/valid_metrics.json")
  log "GPU$gpu $tag RESULT valid20 lite19a ${m}% (infer $(( (SECONDS - t0) / 60 )) min)"
  touch "$W/done"; "$PY" "$OUT/summarize.py" "$OUT" > "$OUT/summary.txt" 2>> "$EV"; }

worker(){ local gpu=$1 s f ok
  if [[ $gpu == "${GPUS[0]}" && ! -f $OUT/seeds/orig/done ]]; then
    infer_score "$gpu" orig "$RW/$N329" || log "GPU$gpu orig reference failed"; fi
  while [[ ! -f $OUT/STOP ]]; do
    s=$(claim_seed "$gpu"); log "GPU$gpu claim seed$s"; ok=1
    for f in "${FOLDS[@]}"; do train_fold "$gpu" "$s" "$f" || { ok=0; break; }; done
    if (( ok )); then infer_score "$gpu" "s$s" "$OUT/seeds/s$s/$N329" || ok=0; fi
    if (( ! ok )); then touch "$OUT/seeds/s$s/failed"; touch "$OUT/seeds/s$s/done"; log "GPU$gpu seed$s marked failed, moving on"; sleep 60; fi
  done; log "GPU$gpu STOP file seen, worker exits"; }

# start-only idle check
for g in "${GPUS[@]}"; do IFS=', ' read -r mem util <<< "$(nvidia-smi -i "$g" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  (( mem <= 64 && util == 0 )) || { log "GPU $g busy before start (mem $mem MiB, util $util%), abort"; exit 2; }; done
rm -f "$OUT/STOP"; log "=== start: GPUs ${GPUS[*]}, next seed $(cat "$OUT/next_seed"), folds ${FOLDS[*]} ==="
for g in "${GPUS[@]}"; do worker "$g" > "$OUT/logs/worker_gpu$g.log" 2>&1 & echo $! > "$OUT/logs/worker_gpu$g.pid"; done
wait; log "=== all workers stopped ==="
