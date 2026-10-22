#!/usr/bin/env bash
# Immediate stop of the R329 fold seed search (workers + their training/inference children).
OUT=/mnt/hdd/lth22/KOMAP2026_runs/model349-lite19a-r70-swa/r329_fold_seed_search
touch "$OUT/STOP"
for p in "$OUT"/logs/worker_gpu*.pid; do [[ -f $p ]] && pkill -TERM -g "$(ps -o pgid= -p "$(cat "$p")" | tr -d ' ')" 2>/dev/null; done
pkill -TERM -f "$OUT/seeds/" 2>/dev/null
echo "[$(date '+%F %T')] stop_now: killed workers" >> "$OUT/logs/events.log"
