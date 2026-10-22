#!/bin/bash
# Round 7b: confirm the per-image magnification result with a second seed, and push the target
# width further. Starts after queue_round7.sh finishes.
cd ~/komap/code
PY=~/komap/venv/bin/python
R=~/komap/runs

while pgrep -f queue_round7.sh > /dev/null; do sleep 60; done

wait_gpu() {
  local ok=0
  while [ $ok -lt 12 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -lt 2000 ]; then ok=$((ok+1)); else ok=0; fi
    sleep 10
  done
}

run() {
  name=$1; shift
  if [ -f $R/$name/last.pt ]; then echo "=== $name already done"; return; fi
  wait_gpu
  echo "=== $name $(date)"
  $PY train.py --out $R/$name "$@" > $R/$name.out 2>&1
  grep -E "valid TTA final" $R/$name.out || tail -1 $R/$name.out
}

ens() {
  name=$1; shift
  wait_gpu
  echo "=== eval $name $(date)"
  $PY predict.py --split valid --out $R/ens_$name "$@" > $R/ens_$name.out 2>&1
  tail -1 $R/ens_$name.out
}

BASE="--arch unet --encoder efficientnet-b4 --norm imagenet --crop 1536 --bs 2 --iters 16000"

run r7_tw4000_seed1 $BASE --target_width 4000 --seed 1
run r7_tw5000       $BASE --target_width 5000
ens tw4000_s01 --runs $R/r7_tw4000 $R/r7_tw4000_seed1
echo "ALL DONE $(date)"
