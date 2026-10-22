#!/bin/bash
# Round 3: morphology-feature input (feat3) vs. a second seed of the best config (noise reference).
# Waits for GPU memory < 2000MB for 2 minutes before each job (same threshold as mission-3 runner.py).
cd ~/komap/code
PY=~/komap/venv/bin/python
R=~/komap/runs

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
  tail -1 $R/$name.out
}

UP2="--arch unet --encoder efficientnet-b4 --norm imagenet --scale 2 --crop 1024 --bs 4 --iters 12000"
run r3_up2_feat3 $UP2 --input feat3
run r3_up2_seed1 $UP2 --seed 1
echo "ALL DONE $(date)"
