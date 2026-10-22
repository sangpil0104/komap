#!/bin/bash
# Round 3b: clDice topology loss on the elongated phases (1 Al3Ni rods, 3 eutectic Si needles).
# Starts after queue_round3.sh finishes, then waits for a free GPU like the other queues.
cd ~/komap/code
PY=~/komap/venv/bin/python
R=~/komap/runs

while pgrep -f queue_round3.sh > /dev/null; do sleep 60; done

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
run r3_up2_cldice $UP2 --cldice_w 0.5 --cldice_classes 1,3 --cldice_iters 8 --cldice_pool 2
echo "ALL DONE $(date)"
