#!/bin/bash
# Round 7: align training with the data and the metric.
#  (1) per-image magnification normalization — 675px images have Al3Ni rods half as thick as
#      1230px ones and needles at the 2px pixel limit, yet a fixed 3x scales them the least
#  (2) uniform image sampling — the metric weights every image equally, area-proportional
#      sampling shows large images 4.5x more often
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
  grep -E "valid TTA final" $R/$name.out || tail -1 $R/$name.out
}

BASE="--arch unet --encoder efficientnet-b4 --norm imagenet --crop 1536 --bs 2 --iters 16000"

run r7_tw4000  $BASE --target_width 4000
run r7_uniform $BASE --scale 3 --sample uniform
# if both help, the combination is the natural next step
run r7_tw4000_uniform $BASE --target_width 4000 --sample uniform
echo "ALL DONE $(date)"
