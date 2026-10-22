#!/bin/bash
# Round 5: confirm the 3x result with more seeds, and probe 4x.
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

ens() {  # ens <name> <run dirs...>
  name=$1; shift
  wait_gpu
  echo "=== ensemble $name $(date)"
  $PY predict.py --split valid --runs "$@" --out $R/ens_$name > $R/ens_$name.out 2>&1
  tail -1 $R/ens_$name.out
}

BASE="--arch unet --encoder efficientnet-b4 --norm imagenet"
UP3="$BASE --scale 3 --crop 1536 --bs 2 --iters 16000"
# 4x probe: same crop/batch as 3x (memory-bound), so the original-pixel field shrinks 512 -> 384
UP4="$BASE --scale 4 --crop 1536 --bs 2 --iters 16000"

run r5_up3_seed1 $UP3 --seed 1
run r5_up3_seed2 $UP3 --seed 2
run r5_up4 $UP4

ens up3_s01      $R/r4_up3 $R/r5_up3_seed1
ens up3_s012     $R/r4_up3 $R/r5_up3_seed1 $R/r5_up3_seed2
ens up3_s012_up4 $R/r4_up3 $R/r5_up3_seed1 $R/r5_up3_seed2 $R/r5_up4
echo "ALL DONE $(date)"
