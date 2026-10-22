#!/bin/bash
# Round 4: (1) push resolution further (3x), (2) more seeds of the current best (2x) for ensembling.
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
UP2="$BASE --scale 2 --crop 1024 --bs 4 --iters 12000"
# 3x: crop 1536 keeps the same 512-original-pixel field of view as UP2; bs 2 to fit 16GB
UP3="$BASE --scale 3 --crop 1536 --bs 2 --iters 16000"

run r4_up3 $UP3
run r4_up2_seed2 $UP2 --seed 2
run r4_up2_seed3 $UP2 --seed 3

ens up2_s01   $R/r2_effb4_up2 $R/r3_up2_seed1
ens up2_s012  $R/r2_effb4_up2 $R/r3_up2_seed1 $R/r4_up2_seed2
ens up2_s0123 $R/r2_effb4_up2 $R/r3_up2_seed1 $R/r4_up2_seed2 $R/r4_up2_seed3
ens up2x4_up3 $R/r2_effb4_up2 $R/r3_up2_seed1 $R/r4_up2_seed2 $R/r4_up2_seed3 $R/r4_up3
echo "ALL DONE $(date)"
