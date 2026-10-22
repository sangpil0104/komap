#!/bin/bash
# Round 6: is 4x limited by resolution or by the shrunken field of view? Plus 4x seeds, EMA, scale-TTA.
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

ens() {  # ens <name> <predict.py args...>
  name=$1; shift
  wait_gpu
  echo "=== eval $name $(date)"
  $PY predict.py --split valid --out $R/ens_$name "$@" > $R/ens_$name.out 2>&1
  tail -1 $R/ens_$name.out
}

BASE="--arch unet --encoder efficientnet-b4 --norm imagenet --ema 0.999"
UP4="$BASE --scale 4 --crop 1536 --bs 2 --iters 16000"
# same 512-original-pixel field as the 3x runs: crop 2048 at 4x, bs 1 with accumulation to keep
# the effective batch at 2 and the same number of optimizer steps
UP4_FOV="$BASE --scale 4 --crop 2048 --bs 1 --accum 2 --iters 32000"

# free experiments first: multi-scale TTA on models we already have
ens tta_up3s1_ms --runs $R/r5_up3_seed1 --scales 0.9,1,1.1
ens tta_up4_ms   --runs $R/r5_up4      --scales 0.9,1,1.1

run r6_up4_fov512 $UP4_FOV
run r6_up4_seed1  $UP4 --seed 1
run r6_up4_seed2  $UP4 --seed 2

ens up4_s012     --runs $R/r5_up4 $R/r6_up4_seed1 $R/r6_up4_seed2
ens up4_s012_ema --runs $R/r6_up4_seed1 $R/r6_up4_seed2 --weights ema.pt
ens up4x3_up3x3  --runs $R/r5_up4 $R/r6_up4_seed1 $R/r6_up4_seed2 $R/r4_up3 $R/r5_up3_seed1 $R/r5_up3_seed2
echo "ALL DONE $(date)"
