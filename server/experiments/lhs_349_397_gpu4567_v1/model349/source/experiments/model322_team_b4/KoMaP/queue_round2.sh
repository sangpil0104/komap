#!/bin/bash
# Round 2 -> pre-committed ensemble rule -> retrain members on train+valid -> test predictions.
# Before every GPU job, waits until GPU memory use stays below 2000MB for 2 minutes
# (the same threshold mission-3's runner.py uses, so the two never start on top of each other).
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

BASE="--arch unet --encoder efficientnet-b4 --norm imagenet"
declare -A CFG
CFG[r1_unet_effb4_imnet]="$BASE"
CFG[r2_effb4_crop768_long]="$BASE --crop 768 --bs 6 --iters 16000"
CFG[r2_effb4_up2]="$BASE --scale 2 --crop 1024 --bs 4 --iters 12000"
CFG[r2_cnxb]="--arch unet --encoder tu-convnext_base --norm imagenet"

# ---- round 2
for n in r2_effb4_crop768_long r2_effb4_up2 r2_cnxb; do run $n ${CFG[$n]}; done

# ---- pre-committed rule: baseline + every round-2 model with valid mIoU >= baseline - 1
MEMBERS=$($PY - <<'EOF'
import json, os
R = os.path.expanduser("~/komap/runs")
base = json.load(open(f"{R}/r1_unet_effb4_imnet/valid.json"))["miou_tta"]
keep = ["r1_unet_effb4_imnet"]
for n in ["r2_effb4_crop768_long", "r2_effb4_up2", "r2_cnxb"]:
    f = f"{R}/{n}/valid.json"
    if os.path.exists(f) and json.load(open(f))["miou_tta"] >= base - 1:
        keep.append(n)
print(" ".join(keep))
EOF
)
echo "ensemble members: $MEMBERS"

wait_gpu
echo "=== valid ensemble $(date)"
$PY predict.py --split valid --runs $(for n in $MEMBERS; do echo -n "$R/$n "; done) --out $R/ens_valid > $R/ens_valid.out 2>&1
tail -1 $R/ens_valid.out

# ---- final: retrain members on train+valid, predict test
for n in $MEMBERS; do run full_$n ${CFG[$n]} --train_splits train,valid; done
wait_gpu
echo "=== test prediction $(date)"
$PY predict.py --split test --runs $(for n in $MEMBERS; do echo -n "$R/full_$n "; done) --out ~/komap/submission > $R/submission.out 2>&1
ls ~/komap/submission | wc -l
echo "ALL DONE $(date)"
