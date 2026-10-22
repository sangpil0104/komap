#!/bin/bash
# Waits until the GPU has had no compute processes for 5 consecutive minutes, then runs round-1 experiments.
cd ~/komap/code
PY=~/komap/venv/bin/python
idle=0
while [ $idle -lt 30 ]; do
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then idle=$((idle+1)); else idle=0; fi
  sleep 10
done
echo "GPU idle, start $(date)"

run() { name=$1; shift; echo "=== $name $(date)"; $PY train.py --out ~/komap/runs/$name "$@" > ~/komap/runs/$name.out 2>&1; tail -1 ~/komap/runs/$name.out; }

run r1_unet_cnxs_imnet --arch unet --encoder tu-convnext_small --norm imagenet
run r1_unet_cnxs_imgnorm --arch unet --encoder tu-convnext_small --norm image
run r1_unet_effb4_imnet --arch unet --encoder efficientnet-b4 --norm imagenet
echo "ALL DONE $(date)"
