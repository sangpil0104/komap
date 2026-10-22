#!/usr/bin/env bash
# Repeat one training config over several seeds and report the spread.
# Portable: no server-specific paths, no GPU-sharing gate by default.
#
#   ./run_seeds.sh                      # per-image magnification (target_width 4000), seeds 2 3 4
#   SEEDS="2 3" ./run_seeds.sh          # pick seeds
#   DATA=/path/to/KoMaP ./run_seeds.sh  # dataset elsewhere
#   CONFIG=fixed3x ./run_seeds.sh       # the 3x-fixed baseline instead
#
# Dataset layout expected under $DATA:  train/{images,masks}  valid/{images,masks}  test/images
set -eu

DATA=${DATA:-$HOME/komap/data}
OUT=${OUT:-./runs}
PY=${PY:-python}
SEEDS=${SEEDS:-"2 3 4"}
CONFIG=${CONFIG:-tw4000}
CROP=${CROP:-1536}
BS=${BS:-2}
ITERS=${ITERS:-16000}
WAIT_GPU=${WAIT_GPU:-0}        # 1 = wait until the GPU is idle before each run (shared machines)
CODE=$(cd "$(dirname "$0")" && pwd)

case "$CONFIG" in
  tw4000) SCALE_ARGS="--target_width 4000" ;;          # per-image magnification normalization
  tw5000) SCALE_ARGS="--target_width 5000" ;;
  fixed3x) SCALE_ARGS="--scale 3" ;;                   # the previous best fixed-scale setting
  *) echo "unknown CONFIG=$CONFIG (tw4000 | tw5000 | fixed3x)"; exit 1 ;;
esac

[ -d "$DATA/train/images" ] || { echo "dataset not found at $DATA"; exit 1; }
$PY - <<'EOF' || { echo "environment incomplete -- see README (pip install -r requirements.txt)"; exit 1; }
import torch, segmentation_models_pytorch, albumentations, cv2
assert torch.cuda.is_available(), "no CUDA device"
free, total = torch.cuda.mem_get_info()
print(f"GPU {torch.cuda.get_device_name(0)}  free {free/2**30:.1f}GB / {total/2**30:.1f}GB")
if total / 2**30 < 15:
    print("WARNING: crop 1536 batch 2 needs ~12.7GB. With less VRAM, lower CROP (e.g. 1024) -- "
          "but then results are NOT comparable to the 1536 runs. Do not drop BS to 1: "
          "BatchNorm with batch 1 broke a previous experiment (79 -> 71).")
EOF

wait_for_gpu() {
  [ "$WAIT_GPU" = "1" ] || return 0
  local ok=0
  while [ $ok -lt 12 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -lt 2000 ]; then ok=$((ok+1)); else ok=0; fi
    sleep 10
  done
}

mkdir -p "$OUT"
for s in $SEEDS; do
  name=${CONFIG}_seed${s}
  if [ -f "$OUT/$name/last.pt" ]; then echo "=== $name already done"; continue; fi
  wait_for_gpu
  echo "=== $name  $(date)"
  $PY "$CODE/train.py" --data "$DATA" --out "$OUT/$name" \
      --arch unet --encoder efficientnet-b4 --norm imagenet \
      $SCALE_ARGS --crop "$CROP" --bs "$BS" --iters "$ITERS" --seed "$s" \
      2>&1 | tee "$OUT/$name.out" | grep -E "valid|Error|Traceback" || true
done

echo
$PY "$CODE/summarize.py" --runs "$OUT"/*/
